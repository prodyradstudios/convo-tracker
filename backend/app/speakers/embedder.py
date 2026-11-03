"""Speaker embedders: ECAPA (SpeechBrain, lazy) and a numpy-only MFCC fallback."""

from __future__ import annotations

import importlib.util
import logging
import threading
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.fft import dct, rfft

from ..audio.io import SAMPLE_RATE

log = logging.getLogger(__name__)


@runtime_checkable
class Embedder(Protocol):
    """Maps 16 kHz float32 mono audio to an L2-normalised 1-D float32 vector."""

    name: str
    default_threshold: float

    def embed(self, audio: np.ndarray) -> np.ndarray: ...


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Return ``vec`` scaled to unit length (zeros stay zeros)."""
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 1e-12 else np.zeros_like(vec)


def embed_windows(
    embedder: Embedder, audio: np.ndarray, win_s: float = 3.0, hop_s: float = 1.5
) -> np.ndarray:
    """Embed ``audio`` in overlapping windows, average and re-normalise (enrollment)."""
    win, hop = int(win_s * SAMPLE_RATE), int(hop_s * SAMPLE_RATE)
    if audio.size <= win:
        return l2_normalize(embedder.embed(audio))
    starts = list(range(0, audio.size - win + 1, hop))
    if starts[-1] + win < audio.size:
        starts.append(audio.size - win)  # cover the tail
    embs = [embedder.embed(audio[s : s + win]) for s in starts]
    return l2_normalize(np.mean(embs, axis=0))


# --------------------------------------------------------------------------
# MFCC fallback
# --------------------------------------------------------------------------
def _mel_filterbank(n_mels: int, n_fft: int, sr: int, fmin: float, fmax: float) -> np.ndarray:
    def hz_to_mel(f: np.ndarray) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mels = np.linspace(hz_to_mel(np.array(fmin)), hz_to_mel(np.array(fmax)), n_mels + 2)
    bins = np.fft.rfftfreq(n_fft, 1.0 / sr)
    hz = mel_to_hz(mels)
    fb = np.zeros((n_mels, bins.size), dtype=np.float64)
    for i in range(n_mels):
        lo, mid, hi = hz[i], hz[i + 1], hz[i + 2]
        up = (bins - lo) / (mid - lo)
        down = (hi - bins) / (hi - mid)
        fb[i] = np.maximum(0.0, np.minimum(up, down))
    return fb


def _deltas(feats: np.ndarray, width: int = 2) -> np.ndarray:
    """Regression deltas along time (axis 0)."""
    padded = np.pad(feats, ((width, width), (0, 0)), mode="edge")
    n = feats.shape[0]
    num = sum(k * (padded[width + k : width + k + n] - padded[width - k : width - k + n])
              for k in range(1, width + 1))
    return num / (2.0 * sum(k * k for k in range(1, width + 1)))


class MfccEmbedder:
    """Weak but dependency-free embedder: pooled MFCC statistics (~80-d).

    40 log-mel bands -> 20 MFCCs (c1..c20) + deltas; mean and std pooled over
    voiced frames. The pooled vector is centred (mean removed across its
    dimensions) and L2-normalised, so cosine similarity behaves like a
    correlation. Intended for tests/offline use only.
    """

    name = "mfcc"
    default_threshold = 0.80

    def __init__(self, n_mels: int = 40, n_mfcc: int = 20) -> None:
        self.n_mfcc = n_mfcc
        self.frame, self.hop, self.n_fft = 400, 160, 512  # 25 ms / 10 ms
        self._fb = _mel_filterbank(n_mels, self.n_fft, SAMPLE_RATE, 20.0, 7600.0)
        self._window = np.hamming(self.frame)

    @property
    def dim(self) -> int:
        return 4 * self.n_mfcc

    def _mfcc(self, audio: np.ndarray) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float64).reshape(-1)
        if audio.size < self.frame:
            audio = np.pad(audio, (0, self.frame - audio.size))
        emphasized = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
        n = 1 + (emphasized.size - self.frame) // self.hop
        idx = np.arange(self.frame)[None, :] + self.hop * np.arange(n)[:, None]
        frames = emphasized[idx] * self._window
        power = np.abs(rfft(frames, n=self.n_fft, axis=1)) ** 2
        energy = power.sum(axis=1)
        # Keep frames within 40 dB of the loudest one (drop silence).
        log_e = 10.0 * np.log10(energy + 1e-12)
        voiced = log_e > log_e.max() - 40.0
        logmel = np.log(power[voiced] @ self._fb.T + 1e-10)
        return dct(logmel, type=2, axis=1, norm="ortho")[:, 1 : self.n_mfcc + 1]

    def embed(self, audio: np.ndarray) -> np.ndarray:
        mfcc = self._mfcc(audio)
        feats = np.hstack([mfcc, _deltas(mfcc)])
        pooled = np.concatenate([feats.mean(axis=0), feats.std(axis=0)])
        pooled -= pooled.mean()
        return l2_normalize(pooled)


# --------------------------------------------------------------------------
# ECAPA (SpeechBrain)
# --------------------------------------------------------------------------
class EcapaEmbedder:
    """SpeechBrain ``spkrec-ecapa-voxceleb`` (192-d). Imports torch lazily."""

    name = "ecapa"
    default_threshold = 0.30
    source = "speechbrain/spkrec-ecapa-voxceleb"

    def __init__(self, savedir: Path, device: str = "cpu") -> None:
        import torch  # noqa: F401  (fail early if missing)

        try:  # SpeechBrain >= 1.0
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:  # SpeechBrain 0.5.x
            from speechbrain.pretrained import EncoderClassifier  # type: ignore

        self._model = EncoderClassifier.from_hparams(
            source=self.source, savedir=str(savedir), run_opts={"device": device}
        )
        self._model.eval()
        self._lock = threading.Lock()

    def embed(self, audio: np.ndarray) -> np.ndarray:
        import torch

        wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).unsqueeze(0)
        with self._lock, torch.no_grad():
            emb = self._model.encode_batch(wav)
        return l2_normalize(emb.squeeze().cpu().numpy())


def _ecapa_available() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "speechbrain"))


def build_embedder(kind: str, data_dir: Path, device: str = "cpu") -> Embedder:
    """Create the embedder selected by ``CT_EMBEDDER`` (auto|ecapa|mfcc)."""
    if kind == "mfcc":
        return MfccEmbedder()
    savedir = Path(data_dir) / "models" / "spkrec-ecapa-voxceleb"
    if kind == "ecapa":
        return EcapaEmbedder(savedir, device)
    if not _ecapa_available():
        log.warning("speechbrain/torch not installed; falling back to the weak MFCC embedder")
        return MfccEmbedder()
    try:
        return EcapaEmbedder(savedir, device)
    except Exception:
        log.warning("could not load ECAPA model; falling back to MFCC embedder", exc_info=True)
        return MfccEmbedder()
