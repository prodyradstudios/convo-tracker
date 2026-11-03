"""Speaker embedders: ECAPA (SpeechBrain, lazy) and a numpy-only MFCC fallback."""

from __future__ import annotations

import importlib.util
import logging
import math
import threading
from fractions import Fraction
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

import numpy as np
from scipy.fft import dct, rfft
from scipy.signal import resample_poly

from ..audio.io import SAMPLE_RATE

log = logging.getLogger(__name__)


@runtime_checkable
class Embedder(Protocol):
    """Maps 16 kHz float32 mono audio to an L2-normalised 1-D float32 vector."""

    name: str
    default_threshold: float

    def embed(self, audio: np.ndarray) -> np.ndarray: ...


# Embedders may also offer ``embed_batch(list[np.ndarray]) -> np.ndarray`` (rows =
# embeddings); :func:`embed_many` uses it when present.

# Loudness normalisation (applied to every clip before it is embedded).
TARGET_DBFS = -23.0
PEAK_LIMIT = 0.99  # never push a sample past this (no clipping)
_NORM_FRAME = 480  # 30 ms at 16 kHz, like the VAD
_VOICED_RANGE_DB = 30.0  # frames within 30 dB of the loudest count as voiced
_SILENCE_RMS = 1e-5  # about -100 dBFS: treat as silence, leave untouched
_MAX_GAIN = 1000.0  # +60 dB at most

# Enrollment augmentation: speed perturbation (pitch and tempo, about -1.8 / +1.65
# semitones) and, when torchaudio is available, pure pitch shifts.
SPEED_FACTORS = (0.9, 1.1)
PITCH_STEPS = (-2.0, 2.0)


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Return ``vec`` scaled to unit length (zeros stay zeros)."""
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 1e-12 else np.zeros_like(vec)


def normalize_loudness(
    audio: np.ndarray, target_dbfs: float = TARGET_DBFS, peak: float = PEAK_LIMIT
) -> np.ndarray:
    """Scale ``audio`` so its voiced frames have an RMS of ``target_dbfs``.

    "Voiced" = 30 ms frames within 30 dB of the loudest frame, so pauses do not
    drag the level down. The gain is limited so no sample exceeds ``peak`` and
    to +60 dB; near-silent input is returned unchanged. Makes embeddings
    independent of how loudly someone spoke or how far they sat from the mic.
    """
    x = np.nan_to_num(np.asarray(audio, dtype=np.float32).reshape(-1))
    if x.size == 0:
        return x
    n = x.size // _NORM_FRAME
    if n:
        frames = x[: n * _NORM_FRAME].reshape(n, _NORM_FRAME).astype(np.float64)
        rms = np.sqrt(np.mean(frames * frames, axis=1))
    else:
        rms = np.sqrt(np.mean(np.square(x, dtype=np.float64), keepdims=True))
    loudest = float(rms.max())
    if loudest < _SILENCE_RMS:
        return x
    voiced = rms >= loudest * 10.0 ** (-_VOICED_RANGE_DB / 20.0)
    level = float(np.sqrt(np.mean(rms[voiced] ** 2)))
    gain = 10.0 ** (target_dbfs / 20.0) / level
    top = float(np.abs(x).max())
    gain = min(gain, peak / top, _MAX_GAIN)
    return (x * np.float32(gain)).astype(np.float32, copy=False)


def prepare_audio(audio: np.ndarray) -> np.ndarray:
    """Shared pre-processing of every embedder: float32 mono, loudness-normalised."""
    return normalize_loudness(audio)


def embed_many(embedder: Embedder, clips: Sequence[np.ndarray]) -> np.ndarray:
    """Embed several clips: one ``embed_batch`` call if supported, else a loop.

    Returns a ``(len(clips), dim)`` float32 array of L2-normalised rows.
    """
    if not clips:
        return np.zeros((0, 0), dtype=np.float32)
    batch = getattr(embedder, "embed_batch", None)
    if callable(batch):
        out = np.asarray(batch(list(clips)), dtype=np.float32)
    else:
        out = np.stack([np.asarray(embedder.embed(c), dtype=np.float32).reshape(-1) for c in clips])
    return np.stack([l2_normalize(row) for row in out])


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
    embs = embed_many(embedder, [audio[s : s + win] for s in starts])
    return l2_normalize(np.mean(embs, axis=0))


# --------------------------------------------------------------------------
# Enrollment augmentation
# --------------------------------------------------------------------------
def speed_perturb(audio: np.ndarray, factor: float) -> np.ndarray:
    """Play ``audio`` ``factor`` times faster: pitch and tempo both scale by ``factor``."""
    ratio = Fraction(factor).limit_denominator(100)
    # Output length = input / factor  ->  up = denominator, down = numerator.
    out = resample_poly(np.asarray(audio, dtype=np.float32), ratio.denominator, ratio.numerator)
    return out.astype(np.float32, copy=False)


def _torchaudio_available() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "torchaudio"))


def pitch_shift(audio: np.ndarray, n_steps: float, n_fft: int = 512) -> np.ndarray:
    """Shift pitch by ``n_steps`` semitones keeping the duration (needs torchaudio).

    Same algorithm as ``torchaudio.functional.pitch_shift`` (phase-vocoder time
    stretch, then resample), but the resampling step uses ``resample_poly`` with
    a small rational ratio: torchaudio's own resampler gets an awkward ratio
    such as 17959/16000 for +2 semitones and takes many seconds per call.
    """
    import torch
    import torchaudio.functional as taf

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    rate = 2.0 ** (-n_steps / 12.0)
    hop = n_fft // 4
    window = torch.hann_window(n_fft)
    wav = torch.from_numpy(np.ascontiguousarray(audio))
    with torch.no_grad():
        spec = torch.stft(wav, n_fft, hop, window=window, return_complex=True)
        advance = torch.linspace(0, math.pi * hop, spec.shape[-2])[..., None]
        stretched = taf.phase_vocoder(spec, rate, advance)
        length = int(round(audio.size / rate))
        slow = torch.istft(stretched, n_fft, hop, window=window, length=length).numpy()
    ratio = Fraction(rate).limit_denominator(64)
    out = resample_poly(slow, ratio.numerator, ratio.denominator)
    if out.size >= audio.size:
        out = out[: audio.size]
    else:
        out = np.pad(out, (0, audio.size - out.size))
    return out.astype(np.float32, copy=False)


def augmented_copies(audio: np.ndarray) -> list[np.ndarray]:
    """Perturbed copies of an enrollment recording (speed 0.9/1.1, pitch -/+2 st)."""
    copies = [speed_perturb(audio, f) for f in SPEED_FACTORS]
    if _torchaudio_available():
        for steps in PITCH_STEPS:
            try:
                copies.append(pitch_shift(audio, steps))
            except Exception:  # optional extra: never fail an enrollment over it
                log.warning("pitch-shift augmentation failed; skipping it", exc_info=True)
    return copies


def enrollment_embeddings(
    embedder: Embedder, speech: np.ndarray, augment: bool = True
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Embedding of an enrollment recording plus embeddings of its augmented copies."""
    main = embed_windows(embedder, speech)
    if not augment:
        return main, []
    extra = [embed_windows(embedder, copy) for copy in augmented_copies(speech)]
    return main, [e for e in extra if np.all(np.isfinite(e)) and np.any(e)]


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
        mfcc = self._mfcc(prepare_audio(audio))
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

        wav = torch.from_numpy(np.ascontiguousarray(prepare_audio(audio))).unsqueeze(0)
        with self._lock, torch.no_grad():
            emb = self._model.encode_batch(wav)
        return l2_normalize(emb.squeeze().cpu().numpy())

    def embed_batch(self, clips: list[np.ndarray]) -> np.ndarray:
        """Embed clips of different lengths in one forward pass (zero-padded + wav_lens)."""
        import torch

        prepared = [prepare_audio(c) for c in clips]
        longest = max(max(c.size for c in prepared), 1)
        wavs = np.zeros((len(prepared), longest), dtype=np.float32)
        for i, c in enumerate(prepared):
            wavs[i, : c.size] = c
        lens = torch.tensor([max(c.size, 1) / longest for c in prepared], dtype=torch.float32)
        with self._lock, torch.no_grad():
            emb = self._model.encode_batch(torch.from_numpy(wavs), wav_lens=lens)
        rows = emb.reshape(len(prepared), -1).cpu().numpy()
        return np.stack([l2_normalize(r) for r in rows])


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
