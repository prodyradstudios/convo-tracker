"""Bounded online change detector; short-window evidence never adapts voiceprints.

A cheap MFCC novelty gate avoids running the full voice model on every hop of
an unchanged voice. Confident enrolled identity changes (or a strongly changed,
unmatched voice) require sustained evidence before splitting the active audio.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..audio.io import SAMPLE_RATE
from ..audio.vad import Segment, VadConfig, frame_db, speech_threshold_db
from .embedder import Embedder, MfccEmbedder, l2_normalize
from .registry import SpeakerRegistry
from .turns import TurnConfig


@dataclass(frozen=True)
class SpeakerChange:
    start: float
    speaker_id: str | None
    score: float
    initial: bool = False


class StreamingTurnDetector:
    """Observe active snapshots in stream order, at most one probe per hop.

    The caller feeds the VAD in hop-sized chunks, including when a WebSocket
    packet holds many seconds. All state is constant-sized. Model work belongs
    in a worker thread; ECAPA serializes inference through its own model lock.
    """

    def __init__(self, embedder: Embedder, registry: SpeakerRegistry,
                 config: TurnConfig, vad: VadConfig) -> None:
        self.embedder, self.registry, self.config, self.vad = embedder, registry, config, vad
        self._cheap = MfccEmbedder()
        self.reset()

    def reset(self) -> None:
        self._next = 0.0
        self._last_probe = -math.inf
        self._identity: str | None = None
        self._baseline: np.ndarray | None = None
        self._fingerprint: np.ndarray | None = None
        self._candidate: tuple[str | None, float, float, int] | None = None
        self._candidate_embedding: np.ndarray | None = None

    def observe(self, segment: Segment) -> SpeakerChange | None:
        c = self.config
        win = max(1, int(c.change_window_s * SAMPLE_RATE))
        if segment.audio.size < win or segment.end + 1e-6 < self._next:
            return None
        self._next = segment.end + c.change_hop_s - self.vad.frame_ms / 1000 - 1e-6
        clip = segment.audio[-win:]
        flags = [frame_db(clip[i:i + self.vad.frame_len]) >
                 speech_threshold_db(self.vad, segment.noise_floor_db)
                 for i in range(0, clip.size, self.vad.frame_len)]
        if np.mean(flags) < 0.8:
            self._candidate = None
            return None
        fingerprint = self._cheap.embed(clip)
        if self._fingerprint is not None and self._candidate is None:
            novelty = 1.0 - float(np.clip(fingerprint @ self._fingerprint, -1, 1))
            if novelty < c.change_threshold and segment.end - self._last_probe < 0.3:
                self._candidate = None
                return None
        self._last_probe = segment.end
        embedding = l2_normalize(self.embedder.embed(clip))
        table = self.registry.score(embedding[None, :], adaptive=False)
        if not table.speaker_ids:
            self._candidate = None
            return None
        row = table.scores[0]
        order = np.argsort(row)
        best = int(order[-1])
        score = float(row[best])
        runner = float(row[order[-2]]) if len(order) > 1 else -1.0
        known = score >= table.threshold and score - runner >= c.change_margin
        # Ambiguous matches are not evidence for Unknown. A below-threshold
        # voice may be Unknown only when it also differs strongly acoustically.
        if not known and score >= table.threshold - c.change_margin:
            self._candidate = None
            return None
        identity = table.speaker_ids[best] if known else None
        if self._baseline is None:
            self._identity, self._baseline, self._fingerprint = identity, embedding, fingerprint
            return SpeakerChange(max(segment.start, segment.end - c.change_window_s),
                                 identity, score, initial=True)
        distance = 1.0 - float(np.clip(embedding @ self._baseline, -1, 1))
        if identity == self._identity and (identity is not None or distance <= c.change_threshold):
            self._baseline, self._fingerprint = embedding, fingerprint
            self._candidate = None
            return None
        if distance < c.change_threshold:
            self._candidate = None
            return None
        stable = (self._candidate_embedding is not None and
                  1.0 - float(np.clip(embedding @ self._candidate_embedding, -1, 1))
                  <= c.change_threshold)
        if self._candidate is None or self._candidate[0] != identity or not stable:
            boundary = self._boundary(segment, win)
            self._candidate = (identity, boundary, segment.end, 1)
            self._candidate_embedding = embedding
            return None
        identity, boundary, first_end, count = self._candidate
        count += 1
        self._candidate = (identity, boundary, first_end, count)
        needed = max(2, math.ceil(c.change_confirm_s / c.change_hop_s))
        if count < needed or segment.end - first_end + c.change_hop_s + 1e-6 < c.change_confirm_s:
            return None
        self._identity, self._baseline, self._fingerprint = identity, embedding, fingerprint
        self._candidate = None
        return SpeakerChange(boundary, identity, score)

    def _boundary(self, segment: Segment, win: int) -> float:
        """Refine the first changed window by a gain-independent spectral step.

        Adjacent 60 ms contexts locate abrupt timbre changes more precisely than
        the centre of a mixed 400 ms identity window. The leading edge is the
        conservative fallback when the transition is outside the search range.
        """
        audio = segment.audio
        context = max(1, int(0.06 * SAMPLE_RATE))
        lo = max(context, audio.size - win - context)
        hi = audio.size - context
        if hi <= lo:
            return max(segment.start, segment.end - win / SAMPLE_RATE)
        best, best_change = lo, -1.0
        for cut in range(lo, hi + 1, self.vad.frame_len):
            spectra = []
            for piece in (audio[cut - context:cut], audio[cut:cut + context]):
                spectrum = np.abs(np.fft.rfft(piece * np.hanning(piece.size), n=2048))
                norm = float(np.linalg.norm(spectrum))
                spectra.append(spectrum / norm if norm > 1e-9 else spectrum * 0)
            change = 1.0 - float(spectra[0] @ spectra[1])
            if change > best_change:
                best, best_change = cut, change
        return segment.start + best / SAMPLE_RATE
