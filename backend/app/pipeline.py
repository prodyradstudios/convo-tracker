"""Per-session processing: VAD -> embed -> identify -> transcribe -> Utterance."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .audio.vad import Segment, Segmenter, VadConfig
from .speakers.embedder import Embedder
from .speakers.registry import SpeakerRegistry
from .transcription.transcriber import Transcriber


@dataclass
class Utterance:
    id: str
    seq: int
    speaker_id: str | None
    speaker_name: str
    color: str
    text: str
    start: float
    end: float
    confidence: float

    def to_message(self) -> dict[str, Any]:
        return {"type": "utterance", **asdict(self)}


class SessionPipeline:
    """State of one listening session.

    :meth:`feed`/:meth:`flush` are cheap and run on the event loop;
    :meth:`process` is heavy and must be called from a worker thread, one
    segment at a time, in order.
    """

    def __init__(
        self,
        embedder: Embedder,
        transcriber: Transcriber,
        registry: SpeakerRegistry,
        vad_config: VadConfig | None = None,
    ) -> None:
        self.embedder = embedder
        self.transcriber = transcriber
        self.registry = registry
        self.segmenter = Segmenter(vad_config)
        self._seq = 0
        self._last_speaker: str | None = None

    def feed(self, samples: np.ndarray) -> list[Segment]:
        return self.segmenter.feed(samples)

    def flush(self) -> list[Segment]:
        return self.segmenter.flush()

    def pop_events(self) -> list[bool]:
        return self.segmenter.pop_events()

    def process(self, segment: Segment) -> Utterance | None:
        """Identify and transcribe a segment; None if the transcript is empty."""
        embedding = self.embedder.embed(segment.audio)
        match = self.registry.identify(embedding, segment.duration, self._last_speaker)
        text = self.transcriber.transcribe(segment.audio).strip()
        if not text:
            return None
        if match.speaker_id is not None:
            self._last_speaker = match.speaker_id
        self._seq += 1
        return Utterance(
            id=uuid.uuid4().hex,
            seq=self._seq,
            speaker_id=match.speaker_id,
            speaker_name=match.speaker_name,
            color=match.color,
            text=text,
            start=round(segment.start, 3),
            end=round(segment.end, 3),
            confidence=match.confidence,
        )
