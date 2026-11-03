"""Per-session processing: VAD -> turns -> embed -> identify -> transcribe -> Utterances."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .audio.io import SAMPLE_RATE
from .audio.vad import Segment, Segmenter, VadConfig
from .speakers.embedder import Embedder
from .speakers.registry import Match, SpeakerRegistry
from .speakers.turns import Turn, TurnConfig, detect_turns
from .transcription.transcriber import Transcriber, Word, words_to_text

# Pieces at least this long are re-identified from their own audio; shorter
# ones keep the label of the turn they came from.
PIECE_REIDENTIFY_S = 1.0


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


@dataclass
class _Piece:
    """Part of a segment spoken by one person; times relative to the segment audio."""

    start: float
    end: float
    text: str
    turn: Turn
    match: Match | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start


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
        turn_config: TurnConfig | None = None,
    ) -> None:
        self.embedder = embedder
        self.transcriber = transcriber
        self.registry = registry
        self.segmenter = Segmenter(vad_config)
        self.vad_config = self.segmenter.config
        self.turn_config = turn_config or TurnConfig()
        self._seq = 0
        self._last_speaker: str | None = None

    def feed(self, samples: np.ndarray) -> list[Segment]:
        return self.segmenter.feed(samples)

    def flush(self) -> list[Segment]:
        return self.segmenter.flush()

    def pop_events(self) -> list[bool]:
        return self.segmenter.pop_events()

    # -- processing -----------------------------------------------------------
    def process(self, segment: Segment) -> list[Utterance]:
        """Identify and transcribe a segment: one utterance per speaker turn, in order.

        Empty transcripts produce no utterance (and consume no ``seq``).
        """
        turns = self._turns(segment)
        if len(turns) > 1:
            pieces, false_split = self._split(segment, turns)
            if not false_split:
                t0 = segment.start
                return [self._emit(t0 + p.start, t0 + p.end, p.text, p.match) for p in pieces]
            # Every turn turned out to be the same person: behave as if unsplit.
            return self._whole(segment, pieces[0].text, adapt=False)
        return self._whole(segment)

    def _turns(self, segment: Segment) -> list[Turn]:
        if segment.speech_seconds < self.turn_config.min_segment_s or len(self.registry) < 2:
            return []
        return detect_turns(
            segment.audio, self.embedder, self.registry, self.turn_config, self.vad_config
        )

    def _whole(
        self, segment: Segment, text: str | None = None, adapt: bool = True
    ) -> list[Utterance]:
        """One utterance for the whole segment (identity from its full embedding)."""
        embedding = self.embedder.embed(segment.audio)
        if text is None:
            text = self.transcriber.transcribe(segment.audio)
        text = text.strip()
        if not text:
            return []
        # Continuity smoothing is about how long the person spoke, not the padded clip.
        match = self.registry.identify(
            embedding, segment.speech_seconds, self._last_speaker, adapt=adapt
        )
        return [self._emit(segment.start, segment.end, text, match)]

    def _split(self, segment: Segment, turns: list[Turn]) -> tuple[list[_Piece], bool]:
        """Per-speaker pieces of a multi-turn segment, identified and merged.

        The flag is True when every turn produced text and re-identification
        merged them all back into one speaker (a false split).
        """
        audio = segment.audio
        words = self._words(audio)
        if words:
            pieces = _pieces_from_words(words, turns, audio.size / SAMPLE_RATE)
        else:  # no word timing: cut the audio at the turn boundaries
            pieces = [
                _Piece(t.start, t.end, self.transcriber.transcribe(_cut(audio, t.start, t.end)), t)
                for t in turns
            ]
        complete = len(pieces) == len(turns) and all(p.text.strip() for p in pieces)
        pieces = [p for p in pieces if p.text.strip()]
        previous = self._last_speaker
        for p in pieces:
            p.text = p.text.strip()
            if p.duration >= PIECE_REIDENTIFY_S:
                embedding = self.embedder.embed(_cut(audio, p.start, p.end))
                p.match = self.registry.identify(embedding, p.duration, previous, adapt=True)
            else:
                p.match = self.registry.match_for(p.turn.speaker_id, p.turn.score)
            if p.match.speaker_id is not None:
                previous = p.match.speaker_id
        merged = _merge_same_speaker(pieces)
        return merged, complete and len(merged) == 1

    def _words(self, audio: np.ndarray) -> list[Word]:
        transcribe_words = getattr(self.transcriber, "transcribe_words", None)
        return list(transcribe_words(audio)) if callable(transcribe_words) else []

    def _emit(self, start: float, end: float, text: str, match: Match | None) -> Utterance:
        """Number and package one utterance; ``start``/``end`` in session seconds."""
        assert match is not None
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
            start=round(start, 3),
            end=round(end, 3),
            confidence=match.confidence,
        )


def _cut(audio: np.ndarray, start: float, end: float) -> np.ndarray:
    a = max(0, int(round(start * SAMPLE_RATE)))
    b = min(audio.size, int(round(end * SAMPLE_RATE)))
    return audio[a:max(a, b)]


def _pieces_from_words(words: list[Word], turns: list[Turn], duration: float) -> list[_Piece]:
    """Assign each word to the turn holding its midpoint; group consecutive words."""
    groups: list[tuple[Turn, list[Word]]] = []
    for w in words:
        mid = (w.start + w.end) / 2.0
        turn = next((t for t in turns if mid < t.end), turns[-1])
        if groups and groups[-1][0] is turn:
            groups[-1][1].append(w)
        else:
            groups.append((turn, [w]))
    pieces = []
    for turn, ws in groups:
        start = min(max(0.0, ws[0].start), duration)
        end = min(max(start, ws[-1].end), duration)
        pieces.append(_Piece(start, end, words_to_text(ws), turn))
    return pieces


def _merge_same_speaker(pieces: list[_Piece]) -> list[_Piece]:
    """Join neighbouring pieces attributed to the same speaker (or both Unknown)."""
    merged: list[_Piece] = []
    for p in pieces:
        assert p.match is not None
        last = merged[-1] if merged else None
        if last is not None and last.match is not None and last.match.speaker_id == p.match.speaker_id:
            total = last.duration + p.duration
            weight = (last.duration / total) if total > 0 else 0.5
            conf = weight * last.match.confidence + (1 - weight) * p.match.confidence
            merged[-1] = _Piece(
                last.start,
                p.end,
                f"{last.text} {p.text}",
                last.turn,
                Match(last.match.speaker_id, last.match.speaker_name, last.match.color,
                      round(conf, 3)),
            )
        else:
            merged.append(p)
    return merged
