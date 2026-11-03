"""Per-session processing: VAD -> turns -> embed -> identify -> transcribe -> Utterances."""

from __future__ import annotations

import math
import uuid
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .audio.io import SAMPLE_RATE
from .audio.vad import Segment, Segmenter, VadConfig
from .speakers.embedder import Embedder
from .speakers.registry import Match, SpeakerRegistry
from .speakers.turns import Turn, TurnAnalysis, TurnConfig, analyze_turns
from .transcription.transcriber import Transcriber, Word, join_text

# Pieces at least this long are re-identified from their own audio; shorter
# ones keep the label of the turn they came from.
PIECE_REIDENTIFY_S = 1.0
_MIN_PIECE_S = 1e-3  # pieces shorter than this (after rounding to ms) take their turn's span


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
    text: str  # raw: Whisper-style leading space kept, so pieces concatenate cleanly
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
        analysis = self._analyze(segment)
        if analysis is not None and len(analysis.turns) > 1:
            pieces, false_split = self._split(segment, analysis.turns)
            if not false_split:
                t0 = segment.start
                return [
                    self._emit(t0 + p.start, t0 + p.end, p.text.strip(), p.match) for p in pieces
                ]
            # Every turn turned out to be the same person: behave as if unsplit.
            return self._whole(segment, analysis.embedding, pieces[0].text, adapt=False)
        embedding = analysis.embedding if analysis is not None else None
        # Adapt only from segments where no window/chunk hinted at a second voice.
        clean = analysis is None or analysis.clean
        return self._whole(segment, embedding, adapt=clean)

    def _analyze(self, segment: Segment) -> TurnAnalysis | None:
        """Turn analysis, or None when splitting does not apply (short / < 2 enrolled)."""
        if segment.speech_seconds < self.turn_config.min_segment_s or len(self.registry) < 2:
            return None
        return analyze_turns(
            segment.audio,
            self.embedder,
            self.registry,
            self.turn_config,
            self.vad_config,
            segment.noise_floor_db,
        )

    def _whole(
        self,
        segment: Segment,
        embedding: np.ndarray | None = None,
        text: str | None = None,
        adapt: bool = True,
    ) -> list[Utterance]:
        """One utterance for the whole segment.

        Identity from ``embedding`` (the turn analysis' chunk mean) when given,
        else from an embedding of the full segment audio.
        """
        if embedding is None:
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
        merged them all back into one speaker (a false split). Pieces are never
        used for adaptation: they may hold the boundary mixture of two voices.
        """
        audio = segment.audio
        words = self._words(audio)
        if words is not None:  # [] = nothing was said: no pieces
            pieces = _pieces_from_words(words, turns, audio.size / SAMPLE_RATE)
        else:  # no word timing: cut the audio at the turn boundaries
            pieces = []
            for t in turns:
                text = self.transcriber.transcribe(_cut(audio, t.start, t.end)).strip()
                pieces.append(_Piece(t.start, t.end, f" {text}" if text else "", t))
        complete = len(pieces) == len(turns) and all(p.text.strip() for p in pieces)
        pieces = [p for p in pieces if p.text.strip()]
        previous = self._last_speaker
        for p in pieces:
            if p.duration >= PIECE_REIDENTIFY_S:
                embedding = self.embedder.embed(_cut(audio, p.start, p.end))
                p.match = self.registry.identify(embedding, p.duration, previous)
            else:
                p.match = self.registry.match_for(p.turn.speaker_id, p.turn.score)
            if p.match.speaker_id is not None:
                previous = p.match.speaker_id
        merged = _merge_same_speaker(pieces)
        return merged, complete and len(merged) == 1

    def _words(self, audio: np.ndarray) -> list[Word] | None:
        """Timed words, ``[]`` if nothing was said, None if the transcriber has no timing."""
        transcribe_words = getattr(self.transcriber, "transcribe_words", None)
        if not callable(transcribe_words):
            return None
        words = transcribe_words(audio)
        return None if words is None else list(words)

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


def _timed_words(words: list[Word]) -> list[tuple[float, float, Word]]:
    """(start, end, word) with finite, ordered times, sorted by midpoint.

    A word with one non-finite time uses the other; with none it borrows the
    time of the word before it (or after it) in the transcriber's order, so its
    text is kept rather than dropped.
    """
    times: list[tuple[float, float] | None] = []
    for w in words:
        ends = [t for t in (w.start, w.end) if math.isfinite(t)]
        times.append((min(ends), max(ends)) if ends else None)
    known = [i for i, t in enumerate(times) if t is not None]
    if not known:
        return [(0.0, 0.0, w) for w in words]
    out = []
    for i, w in enumerate(words):
        t = times[i]
        if t is None:
            j = max((k for k in known if k < i), default=min(known))
            prev = times[j]
            assert prev is not None
            t = (prev[1], prev[1])
        out.append((t[0], t[1], w))
    return sorted(out, key=lambda x: (x[0] + x[1]) / 2.0)


def _pieces_from_words(words: list[Word], turns: list[Turn], duration: float) -> list[_Piece]:
    """Assign each word to the turn holding its midpoint; group consecutive words.

    Pieces come out in time order, inside the clip, non-overlapping and with
    ``start < end``: a piece spans its first to last word (clamped to the clip);
    where two pieces' words overlap they are cut at the turn boundary; a piece
    left with no length takes its turn's span (clipped to its neighbours) or, if
    even that is empty, joins the previous piece.
    """
    groups: list[list] = []  # [turn, words, start, end]
    for start, end, w in _timed_words(words):
        start, end = min(max(start, 0.0), duration), min(max(end, 0.0), duration)
        mid = (start + end) / 2.0
        turn = next((t for t in turns if mid < t.end), turns[-1])
        if groups and groups[-1][0] is turn:
            g = groups[-1]
            g[1].append(w)
            g[2], g[3] = min(g[2], start), max(g[3], end)
        else:
            groups.append([turn, [w], start, end])
    for prev, cur in zip(groups, groups[1:]):
        if cur[2] < prev[3]:  # overlapping words: cut at the turn boundary
            cut = min(max(cur[0].start, prev[2]), cur[3])
            prev[3], cur[2] = min(prev[3], cut), max(cur[2], cut)
    i = 0
    while i < len(groups):
        g = groups[i]
        if g[3] - g[2] < _MIN_PIECE_S:
            lo = groups[i - 1][3] if i > 0 else 0.0
            hi = groups[i + 1][2] if i + 1 < len(groups) else duration
            start, end = max(g[0].start, lo), min(g[0].end, hi)
            if end - start >= _MIN_PIECE_S:
                g[2], g[3] = start, end
            elif len(groups) > 1:
                other = groups[i - 1] if i > 0 else groups[i + 1]
                other[1] = [*other[1], *g[1]] if i > 0 else [*g[1], *other[1]]
                other[2], other[3] = min(other[2], g[2]), max(other[3], g[3])
                del groups[i]
                continue
        i += 1
    return [_Piece(g[2], g[3], join_text(w.text for w in g[1]), g[0]) for g in groups]


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
                join_text([last.text, p.text]),
                last.turn,
                Match(last.match.speaker_id, last.match.speaker_name, last.match.color,
                      round(conf, 3)),
            )
        else:
            merged.append(p)
    return merged
