"""Speech-to-text: faster-whisper (lazy import) or a null placeholder."""

from __future__ import annotations

import importlib.util
import logging
import threading
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Protocol, runtime_checkable

import numpy as np

from ..audio.io import SAMPLE_RATE

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Word:
    """One transcribed word; ``start``/``end`` are seconds from the start of the clip."""

    text: str
    start: float
    end: float


@runtime_checkable
class Transcriber(Protocol):
    """Turns 16 kHz float32 mono audio into text.

    ``transcribe_words`` returns the words with timestamps, ``[]`` when nothing
    was said, or ``None`` when this transcriber has no word timing (the caller
    then cuts the audio itself). Transcribers without the method count as None.
    """

    name: str

    def transcribe(self, audio: np.ndarray) -> str: ...

    def transcribe_words(self, audio: np.ndarray) -> list[Word] | None: ...


def _unspaced(ch: str) -> bool:
    """Characters of scripts written without spaces between words (CJK, Thai, ...)."""
    if unicodedata.east_asian_width(ch) in ("W", "F"):
        return True
    return unicodedata.name(ch, "").startswith(("THAI", "LAO", "KHMER", "MYANMAR", "TIBETAN"))


def join_text(parts: Iterable[str]) -> str:
    """Concatenate pieces of text the way Whisper spaces its words.

    Whisper words carry their own leading space (" Hello", " there.") and CJK
    words carry none, so pieces are concatenated as they are. A space is added
    only where neither side has whitespace and neither boundary character
    belongs to an unspaced script (words from transcribers that strip spaces).
    """
    out = ""
    for part in parts:
        if not part:
            continue
        if out and not (
            out[-1].isspace() or part[0].isspace() or _unspaced(out[-1]) or _unspaced(part[0])
        ):
            out += " "
        out += part
    return out


def words_to_text(words: Iterable[Word]) -> str:
    """Join words into a line of text (Whisper words carry their own spacing)."""
    return join_text(w.text for w in words).strip()


class NullTranscriber:
    """Placeholder used when faster-whisper is unavailable (and in tests)."""

    name = "null"

    def transcribe(self, audio: np.ndarray) -> str:
        return f"[speech {len(audio) / SAMPLE_RATE:.1f}s]"

    def transcribe_words(self, audio: np.ndarray) -> None:
        return None  # no word timing: callers fall back to cutting the audio


class WhisperTranscriber:
    """faster-whisper model, loaded once. Calls are serialised with a lock."""

    name = "whisper"

    def __init__(self, model: str = "base.en", device: str = "cpu") -> None:
        from faster_whisper import WhisperModel

        compute_type = "int8" if device == "cpu" else "float16"
        self._model = WhisperModel(model, device=device, compute_type=compute_type)
        self._language = "en" if model.endswith(".en") else None
        self._lock = threading.Lock()

    def _segments(self, audio: np.ndarray, word_timestamps: bool) -> list:
        with self._lock:
            segments, _info = self._model.transcribe(
                np.ascontiguousarray(audio, dtype=np.float32),
                language=self._language,
                beam_size=1,
                vad_filter=False,
                condition_on_previous_text=False,
                word_timestamps=word_timestamps,
            )
            # ``segments`` is a lazy generator: consume it while holding the lock.
            return list(segments)

    def transcribe(self, audio: np.ndarray) -> str:
        segments = self._segments(audio, word_timestamps=False)
        return " ".join(s.text.strip() for s in segments).strip()

    def transcribe_words(self, audio: np.ndarray) -> list[Word] | None:
        """Words with timestamps; ``[]`` if Whisper heard nothing.

        None (no timing) if a segment has text but no words, so the caller
        falls back to cutting the audio rather than losing that text.
        """
        words: list[Word] = []
        for seg in self._segments(audio, word_timestamps=True):
            seg_words = getattr(seg, "words", None)
            if not seg_words:
                if (getattr(seg, "text", "") or "").strip():
                    return None
                continue
            for w in seg_words:
                if w.word.strip():
                    words.append(Word(w.word, float(w.start), float(w.end)))
        return words


def _whisper_available() -> bool:
    return importlib.util.find_spec("faster_whisper") is not None


def build_transcriber(kind: str, model: str = "base.en", device: str = "cpu") -> Transcriber:
    """Create the transcriber selected by ``CT_TRANSCRIBER`` (auto|whisper|null)."""
    if kind == "null":
        return NullTranscriber()
    if kind == "whisper":
        return WhisperTranscriber(model, device)
    if not _whisper_available():
        log.warning("faster-whisper not installed; transcripts will be placeholders")
        return NullTranscriber()
    try:
        return WhisperTranscriber(model, device)
    except Exception:
        log.warning("could not load Whisper model; using null transcriber", exc_info=True)
        return NullTranscriber()
