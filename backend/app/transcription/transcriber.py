"""Speech-to-text: faster-whisper (lazy import) or a null placeholder."""

from __future__ import annotations

import importlib.util
import logging
import threading
from typing import Protocol, runtime_checkable

import numpy as np

from ..audio.io import SAMPLE_RATE

log = logging.getLogger(__name__)


@runtime_checkable
class Transcriber(Protocol):
    """Turns 16 kHz float32 mono audio into text."""

    name: str

    def transcribe(self, audio: np.ndarray) -> str: ...


class NullTranscriber:
    """Placeholder used when faster-whisper is unavailable (and in tests)."""

    name = "null"

    def transcribe(self, audio: np.ndarray) -> str:
        return f"[speech {len(audio) / SAMPLE_RATE:.1f}s]"


class WhisperTranscriber:
    """faster-whisper model, loaded once. Calls are serialised with a lock."""

    name = "whisper"

    def __init__(self, model: str = "base.en", device: str = "cpu") -> None:
        from faster_whisper import WhisperModel

        compute_type = "int8" if device == "cpu" else "float16"
        self._model = WhisperModel(model, device=device, compute_type=compute_type)
        self._language = "en" if model.endswith(".en") else None
        self._lock = threading.Lock()

    def transcribe(self, audio: np.ndarray) -> str:
        with self._lock:
            segments, _info = self._model.transcribe(
                np.ascontiguousarray(audio, dtype=np.float32),
                language=self._language,
                beam_size=1,
                vad_filter=False,
                condition_on_previous_text=False,
            )
            # ``segments`` is a lazy generator: consume it while holding the lock.
            return " ".join(s.text.strip() for s in segments).strip()


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
