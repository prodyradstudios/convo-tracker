"""Streaming energy VAD and utterance segmenter (see docs/SPEC.md, "Segmentation")."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .io import SAMPLE_RATE

_MIN_DB = -100.0


@dataclass
class Segment:
    """A finished utterance. ``start``/``end`` are seconds since the stream start.

    ``audio`` (and thus ``duration``) includes the pre-roll and tail padding;
    ``speech_duration`` is the VAD's onset-to-last-speech-frame time, when known.
    """

    audio: np.ndarray
    start: float
    end: float
    speech_duration: float | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def speech_seconds(self) -> float:
        """Seconds of actual speech (falls back to ``duration`` if unknown)."""
        return self.duration if self.speech_duration is None else self.speech_duration


@dataclass
class VadConfig:
    """Segmenter tunables. Defaults follow the spec."""

    sample_rate: int = SAMPLE_RATE
    frame_ms: int = 30
    silence_ms: int = 700
    max_utterance_s: float = 15.0
    min_utterance_s: float = 0.5
    start_frames: int = 3
    preroll_ms: int = 300
    tail_ms: int = 200
    noise_init_ms: int = 300
    margin_db: float = 10.0
    min_speech_db: float = -50.0
    noise_alpha: float = 0.05  # slow upward adaptation of the noise floor
    # Cap on the noise floor so speech that starts right at stream start (and
    # thus pollutes the 300 ms initialisation) is still detected.
    max_noise_db: float = -40.0
    # If *every* frame of this window counted as speech, the background has
    # probably risen (real speech always dips between words): raise the floor
    # to the window's quietest frame (minimum statistics).
    noise_window_ms: int = 5000

    @property
    def frame_len(self) -> int:
        return self.sample_rate * self.frame_ms // 1000

    def frames(self, ms: float) -> int:
        """Number of whole frames covering ``ms`` milliseconds (at least 1)."""
        return max(1, int(round(ms / self.frame_ms)))


def frame_db(frame: np.ndarray) -> float:
    """RMS energy of a frame in dBFS (floored at -100)."""
    rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64)))) if frame.size else 0.0
    return max(_MIN_DB, 20.0 * np.log10(rms + 1e-12))


class Segmenter:
    """Cuts a continuous float32 stream into utterances.

    Call :meth:`feed` with chunks of any length and :meth:`flush` at the end.
    Speech start/end transitions are queued and returned by :meth:`pop_events`
    (``True`` = speech started, ``False`` = speech ended).
    """

    def __init__(self, config: VadConfig | None = None) -> None:
        self.config = config or VadConfig()
        c = self.config
        self._flen = c.frame_len
        self._silence_frames = max(1, int(np.ceil(c.silence_ms / c.frame_ms)))
        self._tail_frames = c.frames(c.tail_ms)
        self._init_frames = c.frames(c.noise_init_ms)
        self._max_frames = max(1, int(c.max_utterance_s * 1000 // c.frame_ms))
        self.reset()

    # -- public API -------------------------------------------------------
    def reset(self) -> None:
        """Forget all state, including the stream clock and noise floor."""
        c = self.config
        self._pending = np.zeros(0, dtype=np.float32)
        self._frame_index = 0  # index of the next frame to be processed
        self._noise_db: float | None = None
        self._seen = 0
        self._speech_dbs: deque[float] = deque(maxlen=c.frames(c.noise_window_ms))
        self._history: deque[tuple[int, np.ndarray]] = deque(
            maxlen=c.frames(c.preroll_ms) + c.start_frames
        )
        self._run = 0  # consecutive speech frames while idle
        self._events: list[bool] = []
        self._reset_utterance()

    @property
    def in_speech(self) -> bool:
        return self._active

    @property
    def noise_floor_db(self) -> float | None:
        return self._noise_db

    def pop_events(self) -> list[bool]:
        """Return and clear queued speech start (True) / end (False) events."""
        events, self._events = self._events, []
        return events

    def feed(self, samples: np.ndarray) -> list[Segment]:
        """Process a chunk of samples and return utterances completed by it."""
        samples = np.array(samples, dtype=np.float32).reshape(-1)  # own copy
        if self._pending.size:
            samples = np.concatenate([self._pending, samples])
        n_frames = samples.size // self._flen
        out: list[Segment] = []
        for i in range(n_frames):
            frame = samples[i * self._flen : (i + 1) * self._flen]
            seg = self._process_frame(frame)
            if seg is not None:
                out.append(seg)
        self._pending = samples[n_frames * self._flen :].copy()
        return out

    def flush(self) -> list[Segment]:
        """End the stream: finish any in-progress utterance (including a partial frame)."""
        out: list[Segment] = []
        if self._active:
            if self._pending.size:
                self._frames.append(self._pending)
                if frame_db(self._pending) > self._threshold():
                    self._last_speech = len(self._frames) - 1
            seg = self._finish(trim=True)
            if seg is not None:
                out.append(seg)
            self._events.append(False)
        self._pending = np.zeros(0, dtype=np.float32)
        self._history.clear()
        self._run = 0
        return out

    # -- internals --------------------------------------------------------
    def _reset_utterance(self) -> None:
        self._active = False
        self._frames: list[np.ndarray] = []
        self._utt_start = 0  # frame index of the first frame in self._frames
        self._onset = 0  # index (within self._frames) of the first speech frame
        self._last_speech = -1  # index (within self._frames) of the last speech frame
        self._silence = 0

    def _threshold(self) -> float:
        floor = self._noise_db if self._noise_db is not None else _MIN_DB
        floor = min(floor, self.config.max_noise_db)
        return max(floor + self.config.margin_db, self.config.min_speech_db)

    def _update_noise(self, db: float, is_speech: bool) -> None:
        self._seen += 1
        assert self._noise_db is not None
        if self._seen <= self._init_frames:
            # Initialisation (first 300 ms): track the quietest frame so that
            # speech right at the start does not inflate the floor.
            self._noise_db = min(self._noise_db, db)
        elif not is_speech:
            # Fall quickly, rise slowly (slow EMA of non-speech frames).
            self._speech_dbs.clear()
            alpha = 0.5 if db < self._noise_db else self.config.noise_alpha
            self._noise_db += alpha * (db - self._noise_db)
        else:
            self._speech_dbs.append(db)
            if len(self._speech_dbs) == self._speech_dbs.maxlen:
                # No dip below the threshold for a whole window: the "speech"
                # is most likely a louder background. Re-estimate the floor.
                self._noise_db = max(self._noise_db, min(self._speech_dbs))
                self._speech_dbs.clear()

    def _process_frame(self, frame: np.ndarray) -> Segment | None:
        idx = self._frame_index
        self._frame_index += 1
        db = frame_db(frame)
        if self._noise_db is None:
            self._noise_db = db
        is_speech = db > self._threshold()
        self._update_noise(db, is_speech)

        if not self._active:
            return self._idle_frame(idx, frame, is_speech)

        self._frames.append(frame)
        if is_speech:
            self._last_speech = len(self._frames) - 1
            self._silence = 0
        else:
            self._silence += 1
            if self._silence >= self._silence_frames:
                seg = self._finish(trim=True)
                self._events.append(False)
                return seg
        if len(self._frames) >= self._max_frames:
            # Force-cut; keep listening as a continuation of the same speech.
            seg = self._finish(trim=False)
            self._active = True
            self._utt_start = self._frame_index
            self._onset = 0
            return seg
        return None

    def _idle_frame(self, idx: int, frame: np.ndarray, is_speech: bool) -> Segment | None:
        self._history.append((idx, frame))
        self._run = self._run + 1 if is_speech else 0
        if self._run < self.config.start_frames:
            return None
        frames = list(self._history)
        self._active = True
        self._frames = [f for _, f in frames]
        self._utt_start = frames[0][0]
        self._onset = len(frames) - self.config.start_frames
        self._last_speech = len(frames) - 1
        self._silence = 0
        self._history.clear()
        self._run = 0
        self._events.append(True)
        return None

    def _finish(self, trim: bool) -> Segment | None:
        """Close the current utterance; returns None if it is too short."""
        frames = self._frames
        if trim:
            frames = frames[: self._last_speech + 1 + self._tail_frames]
        speech_frames = self._last_speech + 1 - self._onset
        start_frame = self._utt_start
        self._reset_utterance()
        if not frames:
            return None
        audio = np.concatenate(frames)
        speech_s = speech_frames * self.config.frame_ms / 1000.0
        if speech_s < self.config.min_utterance_s:
            return None
        sr = self.config.sample_rate
        start = start_frame * self._flen / sr
        return Segment(
            audio=audio, start=start, end=start + audio.size / sr, speech_duration=speech_s
        )


def voiced_segments(audio: np.ndarray, config: VadConfig | None = None) -> list[Segment]:
    """VAD segments of a complete recording (each padded with pre-roll and tail)."""
    seg = Segmenter(config)
    return seg.feed(audio) + seg.flush()


def trim_silence(audio: np.ndarray, config: VadConfig | None = None) -> np.ndarray:
    """Return only the voiced parts of ``audio`` (concatenated VAD segments)."""
    segments = voiced_segments(audio, config)
    if not segments:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate([s.audio for s in segments])
