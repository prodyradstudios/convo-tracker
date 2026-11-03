"""Audio decoding helpers. Everything internal is float32 mono 16 kHz in [-1, 1]."""

from __future__ import annotations

import io
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SAMPLE_RATE = 16_000


class AudioDecodeError(ValueError):
    """Raised when uploaded audio cannot be decoded."""


def probe_duration(data: bytes) -> float:
    """Return the duration in seconds of an encoded audio file without decoding it."""
    try:
        info = sf.info(io.BytesIO(data))
    except Exception as exc:  # soundfile raises various error types
        raise AudioDecodeError("could not read audio file (expected WAV)") from exc
    return info.frames / info.samplerate if info.samplerate else 0.0


def to_mono_16k(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Downmix a (frames,) or (frames, channels) array to mono and resample to 16 kHz."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sample_rate != SAMPLE_RATE and audio.size:
        g = gcd(int(sample_rate), SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // g, int(sample_rate) // g)
    audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
    return np.clip(audio, -1.0, 1.0).astype(np.float32, copy=False)


def decode_wav(data: bytes) -> np.ndarray:
    """Decode WAV (or any soundfile-supported) bytes to float32 mono 16 kHz."""
    try:
        audio, sample_rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as exc:
        raise AudioDecodeError("could not decode audio file (expected WAV)") from exc
    return to_mono_16k(audio, sample_rate)


def pcm16_to_float32(data: bytes) -> np.ndarray:
    """Convert raw little-endian PCM16 bytes to float32; a trailing odd byte is dropped."""
    usable = len(data) - (len(data) % 2)
    samples = np.frombuffer(data[:usable], dtype="<i2")
    return samples.astype(np.float32) / 32768.0


def float32_to_pcm16(audio: np.ndarray) -> bytes:
    """Convert float32 audio in [-1, 1] to little-endian PCM16 bytes."""
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    return (clipped * 32767.0).round().astype("<i2").tobytes()


def encode_wav(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Encode float32 audio (mono or (frames, channels)) as a 16-bit PCM WAV."""
    buf = io.BytesIO()
    sf.write(buf, np.asarray(audio, dtype=np.float32), sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()
