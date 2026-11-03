import io

import numpy as np
import pytest
import soundfile as sf

from app.audio.io import (
    AudioDecodeError,
    decode_wav,
    encode_wav,
    float32_to_pcm16,
    pcm16_to_float32,
    probe_duration,
    to_mono_16k,
)
from conftest import SR, dominant_freq, tone


def test_decode_16k_mono_roundtrip():
    audio = tone(1.0, 440)
    out = decode_wav(encode_wav(audio))
    assert out.dtype == np.float32 and out.ndim == 1 and out.size == SR
    assert np.max(np.abs(out - audio)) < 1e-3


def test_decode_44k_stereo_downmix_and_resample():
    t = np.arange(44100 * 2) / 44100
    left = 0.4 * np.sin(2 * np.pi * 440 * t)
    stereo = np.stack([left, left], axis=1)
    out = decode_wav(encode_wav(stereo, 44100))
    assert out.size == pytest.approx(2 * SR, abs=1)
    assert dominant_freq(out) == pytest.approx(440, abs=2)
    assert np.max(np.abs(out)) == pytest.approx(0.4, abs=0.02)


def test_downmix_cancels_out_of_phase_channels():
    left = tone(0.5, 300)
    out = to_mono_16k(np.stack([left, -left], axis=1), SR)
    assert np.max(np.abs(out)) < 1e-6


def test_decode_8k_and_float_wav():
    t = np.arange(8000) / 8000
    buf = io.BytesIO()
    sf.write(buf, (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32), 8000, format="WAV", subtype="FLOAT")
    out = decode_wav(buf.getvalue())
    assert out.size == SR
    assert dominant_freq(out) == pytest.approx(200, abs=2)


def test_decode_garbage_raises():
    with pytest.raises(AudioDecodeError):
        decode_wav(b"definitely not a wav file")
    with pytest.raises(AudioDecodeError):
        probe_duration(b"RIFF....WAVEjunk")


def test_probe_duration():
    assert probe_duration(encode_wav(tone(2.5), SR)) == pytest.approx(2.5)


def test_pcm16_conversion():
    samples = np.array([0, 1, -1, 32767, -32768, 16384], dtype="<i2")
    out = pcm16_to_float32(samples.tobytes())
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, samples / 32768.0)
    assert out.min() >= -1.0 and out.max() < 1.0


def test_pcm16_odd_length_drops_trailing_byte():
    data = np.array([1000, -1000], dtype="<i2").tobytes() + b"\x7f"
    out = pcm16_to_float32(data)
    assert out.size == 2
    assert pcm16_to_float32(b"").size == 0
    assert pcm16_to_float32(b"\x01").size == 0


def test_pcm16_roundtrip():
    audio = tone(0.1, 1000, amp=0.9)
    back = pcm16_to_float32(float32_to_pcm16(audio))
    assert np.max(np.abs(back - audio)) < 1e-4


def _raw_wav_header(sample_rate: int, data: bytes = b"\x00\x10" * 100) -> bytes:
    """A tiny hand-built 16-bit mono PCM WAV with an arbitrary sample rate."""
    import struct

    fmt = struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
    return (
        b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt " + fmt
        + b"data" + struct.pack("<I", len(data)) + data
    )


@pytest.mark.parametrize("rate", [1, 3999, 384_001, 2_000_003, 50_000_017])
def test_absurd_sample_rates_rejected_before_resampling(rate, monkeypatch):
    import app.audio.io as io_mod

    def boom(*args, **kwargs):  # resampling such rates can take GBs of RAM
        raise AssertionError("resample_poly must not be called")

    monkeypatch.setattr(io_mod, "resample_poly", boom)
    wav = _raw_wav_header(rate)
    with pytest.raises(AudioDecodeError, match="sample rate"):
        decode_wav(wav)
    with pytest.raises(AudioDecodeError, match="sample rate"):
        probe_duration(wav)
    with pytest.raises(AudioDecodeError, match="sample rate"):
        to_mono_16k(np.zeros(10, np.float32), rate)


@pytest.mark.parametrize("rate", [4000, 8000, 44100, 384_000])
def test_supported_sample_rates_decode(rate):
    assert decode_wav(_raw_wav_header(rate)).dtype == np.float32


def test_awkward_in_range_rate_resamples_cheaply(monkeypatch):
    import app.audio.io as io_mod

    seen = []
    real = io_mod.resample_poly

    def spy(x, up, down):
        seen.append((up, down))
        return real(x, up, down)

    monkeypatch.setattr(io_mod, "resample_poly", spy)
    out = decode_wav(_raw_wav_header(383_983, b"\x00\x10" * 4000))  # prime rate
    assert max(seen[-1]) <= 1000
    assert out.size == pytest.approx(4000 * 16000 / 383_983, abs=2)
    decode_wav(encode_wav(tone(0.1), 44100))
    assert seen[-1] == (160, 441)  # standard rates stay exact
