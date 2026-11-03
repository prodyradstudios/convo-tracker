import numpy as np
import pytest

from app.audio.vad import Segmenter, VadConfig, frame_db, trim_silence
from conftest import SR, chunks, noise_burst, silence, tone


def run(audio: np.ndarray, config: VadConfig | None = None, chunk: int = 1600):
    seg = Segmenter(config)
    out, events = [], []
    for c in chunks(audio, chunk):
        out += seg.feed(c)
        events += seg.pop_events()
    out += seg.flush()
    events += seg.pop_events()
    return out, events


def test_frame_db():
    assert frame_db(np.zeros(480, dtype=np.float32)) == -100.0
    assert frame_db(np.ones(480, dtype=np.float32)) == pytest.approx(0.0, abs=1e-6)
    assert frame_db(np.full(480, 0.1, dtype=np.float32)) == pytest.approx(-20.0, abs=1e-4)


def test_two_utterances_counts_and_timings():
    audio = np.concatenate([silence(1.0), tone(1.5), silence(1.5), noise_burst(2.0), silence(1.0)])
    segs, events = run(audio)
    assert len(segs) == 2
    assert events == [True, False, True, False]
    a, b = segs
    # 300 ms pre-roll before onset, ~200 ms tail after the last speech frame.
    assert a.start == pytest.approx(1.0 - 0.3, abs=0.04)
    assert a.end == pytest.approx(2.5 + 0.2, abs=0.05)
    assert b.start == pytest.approx(4.0 - 0.3, abs=0.04)
    assert b.end == pytest.approx(6.0 + 0.2, abs=0.05)
    for s in segs:
        assert s.audio.dtype == np.float32
        assert s.audio.size / SR == pytest.approx(s.end - s.start, abs=1e-6)


def test_chunk_size_does_not_matter():
    audio = np.concatenate([silence(0.5), tone(1.0), silence(1.0), tone(0.8, 440), silence(1.0)])
    ref, _ = run(audio, chunk=1600)
    for size in (7, 480, 1000, 16000, audio.size):
        got, _ = run(audio, chunk=size)
        assert [(s.start, s.end) for s in got] == [(s.start, s.end) for s in ref]


def test_short_pause_does_not_split():
    audio = np.concatenate([silence(0.5), tone(1.0), silence(0.4), tone(1.0), silence(1.5)])
    segs, _ = run(audio)
    assert len(segs) == 1


def test_silence_ms_config_splits_shorter_pauses():
    audio = np.concatenate([silence(0.5), tone(1.0), silence(0.4), tone(1.0), silence(1.5)])
    segs, _ = run(audio, VadConfig(silence_ms=300))
    assert len(segs) == 2


def test_max_length_force_cut():
    audio = np.concatenate([silence(0.5), tone(5.0), silence(1.0)])
    segs, events = run(audio, VadConfig(max_utterance_s=2.0))
    assert len(segs) == 3
    assert all(s.duration <= 2.0 + 1e-6 for s in segs[:2])
    # Contiguous: each cut continues exactly where the previous stopped.
    assert segs[1].start == pytest.approx(segs[0].end)
    assert segs[2].start == pytest.approx(segs[1].end)
    assert events == [True, False]  # one continuous speech region


def test_min_length_drop():
    audio = np.concatenate([silence(0.5), tone(0.3), silence(1.0), tone(0.8), silence(1.0)])
    segs, events = run(audio)
    assert len(segs) == 1
    assert segs[0].start == pytest.approx(1.8 - 0.3, abs=0.04)
    assert events == [True, False, True, False]  # events still fire for the dropped blip


def test_too_few_speech_frames_do_not_trigger():
    audio = np.concatenate([silence(0.5), tone(0.04), silence(1.0)])  # spans 2 frames
    segs, events = run(audio)
    assert segs == [] and events == []


def test_flush_emits_in_progress_utterance():
    seg = Segmenter()
    out = seg.feed(np.concatenate([silence(0.5), tone(1.2)]))
    assert out == [] and seg.in_speech
    assert seg.pop_events() == [True]
    flushed = seg.flush()
    assert len(flushed) == 1
    assert flushed[0].end == pytest.approx(1.7, abs=0.01)
    assert seg.pop_events() == [False]
    assert not seg.in_speech
    assert seg.flush() == []


def test_flush_while_idle_is_noop():
    seg = Segmenter()
    seg.feed(silence(1.0))
    assert seg.flush() == [] and seg.pop_events() == []


def test_digital_silence_and_quiet_sound():
    # A -55 dBFS tone in digital silence is below the absolute -50 dBFS floor.
    audio = np.concatenate([np.zeros(SR, np.float32), tone(1.0, amp=0.0025), np.zeros(SR, np.float32)])
    segs, _ = run(audio)
    assert segs == []
    loud = np.concatenate([np.zeros(SR, np.float32), tone(1.0, amp=0.1), np.zeros(SR, np.float32)])
    segs, _ = run(loud)
    assert len(segs) == 1


def test_noise_floor_tracks_steady_background():
    # Steady background noise well above -50 dBFS is not speech...
    background = silence(3.0, level=0.02, seed=3)
    seg = Segmenter()
    assert seg.feed(background) == [] and seg.pop_events() == []
    assert seg.noise_floor_db == pytest.approx(frame_db(background), abs=3.0)
    # ...but a loud tone on top of it is.
    loud = background[:SR] + tone(1.0, amp=0.3)
    out = seg.feed(np.concatenate([loud, background]))
    assert len(out) == 1


def test_speech_at_stream_start():
    audio = np.concatenate([tone(1.0), silence(1.5)])
    segs, _ = run(audio)
    assert len(segs) == 1
    assert segs[0].start == 0.0


def test_reset_restarts_clock():
    seg = Segmenter()
    seg.feed(silence(2.0))
    seg.reset()
    out = seg.feed(np.concatenate([silence(0.5), tone(1.0), silence(1.0)]))
    assert out[0].start == pytest.approx(0.2, abs=0.04)


def test_trim_silence():
    audio = np.concatenate([silence(1.0), tone(1.0), silence(2.0), tone(1.0), silence(1.0)])
    speech = trim_silence(audio)
    assert 2.0 <= speech.size / SR <= 3.2
    assert trim_silence(silence(2.0)).size == 0
