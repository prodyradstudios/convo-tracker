"""Streaming changes use short evidence rather than waiting for an utterance pause."""
import numpy as np
import pytest

from app.audio.vad import Segmenter, VadConfig
from app.config import Settings
from app.speakers.registry import SpeakerRegistry
from app.speakers.streaming import StreamingTurnDetector
from app.speakers.turns import TurnConfig
from conftest import FakeEmbedder, chunks, silence, tone


def observe(audio, *, unknown=False):
    embedder = FakeEmbedder()
    registry = SpeakerRegistry(embedder.name, embedder.default_threshold)
    for name, frequency in (("Alice", 200), ("Bob", 400))[:1 if unknown else 2]:
        registry.enroll(name, embedder.embed(tone(1, frequency)))
    detector = StreamingTurnDetector(embedder, registry, TurnConfig(), VadConfig())
    segmenter = Segmenter()
    out = []
    clock = 0
    for part in chunks(audio, 1600):
        clock += part.size / 16000
        segmenter.feed(part)
        snapshot = segmenter.active_segment(1.0)
        if snapshot is None:
            detector.reset()
            continue
        change = detector.observe(snapshot)
        if change is not None:
            out.append((clock, change, registry.match_for(change.speaker_id, change.score)))
    return out, embedder


def test_fast_change_detected_before_pause_and_boundary_refined():
    out, _ = observe(np.concatenate([silence(.3), tone(1.5, 200), tone(.6, 400)]))
    changes = [(clock, change, match) for clock, change, match in out if not change.initial]
    assert len(changes) == 1
    clock, change, match = changes[0]
    assert match.speaker_name == 'Bob'
    assert clock - 1.8 <= .6 + 1e-6
    assert change.start == pytest.approx(1.8, abs=.1)


def test_gain_jumps_do_not_change_voice_and_gate_limits_full_probes():
    out, embedder = observe(np.concatenate([silence(.3), tone(1, 200, .03),
                                           tone(1, 200, .6), tone(1, 200, .03)]))
    assert all(change.initial for _, change, _ in out)
    assert embedder.calls < 16  # enrollment plus periodic probes, not 10 model calls/s


def test_unregistered_interruption_is_not_attributed_to_enrolled_voice():
    out, _ = observe(np.concatenate([silence(.3), tone(1.5, 200), tone(.7, 400)]), unknown=True)
    changes = [(change, match) for _, change, match in out if not change.initial]
    assert len(changes) == 1
    assert changes[0][0].speaker_id is None
    assert changes[0][1].speaker_name == 'Unknown'


def test_split_preserves_audio_and_short_confirmed_suffix():
    segmenter = Segmenter()
    segmenter.feed(np.concatenate([silence(.3), tone(1.5, 200), tone(.3, 400)]))
    before = segmenter.active_segment()
    prefix = segmenter.split_at(1.8)
    assert prefix is not None and before is not None
    tail = segmenter.active_segment()
    assert tail is not None and tail.speaker_change
    np.testing.assert_array_equal(np.concatenate([prefix.audio, tail.audio]), before.audio)
    suffix = segmenter.flush()
    assert len(suffix) == 1
    assert suffix[0].speech_seconds < .5
    assert prefix.end == suffix[0].start


@pytest.mark.parametrize('name,value', [
    ('WINDOW_S', '.1'), ('HOP_S', '0'), ('CONFIRM_S', '.01'),
    ('MARGIN', 'nan'), ('THRESHOLD', '2.1'),
])
def test_invalid_streaming_settings(name, value):
    variable = 'CT_TURN_CHANGE_' + name
    with pytest.raises(ValueError, match=variable):
        Settings.from_env({variable: value})


def test_streaming_settings_from_environment_are_used_by_state(tmp_path):
    from app.state import build_state
    from conftest import FakeTranscriber
    settings = Settings.from_env({'CT_DATA_DIR': str(tmp_path), 'CT_TURN_CHANGE_THRESHOLD': '.25'})
    state = build_state(settings, FakeEmbedder(), FakeTranscriber())
    assert state.turn_config().change_threshold == .25


def test_unknown_to_known_is_detected():
    out, _ = observe(np.concatenate([silence(.3), tone(1.5, 600), tone(.7, 200)]), unknown=True)
    assert out[0][2].speaker_name == 'Unknown'
    assert any(not change.initial and match.speaker_name == 'Alice' for _, change, match in out)


def test_partial_frame_is_preserved_across_a_split_and_flush():
    audio = np.concatenate([silence(.3), tone(1.5, 200), tone(.315, 400)])
    segmenter = Segmenter()
    segmenter.feed(audio)
    prefix = segmenter.split_at(1.8)
    suffix = segmenter.flush()
    assert prefix is not None and len(suffix) == 1
    expected = audio[round(prefix.start * 16000):]
    np.testing.assert_array_equal(np.concatenate([prefix.audio, suffix[0].audio]), expected)


@pytest.mark.parametrize('name', ['WINDOW_S', 'HOP_S', 'CONFIRM_S', 'MARGIN', 'THRESHOLD'])
@pytest.mark.parametrize('value', ['nan', 'inf', '-inf'])
def test_all_change_settings_reject_nonfinite(name, value):
    variable = 'CT_TURN_CHANGE_' + name
    with pytest.raises(ValueError, match=variable):
        Settings.from_env({variable: value})


def test_two_distinct_unregistered_voices_still_create_a_boundary():
    out, _ = observe(np.concatenate([silence(.3), tone(1.5, 600), tone(.7, 800)]), unknown=True)
    changes = [(change, match) for _, change, match in out if not change.initial]
    assert len(changes) == 1
    assert changes[0][0].start == pytest.approx(1.8, abs=.1)
    assert changes[0][1].speaker_name == 'Unknown'
