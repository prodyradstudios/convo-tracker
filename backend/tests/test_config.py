from pathlib import Path

import pytest

from app.config import Settings
from app.state import build_state
from conftest import FakeEmbedder, FakeTranscriber


def test_defaults():
    s = Settings.from_env({})
    assert s.data_dir == Path("data")
    assert (s.embedder, s.transcriber) == ("auto", "auto")
    assert s.match_threshold is None
    assert (s.silence_ms, s.max_utterance_s, s.min_utterance_s) == (700, 15.0, 0.5)
    assert (s.whisper_model, s.device, s.host, s.port) == ("base.en", "cpu", "127.0.0.1", 8000)


def test_from_env_overrides():
    s = Settings.from_env({
        "CT_DATA_DIR": "/tmp/x", "CT_EMBEDDER": "MFCC", "CT_TRANSCRIBER": "null",
        "CT_MATCH_THRESHOLD": "0.42", "CT_SILENCE_MS": "500", "CT_MAX_UTTERANCE_S": "10",
        "CT_MIN_UTTERANCE_S": "0.3", "CT_WHISPER_MODEL": "small.en", "CT_DEVICE": "cuda",
        "CT_HOST": "0.0.0.0", "CT_PORT": "9000",
    })
    assert s.data_dir == Path("/tmp/x") and s.embedder == "mfcc"
    assert s.match_threshold == 0.42 and s.silence_ms == 500 and s.port == 9000
    assert s.max_utterance_s == 10.0 and s.min_utterance_s == 0.3


@pytest.mark.parametrize("env", [{"CT_EMBEDDER": "gpt"}, {"CT_TRANSCRIBER": "x"}, {"CT_PORT": "abc"}])
def test_invalid_values(env):
    with pytest.raises(ValueError):
        Settings.from_env(env)


def test_threshold_defaults_to_embedder_and_can_be_overridden(tmp_path):
    st = build_state(Settings(data_dir=tmp_path), FakeEmbedder(), FakeTranscriber())
    assert st.registry.threshold == FakeEmbedder.default_threshold
    st = build_state(Settings(data_dir=tmp_path, match_threshold=0.9), FakeEmbedder(), FakeTranscriber())
    assert st.registry.threshold == 0.9
    vad = build_state(
        Settings(data_dir=tmp_path, silence_ms=300), FakeEmbedder(), FakeTranscriber()
    ).vad_config()
    assert vad.silence_ms == 300


@pytest.mark.parametrize(
    ("env", "word"),
    [
        ({"CT_MATCH_THRESHOLD": "nan"}, "CT_MATCH_THRESHOLD"),
        ({"CT_MATCH_THRESHOLD": "1.5"}, "CT_MATCH_THRESHOLD"),
        ({"CT_MATCH_THRESHOLD": "-2"}, "CT_MATCH_THRESHOLD"),
        ({"CT_SILENCE_MS": "0"}, "CT_SILENCE_MS"),
        ({"CT_SILENCE_MS": "-5"}, "CT_SILENCE_MS"),
        ({"CT_MAX_UTTERANCE_S": "inf"}, "CT_MAX_UTTERANCE_S"),
        ({"CT_MAX_UTTERANCE_S": "nan"}, "CT_MAX_UTTERANCE_S"),
        ({"CT_MAX_UTTERANCE_S": "1e9"}, "CT_MAX_UTTERANCE_S"),
        ({"CT_MAX_UTTERANCE_S": "0.4"}, "CT_MAX_UTTERANCE_S"),  # not > min (0.5)
        ({"CT_MIN_UTTERANCE_S": "-1"}, "CT_MIN_UTTERANCE_S"),
        ({"CT_MIN_UTTERANCE_S": "nan"}, "CT_MIN_UTTERANCE_S"),
        ({"CT_PORT": "70000"}, "CT_PORT"),
    ],
)
def test_out_of_range_values_fail_at_startup(env, word):
    with pytest.raises(ValueError, match=word):
        Settings.from_env(env)


def test_edge_values_accepted():
    s = Settings.from_env({"CT_MATCH_THRESHOLD": "-1", "CT_MIN_UTTERANCE_S": "0",
                           "CT_MAX_UTTERANCE_S": "0.1", "CT_SILENCE_MS": "1"})
    assert s.match_threshold == -1.0 and s.min_utterance_s == 0.0
    assert Settings(match_threshold=1.0).match_threshold == 1.0


def test_turn_and_robustness_defaults_and_overrides(tmp_path):
    s = Settings.from_env({})
    assert (s.turn_min_segment_s, s.turn_window_s, s.turn_hop_s, s.turn_min_s) == (2.0, 1.5, 0.5, 1.0)
    assert s.enroll_augment is True and s.adapt is True
    s = Settings.from_env({
        "CT_TURN_MIN_SEGMENT_S": "3", "CT_TURN_WINDOW_S": "2", "CT_TURN_HOP_S": "0.25",
        "CT_TURN_MIN_S": "0", "CT_ENROLL_AUGMENT": "0", "CT_ADAPT": "false",
    })
    assert (s.turn_min_segment_s, s.turn_window_s, s.turn_hop_s, s.turn_min_s) == (3.0, 2.0, 0.25, 0.0)
    assert s.enroll_augment is False and s.adapt is False
    assert Settings.from_env({"CT_ADAPT": "Yes", "CT_ENROLL_AUGMENT": "1"}).adapt is True
    st = build_state(
        Settings(data_dir=tmp_path, turn_hop_s=0.3, adapt=False), FakeEmbedder(), FakeTranscriber()
    )
    turns = st.turn_config()
    assert (turns.min_segment_s, turns.window_s, turns.hop_s, turns.min_turn_s) == (2.0, 1.5, 0.3, 1.0)
    assert st.registry.adapt is False


@pytest.mark.parametrize(
    ("env", "word"),
    [
        ({"CT_TURN_MIN_SEGMENT_S": "-1"}, "CT_TURN_MIN_SEGMENT_S"),
        ({"CT_TURN_MIN_SEGMENT_S": "nan"}, "CT_TURN_MIN_SEGMENT_S"),
        ({"CT_TURN_MIN_SEGMENT_S": "abc"}, "CT_TURN_MIN_SEGMENT_S"),
        ({"CT_TURN_WINDOW_S": "0.1"}, "CT_TURN_WINDOW_S"),
        ({"CT_TURN_WINDOW_S": "inf"}, "CT_TURN_WINDOW_S"),
        ({"CT_TURN_WINDOW_S": "11"}, "CT_TURN_WINDOW_S"),
        ({"CT_TURN_HOP_S": "0"}, "CT_TURN_HOP_S"),
        ({"CT_TURN_HOP_S": "2"}, "CT_TURN_HOP_S"),  # > window (1.5)
        ({"CT_TURN_HOP_S": "nan"}, "CT_TURN_HOP_S"),
        ({"CT_TURN_MIN_S": "-0.5"}, "CT_TURN_MIN_S"),
        ({"CT_TURN_MIN_S": "inf"}, "CT_TURN_MIN_S"),
        ({"CT_ENROLL_AUGMENT": "maybe"}, "CT_ENROLL_AUGMENT"),
        ({"CT_ADAPT": "2"}, "CT_ADAPT"),
    ],
)
def test_invalid_turn_and_robustness_settings(env, word):
    with pytest.raises(ValueError, match=word):
        Settings.from_env(env)


def test_turn_edge_values_accepted():
    s = Settings.from_env({"CT_TURN_WINDOW_S": "0.25", "CT_TURN_HOP_S": "0.25",
                           "CT_TURN_MIN_SEGMENT_S": "0", "CT_TURN_MIN_S": "0"})
    assert s.turn_window_s == s.turn_hop_s == 0.25
    assert Settings(turn_window_s=10.0, turn_hop_s=10.0).turn_hop_s == 10.0
    with pytest.raises(ValueError, match="CT_ADAPT"):
        Settings(adapt=1)  # type: ignore[arg-type]
