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
    vad = build_state(Settings(data_dir=tmp_path, silence_ms=300), FakeEmbedder()).vad_config()
    assert vad.silence_ms == 300
