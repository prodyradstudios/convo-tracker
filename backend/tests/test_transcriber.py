"""Normal startup must transcribe speech or report why it cannot."""

import numpy as np
import pytest

from app.transcription import transcriber as tr


@pytest.mark.parametrize("kind", ["auto", "whisper"])
def test_real_transcription_requires_installed_runtime(monkeypatch, kind):
    monkeypatch.setattr(tr, "_whisper_available", lambda: False)
    with pytest.raises(RuntimeError, match="pip install -r backend/requirements.txt"):
        tr.build_transcriber(kind)


@pytest.mark.parametrize("kind", ["auto", "whisper"])
def test_model_load_failure_is_actionable_and_never_a_placeholder(monkeypatch, kind):
    cause = OSError("model download unavailable")

    def unavailable(model, device):
        raise cause

    monkeypatch.setattr(tr, "_whisper_available", lambda: True)
    monkeypatch.setattr(tr, "WhisperTranscriber", unavailable)
    with pytest.raises(RuntimeError, match="Could not load Whisper speech-to-text model") as error:
        tr.build_transcriber(kind, "base.en", "cpu")
    assert error.value.__cause__ is cause
    assert "CT_WHISPER_MODEL" in str(error.value)


def test_placeholder_requires_explicit_opt_in_and_warns(monkeypatch, caplog):
    def unexpected():
        raise AssertionError("explicit placeholder mode must not load Whisper")

    monkeypatch.setattr(tr, "_whisper_available", unexpected)
    transcriber = tr.build_transcriber("null")
    assert transcriber.transcribe(np.zeros(16000, np.float32)) == "[speech 1.0s]"
    assert "Speech-to-text is disabled by CT_TRANSCRIBER=null" in caplog.text


def test_unknown_transcriber_is_rejected():
    with pytest.raises(ValueError, match="Unknown transcriber"):
        tr.build_transcriber("misspelled")
