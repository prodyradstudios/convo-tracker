"""Whisper adapter integration without downloading a model in unit tests."""
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pytest

from app.audio.vad import Segment
from app.pipeline import SessionPipeline
from app.speakers.registry import SpeakerRegistry
from app.transcription import transcriber as tr
from conftest import FakeEmbedder, tone, tone_runs


def whisper_adapter(monkeypatch):
    """Use the production constructor/adapter with a deterministic model API."""
    module = ModuleType('faster_whisper')
    class Model:
        def __init__(self, model, **kwargs):
            assert model == 'base.en'
            assert kwargs == {'device': 'cpu', 'compute_type': 'int8'}
        def transcribe(self, audio, **kwargs):
            def segments():
                for frequency, start, end in tone_runs(audio):
                    if frequency == 200:
                        text = ' I was speaking.'
                    elif frequency == 400:
                        text = ' Let me interrupt.'
                    else:
                        continue
                    yield SimpleNamespace(text=text, words=[
                        SimpleNamespace(word=text, start=start, end=end),
                    ] if kwargs['word_timestamps'] else None)
            return segments(), SimpleNamespace(language='en')
    module.WhisperModel = Model
    monkeypatch.setitem(sys.modules, 'faster_whisper', module)
    monkeypatch.setattr(tr, '_whisper_available', lambda: True)
    adapter = tr.build_transcriber('auto')
    assert isinstance(adapter, tr.WhisperTranscriber)
    return adapter


def pipeline(transcriber):
    embedder = FakeEmbedder()
    registry = SpeakerRegistry(embedder.name, embedder.default_threshold)
    for name, frequency in [('Alice', 200), ('Bob', 400)]:
        registry.enroll(name, embedder.embed(tone(1, frequency)))
    return SessionPipeline(embedder, transcriber, registry)


def test_default_whisper_adapter_keeps_words_with_online_interrupting_speaker(monkeypatch):
    pipe = pipeline(whisper_adapter(monkeypatch))
    audio = np.concatenate([tone(2, 200), tone(.8, 400), tone(1, 200)])
    completed = []
    for offset in range(0, len(audio), 1600):
        completed.extend(pipe.feed(audio[offset:offset + 1600]))
    completed.extend(pipe.flush())
    utterances = [u for segment in completed for u in pipe.process(segment)]
    assert [(u.speaker_name, u.text) for u in utterances] == [
        ('Alice', 'I was speaking.'), ('Bob', 'Let me interrupt.'), ('Alice', 'I was speaking.'),
    ]
    assert all(not u.text.startswith('[speech ') for u in utterances)


def test_default_whisper_word_timing_keeps_offline_interruption_text_separate(monkeypatch):
    pipe = pipeline(whisper_adapter(monkeypatch))
    audio = np.concatenate([tone(2, 200), tone(.8, 400), tone(2, 200)])
    utterances = pipe.process(Segment(audio, 0, len(audio) / 16000))
    assert [(u.speaker_name, u.text) for u in utterances] == [
        ('Alice', 'I was speaking.'), ('Bob', 'Let me interrupt.'), ('Alice', 'I was speaking.'),
    ]


def test_lazy_inference_failure_is_never_replaced_with_duration_text(monkeypatch):
    adapter = whisper_adapter(monkeypatch)
    def broken(*args, **kwargs):
        def segments():
            raise RuntimeError('inference failed')
            yield
        return segments(), None
    adapter._model.transcribe = broken
    with pytest.raises(RuntimeError, match='inference failed'):
        adapter.transcribe(tone(1, 200))
    with pytest.raises(RuntimeError, match='inference failed'):
        adapter.transcribe_words(tone(1, 200))
