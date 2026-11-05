"""Integration contracts between Whisper output and interrupted speaker turns."""

import threading
from types import SimpleNamespace as NS

import numpy as np
import pytest

from app.audio.vad import Segment
from app.pipeline import SessionPipeline
from app.speakers.registry import SpeakerRegistry
from app.speakers.turns import Turn, TurnAnalysis
from app.transcription.transcriber import WhisperTranscriber
from conftest import FakeEmbedder, tone


def whisper(model):
    transcriber = WhisperTranscriber.__new__(WhisperTranscriber)
    transcriber._model = model
    transcriber._language = 'en'
    transcriber._lock = threading.Lock()
    return transcriber


def pipeline(transcriber, monkeypatch):
    import app.pipeline as module

    embedder = FakeEmbedder()
    registry = SpeakerRegistry(embedder.name, embedder.default_threshold)
    alice = registry.enroll('Alice', embedder.embed(tone(1, 200)))['id']
    bob = registry.enroll('Bob', embedder.embed(tone(1, 400)))['id']
    monkeypatch.setattr(module, 'analyze_turns', lambda *args: TurnAnalysis(
        [Turn(0, 1.2, alice, 1), Turn(1.2, 2.4, bob, 1)], False, None))
    return SessionPipeline(embedder, transcriber, registry)


def test_partial_word_timing_uses_audio_fallback_without_losing_either_speaker(monkeypatch):
    class Model:
        def transcribe(self, audio, **kwargs):
            if kwargs['word_timestamps']:
                return iter([
                    NS(text=' I was saying', words=[NS(word=' I', start=0, end=.3),
                                                   NS(word=' was saying', start=.3, end=1.2)]),
                    NS(text=' Excuse me', words=None),
                ]), NS()
            text = 'I was saying' if np.argmax(abs(np.fft.rfft(audio))) < 360 else 'Excuse me'
            return iter([NS(text=text)]), NS()

    p = pipeline(whisper(Model()), monkeypatch)
    audio = np.concatenate([tone(1.2, 200), tone(1.2, 400)])
    utterances = p.process(Segment(audio, 0, 2.4, speech_duration=2.4))
    assert [(u.speaker_name, u.text) for u in utterances] == [
        ('Alice', 'I was saying'), ('Bob', 'Excuse me')]


def test_lazy_whisper_failure_propagates_without_partial_utterances(monkeypatch):
    class Model:
        def transcribe(self, audio, **kwargs):
            def broken():
                yield NS(text=' I was saying', words=[NS(word=' I', start=0, end=.3)])
                raise RuntimeError('decoder failed')
            return broken(), NS()

    p = pipeline(whisper(Model()), monkeypatch)
    audio = np.concatenate([tone(1.2, 200), tone(1.2, 400)])
    with pytest.raises(RuntimeError, match='decoder failed'):
        p.process(Segment(audio, 0, 2.4, speech_duration=2.4))
    assert p._seq == 0


@pytest.mark.parametrize('word', [
    NS(word=' Excuse me', start=None, end=None),
    NS(word=' Excuse me', end=1.2),
    NS(word=' Excuse me', start=float('nan'), end=1.2),
    NS(word=' Excuse me', start=0, end=float('inf')),
    NS(word=' Excuse me', start=1.2, end=0),
    NS(word=' Excuse me', start='invalid', end=1.2),
    NS(word='   ', start=0, end=1.2),
    NS(start=0, end=1.2),
])
def test_missing_word_timestamp_falls_back_to_audio_instead_of_dropping_text(monkeypatch, word):
    class Model:
        def transcribe(self, audio, **kwargs):
            if kwargs['word_timestamps']:
                return iter([
                    NS(text=' I was saying', words=[NS(word=' I was saying', start=0, end=1.2)]),
                    NS(text=' Excuse me', words=[word]),
                ]), NS()
            text = 'I was saying' if np.argmax(abs(np.fft.rfft(audio))) < 360 else 'Excuse me'
            return iter([NS(text=text)]), NS()

    p = pipeline(whisper(Model()), monkeypatch)
    audio = np.concatenate([tone(1.2, 200), tone(1.2, 400)])
    assert [u.text for u in p.process(Segment(audio, 0, 2.4, speech_duration=2.4))] == [
        'I was saying', 'Excuse me']


@pytest.mark.parametrize('malformed', [NS(start=.5, end=1),
                                       NS(word=None, start=.5, end=1),
                                       NS(word=' ', start=.5, end=1)])
def test_malformed_token_among_valid_words_does_not_return_partial_text(malformed):
    class Model:
        def transcribe(self, audio, **kwargs):
            return iter([NS(text=' Hello world', words=[
                NS(word=' Hello', start=0, end=.5), malformed,
            ])]), NS()

    assert whisper(Model()).transcribe_words(np.zeros(16000, np.float32)) is None
