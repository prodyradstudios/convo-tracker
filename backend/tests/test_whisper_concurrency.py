"""The shared Whisper model must serialize lazy inference across clients."""
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType, SimpleNamespace
import sys
import threading

import numpy as np
import pytest

from app.transcription.transcriber import WhisperTranscriber


def adapter(monkeypatch):
    class Model:
        def __init__(self, *args, **kwargs):
            self.active = 0
            self.maximum_active = 0
            self.completed = 0
            self.fail_next = False
            self.mutex = threading.Lock()

        def transcribe(self, audio, **kwargs):
            with self.mutex:
                self.active += 1
                self.maximum_active = max(self.maximum_active, self.active)
            assert self.active == 1, 'shared model inference overlapped'
            assert audio.dtype == np.float32 and audio.flags.c_contiguous
            assert not kwargs['vad_filter']
            assert not kwargs['condition_on_previous_text']

            def lazy_segments():
                try:
                    # This executes only on iteration. Locking transcribe()
                    # without also consuming its result cannot protect it.
                    assert self.inference_locked(), 'lazy inference ran outside model lock'
                    if self.fail_next:
                        self.fail_next = False
                        raise RuntimeError('inference failed lazily')
                    yield SimpleNamespace(text=' hello', words=[
                        SimpleNamespace(word=' hello', start=0., end=.25),
                    ])
                finally:
                    with self.mutex:
                        self.active -= 1
                        self.completed += 1
            return lazy_segments(), None

    module = ModuleType('faster_whisper')
    module.WhisperModel = Model
    monkeypatch.setitem(sys.modules, 'faster_whisper', module)
    transcriber = WhisperTranscriber()
    transcriber._model.inference_locked = transcriber._lock.locked
    return transcriber


def test_text_and_timed_word_calls_share_lock_through_lazy_inference(monkeypatch):
    transcriber = adapter(monkeypatch)
    clip = np.zeros(4000, dtype=np.float32)
    ready = threading.Barrier(8)

    def call(index):
        if index < 8:
            ready.wait(timeout=2)
        method = transcriber.transcribe if index % 2 else transcriber.transcribe_words
        result = method(clip)
        assert result == 'hello' if isinstance(result, str) else result[0].text == ' hello'

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(call, index) for index in range(32)]
        for future in futures:
            future.result(timeout=5)
    assert transcriber._model.maximum_active == 1
    assert transcriber._model.completed == 32
    assert transcriber._model.active == 0


def test_lazy_failure_releases_shared_model_for_next_client(monkeypatch):
    transcriber = adapter(monkeypatch)
    clip = np.zeros(4000, dtype=np.float32)
    transcriber._model.fail_next = True
    with pytest.raises(RuntimeError, match='inference failed lazily'):
        transcriber.transcribe(clip)

    # A bounded daemon worker makes a missing release fail promptly rather
    # than hanging the test process at executor shutdown.
    finished = threading.Event()
    result = []
    def next_client():
        try:
            result.append(transcriber.transcribe_words(clip))
        finally:
            finished.set()
    thread = threading.Thread(target=next_client, daemon=True)
    thread.start()
    assert finished.wait(2), 'failed inference left the model lock held'
    assert result[0][0].text == ' hello'
    assert transcriber._model.active == 0
    assert transcriber._model.completed == 2
