import numpy as np

from app.audio.vad import Segment
from app.pipeline import SessionPipeline
from app.speakers.registry import SpeakerRegistry
from app.transcription.transcriber import NullTranscriber, build_transcriber
from conftest import FakeEmbedder, FakeTranscriber, tone


def make(transcriber=None) -> tuple[SessionPipeline, SpeakerRegistry]:
    fake = FakeEmbedder()
    reg = SpeakerRegistry(fake.name, fake.default_threshold)
    reg.enroll("Alice", fake.embed(tone(1.0, 200)))
    reg.enroll("Bob", fake.embed(tone(1.0, 400)))
    return SessionPipeline(fake, transcriber or FakeTranscriber(), reg), reg


def test_process_segment():
    p, _ = make()
    u = p.process(Segment(tone(1.0, 400), 1.23456, 2.23456))
    assert u is not None and u.seq == 1 and u.speaker_name == "Bob"
    assert (u.start, u.end) == (1.235, 2.235)
    msg = u.to_message()
    assert msg["type"] == "utterance" and msg["text"] == "tone 400"
    assert p.process(Segment(tone(1.0, 200), 3, 4)).seq == 2


def test_empty_transcript_skipped_without_consuming_seq():
    p, _ = make(FakeTranscriber(silent=(200,)))
    assert p.process(Segment(tone(1.0, 200), 0, 1)) is None
    assert p.process(Segment(tone(1.0, 400), 1, 2)).seq == 1


def test_null_transcriber():
    t = NullTranscriber()
    assert t.transcribe(np.zeros(24000, np.float32)) == "[speech 1.5s]"
    assert build_transcriber("null").name == "null"
