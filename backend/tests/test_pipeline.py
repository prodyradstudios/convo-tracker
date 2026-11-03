import numpy as np
import pytest

from app.audio.vad import Segment
from app.pipeline import SessionPipeline
from app.speakers.registry import SpeakerRegistry
from app.transcription.transcriber import NullTranscriber, build_transcriber
from conftest import FakeEmbedder, FakeTranscriber, dominant_freq, silence, tone


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


class AmbiguousEmbedder:
    """200 Hz -> Alice, 400 Hz -> Bob, 300 Hz -> between them, leaning slightly to Bob."""

    name = "ambiguous"
    default_threshold = 0.5

    def embed(self, audio: np.ndarray) -> np.ndarray:
        a, b = np.eye(4, dtype=np.float32)[:2]
        vec = {2: a, 4: b}.get(int(round(dominant_freq(audio) / 100.0)), 0.70 * a + 0.72 * b)
        return vec / np.linalg.norm(vec)


def test_continuity_smoothing_triggers_for_short_live_segment():

    emb = AmbiguousEmbedder()
    reg = SpeakerRegistry(emb.name, emb.default_threshold)
    reg.enroll("Alice", emb.embed(tone(1.0, 200)))
    reg.enroll("Bob", emb.embed(tone(1.0, 400)))
    p = SessionPipeline(emb, FakeTranscriber(), reg)
    audio = np.concatenate([silence(0.5), tone(1.5, 200), silence(1.0), tone(0.7, 300), silence(1.0)])
    segments = p.feed(audio) + p.flush()
    assert len(segments) == 2
    short = segments[1]
    assert short.duration >= 1.0  # padded with pre-roll and tail...
    assert short.speech_duration == pytest.approx(0.7, abs=0.07)  # ...but only 0.7 s of speech
    assert p.process(segments[0]).speaker_name == "Alice"
    assert p.process(short).speaker_name == "Alice"  # smoothed toward the previous speaker
