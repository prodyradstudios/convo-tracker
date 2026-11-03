import numpy as np
import pytest

from app.audio.vad import Segment
from app.pipeline import SessionPipeline
from app.speakers.registry import SpeakerRegistry
from app.speakers.turns import Turn
from app.transcription.transcriber import NullTranscriber, build_transcriber
from conftest import FakeEmbedder, FakeTranscriber, dominant_freq, silence, tone, tone_runs


def make(transcriber=None) -> tuple[SessionPipeline, SpeakerRegistry]:
    fake = FakeEmbedder()
    reg = SpeakerRegistry(fake.name, fake.default_threshold)
    reg.enroll("Alice", fake.embed(tone(1.0, 200)))
    reg.enroll("Bob", fake.embed(tone(1.0, 400)))
    return SessionPipeline(fake, transcriber or FakeTranscriber(), reg), reg


def test_process_segment():
    p, _ = make()
    [u] = p.process(Segment(tone(1.0, 400), 1.23456, 2.23456))
    assert u.seq == 1 and u.speaker_name == "Bob"
    assert (u.start, u.end) == (1.235, 2.235)
    msg = u.to_message()
    assert msg["type"] == "utterance" and msg["text"] == "tone 400"
    assert [u.seq for u in p.process(Segment(tone(1.0, 200), 3, 4))] == [2]


def test_empty_transcript_skipped_without_consuming_seq():
    p, _ = make(FakeTranscriber(silent=(200,)))
    assert p.process(Segment(tone(1.0, 200), 0, 1)) == []
    assert [u.seq for u in p.process(Segment(tone(1.0, 400), 1, 2))] == [1]


def test_null_transcriber():
    t = NullTranscriber()
    assert t.transcribe(np.zeros(24000, np.float32)) == "[speech 1.5s]"
    assert t.transcribe_words(np.zeros(24000, np.float32)) == []  # no word timing
    assert build_transcriber("null").name == "null"


def test_whisper_transcribe_words_uses_word_timestamps():
    import threading
    from types import SimpleNamespace as NS

    from app.transcription.transcriber import WhisperTranscriber, Word

    calls = []

    class FakeModel:
        def transcribe(self, audio, **kwargs):
            calls.append(kwargs)

            def segments():  # a lazy generator, like faster-whisper's
                yield NS(text=" Hello there.", words=[NS(word=" Hello", start=0.1, end=0.4),
                                                       NS(word=" there.", start=0.4, end=0.8)])
                yield NS(text=" Hi!", words=[NS(word=" Hi!", start=1.0, end=1.2)])
                yield NS(text="", words=None)

            return segments(), NS(language="en")

    w = WhisperTranscriber.__new__(WhisperTranscriber)
    w._model, w._language, w._lock = FakeModel(), "en", threading.Lock()
    words = w.transcribe_words(np.zeros(16000, np.float32))
    assert words == [Word(" Hello", 0.1, 0.4), Word(" there.", 0.4, 0.8), Word(" Hi!", 1.0, 1.2)]
    assert calls[-1]["word_timestamps"] is True and calls[-1]["beam_size"] == 1
    assert w.transcribe(np.zeros(16000, np.float32)) == "Hello there. Hi!"
    assert calls[-1]["word_timestamps"] is False


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
    assert [u.speaker_name for u in p.process(segments[0])] == ["Alice"]
    # smoothed toward the previous speaker
    assert [u.speaker_name for u in p.process(short)] == ["Alice"]


# -- speaker-turn splitting -------------------------------------------------------

def interruption(a: float = 3.0, b: float = 3.0) -> np.ndarray:
    """Alice (200 Hz) then Bob (400 Hz) with no pause, padded like a VAD segment."""
    return np.concatenate([silence(0.3), tone(a, 200), tone(b, 400), silence(0.2)])


def seg(audio: np.ndarray, start: float = 10.0) -> Segment:
    return Segment(audio, start, start + audio.size / 16000, speech_duration=audio.size / 16000 - 0.5)


def test_fake_word_transcriber():
    words = FakeTranscriber().transcribe_words(interruption())
    assert [w.text.strip() for w in words] == ["tone", "200", "tone", "400"]
    assert words[0].start == pytest.approx(0.3, abs=0.11) and words[1].end == pytest.approx(3.3, abs=0.11)
    assert tone_runs(silence(1.0)) == []


def test_interruption_with_word_timestamps_gives_two_utterances():
    p, reg = make()
    alice, bob = (s["id"] for s in reg.list())
    utts = p.process(seg(interruption()))
    assert [(u.seq, u.speaker_id, u.speaker_name, u.text) for u in utts] == [
        (1, alice, "Alice", "tone 200"),
        (2, bob, "Bob", "tone 400"),
    ]
    a, b = utts
    # Snapped to the first/last word of each piece, offset by the segment start.
    assert a.start == pytest.approx(10.3, abs=0.11) and a.end == pytest.approx(13.3, abs=0.11)
    assert b.start == pytest.approx(13.3, abs=0.11) and b.end == pytest.approx(16.3, abs=0.11)
    assert a.confidence == b.confidence == 1.0
    assert a.id != b.id
    # seq continues across segments
    assert [u.seq for u in p.process(seg(np.concatenate([silence(0.3), tone(1.0, 200)]), 20))] == [3]


def test_interruption_without_word_timestamps_cuts_audio():
    p, reg = make(NullTranscriber())
    utts = p.process(seg(interruption()))
    assert [u.speaker_name for u in utts] == ["Alice", "Bob"]
    a, b = utts
    assert a.start == 10.0 and b.end == pytest.approx(16.5)
    assert a.end == b.start == pytest.approx(13.3, abs=0.5)
    assert a.text.startswith("[speech ") and b.text.startswith("[speech ")
    assert [u.seq for u in utts] == [1, 2]


def test_three_turns_in_order():
    p, _ = make()
    audio = np.concatenate([silence(0.3), tone(2.5, 200), tone(2.5, 400), tone(2.5, 200), silence(0.2)])
    utts = p.process(seg(audio, 0.0))
    assert [(u.seq, u.speaker_name) for u in utts] == [(1, "Alice"), (2, "Bob"), (3, "Alice")]
    assert all(x.end <= y.start + 1e-6 for x, y in zip(utts, utts[1:]))


def test_no_split_for_short_segments_or_single_speaker():
    p, _ = make()
    short = np.concatenate([silence(0.3), tone(0.8, 200), tone(0.8, 400), silence(0.2)])
    assert len(p.process(seg(short))) == 1  # < CT_TURN_MIN_SEGMENT_S of speech
    fake = FakeEmbedder()
    reg = SpeakerRegistry(fake.name, fake.default_threshold)
    reg.enroll("Alice", fake.embed(tone(1.0, 200)))
    single = SessionPipeline(fake, FakeTranscriber(), reg)
    assert len(single.process(seg(interruption()))) == 1  # only one person enrolled
    assert fake.calls == 2  # enrollment + whole segment: no window embeddings


def test_empty_pieces_are_dropped():
    p, _ = make(FakeTranscriber(silent=(400,)))
    utts = p.process(seg(interruption()))
    assert [(u.seq, u.speaker_name, u.text) for u in utts] == [(1, "Alice", "tone 200")]
    assert utts[0].end == pytest.approx(13.3, abs=0.11)  # Alice's words only


def test_false_split_is_merged_back(monkeypatch):
    import app.pipeline as pipeline_mod

    # Turn detection claims Alice|Bob, but both halves re-identify as Alice.
    def fake_turns(audio, *args):
        mid = audio.size / 16000 / 2
        alice, bob = (s["id"] for s in reg.list())
        return [Turn(0.0, mid, alice, 0.9), Turn(mid, audio.size / 16000, bob, 0.9)]

    p, reg = make()
    monkeypatch.setattr(pipeline_mod, "detect_turns", fake_turns)
    audio = np.concatenate([silence(0.3), tone(6.0, 200), silence(0.2)])
    [u] = p.process(seg(audio))
    assert (u.speaker_name, u.text) == ("Alice", "tone 200")
    assert (u.start, u.end) == (10.0, 16.5)  # like an unsplit segment


def test_short_piece_keeps_turn_label(monkeypatch):
    import app.pipeline as pipeline_mod

    def fake_turns(audio, *args):
        alice, bob = (s["id"] for s in reg.list())
        end = audio.size / 16000
        return [Turn(0.0, 3.3, alice, 0.9), Turn(3.3, end, bob, 0.8)]

    p, reg = make(NullTranscriber())
    monkeypatch.setattr(pipeline_mod, "detect_turns", fake_turns)
    # Bob's piece is 0.8 s (< 1 s): not re-embedded, labelled from its turn.
    audio = np.concatenate([silence(0.3), tone(3.0, 200), tone(0.6, 200), silence(0.2)])
    utts = p.process(seg(audio))
    assert [(u.speaker_name, u.confidence) for u in utts] == [("Alice", 1.0), ("Bob", 0.8)]


def test_live_utterances_feed_adaptive_prototypes():
    p, reg = make()
    alice, bob = (s["id"] for s in reg.list())
    p.process(seg(np.concatenate([silence(0.3), tone(1.5, 200), silence(0.2)])))
    assert reg.num_adaptive(alice) == 0  # < 2 s of speech
    p.process(seg(np.concatenate([silence(0.3), tone(3.0, 200), silence(0.2)])))
    assert reg.num_adaptive(alice) == 1
    p.process(seg(interruption()))  # each >= 2 s piece is confident on its own
    assert (reg.num_adaptive(alice), reg.num_adaptive(bob)) == (2, 1)
