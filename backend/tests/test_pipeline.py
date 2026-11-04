import numpy as np
import pytest

from app.audio.vad import Segment
from app.pipeline import SessionPipeline
from app.speakers.registry import SpeakerRegistry
from app.speakers.turns import Turn, TurnAnalysis
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
    assert t.transcribe_words(np.zeros(24000, np.float32)) is None  # no word timing
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


def test_short_segments_and_single_enrollment_still_detect_interruptions():
    p, _ = make()
    short = np.concatenate([silence(0.3), tone(0.8, 200), tone(0.8, 400), silence(0.2)])
    assert [u.speaker_name for u in p.process(seg(short))] == ["Alice", "Bob"]
    fake = FakeEmbedder()
    reg = SpeakerRegistry(fake.name, fake.default_threshold)
    reg.enroll("Alice", fake.embed(tone(1.0, 200)))
    single = SessionPipeline(fake, FakeTranscriber(), reg)
    assert [u.speaker_name for u in single.process(seg(interruption()))] == ["Alice", "Unknown"]


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
        turns = [Turn(0.0, mid, alice, 0.9), Turn(mid, audio.size / 16000, bob, 0.9)]
        return TurnAnalysis(turns, False, None)

    p, reg = make()
    monkeypatch.setattr(pipeline_mod, "analyze_turns", fake_turns)
    audio = np.concatenate([silence(0.3), tone(6.0, 200), silence(0.2)])
    [u] = p.process(seg(audio))
    assert (u.speaker_name, u.text) == ("Alice", "tone 200")
    assert (u.start, u.end) == (10.0, 16.5)  # like an unsplit segment


def test_short_piece_keeps_turn_label(monkeypatch):
    import app.pipeline as pipeline_mod

    def fake_turns(audio, *args):
        alice, bob = (s["id"] for s in reg.list())
        end = audio.size / 16000
        return TurnAnalysis([Turn(0.0, 3.3, alice, 0.9), Turn(3.3, end, bob, 0.8)], False, None)

    p, reg = make(NullTranscriber())
    monkeypatch.setattr(pipeline_mod, "analyze_turns", fake_turns)
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
    # Split pieces never adapt (they may hold the boundary mixture of two voices).
    p.process(seg(interruption()))
    assert (reg.num_adaptive(alice), reg.num_adaptive(bob)) == (1, 0)


# -- regressions (adversarial review of the turn splitter) -------------------------------
class NoWordsTranscriber:
    """Whisper-like: hears nothing in the segment; hallucinates on short clips."""

    name = "nowords"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def transcribe_words(self, audio):
        self.calls.append("words")
        return []

    def transcribe(self, audio):
        self.calls.append("text")
        return "Thank you."


def test_no_words_means_nothing_said_not_no_timing():
    tr = NoWordsTranscriber()
    p, _ = make(tr)
    turns = [tone(3.0, 200), tone(3.0, 400), tone(3.0, 200)]
    audio = np.concatenate([silence(0.3), *turns, silence(0.2)])
    assert p.process(seg(audio)) == []
    assert tr.calls == ["words"]  # no per-turn re-transcription


def test_transcribers_without_word_timing_still_cut_audio():
    class OldStyle:  # no transcribe_words at all
        name = "old"

        def transcribe(self, audio):
            return f"{audio.size / 16000:.1f}s"

    for tr in (NullTranscriber(), OldStyle()):
        p, _ = make(tr)
        utts = p.process(seg(interruption()))
        assert [u.speaker_name for u in utts] == ["Alice", "Bob"], tr


def test_whisper_text_without_words_means_no_timing():
    import threading
    from types import SimpleNamespace as NS

    from app.transcription.transcriber import WhisperTranscriber

    class FakeModel:
        def __init__(self, segments):
            self.segments = segments

        def transcribe(self, audio, **kwargs):
            return iter(self.segments), NS(language="en")

    w = WhisperTranscriber.__new__(WhisperTranscriber)
    w._language, w._lock = "en", threading.Lock()
    w._model = FakeModel([])
    assert w.transcribe_words(np.zeros(16000, np.float32)) == []  # heard nothing
    w._model = FakeModel([NS(text=" Hi.", words=None)])
    assert w.transcribe_words(np.zeros(16000, np.float32)) is None  # text but no timing


class FixedWords:
    name = "fixed"

    def __init__(self, words):
        self.words = words

    def transcribe_words(self, audio):
        return list(self.words)

    def transcribe(self, audio):
        return "unused"


def _check_pieces(utts, segment):
    for u in utts:
        assert segment.start <= u.start < u.end <= segment.end, (u.start, u.end)
    for x, y in zip(utts, utts[1:]):
        assert x.end <= y.start


@pytest.mark.parametrize("order", ["sorted", "reversed", "shuffled"])
def test_unsorted_and_nan_word_times_give_ordered_valid_pieces(order):
    from app.transcription.transcriber import Word

    nan = float("nan")
    words = [Word(f" w{i:02d}", 0.4 + 0.3 * i, 0.6 + 0.3 * i) for i in range(20)]
    words += [Word(" n1", nan, nan), Word(" n2", 4.4, nan), Word(" z", 1.0, 1.0)]
    if order == "reversed":
        words = words[::-1]
    elif order == "shuffled":
        words = [words[i] for i in np.random.default_rng(0).permutation(len(words))]
    p, _ = make(FixedWords(words))
    segment = seg(interruption())
    utts = p.process(segment)
    assert [u.speaker_name for u in utts] == ["Alice", "Bob"]
    _check_pieces(utts, segment)
    tokens = [t for u in utts for t in u.text.split()]
    assert sorted(tokens) == sorted(w.text.strip() for w in words)  # nothing lost
    timed = [t for t in tokens if t.startswith("w")]
    assert timed == sorted(timed)  # chronological


def test_degenerate_word_times_never_give_empty_spans():
    from app.transcription.transcriber import Word

    # Every word of Bob's turn is zero-length at the same instant; one Alice word reversed.
    words = [
        Word(" a1", 1.0, 2.0), Word(" a2", 3.0, 2.5), Word(" b1", 5.0, 5.0), Word(" b2", 5.0, 5.0)
    ]
    p, _ = make(FixedWords(words))
    segment = seg(interruption())
    utts = p.process(segment)
    assert [(u.speaker_name, u.text) for u in utts] == [("Alice", "a1 a2"), ("Bob", "b1 b2")]
    _check_pieces(utts, segment)


def test_words_join_without_inventing_spaces():
    from app.transcription.transcriber import Word, words_to_text

    assert words_to_text([Word(" Hello", 0, 1), Word(" there.", 1, 2)]) == "Hello there."
    assert words_to_text([Word("你", 0, 1), Word("好", 1, 2), Word("。", 2, 3)]) == "你好。"
    assert words_to_text([Word("สวัส", 0, 1), Word("ดี", 1, 2)]) == "สวัสดี"
    # Transcribers whose words carry no spacing still get spaces between Latin words.
    assert words_to_text([Word("A1", 0, 1), Word("A2", 1, 2)]) == "A1 A2"


def test_cjk_pieces_of_one_speaker_merge_without_spaces(monkeypatch):
    """Two turns re-identified as the same person: text joins like Whisper's words."""
    import app.pipeline as pipeline_mod
    from app.transcription.transcriber import Word

    def fake_turns(audio, *args):
        alice, bob = (s["id"] for s in reg.list())
        end = audio.size / 16000
        return TurnAnalysis([Turn(0.0, 3.3, alice, 0.9), Turn(3.3, end, bob, 0.9)], False, None)

    words = [
        Word("你", 0.5, 1.5), Word("好", 1.5, 3.0), Word("谢", 3.5, 5.0), Word("谢", 5.0, 6.0)
    ]
    p, reg = make(FixedWords(words))
    monkeypatch.setattr(pipeline_mod, "analyze_turns", fake_turns)
    audio = np.concatenate([silence(0.3), tone(6.0, 200), silence(0.2)])  # all Alice
    [u] = p.process(seg(audio))
    assert u.text == "你好谢谢"


def test_no_adaptation_from_split_or_mixed_segments():
    p, reg = make()
    alice, bob = (s["id"] for s in reg.list())
    p.process(seg(interruption(3.0, 3.0)))
    assert reg.num_adaptive(alice) == reg.num_adaptive(bob) == 0  # split pieces
    # A short but confident Bob interruption is retained, never used for adaptation.
    audio = np.concatenate([tone(3.0, 200), tone(0.8, 400), tone(2.7, 200)])
    utterances = p.process(Segment(audio, 0.0, 6.5, speech_duration=6.5))
    assert [u.speaker_name for u in utterances] == ["Alice", "Bob", "Alice"]
    assert reg.num_adaptive(alice) == reg.num_adaptive(bob) == 0
    # A clean Alice segment does adapt.
    p.process(seg(np.concatenate([silence(0.3), tone(6.0, 200), silence(0.2)])))
    assert reg.num_adaptive(alice) == 1


class SpyEmbedder(FakeEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self.sizes: list[int] = []

    def embed(self, audio):
        self.sizes.append(audio.size)
        return super().embed(audio)


def test_single_voice_segment_embedding_work_is_bounded():
    fake = SpyEmbedder()
    reg = SpeakerRegistry(fake.name, fake.default_threshold)
    reg.enroll("Alice", fake.embed(tone(1.0, 200)))
    reg.enroll("Bob", fake.embed(tone(1.0, 400)))
    p = SessionPipeline(fake, FakeTranscriber(), reg)
    fake.sizes.clear()
    audio = np.concatenate([silence(0.3), tone(14.5, 200), silence(0.2)])  # 15 s
    [u] = p.process(seg(audio))
    assert u.speaker_name == "Alice"
    # Bounded short-window scan plus the coarse single-speaker pass.
    assert max(fake.sizes) == int(1.5 * 16000)
    assert sum(fake.sizes) <= 3 * audio.size


def test_word_timestamps_crossing_cut_do_not_reidentify_from_previous_voice(monkeypatch):
    """A loud old voice inside a mistimed new word must not erase the handoff."""
    import app.pipeline as pipeline_mod
    from app.transcription.transcriber import Word

    p, reg = make(FixedWords([Word(' Alice', 0.2, 1.0), Word(' Bob', 0.5, 2.5)]))
    alice, bob = (s['id'] for s in reg.list())
    monkeypatch.setattr(
        pipeline_mod, 'analyze_turns',
        lambda *args: TurnAnalysis([Turn(0, 1.2, alice, 1), Turn(1.2, 3, bob, 1)], False, None),
    )
    audio = np.concatenate([tone(1.2, 200, amp=0.8), tone(1.8, 400, amp=0.1)])
    utterances = p.process(Segment(audio, 0, 3, speech_duration=3))
    assert [(u.speaker_name, u.text) for u in utterances] == [('Alice', 'Alice'), ('Bob', 'Bob')]
    assert utterances[0].end <= 1.2 <= utterances[1].start


def test_live_interruption_splits_before_silence_or_stop():
    p, reg = make()
    alice, bob = (s['id'] for s in reg.list())
    source = np.concatenate([silence(0.3), tone(2, 200), tone(2, 400)])
    segments = p.feed(source)
    assert segments, 'speaker change must finish Alice without waiting for silence'
    events = p.pop_speaker_events()
    assert [e['speaker_id'] for e in events] == [alice, bob]
    assert events[-1]['start'] == pytest.approx(2.3, abs=0.06)
    utterances = [u for segment in segments for u in p.process(segment)]
    assert [(u.speaker_name, u.text) for u in utterances] == [('Alice', 'tone 200')]
    tail = p.flush()
    assert segments[-1].end == tail[0].start
    assert [(u.speaker_name, u.text) for segment in tail for u in p.process(segment)] == [('Bob', 'tone 400')]


def test_confirmed_online_boundary_disables_continuity_and_adaptation():
    emb = AmbiguousEmbedder()
    reg = SpeakerRegistry(emb.name, emb.default_threshold)
    alice = reg.enroll('Alice', emb.embed(tone(1, 200)))['id']
    reg.enroll('Bob', emb.embed(tone(1, 400)))
    p = SessionPipeline(emb, FakeTranscriber(), reg)
    p.process(Segment(tone(1.5, 200), 0, 1.5))
    [utterance] = p.process(Segment(tone(0.7, 300), 1.5, 2.2, speaker_change=True))
    assert utterance.speaker_name == 'Bob'
    assert reg.num_adaptive(alice) == 0
