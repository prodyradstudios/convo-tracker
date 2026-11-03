"""Speaker-turn detection inside one segment (app/speakers/turns.py)."""

import numpy as np
import pytest

from app.speakers.embedder import l2_normalize
from app.speakers.registry import ScoreTable, SpeakerRegistry
from app.audio.vad import VadConfig
from app.speakers.turns import (
    TurnConfig,
    analyze_turns,
    detect_turns,
    label_runs,
    voiced_frames,
    window_starts,
)
from conftest import SR, FakeEmbedder, silence, tone

A, B, C = 200, 400, 600  # "voices": tone frequencies


class BandEmbedder:
    """Maps each frequency band's energy share to a fixed embedding and mixes them.

    A window holding 70 % voice A and 30 % voice B embeds as 0.7*eA + 0.3*eB
    (normalised), like a real embedder smearing two voices together.
    """

    name = "band"
    default_threshold = 0.5

    def __init__(self) -> None:
        self.batches: list[int] = []

    def embed(self, audio: np.ndarray) -> np.ndarray:
        spec = np.abs(np.fft.rfft(audio)) ** 2
        freqs = np.fft.rfftfreq(audio.size, 1 / SR)
        bands = np.array([spec[(freqs >= f - 50) & (freqs < f + 50)].sum() for f in (A, B, C)])
        vec = np.zeros(8, dtype=np.float32)
        vec[:3] = np.sqrt(bands / bands.sum()) if bands.sum() > 0 else 0.0
        vec[7] = 1e-3  # never all-zero
        return l2_normalize(vec)

    def embed_batch(self, clips: list[np.ndarray]) -> np.ndarray:
        self.batches.append(len(clips))
        return np.stack([self.embed(c) for c in clips])


def enrolled(embedder, *freqs: int) -> tuple[SpeakerRegistry, dict[str | None, str]]:
    reg = SpeakerRegistry(embedder.name, embedder.default_threshold)
    names: dict[str | None, str] = {None: "?"}
    for name, freq in zip("ABC", freqs):
        names[reg.enroll(name, embedder.embed(tone(2.0, freq)))["id"]] = name
    return reg, names


def summary(turns, names) -> list[str]:
    return [names[t.speaker_id] for t in turns]


@pytest.mark.parametrize("embedder", [FakeEmbedder(), BandEmbedder()], ids=["onehot", "band"])
def test_two_voices_without_pause_give_two_turns(embedder):
    reg, names = enrolled(embedder, A, B)
    audio = np.concatenate([tone(3.0, A), tone(3.0, B)])
    turns = detect_turns(audio, embedder, reg)
    assert summary(turns, names) == ["A", "B"]
    assert turns[0].start == 0.0 and turns[-1].end == pytest.approx(6.0)
    assert turns[0].end == turns[1].start
    assert turns[0].end == pytest.approx(3.0, abs=0.5)
    assert all(t.score >= reg.threshold for t in turns)


def test_turns_with_padding_and_three_voices():
    emb = BandEmbedder()
    reg, names = enrolled(emb, A, B, C)
    audio = np.concatenate([silence(0.3), tone(2.5, A), tone(2.5, B), tone(2.5, C), silence(0.2)])
    turns = detect_turns(audio, emb, reg)
    assert summary(turns, names) == ["A", "B", "C"]
    assert turns[0].end == pytest.approx(2.8, abs=0.5)
    assert turns[1].end == pytest.approx(5.3, abs=0.5)


@pytest.mark.parametrize("blip", [0.4, 0.8])
@pytest.mark.parametrize("embedder", [FakeEmbedder(), BandEmbedder()], ids=["onehot", "band"])
def test_short_blip_of_other_voice_is_smoothed_away(embedder, blip):
    reg, names = enrolled(embedder, A, B)
    audio = np.concatenate([tone(3.0, A), tone(blip, B), tone(3.0, A)])
    turns = detect_turns(audio, embedder, reg)
    assert summary(turns, names) == ["A"]
    assert (turns[0].start, turns[0].end) == (0.0, pytest.approx(audio.size / SR))


@pytest.mark.parametrize("embedder", [FakeEmbedder(), BandEmbedder()], ids=["onehot", "band"])
def test_single_speaker_is_one_turn(embedder):
    reg, names = enrolled(embedder, A, B)
    turns = detect_turns(np.concatenate([silence(0.3), tone(6.0, A), silence(0.2)]), embedder, reg)
    assert summary(turns, names) == ["A"]


def test_unknown_voice_gets_its_own_turn():
    emb = BandEmbedder()
    reg, names = enrolled(emb, A, B)  # nobody enrolled at C
    turns = detect_turns(np.concatenate([tone(3.0, A), tone(3.0, C)]), emb, reg)
    assert summary(turns, names) == ["A", "?"]


def test_nothing_to_split():
    emb = BandEmbedder()
    reg, _ = enrolled(emb, A)  # one speaker: no turns
    audio = np.concatenate([tone(3.0, A), tone(3.0, B)])
    assert len(detect_turns(audio, emb, reg)) == 1
    reg, _ = enrolled(emb, A, B)
    assert len(detect_turns(tone(1.8, A), emb, reg)) == 1  # shorter than two windows
    # Mostly silence: windows are skipped, nothing to compare.
    assert len(detect_turns(np.concatenate([tone(0.4, A), silence(3.0)]), emb, reg)) == 1


def test_silent_windows_are_not_embedded():
    emb = BandEmbedder()
    reg, names = enrolled(emb, A, B)
    audio = np.concatenate([tone(3.0, A), silence(3.0), tone(3.0, B)])
    turns = detect_turns(audio, emb, reg)
    assert summary(turns, names) == ["A", "B"]
    assert 3.0 <= turns[0].end <= 6.0  # the cut falls in the pause
    all_windows = len(window_starts(audio.size, int(1.5 * SR), int(0.5 * SR), 40))
    assert sum(emb.batches) < all_windows


def test_window_count_is_capped_and_batched():
    emb = BandEmbedder()
    reg, _ = enrolled(emb, A, B)
    detect_turns(np.concatenate([tone(15.0, A), tone(15.0, B)]), emb, reg)
    # One batch of chunks tiling the segment (they disagree), then one batch of
    # the remaining windows: at most 40 windows in all (hop grew).
    assert len(emb.batches) == 2 and sum(emb.batches) == 40
    starts = window_starts(30 * SR, int(1.5 * SR), int(0.5 * SR), 40)
    assert len(starts) == 40 and starts[0] == 0 and starts[-1] == 30 * SR - int(1.5 * SR)


def test_falls_back_to_embed_when_no_batch_support():
    emb = FakeEmbedder()
    reg, _ = enrolled(emb, A, B)
    calls = emb.calls
    detect_turns(np.concatenate([tone(3.0, A), tone(3.0, B)]), emb, reg)
    # Chunk embeddings are reused: each window is embedded once.
    assert emb.calls - calls == len(window_starts(6 * SR, int(1.5 * SR), int(0.5 * SR), 40))


# -- label smoothing (pure) -----------------------------------------------------
def table(labels: str, threshold: float = 0.5) -> ScoreTable:
    """Windows labelled by letters: A/B = that speaker scores 0.9, '?' = all below threshold."""
    ids = ["a", "b"]
    rows = []
    for ch in labels:
        rows.append({"A": [0.9, 0.1], "B": [0.1, 0.9], "?": [0.3, 0.2]}[ch])
    return ScoreTable(ids, np.asarray(rows, dtype=np.float32), threshold)


def runs_of(labels: str, hop: float = 0.5, **cfg) -> list[tuple[str | None, float, float]]:
    centres = [0.75 + i * hop for i in range(len(labels))]
    duration = centres[-1] + 0.75
    turns = label_runs(table(labels), centres, duration, TurnConfig(**cfg))
    return [(t.speaker_id, round(t.start, 2), round(t.end, 2)) for t in turns]


def test_majority_filter_removes_single_window_flips():
    assert runs_of("AAAABAAAA") == [("a", 0.0, 5.5)]
    assert runs_of("AAAA?AAAA") == [("a", 0.0, 5.5)]


def test_short_runs_merge_into_better_fitting_neighbour():
    # Two windows of B with a 0.4 s hop = 0.8 s < 1.0 s -> merged into A.
    assert runs_of("AAAAABBAAAAA", hop=0.4) == [("a", 0.0, 5.9)]
    # A short Unknown run between two different speakers is the boundary
    # mixture: it is split at its midpoint between them.
    t = ScoreTable(
        ["a", "b"],
        np.asarray([[0.9, 0.1]] * 5 + [[0.3, 0.45]] * 2 + [[0.1, 0.9]] * 5, dtype=np.float32),
        0.5,
    )
    centres = [0.75 + i * 0.4 for i in range(12)]
    turns = label_runs(t, centres, centres[-1] + 0.75, TurnConfig())
    assert [x.speaker_id for x in turns] == ["a", "b"]
    assert turns[0].end == pytest.approx((centres[5] + centres[6]) / 2)


def test_long_enough_runs_and_boundaries():
    assert runs_of("AAAAAABBBBBB") == [("a", 0.0, 3.5), ("b", 3.5, 7.0)]
    assert runs_of("AAAAAABBBBBB", min_turn_s=0.0) == [("a", 0.0, 3.5), ("b", 3.5, 7.0)]
    # A trailing one-window run: the one-sided edge filter fixes it, and a
    # single window is never a turn of its own (even with min_turn_s=0).
    assert runs_of("AAAAAAAAB", hop=0.4) == [("a", 0.0, 4.7)]
    assert runs_of("AAAAAAAAB", hop=0.4, min_turn_s=0.0) == [("a", 0.0, 4.7)]


# -- regressions (adversarial review of the turn splitter) -------------------------------
@pytest.mark.parametrize("duration", [3.5, 3.6, 5.0, 5.1, 8.0, 8.05])
@pytest.mark.parametrize("edge", ["first", "last"])
def test_edge_window_flip_is_smoothed_whatever_the_spacing(duration, edge):
    """One odd first/last window (onset, pre-roll) must not become a 1 s turn.

    The spacing equals the hop exactly for some durations (3.5, 5.0, 8.0) and not
    for others; the result must not depend on it.
    """
    win, hop = int(1.5 * SR), int(0.5 * SR)
    starts = window_starts(int(duration * SR), win, hop, 40)
    centres = [(s + win / 2) / SR for s in starts]
    scores = np.tile([0.9, 0.1], (len(starts), 1)).astype(np.float32)
    scores[0 if edge == "first" else -1] = [0.1, 0.9]
    turns = label_runs(ScoreTable(["a", "b"], scores, 0.3), centres, duration, TurnConfig())
    assert [(t.speaker_id, t.start, t.end) for t in turns] == [("a", 0.0, duration)]


def test_two_window_run_at_exactly_min_turn_is_merged():
    # Two windows at a 0.5 s hop span exactly 1.0 s = min_turn_s: still too short.
    assert runs_of("AAAAABBAAAAA", hop=0.5) == [("a", 0.0, 7.0)]
    # ... also when the spacing is a hair off the hop (linspace rounding).
    assert runs_of("AAAAABBAAAAA", hop=0.5 - 1e-7) == [("a", 0.0, 7.0)]


@pytest.mark.parametrize("unknown", [2, 3])
def test_boundary_mixture_is_split_between_the_two_speakers(unknown):
    """Windows straddling A->B score below threshold for both: no Unknown turn."""
    labels = "AAAAA" + "?" * unknown + "BBBBB"
    turns = runs_of(labels)
    assert [t[0] for t in turns] == ["a", "b"]
    centres = [0.75 + i * 0.5 for i in range(len(labels))]
    mixture_mid = (centres[5] + centres[4 + unknown]) / 2  # midpoint of the Unknown run
    assert turns[0][2] == pytest.approx(mixture_mid, abs=0.25 + 1e-6)


def test_real_unknown_turns_survive_smoothing():
    # Longer than a window between A and B: a third (unenrolled) person.
    assert [t[0] for t in runs_of("AAAAA????BBBBB")] == ["a", None, "b"]
    # Between the same speaker it is not a boundary mixture either.
    assert [t[0] for t in runs_of("AAAAA???AAAAA")] == ["a", None, "a"]


def test_quiet_second_speaker_frames_are_voiced():
    """B 10-28 dB quieter than A but far above the VAD's -50 dBFS: still voiced."""
    vad = VadConfig()
    for gap in (10, 16, 22, 28):
        quiet = tone(4.0, B, amp=0.3 * 10 ** (-gap / 20))
        audio = np.concatenate([silence(0.3), tone(4.0, A), quiet, silence(0.2)])
        v = voiced_frames(audio, vad)
        f = vad.frame_len
        b = v[int(4.3 * SR) // f : int(8.3 * SR) // f]
        assert b.mean() > 0.95, (gap, b.mean())
        # Same with the segmenter's noise floor and no padding at all.
        v = voiced_frames(np.concatenate([tone(4.0, A), quiet]), vad, noise_floor_db=-60.0)
        assert v[int(4.0 * SR) // f :].mean() > 0.95
    # Background noise alone is not voiced.
    assert not voiced_frames(silence(2.0), vad, noise_floor_db=-60.0).any()


def test_quiet_second_speaker_gets_a_turn():
    emb = BandEmbedder()
    reg, names = enrolled(emb, A, B)
    quiet = tone(4.0, B, amp=0.3 * 10 ** (-28 / 20))
    audio = np.concatenate([silence(0.3), tone(4.0, A), quiet, silence(0.2)])
    assert summary(detect_turns(audio, emb, reg), names) == ["A", "B"]


def test_chunk_precheck_skips_windows_for_one_clear_voice():
    emb = BandEmbedder()
    reg, names = enrolled(emb, A, B)
    audio = np.concatenate([silence(0.3), tone(14.5, A), silence(0.2)])  # 15 s
    result = analyze_turns(audio, emb, reg)
    assert emb.batches == [10]  # ten 1.5 s chunks, no other sliding windows
    assert summary(result.turns, names) == ["A"] and result.clean
    np.testing.assert_allclose(result.embedding, emb.embed(tone(2.0, A)), atol=1e-2)
    # Two voices: the chunks disagree, so the fine windows run as well.
    emb.batches.clear()
    result = analyze_turns(np.concatenate([tone(6.0, A), tone(6.0, B)]), emb, reg)
    assert len(emb.batches) == 2 and summary(result.turns, names) == ["A", "B"]
    assert sum(emb.batches) == len(window_starts(12 * SR, int(1.5 * SR), int(0.5 * SR), 40))
    assert not result.clean


def test_disagreeing_window_marks_segment_unclean():
    """A blip of another voice is smoothed away but the segment is not 'clean'."""
    emb = BandEmbedder()
    reg, names = enrolled(emb, A, B)
    # 6.5 s: chunks at 0, 1.5, 3.0 ... s; the one at 3.0 s is half B (ambiguous).
    result = analyze_turns(np.concatenate([tone(3.0, A), tone(0.8, B), tone(2.7, A)]), emb, reg)
    assert summary(result.turns, names) == ["A"]
    assert not result.clean
