"""Speaker-turn detection inside one segment (app/speakers/turns.py)."""

import numpy as np
import pytest

from app.speakers.embedder import l2_normalize
from app.speakers.registry import ScoreTable, SpeakerRegistry
from app.speakers.turns import TurnConfig, detect_turns, label_runs, window_starts
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
    assert emb.batches == [40]  # one batch, at most 40 windows (hop grew)
    starts = window_starts(30 * SR, int(1.5 * SR), int(0.5 * SR), 40)
    assert len(starts) == 40 and starts[0] == 0 and starts[-1] == 30 * SR - int(1.5 * SR)


def test_falls_back_to_embed_when_no_batch_support():
    emb = FakeEmbedder()
    reg, _ = enrolled(emb, A, B)
    calls = emb.calls
    detect_turns(np.concatenate([tone(3.0, A), tone(3.0, B)]), emb, reg)
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
    # A short Unknown run between A and B joins the neighbour that fits it better.
    t = ScoreTable(
        ["a", "b"],
        np.asarray([[0.9, 0.1]] * 5 + [[0.3, 0.45]] * 2 + [[0.1, 0.9]] * 5, dtype=np.float32),
        0.5,
    )
    centres = [0.75 + i * 0.4 for i in range(12)]
    turns = label_runs(t, centres, centres[-1] + 0.75, TurnConfig())
    assert [x.speaker_id for x in turns] == ["a", "b"]
    assert turns[0].end == pytest.approx((centres[4] + centres[5]) / 2)


def test_long_enough_runs_and_boundaries():
    assert runs_of("AAAAAABBBBBB") == [("a", 0.0, 3.5), ("b", 3.5, 7.0)]
    assert runs_of("AAAAAABBBBBB", min_turn_s=0.0) == [("a", 0.0, 3.5), ("b", 3.5, 7.0)]
    # A trailing one-window run (0.2 + 0.75 s): the filter cannot fix an edge,
    # the length rule does.
    assert runs_of("AAAAAAAAB", hop=0.4) == [("a", 0.0, 4.7)]
    assert runs_of("AAAAAAAAB", hop=0.4, min_turn_s=0.0)[-1] == ("b", 3.75, 4.7)
