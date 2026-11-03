import json
import threading

import numpy as np
import pytest

from app.speakers.registry import (
    PALETTE,
    UNKNOWN_COLOR,
    DuplicateNameError,
    SpeakerNotFoundError,
    SpeakerRegistry,
)


def vec(*values: float) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


@pytest.fixture
def path(tmp_path):
    return tmp_path / "data" / "speakers.json"


@pytest.fixture
def reg(path) -> SpeakerRegistry:
    return SpeakerRegistry("fake", threshold=0.5, path=path)


def test_enroll_and_list(reg):
    s = reg.enroll("  Alice ", vec(1, 0, 0))
    assert s["name"] == "Alice" and s["num_samples"] == 1
    assert len(s["id"]) == 32 and s["color"] == PALETTE[0]
    assert isinstance(s["created_at"], float)
    assert reg.list() == [s]
    assert reg.get(s["id"]) == s


def test_duplicate_name_case_insensitive(reg):
    reg.enroll("Alice", vec(1, 0, 0))
    assert reg.has_name(" alice ")
    with pytest.raises(DuplicateNameError):
        reg.enroll("ALICE", vec(0, 1, 0))


def test_colors_cycle_and_reuse_freed(reg):
    ids = [reg.enroll(f"p{i}", vec(1, 0, 0))["id"] for i in range(len(PALETTE) + 1)]
    colors = [reg.get(i)["color"] for i in ids]
    assert colors[: len(PALETTE)] == list(PALETTE)
    assert colors[-1] == PALETTE[0]
    reg.delete(ids[3])
    assert reg.enroll("new", vec(1, 0, 0))["color"] == PALETTE[3]


def test_identify_best_and_unknown(reg):
    a = reg.enroll("Alice", vec(1, 0, 0))["id"]
    b = reg.enroll("Bob", vec(0, 1, 0))["id"]
    m = reg.identify(vec(0.9, 0.1, 0), duration=2.0)
    assert m.speaker_id == a and m.speaker_name == "Alice"
    assert m.confidence == round(0.9 / np.hypot(0.9, 0.1), 3)
    assert reg.identify(vec(0.1, 1, 0), 2.0).speaker_id == b
    unknown = reg.identify(vec(0, 0, 1), 2.0)
    assert unknown.speaker_id is None and unknown.speaker_name == "Unknown"
    assert unknown.color == UNKNOWN_COLOR and unknown.confidence == 0.0


def test_identify_empty_registry(reg):
    m = reg.identify(vec(1, 0, 0), 2.0)
    assert m.speaker_id is None and m.confidence == 0.0


def test_add_sample_updates_centroid(reg):
    a = reg.enroll("Alice", vec(1, 0, 0))["id"]
    assert reg.add_sample(a, vec(0, 1, 0))["num_samples"] == 2
    m = reg.identify(vec(1, 1, 0), 2.0)
    assert m.speaker_id == a and m.confidence == pytest.approx(1.0)
    with pytest.raises(SpeakerNotFoundError):
        reg.add_sample("nope", vec(1, 0, 0))


def test_delete_and_clear(reg):
    a = reg.enroll("Alice", vec(1, 0, 0))["id"]
    reg.enroll("Bob", vec(0, 1, 0))
    reg.delete(a)
    assert [s["name"] for s in reg.list()] == ["Bob"]
    with pytest.raises(SpeakerNotFoundError):
        reg.delete(a)
    reg.clear()
    assert reg.list() == [] and len(reg) == 0


def test_continuity_smoothing(reg):
    a = reg.enroll("Alice", vec(1, 0, 0))["id"]
    b = reg.enroll("Bob", vec(0, 1, 0))["id"]
    close = vec(0.71, 0.69, 0)  # Alice slightly ahead (diff ~0.02)
    # Short utterance: previous speaker Bob is in the near-tied pair -> Bob.
    m = reg.identify(close, duration=0.6, previous_id=b)
    assert m.speaker_id == b
    assert m.confidence == round(0.69 / np.hypot(0.71, 0.69), 3)
    # Long utterance, no previous, or previous not in the pair -> plain best.
    assert reg.identify(close, duration=1.5, previous_id=b).speaker_id == a
    assert reg.identify(close, duration=0.6).speaker_id == a
    assert reg.identify(close, duration=0.6, previous_id="someone-else").speaker_id == a
    # Clear winner (diff > 0.05) is not overridden.
    assert reg.identify(vec(0.8, 0.6, 0), duration=0.6, previous_id=b).speaker_id == a


def test_persistence_roundtrip(path, reg):
    a = reg.enroll("Alice", vec(1, 0, 0))
    reg.add_sample(a["id"], vec(0.8, 0.2, 0))
    reg.enroll("Bob", vec(0, 1, 0))
    data = json.loads(path.read_text())
    assert {s["embedder"] for s in data["speakers"]} == {"fake"}
    assert not list(path.parent.glob("*.tmp"))  # atomic write leaves no temp file

    again = SpeakerRegistry("fake", 0.5, path)
    assert again.list() == reg.list()
    assert again.identify(vec(1, 0, 0), 2.0).speaker_id == a["id"]


def test_embedder_mismatch_is_discarded(path, reg, caplog):
    reg.enroll("Alice", vec(1, 0, 0))
    with caplog.at_level("WARNING"):
        other = SpeakerRegistry("ecapa", 0.3, path)
    assert other.list() == []
    assert "different embedder" in caplog.text
    # Foreign records survive on disk so switching back restores them.
    other.enroll("Zed", vec(0, 0, 1))
    assert [s["name"] for s in SpeakerRegistry("fake", 0.5, path).list()] == ["Alice"]
    assert [s["name"] for s in SpeakerRegistry("ecapa", 0.3, path).list()] == ["Zed"]


def test_corrupt_file_is_moved_aside(path):
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    reg = SpeakerRegistry("fake", 0.5, path)
    assert reg.list() == []
    assert list(path.parent.glob("speakers.corrupt-*.json"))


@pytest.mark.parametrize("content", ['{"speakers": null}', '[]', '{"speakers": "x"}', "null"])
def test_wrong_top_level_shape_is_moved_aside(path, content):
    path.parent.mkdir(parents=True)
    path.write_text(content)
    assert SpeakerRegistry("fake", 0.5, path).list() == []
    assert list(path.parent.glob("speakers.corrupt-*.json"))


def test_invalid_entries_are_skipped(path, caplog):
    def entry(id_, samples, **kw):
        return {"embedder": "fake", "id": id_, "name": id_, "color": "#000000",
                "created_at": 1.0, "samples": samples, **kw}

    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"speakers": [
        "x",
        {"embedder": "fake", "name": "no id"},
        entry("ragged", [[1, 2], [1, 2, 3]]),
        entry("matrix", [[[1, 2]]]),
        entry("empty", []),
        entry("nan", [[1e39, 0]]),  # overflows float32 -> inf
        entry("strs", ["ab"]),
        entry("badtime", [[1, 0]], created_at="soon"),
        entry("good", [[1, 0]]),
        entry("otherdim", [[1, 0, 0]]),
        entry("good2", [[0, 1]]),
    ]}))
    with caplog.at_level("WARNING"):
        reg = SpeakerRegistry("fake", 0.5, path)
    assert [s["name"] for s in reg.list()] == ["good", "good2"]
    assert "invalid" in caplog.text
    assert reg.identify(vec(1, 0), 2.0).speaker_name == "good"
    assert reg.identify(vec(1, 0, 0, 0), 2.0).speaker_id is None  # wrong size: no crash


def test_failed_save_leaves_memory_unchanged(path, reg, monkeypatch):
    import app.speakers.registry as registry_mod

    alice = reg.enroll("Alice", vec(1, 0, 0))
    before = reg.list()

    def broken_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(registry_mod.os, "replace", broken_replace)
    with pytest.raises(OSError):
        reg.enroll("Bob", vec(0, 1, 0))
    with pytest.raises(OSError):
        reg.add_sample(alice["id"], vec(0, 0, 1))
    with pytest.raises(OSError):
        reg.delete(alice["id"])
    with pytest.raises(OSError):
        reg.clear()
    assert reg.list() == before  # memory still matches disk
    assert reg.identify(vec(1, 0, 0), 2.0).confidence == 1.0  # sample not added
    assert not list(path.parent.glob("*.tmp"))
    monkeypatch.undo()
    assert reg.enroll("Bob", vec(0, 1, 0))["name"] == "Bob"  # retry is not a 409
    assert SpeakerRegistry("fake", 0.5, path).list() == reg.list()


def test_in_memory_registry_has_no_file(tmp_path):
    reg = SpeakerRegistry("fake", 0.5, None)
    reg.enroll("Alice", vec(1, 0))
    assert reg.list()[0]["name"] == "Alice"


def test_thread_safety(path, reg):
    errors: list[Exception] = []

    def worker(i: int) -> None:
        try:
            for j in range(10):
                reg.enroll(f"t{i}-{j}", vec(1, i, j))
                reg.identify(vec(1, 0, 0), 2.0)
                reg.list()
        except Exception as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(reg) == 80
    assert len(SpeakerRegistry("fake", 0.5, path).list()) == 80


def test_concurrent_duplicate_enroll_only_one_wins(reg):
    results: list[str] = []

    def worker() -> None:
        try:
            reg.enroll("Same", vec(1, 0))
            results.append("ok")
        except DuplicateNameError:
            results.append("dup")

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results.count("ok") == 1 and len(reg) == 1


# -- augmented prototypes ---------------------------------------------------------------
def unit(*values: float) -> np.ndarray:
    v = vec(*values)
    return v / np.linalg.norm(v)


def test_augmented_embeddings_are_stored_and_persisted(path, reg):
    a = reg.enroll("Alice", vec(1, 0, 0), [vec(0.8, 0.6, 0), vec(0.8, 0, 0.6)])
    assert a["num_samples"] == 1  # real samples only
    assert reg.add_sample(a["id"], vec(0.9, 0.1, 0))["num_samples"] == 2  # no augmentation
    entry = json.loads(path.read_text())["speakers"][0]
    assert len(entry["samples"]) == 2
    assert [len(group) for group in entry["augmented"]] == [2, 0]
    assert entry["augmented"][0][0] == pytest.approx([0.8, 0.6, 0.0])
    again = SpeakerRegistry("fake", 0.5, path)
    assert again.list() == reg.list()
    q = unit(0.8, 0.6, 0)
    assert again.identify(q, 2.0).confidence == reg.identify(q, 2.0).confidence


def test_old_files_without_augmented_field_still_load(path):
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"version": 1, "speakers": [
        {"id": "a1", "name": "Alice", "color": "#e6194b", "created_at": 1.0,
         "embedder": "fake", "samples": [[1, 0, 0]]},
    ]}))
    reg = SpeakerRegistry("fake", 0.5, path)
    assert [s["name"] for s in reg.list()] == ["Alice"]
    assert reg.identify(vec(1, 0, 0), 2.0).confidence == 1.0
    reg.add_sample("a1", vec(0, 1, 0), [vec(0, 0.8, 0.6)])
    entry = json.loads(path.read_text())["speakers"][0]
    assert [len(g) for g in entry["augmented"]] == [0, 1]


def test_invalid_augmented_entries_are_skipped(path, caplog):
    def entry(id_, augmented):
        return {"embedder": "fake", "id": id_, "name": id_, "color": "#000000",
                "created_at": 1.0, "samples": [[1, 0]], "augmented": augmented}

    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"speakers": [
        entry("wrongcount", [[], []]),
        entry("notlist", "x"),
        entry("notgroups", [[1, 0]]),
        entry("wrongdim", [[[1, 0, 0]]]),
        entry("zero", [[[0, 0]]]),
        entry("huge", [[[1e39, 0]]]),
        entry("ok", [[[0.6, 0.8]]]),
        entry("empty-ok", [[]]),
    ]}))
    with caplog.at_level("WARNING"):
        reg = SpeakerRegistry("fake", 0.5, path)
    assert [s["name"] for s in reg.list()] == ["ok", "empty-ok"]
    assert "invalid" in caplog.text


def test_prototype_scoring_uses_top2_of_all_prototypes(reg):
    a = reg.enroll("Alice", vec(1, 0, 0), [unit(0.6, 0.8, 0)])["id"]
    reg.enroll("Bob", vec(0, 0, 1))
    q = unit(0.6, 0.8, 0)
    # centroid: 0.6; prototypes {sample: 0.6, augmented: 1.0} -> mean of top 2 = 0.8
    m = reg.identify(q, 2.0)
    assert m.speaker_id == a and m.confidence == pytest.approx(0.8)
    # The centroid still wins when it is the better match.
    reg2 = SpeakerRegistry("fake", 0.5)
    reg2.enroll("Carol", vec(1, 0, 0))
    cid = reg2.list()[0]["id"]
    reg2.add_sample(cid, vec(0, 1, 0))
    assert reg2.identify(vec(1, 1, 0), 2.0).confidence == pytest.approx(1.0)
    table = reg.score(np.stack([q, vec(0, 0, 1)]))
    assert table.scores.shape == (2, 2) and table.threshold == 0.5
    assert table.scores[1, 1] == pytest.approx(1.0)


ALICE = (120, [(700, 150), (1200, 200), (2600, 300)])
BOB = (210, [(400, 150), (2000, 200), (2900, 300)])
CAROL = (160, [(550, 150), (1600, 200), (2700, 300)])


def test_augmentation_helps_pitch_shifted_voice():
    from app.speakers.embedder import MfccEmbedder, enrollment_embeddings, speed_perturb
    from conftest import voice

    emb = MfccEmbedder()
    plain = SpeakerRegistry(emb.name, emb.default_threshold)
    augmented = SpeakerRegistry(emb.name, emb.default_threshold)
    ids = {}
    for name, (f0, formants), seed in (("Alice", ALICE, 1), ("Bob", BOB, 2), ("Carol", CAROL, 3)):
        main, extra = enrollment_embeddings(emb, voice(f0, formants, seconds=6.0, seed=seed))
        assert len(extra) >= 2
        plain.enroll(name, main)
        ids[name] = augmented.enroll(name, main, extra)["id"]
    for name, (f0, formants) in (("Alice", ALICE), ("Bob", BOB), ("Carol", CAROL)):
        # Same person, different take, speaking about 1.65 semitones higher.
        query = emb.embed(speed_perturb(voice(f0, formants, seconds=3.0, seed=11), 1.1))
        before = plain.score(query[None, :])  # same speaker order as `augmented`
        after = augmented.score(query[None, :])
        col = after.speaker_ids.index(ids[name])
        centroid_only = float(before.scores[0, col])
        with_protos = float(after.scores[0, col])
        assert with_protos > centroid_only + 0.05, name
        assert after.scores[0].argmax() == col  # still the right person
        assert augmented.identify(query, 3.0).speaker_id == ids[name]


# -- adaptive prototypes ---------------------------------------------------------------
@pytest.fixture
def pair(path):
    reg = SpeakerRegistry("fake", 0.5, path)
    a = reg.enroll("Alice", vec(1, 0, 0))["id"]
    b = reg.enroll("Bob", vec(0, 1, 0))["id"]
    return reg, a, b


def test_adaptive_prototype_added_only_above_strict_bar(pair):
    reg, a, b = pair
    confident = unit(0.9, 0.3, 0.1)  # Alice ~0.94, Bob ~0.31
    assert reg.identify(confident, 3.0).speaker_id == a
    assert reg.num_adaptive(a) == 0  # adapt not requested
    reg.identify(confident, 1.9, adapt=True)
    assert reg.num_adaptive(a) == 0  # too short (< 2 s of speech)
    reg.identify(unit(0.62, 0.0, 0.78), 3.0, adapt=True)  # 0.62 < threshold + 0.15
    reg.identify(unit(0.75, 0.7, 0.0), 3.0, adapt=True)  # margin over Bob ~0.05 < 0.10
    reg.identify(unit(0.3, 0.1, 0.95), 3.0, adapt=True)  # Unknown
    assert reg.num_adaptive(a) == reg.num_adaptive(b) == 0
    reg.identify(confident, 2.0, adapt=True)
    assert reg.num_adaptive(a) == 1 and reg.num_adaptive(b) == 0


def test_adaptive_prototypes_raise_scores_capped_and_not_persisted(path, pair):
    reg, a, b = pair
    confident = unit(0.9, 0.3, 0.1)
    near = unit(0.85, 0.35, 0.15)
    before = reg.identify(near, 2.0).confidence
    reg.identify(confident, 3.0, adapt=True)
    assert reg.identify(near, 2.0).confidence > before
    for _ in range(25):
        reg.identify(confident, 3.0, adapt=True)
    assert reg.num_adaptive(a) == 20  # FIFO cap
    data = json.loads(path.read_text())
    assert all("adaptive" not in s for s in data["speakers"])
    assert SpeakerRegistry("fake", 0.5, path).num_adaptive(a) == 0  # memory only


def test_adaptive_prototypes_cleared_on_delete(pair):
    reg, a, b = pair
    reg.identify(unit(0.9, 0.3, 0.1), 3.0, adapt=True)
    reg.identify(unit(0.3, 0.9, 0.1), 3.0, adapt=True)
    assert reg.num_adaptive(a) == reg.num_adaptive(b) == 1
    reg.delete(a)
    assert reg.num_adaptive(a) == 0 and reg.num_adaptive(b) == 1
    reg.clear()
    assert reg.num_adaptive(b) == 0


def test_adaptation_can_be_disabled(path):
    reg = SpeakerRegistry("fake", 0.5, path, adapt=False)
    a = reg.enroll("Alice", vec(1, 0, 0))["id"]
    reg.enroll("Bob", vec(0, 1, 0))
    reg.identify(unit(0.9, 0.3, 0.1), 3.0, adapt=True)
    assert reg.num_adaptive(a) == 0


def test_adaptation_is_thread_safe(pair):
    reg, a, _ = pair
    errors: list[Exception] = []

    def worker() -> None:
        try:
            for _ in range(50):
                reg.identify(unit(0.9, 0.3, 0.1), 3.0, adapt=True)
        except Exception as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and reg.num_adaptive(a) == 20


def test_bad_augmented_embeddings_are_dropped_on_enroll(reg):
    a = reg.enroll("Alice", vec(1, 0, 0), [vec(1, 0), vec(np.nan, 0, 0), vec(0, 0, 0), vec(0, 2, 0)])
    q = unit(0, 1, 0)
    # Only (0, 1, 0) survived (normalised): top-2 mean = (0 + 1) / 2.
    assert reg.identify(q, 2.0).confidence == pytest.approx(0.5)
    assert reg.get(a["id"])["num_samples"] == 1
