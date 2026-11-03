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
