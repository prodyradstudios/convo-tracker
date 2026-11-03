import numpy as np
import pytest

from app.speakers.embedder import MfccEmbedder, build_embedder, embed_windows, l2_normalize
from conftest import FakeEmbedder, SR, silence, tone, voice

ALICE = (120, [(700, 150), (1200, 200), (2600, 300)])
BOB = (210, [(400, 150), (2000, 200), (2900, 300)])
CAROL = (160, [(550, 150), (1600, 200), (2700, 300)])


@pytest.fixture(scope="module")
def emb() -> MfccEmbedder:
    return MfccEmbedder()


def test_shape_norm_and_determinism(emb):
    audio = voice(*ALICE)
    a, b = emb.embed(audio), emb.embed(audio.copy())
    assert a.shape == (emb.dim,) and a.dtype == np.float32
    assert np.linalg.norm(a) == pytest.approx(1.0, abs=1e-5)
    np.testing.assert_array_equal(a, b)


def test_loudness_invariant(emb):
    audio = voice(*ALICE)
    assert float(emb.embed(audio) @ emb.embed(audio * 0.25)) > 0.99


def test_same_voice_scores_higher_than_different_voices(emb):
    a1, a2 = emb.embed(voice(*ALICE, seed=1)), emb.embed(voice(*ALICE, seconds=2.0, seed=2))
    b1, b2 = emb.embed(voice(*BOB, seed=3)), emb.embed(voice(*BOB, seed=4))
    c1 = emb.embed(voice(*CAROL, seed=5))
    same = min(a1 @ a2, b1 @ b2)
    different = max(a1 @ b1, a1 @ c1, b1 @ c1)
    assert same > emb.default_threshold
    assert different < emb.default_threshold
    assert same - different > 0.2


def test_handles_tiny_and_silent_input(emb):
    for audio in (np.zeros(100, np.float32), np.zeros(SR, np.float32), silence(0.02)):
        out = emb.embed(audio)
        assert out.shape == (emb.dim,)
        assert np.all(np.isfinite(out))


def test_embed_windows_averages_and_normalises():
    fake = FakeEmbedder()
    audio = np.concatenate([tone(4.0, 200), tone(4.0, 200)])
    out = embed_windows(fake, audio)
    assert np.linalg.norm(out) == pytest.approx(1.0)
    # 8 s with 3 s windows / 1.5 s hop -> starts 0,1.5,3,4.5 + tail window at 5.0
    assert fake.calls == 5
    short = FakeEmbedder()
    embed_windows(short, tone(2.0, 200))
    assert short.calls == 1


def test_l2_normalize_zero():
    assert np.all(l2_normalize(np.zeros(4)) == 0)


def test_build_embedder_auto_falls_back(monkeypatch, tmp_path, caplog):
    import app.speakers.embedder as mod

    monkeypatch.setattr(mod, "_ecapa_available", lambda: False)
    with caplog.at_level("WARNING"):
        e = build_embedder("auto", tmp_path)
    assert e.name == "mfcc"
    assert "MFCC" in caplog.text
    assert build_embedder("mfcc", tmp_path).name == "mfcc"
