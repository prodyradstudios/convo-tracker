import numpy as np
import pytest

from app.speakers.embedder import (
    PEAK_LIMIT,
    TARGET_DBFS,
    MfccEmbedder,
    augmented_copies,
    build_embedder,
    embed_many,
    embed_windows,
    enrollment_embeddings,
    l2_normalize,
    normalize_loudness,
    pitch_shift,
    speed_perturb,
)
from conftest import SR, FakeEmbedder, dominant_freq, silence, tone, voice

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


# -- loudness normalisation ---------------------------------------------------------
def voiced_dbfs(x: np.ndarray) -> float:
    frames = x[: x.size // 480 * 480].reshape(-1, 480).astype(np.float64)
    rms = np.sqrt((frames**2).mean(axis=1))
    voiced = rms >= rms.max() * 10 ** (-30 / 20)
    return 20 * np.log10(np.sqrt((rms[voiced] ** 2).mean()))


def test_normalize_loudness_targets_voiced_rms_without_clipping():
    speech = np.concatenate([silence(1.0), voice(*ALICE), silence(1.0)])
    for scale in (0.01, 0.1, 1.0, 3.0):
        out = normalize_loudness(speech * scale)
        assert out.dtype == np.float32 and out.shape == speech.shape
        assert np.abs(out).max() <= PEAK_LIMIT + 1e-6
    # A plain tone has a low crest factor: it reaches the target exactly.
    t = normalize_loudness(tone(2.0, 300, amp=0.01))
    assert voiced_dbfs(t) == pytest.approx(TARGET_DBFS, abs=0.1)
    # Pauses do not drag the level down: same voiced level with or without them.
    padded = normalize_loudness(np.concatenate([silence(3.0), tone(1.0, 300, 0.01), silence(3.0)]))
    assert voiced_dbfs(padded) == pytest.approx(TARGET_DBFS, abs=0.1)


def test_normalize_loudness_peak_limit():
    # One loud click on a quiet tone: the click limits the gain (no clipping).
    x = tone(2.0, 300, amp=0.001)
    x[100] = 0.5
    out = normalize_loudness(x)
    assert np.abs(out).max() == pytest.approx(PEAK_LIMIT, rel=1e-4)


def test_normalize_loudness_silence_is_safe():
    for x in (np.zeros(0, np.float32), np.zeros(16000, np.float32), np.full(16000, 1e-7, np.float32)):
        out = normalize_loudness(x)
        assert np.all(np.isfinite(out)) and np.abs(out).max(initial=0) <= 1e-6
    nan = np.full(1000, np.nan, np.float32)
    assert np.all(np.isfinite(normalize_loudness(nan)))


def test_scaled_copies_embed_identically(emb):
    audio = np.concatenate([silence(0.3), voice(*BOB, seed=7), silence(0.3)])
    ref = emb.embed(audio)
    for scale in (0.002, 0.05, 0.5, 2.5):
        assert float(ref @ emb.embed(audio * scale)) > 0.9999


# -- batching ------------------------------------------------------------------------
def test_embed_many_uses_batch_when_available():
    class Batchy(FakeEmbedder):
        def __init__(self):
            super().__init__()
            self.batches = []

        def embed_batch(self, clips):
            self.batches.append(len(clips))
            return np.stack([2.0 * FakeEmbedder.embed(self, c) for c in clips])  # not unit

    clips = [tone(1.0, 200), tone(1.0, 400), tone(0.5, 300)]
    b = Batchy()
    out = embed_many(b, clips)
    assert b.batches == [3] and out.shape == (3, FakeEmbedder.dim)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, rtol=1e-6)
    plain = FakeEmbedder()
    np.testing.assert_allclose(embed_many(plain, clips), out)
    assert plain.calls == 3
    assert embed_many(plain, []).shape[0] == 0


# -- enrollment augmentation ---------------------------------------------------------------
def test_speed_perturb_changes_pitch_and_length():
    x = tone(2.0, 300)
    fast, slow = speed_perturb(x, 1.1), speed_perturb(x, 0.9)
    assert fast.size == pytest.approx(x.size / 1.1, abs=2)
    assert slow.size == pytest.approx(x.size / 0.9, abs=2)
    assert dominant_freq(fast) == pytest.approx(330, abs=2)
    assert dominant_freq(slow) == pytest.approx(270, abs=2)


def test_pitch_shift_keeps_duration():
    pytest.importorskip("torchaudio")
    x = tone(2.0, 300)
    up, down = pitch_shift(x, 2.0), pitch_shift(x, -2.0)
    assert up.size == down.size == x.size
    assert dominant_freq(up) == pytest.approx(300 * 2 ** (2 / 12), abs=3)
    assert dominant_freq(down) == pytest.approx(300 * 2 ** (-2 / 12), abs=3)


def test_augmented_copies_and_enrollment_embeddings(monkeypatch, emb):
    import app.speakers.embedder as mod

    speech = voice(*ALICE, seconds=4.0)
    monkeypatch.setattr(mod, "_torchaudio_available", lambda: False)
    assert len(augmented_copies(speech)) == 2  # speed 0.9 / 1.1 only
    monkeypatch.undo()

    def broken(audio, steps):
        raise RuntimeError("no pitch shift today")

    monkeypatch.setattr(mod, "_torchaudio_available", lambda: True)
    monkeypatch.setattr(mod, "pitch_shift", broken)
    assert len(augmented_copies(speech)) == 2  # failures are skipped, not raised
    monkeypatch.undo()

    main, extra = enrollment_embeddings(emb, speech)
    np.testing.assert_allclose(main, embed_windows(emb, speech))
    expected = 4 if mod._torchaudio_available() else 2
    assert len(extra) == expected
    for e in extra:
        assert e.shape == main.shape and np.linalg.norm(e) == pytest.approx(1.0, abs=1e-5)
        assert 0.5 < float(e @ main) < 0.9999  # similar voice, not identical
    assert enrollment_embeddings(emb, speech, augment=False)[1] == []


def test_ecapa_embed_batch_pads_and_passes_wav_lens():
    torch = pytest.importorskip("torch")
    import threading

    from app.speakers.embedder import EcapaEmbedder

    seen = {}

    class FakeModel:
        def encode_batch(self, wavs, wav_lens=None):
            seen["wavs"], seen["lens"] = wavs, wav_lens
            n = wavs.shape[0]
            # (batch, 1, dim) like SpeechBrain; row i = [i + 1, energy-ish]
            rows = [[i + 1.0, float(wavs[i].abs().sum())] for i in range(n)]
            return torch.tensor(rows).unsqueeze(1)

    e = EcapaEmbedder.__new__(EcapaEmbedder)
    e._model, e._lock = FakeModel(), threading.Lock()
    clips = [tone(1.0, 200, amp=0.01), tone(0.5, 300, amp=0.5)]
    out = e.embed_batch(clips)
    assert out.shape == (2, 2)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, rtol=1e-6)
    assert tuple(seen["wavs"].shape) == (2, 16000)
    assert seen["lens"].tolist() == pytest.approx([1.0, 0.5])
    assert float(seen["wavs"][1, 8000:].abs().max()) == 0.0  # zero padding
    # Both clips were loudness-normalised before the model saw them.
    rms = [float(seen["wavs"][i, : c.size].pow(2).mean().sqrt()) for i, c in enumerate(clips)]
    assert rms == pytest.approx([10 ** (-23 / 20)] * 2, rel=0.02)
    single = e.embed(clips[0])
    assert single.shape == (2,)
