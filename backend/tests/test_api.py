import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.audio.io import encode_wav
from app.config import Settings
from app.main import create_app
from app.speakers.embedder import MfccEmbedder
from conftest import FakeTranscriber, silence, tone, voice


def speech_wav(freq: float = 200, seconds: float = 3.0) -> bytes:
    audio = np.concatenate([silence(0.5), tone(seconds, freq), silence(0.5)])
    return encode_wav(audio)


def enroll(client: TestClient, name: str, wav: bytes | None = None):
    files = {"audio": ("rec.wav", wav if wav is not None else speech_wav(), "audio/wav")}
    return client.post("/api/speakers", data={"name": name}, files=files)


@pytest.fixture
def client(make_client) -> TestClient:
    return make_client()


def test_health(client):
    assert client.get("/api/health").json() == {
        "status": "ok",
        "embedder": "fake",
        "transcriber": "fake",
    }


def test_health_auto_mode_without_ml(monkeypatch, tmp_path, caplog):
    import app.speakers.embedder as emb_mod
    import app.transcription.transcriber as tr_mod

    monkeypatch.setattr(emb_mod, "_ecapa_available", lambda: False)
    monkeypatch.setattr(tr_mod, "_whisper_available", lambda: False)
    settings = Settings(data_dir=tmp_path, embedder="auto", transcriber="auto")
    with caplog.at_level("WARNING"):
        with TestClient(create_app(settings, frontend_dir=None)) as c:
            body = c.get("/api/health").json()
    assert body == {"status": "ok", "embedder": "mfcc", "transcriber": "null"}
    assert "falling back" in caplog.text and "faster-whisper" in caplog.text


def test_enroll_list_and_speaker_shape(client):
    r = enroll(client, "  Alice  ")
    assert r.status_code == 201, r.text
    s = r.json()
    assert set(s) == {"id", "name", "color", "num_samples", "created_at"}
    assert s["name"] == "Alice" and s["num_samples"] == 1
    assert s["color"].startswith("#") and len(s["color"]) == 7
    assert client.get("/api/speakers").json() == {"speakers": [s]}


def test_enroll_assigns_distinct_colors(client):
    colors = {enroll(client, f"P{i}", speech_wav(100 * (i + 1))).json()["color"] for i in range(3)}
    assert len(colors) == 3


@pytest.mark.parametrize("name", ["", "   ", "x" * 41])
def test_enroll_bad_name(client, name):
    r = enroll(client, name)
    assert r.status_code == 400 and "name" in r.json()["detail"]


def test_enroll_name_limits(client):
    assert enroll(client, "x" * 40).status_code == 201
    assert enroll(client, "y").status_code == 201


def test_enroll_missing_fields(client):
    r = client.post("/api/speakers", files={"audio": ("a.wav", speech_wav(), "audio/wav")})
    assert r.status_code == 400
    r = client.post("/api/speakers", data={"name": "Alice"})
    assert r.status_code == 400 and "audio" in r.json()["detail"]


def test_enroll_duplicate_name(client):
    assert enroll(client, "Alice").status_code == 201
    r = enroll(client, " alice ")
    assert r.status_code == 409 and "detail" in r.json()


def test_enroll_unreadable_audio(client):
    r = enroll(client, "Alice", b"this is not audio at all")
    assert r.status_code == 400
    assert enroll(client, "Alice", b"").status_code == 400


def test_enroll_too_little_speech(client):
    wav = encode_wav(np.concatenate([silence(1.0), tone(1.0), silence(3.0)]))
    r = enroll(client, "Alice", wav)
    assert r.status_code == 400 and "speech" in r.json()["detail"]
    r = enroll(client, "Alice", encode_wav(silence(5.0)))
    assert r.status_code == 400


def test_enroll_too_large(client):
    r = enroll(client, "Alice", b"\0" * (10 * 1024 * 1024 + 1))
    assert r.status_code == 413


def test_enroll_too_long(client):
    wav = encode_wav(tone(61.0, 200, amp=0.1)[::2], 8000)  # 61 s at 8 kHz (~1 MB)
    r = enroll(client, "Alice", wav)
    assert r.status_code == 413


def test_enroll_44k_stereo(client):
    t = np.arange(int(44100 * 3.5)) / 44100
    mono = 0.3 * np.sin(2 * np.pi * 300 * t)
    stereo = np.stack([mono, mono * 0.8], axis=1)
    r = enroll(client, "Stereo", encode_wav(stereo, 44100))
    assert r.status_code == 201, r.text


def test_add_sample(client):
    sid = enroll(client, "Alice").json()["id"]
    r = client.post(
        f"/api/speakers/{sid}/samples", files={"audio": ("a.wav", speech_wav(), "audio/wav")}
    )
    assert r.status_code == 200 and r.json()["num_samples"] == 2
    r = client.post(
        "/api/speakers/nope/samples", files={"audio": ("a.wav", speech_wav(), "audio/wav")}
    )
    assert r.status_code == 404 and "detail" in r.json()
    r = client.post(f"/api/speakers/{sid}/samples", files={"audio": ("a.wav", b"xx", "audio/wav")})
    assert r.status_code == 400
    assert client.get("/api/speakers").json()["speakers"][0]["num_samples"] == 2


def test_delete_one_and_all(client):
    a = enroll(client, "Alice").json()["id"]
    enroll(client, "Bob", speech_wav(400))
    r = client.delete(f"/api/speakers/{a}")
    assert r.status_code == 204 and r.content == b""
    assert client.delete(f"/api/speakers/{a}").status_code == 404
    assert [s["name"] for s in client.get("/api/speakers").json()["speakers"]] == ["Bob"]
    assert client.delete("/api/speakers").status_code == 204
    assert client.get("/api/speakers").json() == {"speakers": []}


def test_speakers_persist_across_restart(make_client):
    first = make_client()
    sid = enroll(first, "Alice").json()["id"]
    second = make_client()
    assert [s["id"] for s in second.get("/api/speakers").json()["speakers"]] == [sid]


def test_real_mfcc_enrollment(make_client):
    client = make_client(embedder=MfccEmbedder(), transcriber=FakeTranscriber())
    wav = encode_wav(np.concatenate([silence(0.3), voice(120, [(700, 150), (1200, 200)], 4.0)]))
    assert enroll(client, "Alice", wav).status_code == 201


def test_static_frontend_mounted_after_api(tmp_path, settings):
    front = tmp_path / "frontend"
    front.mkdir()
    (front / "index.html").write_text("<h1>hello</h1>")
    app = create_app(settings, embedder=MfccEmbedder(), transcriber=FakeTranscriber(), frontend_dir=front)
    with TestClient(app) as c:
        assert c.get("/").text == "<h1>hello</h1>"
        assert c.get("/api/health").json()["status"] == "ok"
        assert c.get("/missing.js").status_code == 404


def test_missing_frontend_dir_does_not_crash(tmp_path, settings):
    app = create_app(settings, embedder=MfccEmbedder(), frontend_dir=tmp_path / "nope")
    with TestClient(app) as c:
        assert c.get("/api/health").status_code == 200
