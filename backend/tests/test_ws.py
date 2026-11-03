import json
import logging

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.audio.io import encode_wav, float32_to_pcm16
from conftest import FakeTranscriber, chunks, silence, tone

CHUNK = 1600  # 100 ms, like the browser


def enroll(client: TestClient, name: str, freq: float) -> str:
    wav = encode_wav(np.concatenate([silence(0.3), tone(3.0, freq), silence(0.3)]))
    r = client.post("/api/speakers", data={"name": name}, files={"audio": ("a.wav", wav, "audio/wav")})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def send_audio(ws, audio: np.ndarray) -> None:
    for c in chunks(audio, CHUNK):
        ws.send_bytes(float32_to_pcm16(c))


def receive_until_stopped(ws, limit: int = 100) -> list[dict]:
    messages = []
    for _ in range(limit):
        msg = ws.receive_json()
        messages.append(msg)
        if msg == {"type": "status", "state": "stopped"}:
            return messages
    raise AssertionError(f"no stopped status in {messages}")


def utterances(messages: list[dict]) -> list[dict]:
    return [m for m in messages if m["type"] == "utterance"]


@pytest.fixture
def client(make_client) -> TestClient:
    return make_client()


def conversation() -> np.ndarray:
    # Alice (200 Hz) 1.0-2.5 s, Bob (400 Hz) 4.0-5.5 s, stop shortly after.
    return np.concatenate([silence(1.0), tone(1.5, 200), silence(1.5), tone(1.5, 400), silence(0.3)])


def test_session_end_to_end(client):
    alice, bob = enroll(client, "Alice", 200), enroll(client, "Bob", 400)
    colors = {s["id"]: s["color"] for s in client.get("/api/speakers").json()["speakers"]}
    with client.websocket_connect("/ws/session") as ws:
        ws.send_text(json.dumps({"type": "start"}))
        assert ws.receive_json() == {"type": "status", "state": "listening"}
        send_audio(ws, conversation())
        ws.send_text(json.dumps({"type": "stop"}))
        messages = receive_until_stopped(ws)

    assert messages[-1] == {"type": "status", "state": "stopped"}
    speech = [m["active"] for m in messages if m["type"] == "speech"]
    assert speech == [True, False, True, False]
    utts = utterances(messages)
    assert [u["seq"] for u in utts] == [1, 2]
    a, b = utts
    assert set(a) == {"type", "id", "seq", "speaker_id", "speaker_name", "color", "text",
                      "start", "end", "confidence"}
    assert (a["speaker_id"], a["speaker_name"], a["color"]) == (alice, "Alice", colors[alice])
    assert (b["speaker_id"], b["speaker_name"], b["color"]) == (bob, "Bob", colors[bob])
    assert a["text"] == "tone 200" and b["text"] == "tone 400"
    assert a["confidence"] == 1.0
    assert a["start"] == pytest.approx(0.7, abs=0.05) and a["end"] == pytest.approx(2.7, abs=0.06)
    assert b["start"] == pytest.approx(3.7, abs=0.05)
    assert b["end"] <= 5.8 + 1e-6  # flushed at stop
    assert a["id"] != b["id"]


def test_unknown_speaker_and_empty_transcript(make_client):
    client = make_client(transcriber=FakeTranscriber(silent=(500,)))
    enroll(client, "Alice", 200)
    audio = np.concatenate([
        silence(0.5), tone(1.0, 700), silence(1.0),  # nobody enrolled at 700 Hz
        tone(1.0, 500), silence(1.0),  # transcript empty -> dropped
        tone(1.0, 200), silence(1.0),
    ])
    with client.websocket_connect("/ws/session") as ws:
        ws.send_text('{"type":"start"}')
        ws.receive_json()
        send_audio(ws, audio)
        ws.send_text('{"type":"stop"}')
        utts = utterances(receive_until_stopped(ws))
    assert [u["seq"] for u in utts] == [1, 2]  # no gap for the dropped one
    unknown, alice = utts
    assert unknown["speaker_id"] is None
    assert unknown["speaker_name"] == "Unknown" and unknown["color"] == "#888888"
    assert unknown["confidence"] == 0.0
    assert alice["speaker_name"] == "Alice"


def test_order_preserved_with_slow_processing(make_client):
    client = make_client(transcriber=FakeTranscriber(delays={200: 0.3}))
    enroll(client, "Alice", 200)
    enroll(client, "Bob", 400)
    with client.websocket_connect("/ws/session") as ws:
        ws.send_text('{"type":"start"}')
        ws.receive_json()
        send_audio(ws, conversation())
        ws.send_text('{"type":"stop"}')
        utts = utterances(receive_until_stopped(ws))
    assert [(u["seq"], u["speaker_name"]) for u in utts] == [(1, "Alice"), (2, "Bob")]


def test_bytes_before_start_are_ignored(client):
    enroll(client, "Alice", 200)
    with client.websocket_connect("/ws/session") as ws:
        send_audio(ws, conversation())
        ws.send_text('{"type":"start"}')
        assert ws.receive_json() == {"type": "status", "state": "listening"}
        send_audio(ws, silence(1.0))
        ws.send_text('{"type":"stop"}')
        assert receive_until_stopped(ws) == [{"type": "status", "state": "stopped"}]


def test_bad_json_and_unknown_type_keep_connection_open(client):
    with client.websocket_connect("/ws/session") as ws:
        ws.send_text("{not json")
        msg = ws.receive_json()
        assert msg["type"] == "error" and msg["message"]
        ws.send_text('["start"]')
        assert ws.receive_json()["type"] == "error"
        ws.send_text('{"type":"dance"}')
        assert ws.receive_json()["type"] == "error"
        ws.send_text('{"type":"start"}')
        assert ws.receive_json() == {"type": "status", "state": "listening"}


def test_stop_without_start(client):
    with client.websocket_connect("/ws/session") as ws:
        ws.send_text('{"type":"stop"}')
        assert ws.receive_json() == {"type": "status", "state": "stopped"}


def test_restart_resets_seq_and_clock(client):
    enroll(client, "Alice", 200)
    one = np.concatenate([silence(0.5), tone(1.0, 200), silence(1.0)])
    with client.websocket_connect("/ws/session") as ws:
        for _ in range(2):
            ws.send_text('{"type":"start"}')
            assert ws.receive_json()["state"] == "listening"
            send_audio(ws, one)
            ws.send_text('{"type":"stop"}')
            utts = utterances(receive_until_stopped(ws))
            assert [u["seq"] for u in utts] == [1]
            assert utts[0]["start"] == pytest.approx(0.2, abs=0.05)


def test_odd_sized_frames_are_accepted(client):
    enroll(client, "Alice", 200)
    pcm = float32_to_pcm16(np.concatenate([silence(0.5), tone(1.0, 200), silence(1.0)]))
    with client.websocket_connect("/ws/session") as ws:
        ws.send_text('{"type":"start"}')
        ws.receive_json()
        ws.send_bytes(b"\x01")  # single odd byte: dropped
        for i in range(0, len(pcm), 3200):
            ws.send_bytes(pcm[i : i + 3200])
        ws.send_text('{"type":"stop"}')
        assert len(utterances(receive_until_stopped(ws))) == 1


def test_disconnect_mid_session_is_clean(make_client, caplog):
    client = make_client(transcriber=FakeTranscriber(delays={200: 0.2}))
    enroll(client, "Alice", 200)
    with caplog.at_level(logging.WARNING):
        with client.websocket_connect("/ws/session") as ws:
            ws.send_text('{"type":"start"}')
            ws.receive_json()
            send_audio(ws, np.concatenate([silence(0.5), tone(1.0, 200), silence(1.0), tone(3.0, 200)]))
            # leave without stop while an utterance is processing and speech is active
        # Connection still usable afterwards.
        with client.websocket_connect("/ws/session") as ws:
            ws.send_text('{"type":"stop"}')
            assert ws.receive_json()["state"] == "stopped"
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert client.get("/api/health").status_code == 200


def test_processing_error_is_reported(make_client):
    class Broken:
        name = "broken"
        default_threshold = 0.5

        def embed(self, audio):
            raise RuntimeError("boom")

    client = make_client(embedder=Broken())
    with client.websocket_connect("/ws/session") as ws:
        ws.send_text('{"type":"start"}')
        ws.receive_json()
        send_audio(ws, np.concatenate([silence(0.5), tone(1.0, 200), silence(1.0)]))
        ws.send_text('{"type":"stop"}')
        messages = receive_until_stopped(ws)
    assert any(m["type"] == "error" for m in messages)
    assert utterances(messages) == []


def test_receive_loop_not_blocked_by_slow_processing(make_client):
    client = make_client(transcriber=FakeTranscriber(delays={200: 0.5}))
    enroll(client, "Alice", 200)
    enroll(client, "Bob", 400)
    with client.websocket_connect("/ws/session") as ws:
        ws.send_text('{"type":"start"}')
        ws.receive_json()
        send_audio(ws, conversation())
        ws.send_text('{"type":"stop"}')
        messages = receive_until_stopped(ws)
    kinds = [(m["type"], m.get("active")) for m in messages]
    first_utterance = kinds.index(("utterance", None))
    # Both speech regions were detected while the first utterance was still processing.
    assert kinds[:first_utterance].count(("speech", True)) == 2
