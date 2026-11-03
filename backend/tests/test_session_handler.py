"""SessionHandler driven by an in-memory WebSocket so hangs fail fast instead of blocking."""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any, Callable

import numpy as np
import pytest
from starlette.websockets import WebSocketState

from app.api import session as session_mod
from app.api.session import BACKLOG_DROP_MESSAGE, MAX_BACKLOG, SessionHandler
from app.audio.io import float32_to_pcm16
from app.config import Settings
from app.speakers.registry import SpeakerRegistry
from app.state import AppState
from conftest import FakeEmbedder, chunks, silence, tone

CONNECTED, DISCONNECTED = WebSocketState.CONNECTED, WebSocketState.DISCONNECTED


class FakeWS:
    def __init__(self) -> None:
        self.inbox: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.sent: list[dict[str, Any]] = []
        self.application_state = CONNECTED
        self.client_state = CONNECTED
        self.close_code: int | None = None

    async def accept(self) -> None:
        pass

    async def receive(self) -> dict[str, Any]:
        return await self.inbox.get()

    async def send_json(self, data: dict[str, Any]) -> None:
        if self.client_state != CONNECTED:
            raise RuntimeError("socket closed")
        self.sent.append(data)

    async def close(self, code: int = 1000) -> None:
        self.close_code = code
        self.application_state = DISCONNECTED

    # -- client side helpers ------------------------------------------------
    def text(self, kind: str) -> None:
        self.inbox.put_nowait({"type": "websocket.receive", "text": json.dumps({"type": kind})})

    def audio(self, audio: np.ndarray) -> None:
        for c in chunks(audio, 1600):
            self.inbox.put_nowait({"type": "websocket.receive", "bytes": float32_to_pcm16(c)})

    def disconnect(self) -> None:
        self.client_state = DISCONNECTED  # sends fail from now on
        self.inbox.put_nowait({"type": "websocket.disconnect", "code": 1001})

    def of_type(self, kind: str) -> list[dict[str, Any]]:
        return [m for m in self.sent if m["type"] == kind]


class GatedTranscriber:
    """Blocks in transcribe() until ``gate`` is set."""

    name = "gated"

    def __init__(self, open_: bool = False) -> None:
        self.gate = threading.Event()
        if open_:
            self.gate.set()
        self.calls = 0

    def transcribe(self, audio: np.ndarray) -> str:
        self.calls += 1
        assert self.gate.wait(10), "gate never opened"
        return "words"

    def transcribe_words(self, audio: np.ndarray) -> None:
        return None  # no word timing


def make_handler(transcriber) -> tuple[SessionHandler, FakeWS]:
    fake = FakeEmbedder()
    registry = SpeakerRegistry(fake.name, fake.default_threshold)
    state = AppState(Settings(embedder="mfcc", transcriber="null"), fake, transcriber, registry)
    ws = FakeWS()
    return SessionHandler(ws, state), ws  # type: ignore[arg-type]


async def wait_until(cond: Callable[[], bool], timeout: float = 2.0) -> None:
    async def poll() -> None:
        while not cond():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout)


def utt(seconds: float = 1.0, freq: float = 200) -> np.ndarray:
    return np.concatenate([silence(0.3), tone(seconds, freq), silence(1.0)])


def kinds(ws: FakeWS) -> list[Any]:
    out = []
    for m in ws.sent:
        if m["type"] == "status":
            out.append(m["state"])
        elif m["type"] == "speech":
            out.append(m["active"])
        else:
            out.append(m["type"])
    return out


def test_client_gone_during_restart_does_not_hang():
    async def scenario() -> None:
        tr = GatedTranscriber()
        handler, ws = make_handler(tr)
        task = asyncio.create_task(handler.run())
        ws.text("start")
        ws.audio(utt())
        ws.text("stop")
        ws.text("start")
        await wait_until(lambda: tr.calls >= 1)  # worker busy with the old utterance
        ws.disconnect()
        tr.gate.set()  # worker's next send fails: client is gone
        await asyncio.wait_for(task, 3)  # used to block forever in queue.join()

    asyncio.run(scenario())


def test_restart_never_blocks_receive_loop_and_keeps_order():
    async def scenario() -> None:
        tr = GatedTranscriber()
        handler, ws = make_handler(tr)
        task = asyncio.create_task(handler.run())
        ws.text("start")
        ws.audio(utt())
        ws.text("stop")
        ws.text("start")
        ws.audio(utt(freq=400))
        # Every inbound frame is consumed while the old utterance is still processing.
        await wait_until(lambda: ws.inbox.empty() and tr.calls >= 1)
        await asyncio.sleep(0.05)
        assert ws.inbox.empty()
        assert ws.of_type("utterance") == [] and kinds(ws).count("listening") == 1
        tr.gate.set()
        ws.text("stop")
        await wait_until(lambda: kinds(ws).count("stopped") == 2)
        assert kinds(ws) == [
            "listening", True, False, "utterance", "stopped",
            "listening", True, False, "utterance", "stopped",
        ]
        assert [u["seq"] for u in ws.of_type("utterance")] == [1, 1]
        ws.disconnect()
        await asyncio.wait_for(task, 3)

    asyncio.run(scenario())


def test_start_mid_utterance_flushes_previous_utterance():
    async def scenario() -> None:
        handler, ws = make_handler(GatedTranscriber(open_=True))
        task = asyncio.create_task(handler.run())
        ws.text("start")
        ws.audio(np.concatenate([silence(0.5), tone(1.5, 200)]))  # still talking
        ws.text("start")
        ws.text("stop")
        await wait_until(lambda: "stopped" in kinds(ws))
        assert kinds(ws) == ["listening", True, False, "utterance", "listening", "stopped"]
        ws.disconnect()
        await asyncio.wait_for(task, 3)

    asyncio.run(scenario())


def test_worker_crash_ends_session():
    async def scenario() -> None:
        handler, ws = make_handler(GatedTranscriber(open_=True))

        async def broken(item):
            raise RuntimeError("worker bug")

        handler._process = broken  # type: ignore[method-assign]
        task = asyncio.create_task(handler.run())
        ws.text("start")
        ws.audio(utt())
        await asyncio.wait_for(task, 3)  # no client disconnect needed
        assert ws.close_code == 1011

    asyncio.run(scenario())


def test_backlog_is_capped_by_dropping_oldest_utterance(monkeypatch):
    monkeypatch.setattr(session_mod, "_BACKLOG_WARN", 1000)  # keep the log quiet
    total = MAX_BACKLOG + 5

    async def scenario() -> None:
        tr = GatedTranscriber()
        handler, ws = make_handler(tr)
        task = asyncio.create_task(handler.run())
        ws.text("start")
        for _ in range(total):
            ws.audio(utt(0.6))
        await wait_until(lambda: ws.inbox.empty() and tr.calls >= 1)
        await asyncio.sleep(0.05)
        assert handler._pending_work <= MAX_BACKLOG
        errors = ws.of_type("error")
        assert errors and all(e["message"] == BACKLOG_DROP_MESSAGE for e in errors)
        ws.text("stop")
        ws.text("start")  # markers are never dropped
        ws.text("stop")
        tr.gate.set()
        await wait_until(lambda: kinds(ws).count("stopped") == 2, timeout=10)
        utts = ws.of_type("utterance")
        assert len(utts) + len(ws.of_type("error")) == total
        assert len(utts) in (MAX_BACKLOG, MAX_BACKLOG + 1)
        assert [u["seq"] for u in utts] == list(range(1, len(utts) + 1))
        # The newest utterances survive; the oldest waiting ones were dropped.
        assert utts[-1]["start"] == pytest.approx((total - 1) * 1.9, abs=0.1)
        assert kinds(ws)[-3:] == ["stopped", "listening", "stopped"]
        ws.disconnect()
        await asyncio.wait_for(task, 3)

    asyncio.run(scenario())
