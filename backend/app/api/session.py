"""WebSocket ``/ws/session``: live audio in, labelled utterances out."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections import deque
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from ..audio.io import pcm16_to_float32
from ..audio.vad import Segment
from ..pipeline import SessionPipeline
from ..state import AppState
from .deps import get_state

log = logging.getLogger(__name__)
router = APIRouter()

_BACKLOG_WARN = 20
# Max utterances waiting for the worker; beyond this the oldest is dropped.
MAX_BACKLOG = 40
BACKLOG_DROP_MESSAGE = "Falling behind: dropped an utterance"


@dataclass
class _Work:
    pipeline: SessionPipeline
    segment: Segment


@dataclass
class _Started:
    """Queue marker: everything before it is processed; announce 'listening'."""

    pipeline: SessionPipeline


@dataclass
class _Stopped:
    """Queue marker: everything before it is processed; announce 'stopped'."""


_Item = _Work | _Started | _Stopped


class _ClientGone(Exception):
    """The client disconnected while we were sending."""


class SessionHandler:
    """One connection: a receive loop plus a single ML worker task.

    The receive loop never waits for the worker. Status changes go through the
    worker's queue as markers so they stay ordered with utterances, e.g. the
    old run's 'stopped' always precedes a new run's 'listening'.
    """

    def __init__(self, ws: WebSocket, state: AppState) -> None:
        self.ws = ws
        self.state = state
        self.pipeline: SessionPipeline | None = None
        self._items: deque[_Item] = deque()
        self._pending_work = 0  # number of _Work items in self._items
        self._wakeup = asyncio.Event()
        self._send_lock = asyncio.Lock()
        # Pipeline whose 'listening' has been sent: its speech events go out
        # directly. Events of a not-yet-announced pipeline wait in its segmenter
        # until the worker announces it.
        self._live: SessionPipeline | None = None
        self._worker_task: asyncio.Task[None] | None = None

    async def run(self) -> None:
        await self.ws.accept()
        receiver = asyncio.create_task(self._receive_loop(), name="ct-session-receive")
        worker = self._worker_task = asyncio.create_task(self._worker(), name="ct-session-worker")
        try:
            # Whichever ends first (client gone, worker failure) ends the session.
            await asyncio.wait({receiver, worker}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in (receiver, worker):
                task.cancel()
            # Not gather(): if *we* are cancelled meanwhile, gather would re-raise a
            # child's CancelledError instead of ours (anyio then won't absorb it).
            await asyncio.wait({receiver, worker})
        for name, task in (("receive loop", receiver), ("worker", worker)):
            exc = None if task.cancelled() else task.exception()
            if exc is not None and not isinstance(exc, (WebSocketDisconnect, _ClientGone)):
                log.error("session %s failed", name, exc_info=exc)
                await self._close(1011)
                return
        await self._close(1000)

    async def _close(self, code: int) -> None:
        connected = WebSocketState.CONNECTED
        if self.ws.application_state == connected and self.ws.client_state == connected:
            with contextlib.suppress(Exception):
                await self.ws.close(code)

    # -- receiving ------------------------------------------------------------
    async def _receive_loop(self) -> None:
        while True:
            message = await self.ws.receive()
            if message["type"] == "websocket.disconnect":
                return
            if message.get("bytes") is not None:
                await self._on_audio(message["bytes"])
            elif message.get("text") is not None:
                await self._on_text(message["text"])

    async def _on_text(self, text: str) -> None:
        try:
            msg = json.loads(text)
        except ValueError:
            await self._send({"type": "error", "message": "invalid JSON message"})
            return
        kind = msg.get("type") if isinstance(msg, dict) else None
        if kind == "start":
            await self._start()
        elif kind == "stop":
            await self._stop()
        else:
            await self._send({"type": "error", "message": f"unknown message type: {kind!r}"})

    async def _start(self) -> None:
        # A start while listening implicitly ends the current run: its
        # in-progress utterance is flushed (and speech:false sent) first.
        if self.pipeline is not None:
            await self._end_pipeline(self.pipeline)
        st = self.state
        self.pipeline = SessionPipeline(
            st.embedder, st.transcriber, st.registry, st.vad_config()
        )
        self._enqueue(_Started(self.pipeline))

    async def _stop(self) -> None:
        pipeline, self.pipeline = self.pipeline, None
        if pipeline is not None:
            await self._end_pipeline(pipeline)
        self._enqueue(_Stopped())

    async def _end_pipeline(self, pipeline: SessionPipeline) -> None:
        for segment in pipeline.flush():
            await self._enqueue_work(_Work(pipeline, segment))
        await self._send_events(pipeline)

    async def _on_audio(self, data: bytes) -> None:
        pipeline = self.pipeline
        if pipeline is None:
            return  # not started: ignore
        segments = pipeline.feed(pcm16_to_float32(data))
        await self._send_events(pipeline)
        for segment in segments:
            await self._enqueue_work(_Work(pipeline, segment))

    async def _send_events(self, pipeline: SessionPipeline) -> None:
        if pipeline is not self._live:
            return  # sent by the worker right after this pipeline's 'listening'
        async with self._send_lock:
            for active in pipeline.pop_events():
                await self._send_unlocked({"type": "speech", "active": active})

    # -- queue ----------------------------------------------------------------
    def _enqueue(self, item: _Item) -> None:
        self._items.append(item)
        self._wakeup.set()

    async def _enqueue_work(self, item: _Work) -> None:
        self._enqueue(item)
        self._pending_work += 1
        if self._pending_work > _BACKLOG_WARN:
            log.warning("session backlog: %d utterances waiting", self._pending_work)
        if self._pending_work > MAX_BACKLOG:
            # Drop the oldest waiting utterance; status markers are never dropped.
            oldest = next(i for i in self._items if isinstance(i, _Work))
            self._items.remove(oldest)
            self._pending_work -= 1
            log.warning("session backlog over %d: dropped an utterance", MAX_BACKLOG)
            await self._send({"type": "error", "message": BACKLOG_DROP_MESSAGE})

    async def _next_item(self) -> _Item:
        while not self._items:
            self._wakeup.clear()
            await self._wakeup.wait()
        item = self._items.popleft()
        if isinstance(item, _Work):
            self._pending_work -= 1
        return item

    # -- processing -----------------------------------------------------------
    async def _worker(self) -> None:
        while True:
            item = await self._next_item()
            if isinstance(item, _Started):
                async with self._send_lock:
                    await self._send_unlocked({"type": "status", "state": "listening"})
                    self._live = item.pipeline
                    for active in item.pipeline.pop_events():
                        await self._send_unlocked({"type": "speech", "active": active})
            elif isinstance(item, _Stopped):
                await self._send({"type": "status", "state": "stopped"})
            else:
                await self._process(item)

    async def _process(self, item: _Work) -> None:
        try:
            utterance = await asyncio.to_thread(item.pipeline.process, item.segment)
        except Exception:
            log.exception("failed to process utterance")
            await self._send({"type": "error", "message": "failed to process utterance"})
            return
        if utterance is not None:
            await self._send(utterance.to_message())

    async def _send(self, payload: dict[str, Any]) -> None:
        async with self._send_lock:
            await self._send_unlocked(payload)

    async def _send_unlocked(self, payload: dict[str, Any]) -> None:
        if self.ws.application_state != WebSocketState.CONNECTED:
            raise _ClientGone
        try:
            await self.ws.send_json(payload)
        except (WebSocketDisconnect, RuntimeError, OSError) as exc:
            raise _ClientGone from exc


@router.websocket("/ws/session")
async def session_ws(ws: WebSocket) -> None:
    await SessionHandler(ws, get_state(ws)).run()
