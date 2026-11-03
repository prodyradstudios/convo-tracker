"""WebSocket ``/ws/session``: live audio in, labelled utterances out."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
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


@dataclass
class _Work:
    pipeline: SessionPipeline
    segment: Segment


@dataclass
class _Stopped:
    """Queue marker: everything before it is processed; announce 'stopped'."""


class _ClientGone(Exception):
    """The client disconnected while we were sending."""


class SessionHandler:
    """One connection: a receive loop plus a single ML worker task."""

    def __init__(self, ws: WebSocket, state: AppState) -> None:
        self.ws = ws
        self.state = state
        self.pipeline: SessionPipeline | None = None
        self.queue: asyncio.Queue[_Work | _Stopped] = asyncio.Queue()
        self._send_lock = asyncio.Lock()
        self._worker_task: asyncio.Task[None] | None = None

    async def run(self) -> None:
        await self.ws.accept()
        worker = self._worker_task = asyncio.create_task(self._worker(), name="ct-session-worker")
        try:
            await self._receive_loop()
        except (WebSocketDisconnect, _ClientGone):
            pass
        finally:
            worker.cancel()
            with contextlib.suppress(asyncio.CancelledError, _ClientGone, WebSocketDisconnect):
                await worker

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
        # Let a previous run finish so its results/status precede the new session.
        if self._worker_task is not None and not self._worker_task.done():
            await self.queue.join()
        st = self.state
        self.pipeline = SessionPipeline(
            st.embedder, st.transcriber, st.registry, st.vad_config()
        )
        await self._send({"type": "status", "state": "listening"})

    async def _stop(self) -> None:
        pipeline, self.pipeline = self.pipeline, None
        if pipeline is not None:
            for segment in pipeline.flush():
                self.queue.put_nowait(_Work(pipeline, segment))
            await self._send_events(pipeline)
        self.queue.put_nowait(_Stopped())

    async def _on_audio(self, data: bytes) -> None:
        pipeline = self.pipeline
        if pipeline is None:
            return  # not started: ignore
        segments = pipeline.feed(pcm16_to_float32(data))
        await self._send_events(pipeline)
        for segment in segments:
            self.queue.put_nowait(_Work(pipeline, segment))
        if self.queue.qsize() > _BACKLOG_WARN:
            log.warning("session backlog: %d utterances waiting", self.queue.qsize())

    async def _send_events(self, pipeline: SessionPipeline) -> None:
        for active in pipeline.pop_events():
            await self._send({"type": "speech", "active": active})

    # -- processing -----------------------------------------------------------
    async def _worker(self) -> None:
        while True:
            item = await self.queue.get()
            try:
                if isinstance(item, _Stopped):
                    await self._send({"type": "status", "state": "stopped"})
                else:
                    await self._process(item)
            finally:
                self.queue.task_done()

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
        if self.ws.application_state != WebSocketState.CONNECTED:
            raise _ClientGone
        async with self._send_lock:
            try:
                await self.ws.send_json(payload)
            except (WebSocketDisconnect, RuntimeError, OSError) as exc:
                raise _ClientGone from exc


@router.websocket("/ws/session")
async def session_ws(ws: WebSocket) -> None:
    await SessionHandler(ws, get_state(ws)).run()
