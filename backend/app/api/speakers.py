"""REST routes for speaker enrollment (prefix ``/api``)."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from starlette.datastructures import FormData, UploadFile
from starlette.types import Message

from ..audio.io import AudioDecodeError, decode_wav, probe_duration
from ..audio.vad import voiced_segments
from ..speakers.embedder import embed_windows
from ..speakers.registry import DuplicateNameError, SpeakerNotFoundError
from ..state import AppState
from .deps import get_state

router = APIRouter()

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
# Whole multipart request: the file plus boundaries, headers and the name field.
MAX_REQUEST_BYTES = MAX_UPLOAD_BYTES + 64 * 1024
MAX_DURATION_S = 60.0
MIN_SPEECH_S = 2.0
MAX_NAME_LEN = 40


@router.get("/health")
def health(state: AppState = Depends(get_state)) -> dict[str, str]:
    return {
        "status": "ok",
        "embedder": state.embedder.name,
        "transcriber": state.transcriber.name,
    }


@router.get("/speakers")
def list_speakers(state: AppState = Depends(get_state)) -> dict[str, list]:
    return {"speakers": state.registry.list()}


# The upload routes parse the form themselves (instead of Form/File params) so
# an oversized body is refused before it is read; see _read_form.
@router.post("/speakers", status_code=201)
async def create_speaker(request: Request, state: AppState = Depends(get_state)) -> dict:
    form = await _read_form(request)
    try:
        name = form.get("name")
        name = _validate_name(name if isinstance(name, str) else "")
        if state.registry.has_name(name):
            raise HTTPException(409, f"a person named {name!r} is already enrolled")
        embedding = await _embed_upload(_file_field(form), state)
    finally:
        await form.close()
    try:
        return state.registry.enroll(name, embedding)
    except DuplicateNameError:
        raise HTTPException(409, f"a person named {name!r} is already enrolled") from None


@router.post("/speakers/{speaker_id}/samples")
async def add_sample(
    speaker_id: str, request: Request, state: AppState = Depends(get_state)
) -> dict:
    form = await _read_form(request)
    try:
        _ensure_exists(state, speaker_id)
        embedding = await _embed_upload(_file_field(form), state)
    finally:
        await form.close()
    try:
        return state.registry.add_sample(speaker_id, embedding)
    except SpeakerNotFoundError:
        raise HTTPException(404, "speaker not found") from None


@router.delete("/speakers/{speaker_id}", status_code=204)
def delete_speaker(speaker_id: str, state: AppState = Depends(get_state)) -> Response:
    try:
        state.registry.delete(speaker_id)
    except SpeakerNotFoundError:
        raise HTTPException(404, "speaker not found") from None
    return Response(status_code=204)


@router.delete("/speakers", status_code=204)
def delete_all(state: AppState = Depends(get_state)) -> Response:
    state.registry.clear()
    return Response(status_code=204)


# -- helpers ----------------------------------------------------------------
def _too_large() -> HTTPException:
    return HTTPException(413, "audio file is larger than 10 MB")


async def _read_form(request: Request) -> FormData:
    """Parse the multipart body, refusing (413) bodies over MAX_REQUEST_BYTES early.

    A declared Content-Length is checked before reading anything; a chunked
    body is cut off as soon as it exceeds the limit.
    """
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > MAX_REQUEST_BYTES:
        raise _too_large()
    received = 0

    async def capped_receive() -> Message:
        nonlocal received
        message = await request.receive()
        if message["type"] == "http.request":
            received += len(message.get("body", b""))
            if received > MAX_REQUEST_BYTES:
                raise _too_large()
        return message

    return await Request(request.scope, capped_receive).form(max_files=4, max_fields=16)


def _file_field(form: FormData) -> UploadFile | None:
    audio = form.get("audio")
    return audio if isinstance(audio, UploadFile) else None


def _validate_name(name: str) -> str:
    name = (name or "").strip()
    if not 1 <= len(name) <= MAX_NAME_LEN:
        raise HTTPException(400, f"name must be 1-{MAX_NAME_LEN} characters")
    return name


def _ensure_exists(state: AppState, speaker_id: str) -> None:
    try:
        state.registry.get(speaker_id)
    except SpeakerNotFoundError:
        raise HTTPException(404, "speaker not found") from None


async def _embed_upload(upload: UploadFile | None, state: AppState) -> np.ndarray:
    """Validate an uploaded recording and return its enrollment embedding."""
    if upload is None:
        raise HTTPException(400, "missing 'audio' file")
    data = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "audio file is larger than 10 MB")
    if not data:
        raise HTTPException(400, "audio file is empty")
    try:
        if probe_duration(data) > MAX_DURATION_S:
            raise HTTPException(413, "recording is longer than 60 seconds")
        audio = await run_in_threadpool(decode_wav, data)
    except AudioDecodeError as exc:
        raise HTTPException(400, str(exc)) from None
    segments = await run_in_threadpool(voiced_segments, audio, state.vad_config())
    # Count actual speech (onset to last voiced frame), not the pre-roll/tail
    # padding; the padded audio is still what gets embedded.
    if sum(s.speech_seconds for s in segments) < MIN_SPEECH_S:
        raise HTTPException(
            400, f"not enough speech: need at least {MIN_SPEECH_S:.0f} s, please speak longer"
        )
    speech = np.concatenate([s.audio for s in segments])
    return await run_in_threadpool(embed_windows, state.embedder, speech)
