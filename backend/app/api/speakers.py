"""REST routes for speaker enrollment (prefix ``/api``)."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile
from fastapi.concurrency import run_in_threadpool

from ..audio.io import SAMPLE_RATE, AudioDecodeError, decode_wav, probe_duration
from ..audio.vad import trim_silence
from ..speakers.embedder import embed_windows
from ..speakers.registry import DuplicateNameError, SpeakerNotFoundError
from ..state import AppState
from .deps import get_state

router = APIRouter()

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
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


@router.post("/speakers", status_code=201)
async def create_speaker(
    name: str = Form(""),
    audio: UploadFile | None = File(None),
    state: AppState = Depends(get_state),
) -> dict:
    name = _validate_name(name)
    if state.registry.has_name(name):
        raise HTTPException(409, f"a person named {name!r} is already enrolled")
    embedding = await _embed_upload(audio, state)
    try:
        return state.registry.enroll(name, embedding)
    except DuplicateNameError:
        raise HTTPException(409, f"a person named {name!r} is already enrolled") from None


@router.post("/speakers/{speaker_id}/samples")
async def add_sample(
    speaker_id: str,
    audio: UploadFile | None = File(None),
    state: AppState = Depends(get_state),
) -> dict:
    _ensure_exists(state, speaker_id)
    embedding = await _embed_upload(audio, state)
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
    speech = await run_in_threadpool(trim_silence, audio, state.vad_config())
    if speech.size / SAMPLE_RATE < MIN_SPEECH_S:
        raise HTTPException(
            400, f"not enough speech: need at least {MIN_SPEECH_S:.0f} s, please speak longer"
        )
    return await run_in_threadpool(embed_windows, state.embedder, speech)
