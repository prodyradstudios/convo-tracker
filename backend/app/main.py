"""FastAPI application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .api import session, speakers
from .config import Settings
from .speakers.embedder import Embedder
from .state import build_state
from .transcription.transcriber import Transcriber

log = logging.getLogger(__name__)

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"


def create_app(
    settings: Settings | None = None,
    *,
    embedder: Embedder | None = None,
    transcriber: Transcriber | None = None,
    frontend_dir: Path | None = FRONTEND_DIR,
) -> FastAPI:
    """Build the app. ``embedder``/``transcriber`` may be injected (tests)."""
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.ct = build_state(settings, embedder, transcriber)
        st = app.state.ct
        log.info(
            "Convo Tracker ready: embedder=%s transcriber=%s threshold=%.2f speakers=%d",
            st.embedder.name, st.transcriber.name, st.registry.threshold, len(st.registry),
        )
        yield

    app = FastAPI(title="Convo Tracker", lifespan=lifespan)
    app.include_router(speakers.router, prefix="/api")
    app.include_router(session.router)
    if frontend_dir is not None and Path(frontend_dir).is_dir():
        app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
    else:
        log.warning("frontend directory %s not found; serving API only", frontend_dir)
    return app


app = create_app()
