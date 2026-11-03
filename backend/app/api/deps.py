"""Shared helpers for route handlers."""

from __future__ import annotations

from starlette.requests import HTTPConnection

from ..state import AppState


def get_state(conn: HTTPConnection) -> AppState:
    """The AppState stored on the application at startup."""
    return conn.app.state.ct
