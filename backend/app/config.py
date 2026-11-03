"""Runtime settings, read from ``CT_*`` environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

_CHOICES = {
    "embedder": ("auto", "ecapa", "mfcc"),
    "transcriber": ("auto", "whisper", "null"),
}


def _get(env: Mapping[str, str], key: str, default: str) -> str:
    value = env.get(key, "").strip()
    return value if value else default


def _num(env: Mapping[str, str], key: str, default: float, cast: type) -> float:
    raw = _get(env, key, "")
    if not raw:
        return default
    try:
        return cast(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be a number, got {raw!r}") from exc


@dataclass
class Settings:
    """All tunables of the backend. Defaults match docs/SPEC.md."""

    data_dir: Path = field(default_factory=lambda: Path("data"))
    embedder: str = "auto"
    transcriber: str = "auto"
    match_threshold: float | None = None  # None -> embedder.default_threshold
    silence_ms: int = 700
    max_utterance_s: float = 15.0
    min_utterance_s: float = 0.5
    whisper_model: str = "base.en"
    device: str = "cpu"
    host: str = "127.0.0.1"
    port: int = 8000

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        for attr, choices in _CHOICES.items():
            value = getattr(self, attr).lower()
            if value not in choices:
                raise ValueError(f"CT_{attr.upper()} must be one of {choices}, got {value!r}")
            setattr(self, attr, value)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        """Build settings from environment variables (or a given mapping)."""
        env = os.environ if env is None else env
        threshold = _get(env, "CT_MATCH_THRESHOLD", "")
        return cls(
            data_dir=Path(_get(env, "CT_DATA_DIR", "data")),
            embedder=_get(env, "CT_EMBEDDER", "auto"),
            transcriber=_get(env, "CT_TRANSCRIBER", "auto"),
            match_threshold=_num(env, "CT_MATCH_THRESHOLD", 0.0, float) if threshold else None,
            silence_ms=int(_num(env, "CT_SILENCE_MS", 700, int)),
            max_utterance_s=_num(env, "CT_MAX_UTTERANCE_S", 15.0, float),
            min_utterance_s=_num(env, "CT_MIN_UTTERANCE_S", 0.5, float),
            whisper_model=_get(env, "CT_WHISPER_MODEL", "base.en"),
            device=_get(env, "CT_DEVICE", "cpu"),
            host=_get(env, "CT_HOST", "127.0.0.1"),
            port=int(_num(env, "CT_PORT", 8000, int)),
        )
