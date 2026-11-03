"""Runtime settings, read from ``CT_*`` environment variables."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

MAX_UTTERANCE_LIMIT_S = 300.0
MIN_TURN_WINDOW_S = 0.25
MAX_TURN_WINDOW_S = 10.0

_CHOICES = {
    "embedder": ("auto", "ecapa", "mfcc"),
    "transcriber": ("auto", "whisper", "null"),
}


def _get(env: Mapping[str, str], key: str, default: str) -> str:
    value = env.get(key, "").strip()
    return value if value else default


_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")


def _flag(env: Mapping[str, str], key: str, default: bool) -> bool:
    raw = _get(env, key, "").lower()
    if not raw:
        return default
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    raise ValueError(f"{key} must be 1 or 0 (or true/false), got {raw!r}")


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
    # Speaker-turn splitting inside a segment (speakers/turns.py).
    turn_min_segment_s: float = 2.0
    turn_window_s: float = 1.5
    turn_hop_s: float = 0.5
    turn_min_s: float = 1.0
    # Robustness to voice variation (speakers/embedder.py, speakers/registry.py).
    enroll_augment: bool = True
    adapt: bool = True

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        for attr, choices in _CHOICES.items():
            value = getattr(self, attr).lower()
            if value not in choices:
                raise ValueError(f"CT_{attr.upper()} must be one of {choices}, got {value!r}")
            setattr(self, attr, value)
        self._validate_numbers()

    def _validate_numbers(self) -> None:
        """Fail at startup (ValueError) on non-finite or out-of-range tunables."""

        def check(env: str, value: float, ok: bool, expected: str) -> None:
            if not (math.isfinite(value) and ok):
                raise ValueError(f"{env} must be {expected}, got {value!r}")

        if self.match_threshold is not None:
            t = self.match_threshold
            check("CT_MATCH_THRESHOLD", t, -1.0 <= t <= 1.0, "a number in [-1, 1]")
        check("CT_SILENCE_MS", self.silence_ms, self.silence_ms > 0, "> 0")
        lo, hi = self.min_utterance_s, self.max_utterance_s
        check("CT_MIN_UTTERANCE_S", lo, lo >= 0, ">= 0")
        check(
            "CT_MAX_UTTERANCE_S",
            hi,
            lo < hi <= MAX_UTTERANCE_LIMIT_S,
            f"> CT_MIN_UTTERANCE_S ({lo}) and <= {MAX_UTTERANCE_LIMIT_S:.0f}",
        )
        check("CT_PORT", self.port, 0 <= self.port <= 65535, "a port number (0-65535)")
        seg = self.turn_min_segment_s
        check("CT_TURN_MIN_SEGMENT_S", seg, seg >= 0, ">= 0")
        win, hop = self.turn_window_s, self.turn_hop_s
        check(
            "CT_TURN_WINDOW_S",
            win,
            MIN_TURN_WINDOW_S <= win <= MAX_TURN_WINDOW_S,
            f"in [{MIN_TURN_WINDOW_S}, {MAX_TURN_WINDOW_S:.0f}]",
        )
        check("CT_TURN_HOP_S", hop, 0 < hop <= win, f"> 0 and <= CT_TURN_WINDOW_S ({win})")
        check("CT_TURN_MIN_S", self.turn_min_s, self.turn_min_s >= 0, ">= 0")
        for env, flag in (("CT_ENROLL_AUGMENT", self.enroll_augment), ("CT_ADAPT", self.adapt)):
            if not isinstance(flag, bool):
                raise ValueError(f"{env} must be a boolean, got {flag!r}")

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
            turn_min_segment_s=_num(env, "CT_TURN_MIN_SEGMENT_S", 2.0, float),
            turn_window_s=_num(env, "CT_TURN_WINDOW_S", 1.5, float),
            turn_hop_s=_num(env, "CT_TURN_HOP_S", 0.5, float),
            turn_min_s=_num(env, "CT_TURN_MIN_S", 1.0, float),
            enroll_augment=_flag(env, "CT_ENROLL_AUGMENT", True),
            adapt=_flag(env, "CT_ADAPT", True),
        )
