"""Enrolled speakers: thread-safe store, cosine identification, JSON persistence."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .embedder import l2_normalize

log = logging.getLogger(__name__)

PALETTE = (
    "#e6194b",  # red
    "#3cb44b",  # green
    "#4363d8",  # blue
    "#f58231",  # orange
    "#911eb4",  # purple
    "#42d4f4",  # cyan
    "#f032e6",  # magenta
    "#9a6324",  # brown
)
UNKNOWN_NAME = "Unknown"
UNKNOWN_COLOR = "#888888"
SHORT_UTTERANCE_S = 1.0
CONTINUITY_MARGIN = 0.05


class DuplicateNameError(ValueError):
    """A speaker with this name (case-insensitive) already exists."""


class SpeakerNotFoundError(KeyError):
    """No speaker with the given id."""


@dataclass
class SpeakerRecord:
    id: str
    name: str
    color: str
    created_at: float
    samples: list[np.ndarray] = field(default_factory=list)

    @property
    def centroid(self) -> np.ndarray:
        return l2_normalize(np.mean(self.samples, axis=0))

    def to_public(self) -> dict[str, Any]:
        """The ``Speaker`` JSON object of the API."""
        return {
            "id": self.id,
            "name": self.name,
            "color": self.color,
            "num_samples": len(self.samples),
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class Match:
    speaker_id: str | None
    speaker_name: str
    color: str
    confidence: float


class SpeakerRegistry:
    """Holds enrolled speakers. All public methods are thread-safe."""

    def __init__(self, embedder_name: str, threshold: float, path: Path | None = None) -> None:
        self.embedder_name = embedder_name
        self.threshold = threshold
        self.path = Path(path) if path is not None else None
        self._lock = threading.RLock()
        self._speakers: dict[str, SpeakerRecord] = {}
        self._foreign: list[dict[str, Any]] = []  # other embedders' records, kept on disk
        if self.path is not None:
            self._load()

    # -- queries ------------------------------------------------------------
    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [s.to_public() for s in self._speakers.values()]

    def get(self, speaker_id: str) -> dict[str, Any]:
        with self._lock:
            return self._require(speaker_id).to_public()

    def has_name(self, name: str) -> bool:
        key = name.strip().casefold()
        with self._lock:
            return any(s.name.casefold() == key for s in self._speakers.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._speakers)

    # -- mutations ----------------------------------------------------------
    def enroll(self, name: str, embedding: np.ndarray) -> dict[str, Any]:
        """Create a speaker from one sample embedding. Raises DuplicateNameError."""
        name = name.strip()
        with self._lock:
            if self.has_name(name):
                raise DuplicateNameError(name)
            rec = SpeakerRecord(
                id=uuid.uuid4().hex,
                name=name,
                color=self._next_color(),
                created_at=time.time(),
                samples=[l2_normalize(embedding)],
            )
            self._speakers[rec.id] = rec
            self._save()
            return rec.to_public()

    def add_sample(self, speaker_id: str, embedding: np.ndarray) -> dict[str, Any]:
        with self._lock:
            rec = self._require(speaker_id)
            rec.samples.append(l2_normalize(embedding))
            self._save()
            return rec.to_public()

    def delete(self, speaker_id: str) -> None:
        with self._lock:
            self._require(speaker_id)
            del self._speakers[speaker_id]
            self._save()

    def clear(self) -> None:
        with self._lock:
            self._speakers.clear()
            self._foreign.clear()
            self._save()

    # -- identification -----------------------------------------------------
    def identify(
        self, embedding: np.ndarray, duration: float, previous_id: str | None = None
    ) -> Match:
        """Best-matching speaker by cosine similarity, or Unknown below threshold.

        For short utterances (< 1 s) where the top two are within 0.05 and the
        previous identified speaker is one of them, the previous speaker wins.
        """
        emb = l2_normalize(embedding)
        with self._lock:
            scored = sorted(
                ((float(np.dot(emb, s.centroid)), s) for s in self._speakers.values()),
                key=lambda t: t[0],
                reverse=True,
            )
        if not scored:
            return Match(None, UNKNOWN_NAME, UNKNOWN_COLOR, 0.0)
        best_score, best = scored[0]
        if (
            duration < SHORT_UTTERANCE_S
            and previous_id is not None
            and len(scored) > 1
            and best_score - scored[1][0] < CONTINUITY_MARGIN
            and scored[1][1].id == previous_id
            and scored[1][0] >= self.threshold
        ):
            best_score, best = scored[1]
        if best_score >= self.threshold:
            return Match(best.id, best.name, best.color, round(best_score, 3))
        return Match(None, UNKNOWN_NAME, UNKNOWN_COLOR, round(best_score, 3))

    # -- internals ----------------------------------------------------------
    def _require(self, speaker_id: str) -> SpeakerRecord:
        rec = self._speakers.get(speaker_id)
        if rec is None:
            raise SpeakerNotFoundError(speaker_id)
        return rec

    def _next_color(self) -> str:
        """Least-used palette color, ties broken by palette order (cycles)."""
        used = [s.color for s in self._speakers.values()]
        return min(PALETTE, key=lambda c: (used.count(c), PALETTE.index(c)))

    def _load(self) -> None:
        assert self.path is not None
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            entries = data["speakers"]
        except (OSError, ValueError, KeyError, TypeError):
            backup = self.path.with_suffix(f".corrupt-{int(time.time())}.json")
            log.error("could not parse %s; moved it to %s", self.path, backup)
            os.replace(self.path, backup)
            return
        dropped = 0
        for entry in entries:
            if entry.get("embedder") != self.embedder_name:
                self._foreign.append(entry)
                dropped += 1
                continue
            rec = SpeakerRecord(
                id=entry["id"],
                name=entry["name"],
                color=entry["color"],
                created_at=float(entry["created_at"]),
                samples=[np.asarray(e, dtype=np.float32) for e in entry["samples"]],
            )
            if rec.samples:
                self._speakers[rec.id] = rec
        if dropped:
            log.warning(
                "ignoring %d enrolled speaker(s) made with a different embedder than %r; "
                "re-enroll them",
                dropped,
                self.embedder_name,
            )

    def _save(self) -> None:
        """Atomically write all speakers to disk (tmp file + os.replace)."""
        if self.path is None:
            return
        entries = [
            {
                "id": s.id,
                "name": s.name,
                "color": s.color,
                "created_at": s.created_at,
                "embedder": self.embedder_name,
                "samples": [e.astype(float).round(6).tolist() for e in s.samples],
            }
            for s in self._speakers.values()
        ]
        payload = {"version": 1, "speakers": entries + self._foreign}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".speakers-", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
