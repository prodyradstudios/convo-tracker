"""Enrolled speakers: thread-safe store, cosine identification, JSON persistence."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
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
        self._warned_dim = False
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
    # Each mutation builds the new state, saves it, and only then commits it to
    # memory, so a failed save leaves memory matching what is on disk.
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
            self._commit({**self._speakers, rec.id: rec}, self._foreign)
            return rec.to_public()

    def add_sample(self, speaker_id: str, embedding: np.ndarray) -> dict[str, Any]:
        with self._lock:
            rec = self._require(speaker_id)
            rec = replace(rec, samples=[*rec.samples, l2_normalize(embedding)])
            self._commit({**self._speakers, rec.id: rec}, self._foreign)
            return rec.to_public()

    def delete(self, speaker_id: str) -> None:
        with self._lock:
            self._require(speaker_id)
            speakers = {k: v for k, v in self._speakers.items() if k != speaker_id}
            self._commit(speakers, self._foreign)

    def clear(self) -> None:
        with self._lock:
            self._commit({}, [])

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
            speakers = [s for s in self._speakers.values() if s.samples[0].shape == emb.shape]
            if len(speakers) < len(self._speakers) and not self._warned_dim:
                self._warned_dim = True
                log.warning("skipping enrolled speaker(s) whose embedding size differs; re-enroll")
            scored = sorted(
                ((float(np.dot(emb, s.centroid)), s) for s in speakers),
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

    def _commit(self, speakers: dict[str, SpeakerRecord], foreign: list[dict[str, Any]]) -> None:
        """Persist the given state, then make it current (unchanged if saving fails)."""
        self._save(speakers, foreign)
        self._speakers, self._foreign = speakers, foreign

    def _move_aside(self, reason: str) -> None:
        assert self.path is not None
        backup = self.path.with_suffix(f".corrupt-{int(time.time())}.json")
        log.error("could not load %s (%s); moved it to %s", self.path, reason, backup)
        os.replace(self.path, backup)

    def _load(self) -> None:
        assert self.path is not None
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            self._move_aside(f"invalid JSON: {exc}")
            return
        entries = data.get("speakers") if isinstance(data, dict) else None
        if not isinstance(entries, list):
            self._move_aside("expected an object with a 'speakers' list")
            return
        dropped = skipped = 0
        dim: int | None = None
        for entry in entries:
            if not isinstance(entry, dict):
                skipped += 1
                continue
            if entry.get("embedder") != self.embedder_name:
                self._foreign.append(entry)
                dropped += 1
                continue
            try:
                rec = _parse_record(entry)
            except (KeyError, TypeError, ValueError) as exc:
                log.warning("skipping invalid speaker entry in %s: %s", self.path, exc)
                skipped += 1
                continue
            if dim is None:
                dim = rec.samples[0].size
            if rec.samples[0].size != dim or rec.id in self._speakers:
                log.warning("skipping inconsistent speaker entry %r in %s", rec.name, self.path)
                skipped += 1
                continue
            self._speakers[rec.id] = rec
        if skipped:
            log.warning("skipped %d invalid speaker entr(ies) in %s", skipped, self.path)
        if dropped:
            log.warning(
                "ignoring %d enrolled speaker(s) made with a different embedder than %r; "
                "re-enroll them",
                dropped,
                self.embedder_name,
            )

    def _save(self, speakers: dict[str, SpeakerRecord], foreign: list[dict[str, Any]]) -> None:
        """Atomically write ``speakers`` + ``foreign`` to disk (tmp file + os.replace)."""
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
            for s in speakers.values()
        ]
        payload = {"version": 1, "speakers": entries + foreign}
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


def _parse_record(entry: dict[str, Any]) -> SpeakerRecord:
    """Build a record from a stored entry; raises on any malformed field."""
    samples = [np.asarray(e, dtype=np.float64) for e in entry["samples"]]
    if not samples:
        raise ValueError("no samples")
    shape = samples[0].shape
    if len(shape) != 1 or shape[0] == 0:
        raise ValueError("samples must be non-empty 1-D vectors")
    for e in samples:
        if e.shape != shape:
            raise ValueError("samples differ in size")
        if not np.all(np.abs(e) < 1e30) or not np.any(e):
            raise ValueError("samples must be finite and non-zero")
    for key in ("id", "name", "color"):
        if not isinstance(entry[key], str) or not entry[key]:
            raise ValueError(f"{key!r} must be a non-empty string")
    created_at = float(entry["created_at"])
    if not np.isfinite(created_at):
        raise ValueError("'created_at' must be finite")
    return SpeakerRecord(
        id=entry["id"],
        name=entry["name"],
        color=entry["color"],
        created_at=created_at,
        samples=[e.astype(np.float32) for e in samples],
    )
