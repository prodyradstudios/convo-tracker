"""Enrolled speakers: thread-safe store, cosine identification, JSON persistence."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field, replace
from functools import cached_property
from pathlib import Path
from typing import Any, Sequence

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
# Prototype scoring: mean of the best TOP_K similarities to a speaker's prototypes.
TOP_K = 2
# Adaptive prototypes: confident live utterances become extra (in-memory) prototypes.
# Every gate uses the *static* score (centroid + enrolled/augmented prototypes, no
# adaptive ones), so adaptive prototypes can never vouch for their own successors.
ADAPT_MIN_SPEECH_S = 2.0
ADAPT_SCORE_MARGIN = 0.15  # static score must be >= threshold + this
ADAPT_RIVAL_MARGIN = 0.10  # ... and beat the second-best speaker's static score by this much
ADAPT_SOLO_MARGIN = 0.25  # with nobody else enrolled (no rival): static >= threshold + this
ADAPT_MAX_PROTOTYPES = 20


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
    # augmented[i] = embeddings of perturbed copies of sample i (may be empty).
    augmented: list[list[np.ndarray]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.augmented) < len(self.samples):
            self.augmented = [*self.augmented, *([] for _ in self.samples[len(self.augmented):])]

    # Records are never mutated (mutations build a new one with replace()), so
    # these derived arrays can be cached.
    @cached_property
    def centroid(self) -> np.ndarray:
        return l2_normalize(np.mean(self.samples, axis=0))

    @cached_property
    def prototypes(self) -> np.ndarray:
        """Real sample embeddings plus their augmented copies, one per row."""
        rows = [*self.samples, *(e for aug in self.augmented for e in aug)]
        return np.stack([l2_normalize(r) for r in rows])

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


@dataclass(frozen=True)
class ScoreTable:
    """Scores of several embeddings against every (compatible) enrolled speaker."""

    speaker_ids: list[str]
    scores: np.ndarray  # (n_embeddings, n_speakers)
    threshold: float


def _score_matrix(embs: np.ndarray, centroid: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    """max(cos to centroid, mean of the top-2 cos to any prototype), per row of ``embs``."""
    cent = embs @ centroid
    sims = embs @ prototypes.T
    k = min(TOP_K, sims.shape[1])
    top = np.sort(sims, axis=1)[:, -k:].mean(axis=1)
    return np.maximum(cent, top)


class SpeakerRegistry:
    """Holds enrolled speakers. All public methods are thread-safe."""

    def __init__(
        self,
        embedder_name: str,
        threshold: float,
        path: Path | None = None,
        *,
        adapt: bool = True,
    ) -> None:
        self.embedder_name = embedder_name
        self.threshold = threshold
        self.path = Path(path) if path is not None else None
        self.adapt = adapt
        self._lock = threading.RLock()
        self._speakers: dict[str, SpeakerRecord] = {}
        # In-memory only: confident live embeddings per speaker (FIFO).
        self._adaptive: dict[str, deque[np.ndarray]] = {}
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
    def enroll(
        self,
        name: str,
        embedding: np.ndarray,
        augmented: Sequence[np.ndarray] = (),
    ) -> dict[str, Any]:
        """Create a speaker from one sample embedding (plus its augmented copies).

        Raises DuplicateNameError.
        """
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
                augmented=[_clean_augmented(embedding, augmented)],
            )
            self._commit({**self._speakers, rec.id: rec}, self._foreign)
            return rec.to_public()

    def add_sample(
        self,
        speaker_id: str,
        embedding: np.ndarray,
        augmented: Sequence[np.ndarray] = (),
    ) -> dict[str, Any]:
        with self._lock:
            rec = self._require(speaker_id)
            rec = replace(
                rec,
                samples=[*rec.samples, l2_normalize(embedding)],
                augmented=[*rec.augmented, _clean_augmented(embedding, augmented)],
            )
            self._commit({**self._speakers, rec.id: rec}, self._foreign)
            return rec.to_public()

    def delete(self, speaker_id: str) -> None:
        with self._lock:
            self._require(speaker_id)
            speakers = {k: v for k, v in self._speakers.items() if k != speaker_id}
            self._commit(speakers, self._foreign)
            self._adaptive.pop(speaker_id, None)

    def clear(self) -> None:
        with self._lock:
            self._commit({}, [])
            self._adaptive.clear()

    def num_adaptive(self, speaker_id: str) -> int:
        """Number of adaptive (in-memory) prototypes currently held for a speaker."""
        with self._lock:
            return len(self._adaptive.get(speaker_id, ()))

    # -- identification -----------------------------------------------------
    def score(self, embeddings: np.ndarray, *, adaptive: bool = True) -> ScoreTable:
        """Score each row of ``embeddings`` against every enrolled speaker.

        score = max(cos(u, centroid), mean of the top-2 cos(u, prototype)) where a
        speaker's prototypes are their real sample embeddings, the augmented
        copies made at enrollment and (``adaptive=True``) any adaptive
        prototypes learnt live. ``adaptive=False`` gives the *static* score.

        Adaptive prototypes only raise a score whose static part already clears
        the threshold: they sharpen matches backed by enrolled evidence but can
        never turn an Unknown voice into a match.
        """
        embs = np.atleast_2d(np.asarray(embeddings, dtype=np.float32))
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = np.divide(embs, norms, out=np.zeros_like(embs), where=norms > 1e-12)
        with self._lock:
            speakers = self._compatible(embs.shape[1])
            extra = [self._adaptive.get(s.id) if adaptive else None for s in speakers]
            extra = [np.stack(list(q)) if q else None for q in extra]
            threshold = self.threshold
        cols = []
        for s, ad in zip(speakers, extra):
            static = _score_matrix(embs, s.centroid, s.prototypes)
            if ad is not None:
                boosted = _score_matrix(embs, s.centroid, np.vstack([s.prototypes, ad]))
                static = np.where(static >= threshold, boosted, static)
            cols.append(static)
        scores = np.stack(cols, axis=1) if cols else np.zeros((embs.shape[0], 0), np.float32)
        return ScoreTable([s.id for s in speakers], scores, threshold)

    def identify(
        self,
        embedding: np.ndarray,
        duration: float,
        previous_id: str | None = None,
        *,
        adapt: bool = False,
    ) -> Match:
        """Best-matching speaker by prototype/centroid score, or Unknown below threshold.

        For short utterances (< 1 s) where the top two are within 0.05 and the
        previous identified speaker is one of them, the previous speaker wins.
        With ``adapt=True`` (whole, clean live segments) a very confident match
        of at least 2 s of speech, judged on static scores (see
        :meth:`_maybe_adapt`), also becomes an adaptive prototype of that speaker.
        """
        emb = l2_normalize(embedding)
        table = self.score(emb[None, :])
        with self._lock:
            scored = sorted(
                (
                    (float(score), self._speakers[sid])
                    for sid, score in zip(table.speaker_ids, table.scores[0])
                    if sid in self._speakers
                ),
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
        if best_score < self.threshold:
            return Match(None, UNKNOWN_NAME, UNKNOWN_COLOR, round(best_score, 3))
        if adapt and self.adapt and duration >= ADAPT_MIN_SPEECH_S:
            self._maybe_adapt(best, emb)
        return Match(best.id, best.name, best.color, round(best_score, 3))

    def match_for(self, speaker_id: str | None, score: float) -> Match:
        """A Match for a known speaker id and score (Unknown if gone or below threshold)."""
        with self._lock:
            rec = self._speakers.get(speaker_id) if speaker_id is not None else None
            if rec is None or score < self.threshold:
                return Match(None, UNKNOWN_NAME, UNKNOWN_COLOR, round(float(score), 3))
            return Match(rec.id, rec.name, rec.color, round(float(score), 3))

    def _maybe_adapt(self, best: SpeakerRecord, emb: np.ndarray) -> bool:
        """Add ``emb`` as an adaptive prototype of ``best`` if it clears the strict bar.

        Judged on static scores only (no adaptive prototypes): static score >=
        threshold + 0.15 and >= 0.10 above every other speaker's static score
        (threshold + 0.25 when nobody else is enrolled), and cos to the centroid
        >= threshold. The caller checks the speech duration.
        """
        static = self.score(emb[None, :], adaptive=False)
        if best.id not in static.speaker_ids:
            return False
        col = static.speaker_ids.index(best.id)
        own = float(static.scores[0, col])
        rivals = np.delete(static.scores[0], col)
        if rivals.size:
            ok = own >= self.threshold + ADAPT_SCORE_MARGIN and (
                own - float(rivals.max()) >= ADAPT_RIVAL_MARGIN
            )
        else:
            ok = own >= self.threshold + ADAPT_SOLO_MARGIN
        if not ok or float(emb @ best.centroid) < self.threshold:
            return False
        with self._lock:
            if best.id not in self._speakers:
                return False  # deleted meanwhile
            queue = self._adaptive.setdefault(best.id, deque(maxlen=ADAPT_MAX_PROTOTYPES))
            queue.append(emb.astype(np.float32, copy=True))
        return True

    def _compatible(self, dim: int) -> list[SpeakerRecord]:
        """Speakers whose embeddings have size ``dim`` (call with the lock held)."""
        speakers = [s for s in self._speakers.values() if s.samples[0].size == dim]
        if len(speakers) < len(self._speakers) and not self._warned_dim:
            self._warned_dim = True
            log.warning("skipping enrolled speaker(s) whose embedding size differs; re-enroll")
        return speakers

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
                "augmented": [
                    [e.astype(float).round(6).tolist() for e in aug] for aug in s.augmented
                ],
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


def _clean_augmented(embedding: np.ndarray, augmented: Sequence[np.ndarray]) -> list[np.ndarray]:
    """Normalised augmented embeddings; drops any of the wrong size, non-finite or zero."""
    size = np.asarray(embedding).size
    out = []
    for e in augmented:
        e = np.asarray(e, dtype=np.float32).reshape(-1)
        if e.size == size and np.all(np.isfinite(e)) and np.any(e):
            out.append(l2_normalize(e))
    return out


def _check_vectors(vectors: list[np.ndarray], shape: tuple[int, ...], what: str) -> None:
    for e in vectors:
        if e.shape != shape:
            raise ValueError(f"{what} differ in size")
        if not np.all(np.abs(e) < 1e30) or not np.any(e):
            raise ValueError(f"{what} must be finite and non-zero")


def _parse_record(entry: dict[str, Any]) -> SpeakerRecord:
    """Build a record from a stored entry; raises on any malformed field."""
    samples = [np.asarray(e, dtype=np.float64) for e in entry["samples"]]
    if not samples:
        raise ValueError("no samples")
    shape = samples[0].shape
    if len(shape) != 1 or shape[0] == 0:
        raise ValueError("samples must be non-empty 1-D vectors")
    _check_vectors(samples, shape, "samples")
    # Optional (files written before augmentation existed have no such field).
    raw_aug = entry.get("augmented")
    if raw_aug is None:
        augmented: list[list[np.ndarray]] = [[] for _ in samples]
    else:
        if not isinstance(raw_aug, list) or len(raw_aug) != len(samples):
            raise ValueError("'augmented' must be a list with one entry per sample")
        if not all(isinstance(group, list) for group in raw_aug):
            raise ValueError("'augmented' entries must be lists of vectors")
        augmented = [[np.asarray(e, dtype=np.float64) for e in group] for group in raw_aug]
        for group in augmented:
            _check_vectors(group, shape, "augmented embeddings")
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
        augmented=[[e.astype(np.float32) for e in group] for group in augmented],
    )
