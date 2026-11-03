"""Speaker turns inside one VAD segment (change-point detection).

The segmenter only cuts on pauses, so when someone interrupts or replies
without a pause both voices end up in one segment. :func:`detect_turns` slides a
short window over the segment, labels each window with its best-matching
enrolled speaker, smooths the labels and returns contiguous turns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..audio.io import SAMPLE_RATE
from ..audio.vad import VadConfig, frame_db
from .embedder import Embedder, embed_many
from .registry import ScoreTable, SpeakerRegistry


@dataclass(frozen=True)
class TurnConfig:
    """Tunables of turn splitting (``CT_TURN_*``)."""

    min_segment_s: float = 2.0  # only look for turns in segments with this much speech
    window_s: float = 1.5
    hop_s: float = 0.5
    min_turn_s: float = 1.0  # shorter runs are merged into a neighbour
    max_windows: int = 40  # per segment; the hop grows if needed
    min_voiced_fraction: float = 0.5  # windows with fewer voiced frames are skipped


@dataclass(frozen=True)
class Turn:
    """``start``/``end`` are seconds relative to the segment audio."""

    start: float
    end: float
    speaker_id: str | None  # None = Unknown
    score: float  # mean score of the turn's windows for that speaker


def window_starts(n_samples: int, win: int, hop: int, max_windows: int) -> list[int]:
    """Evenly spaced window starts covering ``n_samples`` (spacing <= ``hop``)."""
    if n_samples <= win:
        return [0]
    n = math.ceil((n_samples - win) / hop) + 1
    n = max(2, min(n, max_windows))
    return [int(round(x)) for x in np.linspace(0, n_samples - win, n)]


def voiced_frames(audio: np.ndarray, vad: VadConfig) -> np.ndarray:
    """Per-frame voiced flags using the VAD's frame size and energy margins.

    The segment's own background level is estimated as its 10th-percentile
    frame energy; a frame is voiced if it is ``margin_db`` above that (but
    never demanding more than 20 dB below the loudest frame) and above the
    VAD's absolute minimum.
    """
    flen = vad.frame_len
    n = audio.size // flen
    if n == 0:
        return np.ones(1, dtype=bool)
    dbs = np.array([frame_db(audio[i * flen : (i + 1) * flen]) for i in range(n)])
    floor = float(np.percentile(dbs, 10))
    threshold = max(vad.min_speech_db, min(floor + vad.margin_db, float(dbs.max()) - 20.0))
    return dbs > threshold


def detect_turns(
    audio: np.ndarray,
    embedder: Embedder,
    registry: SpeakerRegistry,
    config: TurnConfig | None = None,
    vad: VadConfig | None = None,
) -> list[Turn]:
    """Split ``audio`` into speaker turns covering it end to end.

    Returns a single turn (speaker None) when there is nothing to split: audio
    shorter than two windows, fewer than two enrolled speakers or fewer than
    two voiced windows.
    """
    config = config or TurnConfig()
    vad = vad or VadConfig()
    sr = SAMPLE_RATE
    duration = audio.size / sr
    single = [Turn(0.0, duration, None, 0.0)]
    win, hop = int(config.window_s * sr), max(1, int(config.hop_s * sr))
    if audio.size < win + hop or len(registry) < 2:
        return single

    voiced = voiced_frames(audio, vad)
    flen = vad.frame_len
    starts = []
    for s in window_starts(audio.size, win, hop, config.max_windows):
        flags = voiced[s // flen : max(s // flen + 1, (s + win) // flen)]
        if flags.size and flags.mean() >= config.min_voiced_fraction:
            starts.append(s)
    if len(starts) < 2:
        return single

    table = registry.score(embed_many(embedder, [audio[s : s + win] for s in starts]))
    if len(table.speaker_ids) < 2:
        return single
    centres = [(s + win / 2) / sr for s in starts]
    return label_runs(table, centres, duration, config)


def label_runs(
    table: ScoreTable,
    centres: list[float],
    duration: float,
    config: TurnConfig,
) -> list[Turn]:
    """Turn per-window scores into smoothed turns (pure function, see module doc).

    1. Each window gets its best speaker, or Unknown below the threshold.
    2. A window whose two neighbours agree with each other takes their label
       (majority over 3 windows).
    3. Runs shorter than ``min_turn_s`` are merged, shortest first, into the
       neighbour whose speaker fits the run's windows better (an Unknown
       neighbour counts as scoring exactly the threshold).
    Boundaries sit midway between the centres of the last window of one run
    and the first of the next; the first turn starts at 0, the last ends at
    ``duration``.
    """
    scores = table.scores
    ids = table.speaker_ids
    best = scores.argmax(axis=1)
    labels: list[int | None] = [
        int(b) if scores[i, b] >= table.threshold else None for i, b in enumerate(best)
    ]
    for i in range(1, len(labels) - 1):
        if labels[i - 1] == labels[i + 1] != labels[i]:
            labels[i] = labels[i - 1]

    def fit(label: int | None, windows: list[int]) -> float:
        if label is None:
            return table.threshold
        return float(scores[windows, label].mean())

    runs = _runs(labels)
    while len(runs) > 1:
        spans = _spans(runs, centres, duration)
        short = [i for i, (a, b) in enumerate(spans) if b - a < config.min_turn_s]
        if not short:
            break
        i = min(short, key=lambda k: spans[k][1] - spans[k][0])
        windows = runs[i][1]
        neighbours = [j for j in (i - 1, i + 1) if 0 <= j < len(runs)]
        target = max(neighbours, key=lambda j: fit(runs[j][0], windows))
        for w in windows:
            labels[w] = runs[target][0]
        runs = _runs(labels)

    spans = _spans(runs, centres, duration)
    turns = []
    for (label, windows), (a, b) in zip(runs, spans):
        if label is None:
            score = float(scores[windows].max(axis=1).mean())
        else:
            score = fit(label, windows)
        turns.append(Turn(a, b, ids[label] if label is not None else None, score))
    return turns


def _runs(labels: list[int | None]) -> list[tuple[int | None, list[int]]]:
    runs: list[tuple[int | None, list[int]]] = []
    for i, label in enumerate(labels):
        if runs and runs[-1][0] == label:
            runs[-1][1].append(i)
        else:
            runs.append((label, [i]))
    return runs


def _spans(
    runs: list[tuple[int | None, list[int]]], centres: list[float], duration: float
) -> list[tuple[float, float]]:
    cuts = [
        (centres[prev[1][-1]] + centres[nxt[1][0]]) / 2.0 for prev, nxt in zip(runs, runs[1:])
    ]
    edges = [0.0, *cuts, duration]
    return list(zip(edges, edges[1:]))
