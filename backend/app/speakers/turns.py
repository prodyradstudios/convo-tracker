"""Speaker turns inside one VAD segment (change-point detection).

The segmenter only cuts on pauses, so when someone interrupts or replies
without a pause both voices end up in one segment. :func:`analyze_turns` first
checks short windows for sustained high-confidence identity changes and large
cosine-distance jumps. Each pass has a separate ``max_windows`` budget, and
sparse short windows cannot establish sustained evidence. If no strong change
is found, the conservative pass first embeds only the windows that tile the
segment end to end ("chunks": about the segment's length of audio
in total). If they all confidently agree on one speaker it stops there
(one turn; the chunks' mean embedding stands for the whole segment).
Otherwise it embeds the remaining windows, labels each window with its
best-matching enrolled speaker, smooths the labels and returns contiguous turns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..audio.io import SAMPLE_RATE
from ..audio.vad import VadConfig, frame_db, speech_threshold_db
from .embedder import Embedder, embed_many, l2_normalize
from .registry import ScoreTable, SpeakerRegistry

# Chunk pre-check: every chunk's best speaker (static score) must be the same,
# at or above the threshold and this far ahead of the runner-up.
CHUNK_MARGIN = 0.10
# Level-step refinement: when two adjacent turns differ this much in level, the
# louder voice dominates every window it is in (the cut lands late on the quiet
# side), so the cut moves to the sharpest level step near it.
LEVEL_STEP_DB = 10.0
LEVEL_CONTEXT_S = 0.4
_EPS = 1e-6  # tolerance for comparing spans (window spacing is not exactly the hop)


@dataclass(frozen=True)
class TurnConfig:
    """Tunables of turn splitting (``CT_TURN_*``)."""

    min_segment_s: float = 2.0  # only look for turns in segments with this much speech
    window_s: float = 1.5
    hop_s: float = 0.5
    min_turn_s: float = 1.0  # shorter runs are merged into a neighbour
    max_windows: int = 40  # per segment; the hop grows if needed
    min_voiced_fraction: float = 0.5  # windows with fewer voiced frames are skipped
    # Separate bounded scan for strong, short interruptions.
    change_window_s: float = 0.4
    change_hop_s: float = 0.1
    change_margin: float = 0.15
    change_confirm_s: float = 0.2
    change_threshold: float = 0.35  # cosine distance between voices


@dataclass(frozen=True)
class Turn:
    """``start``/``end`` are seconds relative to the segment audio."""

    start: float
    end: float
    speaker_id: str | None  # None = Unknown
    score: float  # mean score of the turn's windows for that speaker


@dataclass(frozen=True)
class TurnAnalysis:
    """Result of :func:`analyze_turns`.

    ``clean``: every analysed chunk/window agreed on one label (no hint of a
    second voice). ``embedding``: normalised mean of the chunk embeddings, a
    stand-in for the whole-segment embedding (None if no chunk was voiced or
    nothing was analysed).
    """

    turns: list[Turn]
    clean: bool
    embedding: np.ndarray | None


def window_starts(n_samples: int, win: int, hop: int, max_windows: int) -> list[int]:
    """Evenly spaced window starts covering ``n_samples`` (spacing <= ``hop``)."""
    if n_samples <= win:
        return [0]
    n = math.ceil((n_samples - win) / hop) + 1
    n = max(2, min(n, max_windows))
    return [int(round(x)) for x in np.linspace(0, n_samples - win, n)]


def frame_dbs(audio: np.ndarray, vad: VadConfig) -> np.ndarray:
    """Energy (dBFS) of each whole VAD frame of ``audio``."""
    flen = vad.frame_len
    return np.array(
        [frame_db(audio[i * flen : (i + 1) * flen]) for i in range(audio.size // flen)]
    )


def voiced_frames(
    audio: np.ndarray, vad: VadConfig, noise_floor_db: float | None = None
) -> np.ndarray:
    """Per-frame voiced flags using the VAD's frame size and absolute criteria.

    A frame is voiced if its energy clears the VAD's speech threshold
    (``noise floor + margin_db``, never below ``min_speech_db``). The floor is
    the segmenter's noise floor when known, else the segment's 2nd-percentile
    frame energy (its pre-roll/tail), capped like the VAD's. Nothing is
    relative to the loudest frame, so a speaker far quieter than the other is
    still voiced as long as the VAD would call them speech.
    """
    return _voiced(frame_dbs(audio, vad), vad, noise_floor_db)


def _voiced(dbs: np.ndarray, vad: VadConfig, noise_floor_db: float | None) -> np.ndarray:
    if dbs.size == 0:
        return np.ones(1, dtype=bool)
    floor = noise_floor_db if noise_floor_db is not None else float(np.percentile(dbs, 2))
    return dbs > speech_threshold_db(vad, floor)


def _voiced_fraction(voiced: np.ndarray, start: int, end: int, flen: int) -> float:
    flags = voiced[start // flen : max(start // flen + 1, end // flen)]
    return float(flags.mean()) if flags.size else 0.0


def detect_turns(
    audio: np.ndarray,
    embedder: Embedder,
    registry: SpeakerRegistry,
    config: TurnConfig | None = None,
    vad: VadConfig | None = None,
    noise_floor_db: float | None = None,
) -> list[Turn]:
    """Split ``audio`` into speaker turns covering it end to end (see :func:`analyze_turns`)."""
    return analyze_turns(audio, embedder, registry, config, vad, noise_floor_db).turns


def analyze_turns(
    audio: np.ndarray,
    embedder: Embedder,
    registry: SpeakerRegistry,
    config: TurnConfig | None = None,
    vad: VadConfig | None = None,
    noise_floor_db: float | None = None,
) -> TurnAnalysis:
    """Turns of ``audio``, whether the segment looked like one clean voice, and its embedding.

    Short-window changes are checked before the conservative coarse pass.
    A registered voice can change to Unknown even with only one enrollment.
    With no reliable change, short clips return a single unlabelled turn for
    the pipeline to identify from the whole clip.
    """
    config = config or TurnConfig()
    vad = vad or VadConfig()
    sr = SAMPLE_RATE
    duration = audio.size / sr
    single = [Turn(0.0, duration, None, 0.0)]
    win, hop = int(config.window_s * sr), max(1, int(config.hop_s * sr))
    if len(registry) < 1:
        return TurnAnalysis(single, True, None)

    dbs = frame_dbs(audio, vad)
    voiced = _voiced(dbs, vad, noise_floor_db)
    flen = vad.frame_len
    fast = _strong_changes(audio, embedder, registry, config, voiced, flen)
    if fast is not None:
        turns = refine_by_level(fast.turns, dbs, voiced, vad.frame_ms / 1000.0,
                                config.change_window_s / 2)
        return TurnAnalysis(turns, fast.clean, fast.embedding)
    if audio.size < win + hop or len(registry) < 2:
        return TurnAnalysis(single, True, None)
    starts = [
        s
        for s in window_starts(audio.size, win, hop, config.max_windows)
        if _voiced_fraction(voiced, s, s + win, flen) >= config.min_voiced_fraction
    ]
    if not starts:
        return TurnAnalysis(single, True, None)

    # 1. Cheap pre-check on "chunks": a subset of the windows, about one window
    #    length apart (tiling the segment), i.e. about one whole-segment
    #    embedding's worth of audio. Their embeddings are reused by the fine pass.
    def embed(indices: list[int]) -> None:
        clips = [audio[starts[i] : starts[i] + win] for i in indices]
        embs.update(zip(indices, embed_many(embedder, clips)))

    embs: dict[int, np.ndarray] = {}
    chunk_idx = _chunk_indices(starts, win)
    embed(chunk_idx)
    chunk_embs = np.stack([embs[i] for i in chunk_idx])
    embedding = l2_normalize(chunk_embs.mean(axis=0))
    agreed = _agreed_speaker(registry.score(chunk_embs, adaptive=False))
    if agreed is not None:
        sid, score = agreed
        return TurnAnalysis([Turn(0.0, duration, sid, score)], True, embedding)
    if len(starts) < 2:
        return TurnAnalysis(single, True, embedding)

    # 2. Fine analysis: all (overlapping) windows, labelled and smoothed.
    rest = [i for i in range(len(starts)) if i not in embs]
    if rest:
        embed(rest)
    table = registry.score(np.stack([embs[i] for i in range(len(starts))]))
    if len(table.speaker_ids) < 2:
        return TurnAnalysis(single, True, embedding)
    centres = [(s + win / 2) / sr for s in starts]
    clean = len(set(window_labels(table))) == 1
    turns = label_runs(table, centres, duration, config)
    turns = refine_by_level(turns, dbs, voiced, vad.frame_ms / 1000.0, config.window_s / 2)
    return TurnAnalysis(turns, clean, embedding)


def _strong_changes(
    audio: np.ndarray,
    embedder: Embedder,
    registry: SpeakerRegistry,
    config: TurnConfig,
    voiced: np.ndarray,
    flen: int,
) -> TurnAnalysis | None:
    """Protect sustained, well-separated short turns from coarse smoothing.

    This pass has its own max_windows budget. Require both a confident identity
    and a large embedding change; boundary mixtures and isolated flips cannot
    become a new turn. No strong change leaves the coarse analysis unchanged.
    """
    sr = SAMPLE_RATE
    win = max(1, int(config.change_window_s * sr))
    hop = max(1, int(config.change_hop_s * sr))
    if audio.size < win + hop:
        return None
    starts = [s for s in window_starts(audio.size, win, hop, config.max_windows)
              if _voiced_fraction(voiced, s, s + win, flen) >= config.min_voiced_fraction]
    if len(starts) < 2:
        return None
    # Sparse observations do not establish a sustained short voice change.
    # The cap or skipped silence can create gaps; use coarse analysis then.
    if any(b - a > win for a, b in zip(starts, starts[1:])):
        return None
    embs = embed_many(embedder, [audio[s:s + win] for s in starts])
    table = registry.score(embs, adaptive=False)
    if not table.speaker_ids:
        return None
    centres = [(s + win / 2) / sr for s in starts]
    spacing = min(config.change_hop_s, config.change_window_s)
    # -1 is inconclusive; None is a confidently unenrolled voice. A gap below
    # threshold keeps ordinary boundary mixtures from becoming Unknown turns.
    labels: list[int | None] = []
    for row in table.scores:
        order = np.argsort(row)
        best = int(order[-1])
        runner = float(row[order[-2]]) if len(order) > 1 else -1.0
        if row[best] >= table.threshold and row[best] - runner >= config.change_margin:
            labels.append(best)
        elif row[best] < table.threshold - config.change_margin:
            labels.append(None)
        else:
            labels.append(-1)
    accepted: list[tuple[int | None, list[int]]] = []
    for label, indices in _runs(labels):
        if label == -1 or len(indices) < 2:
            continue
        evidence = centres[indices[-1]] - centres[indices[0]] + spacing
        if evidence + _EPS < config.change_confirm_s:
            continue
        if accepted and accepted[-1][0] == label:
            accepted[-1][1].extend(indices)
            continue
        if accepted:
            previous = l2_normalize(embs[accepted[-1][1]].mean(axis=0))
            candidate = l2_normalize(embs[indices].mean(axis=0))
            if 1.0 - float(np.dot(previous, candidate)) < config.change_threshold:
                continue
        accepted.append((label, list(indices)))
    if len(accepted) < 2:
        return None
    spans = _spans(accepted, centres, audio.size / sr)
    turns = []
    for (label, indices), (start, end) in zip(accepted, spans):
        score = float(table.scores[indices].max(axis=1).mean()) if label is None else float(
            table.scores[indices, label].mean())
        turns.append(Turn(start, end, table.speaker_ids[label] if label is not None else None, score))
    return TurnAnalysis(turns, False, l2_normalize(embs.mean(axis=0)))


def refine_by_level(
    turns: list[Turn], dbs: np.ndarray, voiced: np.ndarray, frame_s: float, reach_s: float
) -> list[Turn]:
    """Move each cut between turns >= 10 dB apart to the sharpest level step near it.

    Window embeddings are dominated by the louder voice, so with a large level
    gap the cut lands up to half a window inside the quiet turn. The step is
    searched within ``reach_s`` of the cut: the frame where the median level of
    the 0.4 s before minus that of the 0.4 s after is largest (in the expected
    direction, and at least half the gap between the two turns' levels).
    """
    if len(turns) < 2 or dbs.size == 0 or voiced.size != dbs.size:
        return turns
    ctx = max(1, int(round(LEVEL_CONTEXT_S / frame_s)))
    reach = int(round(reach_s / frame_s))

    def level(t: Turn) -> float | None:
        a, b = int(t.start / frame_s), int(t.end / frame_s)
        vals = dbs[a:b][voiced[a:b]]
        return float(np.median(vals)) if vals.size else None

    out = list(turns)
    for i in range(len(out) - 1):
        left, right = out[i], out[i + 1]
        la, lb = level(left), level(right)
        if la is None or lb is None or abs(la - lb) < LEVEL_STEP_DB:
            continue
        sign = 1.0 if la > lb else -1.0
        cut = int(round(left.end / frame_s))
        lo = max(cut - reach, int(np.ceil(left.start / frame_s)) + ctx)
        hi = min(cut + reach, int(right.end / frame_s) - ctx)
        if hi < lo:
            continue
        ks = np.arange(lo, hi + 1)
        steps = np.array(
            [sign * (np.median(dbs[k - ctx : k]) - np.median(dbs[k : k + ctx])) for k in ks]
        )
        j = int(steps.argmax())
        if steps[j] >= abs(la - lb) / 2:
            # The medians make a plateau around the true step: take its middle.
            near = steps >= steps[j] - 1.0
            a = b = j
            while a > 0 and near[a - 1]:
                a -= 1
            while b < len(ks) - 1 and near[b + 1]:
                b += 1
            t = float(ks[(a + b) // 2]) * frame_s
            if left.start < t < right.end:
                out[i] = Turn(left.start, t, left.speaker_id, left.score)
                out[i + 1] = Turn(t, right.end, right.speaker_id, right.score)
    return out


def _chunk_indices(starts: list[int], win: int) -> list[int]:
    """Windows chosen to cover the segment end to end with as little overlap as possible.

    Each next chunk is the furthest window starting no later than the end of
    the previous one, so there are no gaps (a short turn cannot hide in one).
    """
    idx = [0]
    while idx[-1] < len(starts) - 1:
        cur = idx[-1]
        nxt = max((i for i in range(cur + 1, len(starts)) if starts[i] <= starts[cur] + win),
                  default=cur + 1)
        idx.append(nxt)
    return idx


def _agreed_speaker(table: ScoreTable) -> tuple[str, float] | None:
    """(speaker id, mean static score) if >= 2 chunks all confidently pick one speaker."""
    scores = table.scores
    if scores.shape[0] < 2 or scores.shape[1] < 2:
        return None
    best = scores.argmax(axis=1)
    if np.any(best != best[0]):
        return None
    ordered = np.sort(scores, axis=1)
    if np.any(ordered[:, -1] < table.threshold) or np.any(
        ordered[:, -1] - ordered[:, -2] < CHUNK_MARGIN
    ):
        return None
    return table.speaker_ids[int(best[0])], float(ordered[:, -1].mean())


def window_labels(table: ScoreTable) -> list[int | None]:
    """Each window's best speaker (column index), or None below the threshold."""
    best = table.scores.argmax(axis=1)
    return [int(b) if table.scores[i, b] >= table.threshold else None for i, b in enumerate(best)]


def label_runs(
    table: ScoreTable,
    centres: list[float],
    duration: float,
    config: TurnConfig,
) -> list[Turn]:
    """Turn per-window scores into smoothed turns (pure function, see module doc).

    1. Each window gets its best speaker, or Unknown below the threshold.
    2. A window whose two neighbours agree with each other takes their label
       (majority over 3 windows); the first/last window takes the label of
       its two inner neighbours when they agree (one-sided).
    3. Repeat until stable:
       a. an Unknown run no longer than ``window_s`` between two *different*
          speakers is the boundary mixture: it is split at its midpoint
          between them (an odd middle window goes to the better fit);
       b. otherwise the shortest run that spans ``<= min_turn_s`` or holds a
          single window is merged into the neighbour whose speaker fits its
          windows better (an Unknown neighbour counts as scoring exactly
          the threshold).
    Boundaries sit midway between the centres of the last window of one run
    and the first of the next; the first turn starts at 0, the last ends at
    ``duration``.
    """
    scores = table.scores
    ids = table.speaker_ids
    raw = window_labels(table)
    labels = list(raw)
    for i in range(1, len(labels) - 1):
        if labels[i - 1] == labels[i + 1] != labels[i]:
            labels[i] = labels[i - 1]
    if len(labels) >= 3:
        if labels[1] == labels[2] != labels[0]:
            labels[0] = labels[1]
        if labels[-2] == labels[-3] != labels[-1]:
            labels[-1] = labels[-2]

    def fit(label: int | None, windows: list[int]) -> float:
        if label is None:
            return table.threshold
        return float(scores[windows, label].mean())

    while True:
        runs = _runs(labels)
        if len(runs) < 2:
            break
        spans = _spans(runs, centres, duration)
        width = [b - a for a, b in spans]
        mixture = next(
            (
                i
                for i in range(1, len(runs) - 1)
                if runs[i][0] is None
                and runs[i - 1][0] is not None
                and runs[i + 1][0] is not None
                and runs[i - 1][0] != runs[i + 1][0]
                and width[i] <= config.window_s + _EPS
            ),
            None,
        )
        if mixture is not None:
            left, right = runs[mixture - 1][0], runs[mixture + 1][0]
            assert left is not None and right is not None
            windows = runs[mixture][1]
            half = len(windows) // 2
            for w in windows[:half]:
                labels[w] = left
            for w in windows[len(windows) - half :]:
                labels[w] = right
            if len(windows) % 2:
                w = windows[half]
                labels[w] = left if scores[w, left] >= scores[w, right] else right
            continue
        short = [
            i
            for i in range(len(runs))
            if width[i] <= config.min_turn_s + _EPS or len(runs[i][1]) < 2
        ]
        if not short:
            break
        i = min(short, key=lambda k: width[k])
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
            # Score the turn on its own windows, not on absorbed neighbours' ones.
            own = [w for w in windows if raw[w] == label] or windows
            score = fit(label, own)
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
