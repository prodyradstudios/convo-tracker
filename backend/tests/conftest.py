"""Shared fixtures and synthetic-audio helpers. No ML packages required."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.audio.io import SAMPLE_RATE
from app.config import Settings
from app.main import create_app
from app.speakers.embedder import l2_normalize
from app.transcription.transcriber import Word

SR = SAMPLE_RATE


# -- synthetic audio --------------------------------------------------------
def silence(seconds: float, level: float = 1e-3, seed: int = 0) -> np.ndarray:
    """Low-level white noise (a realistic 'silent' room, about -60 dBFS)."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, level, int(seconds * SR)).astype(np.float32)


def tone(seconds: float, freq: float = 220.0, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def noise_burst(seconds: float, amp: float = 0.2, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-amp, amp, int(seconds * SR)).astype(np.float32)


def voice(
    f0: float,
    formants: list[tuple[float, float]],
    seconds: float = 3.0,
    seed: int = 0,
) -> np.ndarray:
    """Crude vowel-like 'voice': harmonics of a vibrato-modulated f0 shaped by formants."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    f = f0 * (1 + 0.03 * np.sin(2 * np.pi * (4 + rng.random()) * t + rng.random() * 6))
    phase = 2 * np.pi * np.cumsum(f) / SR
    x = np.zeros_like(t)
    for k in range(1, int(7000 / f0)):
        amp = sum(np.exp(-(((k * f0) - fc) / bw) ** 2) for fc, bw in formants) + 0.01
        x += amp * np.sin(k * phase + rng.random() * 6) / np.sqrt(k)
    x += rng.normal(0, 0.01, x.size)
    return (0.3 * x / np.abs(x).max()).astype(np.float32)


def chunks(audio: np.ndarray, size: int = 1600) -> Iterator[np.ndarray]:
    for i in range(0, audio.size, size):
        yield audio[i : i + size]


# -- fakes ------------------------------------------------------------------
def dominant_freq(audio: np.ndarray) -> float:
    spec = np.abs(np.fft.rfft(audio))
    return float(np.fft.rfftfreq(audio.size, 1 / SR)[int(np.argmax(spec))])


class FakeEmbedder:
    """Deterministic: one-hot vector at the 100 Hz bucket of the dominant frequency."""

    name = "fake"
    default_threshold = 0.5
    dim = 16

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, audio: np.ndarray) -> np.ndarray:
        self.calls += 1
        vec = np.zeros(self.dim, dtype=np.float32)
        vec[int(round(dominant_freq(audio) / 100.0)) % self.dim] = 1.0
        return l2_normalize(vec)


class FakeTranscriber:
    """Returns 'tone <freq>' (rounded to 100 Hz); empty for frequencies in ``silent``."""

    name = "fake"

    def __init__(self, silent: tuple[int, ...] = (), delays: dict[int, float] | None = None):
        self.silent = silent
        self.delays = delays or {}

    def transcribe(self, audio: np.ndarray) -> str:
        freq = int(round(dominant_freq(audio) / 100.0)) * 100
        time.sleep(self.delays.get(freq, 0.0))
        return "" if freq in self.silent else f"tone {freq}"

    def transcribe_words(self, audio: np.ndarray) -> list[Word]:
        """Words "tone", "<freq>" for every run of one tone (first/second half of the run)."""
        freq = int(round(dominant_freq(audio) / 100.0)) * 100
        time.sleep(self.delays.get(freq, 0.0))
        words: list[Word] = []
        for f, start, end in tone_runs(audio):
            if f not in self.silent:
                mid = (start + end) / 2
                words += [Word(" tone", start, mid), Word(f" {f}", mid, end)]
        return words


def tone_runs(audio: np.ndarray, block_s: float = 0.1) -> list[tuple[int, float, float]]:
    """(freq rounded to 100 Hz, start s, end s) of each run of one tone, ignoring quiet blocks."""
    block = int(block_s * SR)
    runs: list[list] = []
    for i in range(audio.size // block):
        chunk = audio[i * block : (i + 1) * block]
        if np.sqrt(np.mean(chunk.astype(np.float64) ** 2)) < 0.01:
            continue
        f = int(round(dominant_freq(chunk) / 100.0)) * 100
        t0, t1 = i * block_s, (i + 1) * block_s
        if runs and runs[-1][0] == f and abs(runs[-1][2] - t0) < 1e-9:
            runs[-1][2] = t1
        else:
            runs.append([f, t0, t1])
    return [(f, round(a, 3), round(b, 3)) for f, a, b in runs if b - a >= 0.2]


# -- fixtures ---------------------------------------------------------------
@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data", embedder="mfcc", transcriber="null")


@pytest.fixture
def make_client(settings: Settings) -> Iterator[Callable[..., TestClient]]:
    """Factory: ``make_client(embedder=..., transcriber=...)`` -> started TestClient."""
    clients: list[TestClient] = []

    def factory(**kwargs) -> TestClient:
        kwargs.setdefault("embedder", FakeEmbedder())
        kwargs.setdefault("transcriber", FakeTranscriber())
        kwargs.setdefault("frontend_dir", None)
        client = TestClient(create_app(kwargs.pop("settings", settings), **kwargs))
        client.__enter__()
        clients.append(client)
        return client

    yield factory
    for client in clients:
        client.__exit__(None, None, None)
