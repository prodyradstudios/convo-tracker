"""Application state built once at startup and shared by all requests."""

from __future__ import annotations

from dataclasses import dataclass

from .audio.vad import VadConfig
from .config import Settings
from .speakers.embedder import Embedder, build_embedder
from .speakers.registry import SpeakerRegistry
from .speakers.turns import TurnConfig
from .transcription.transcriber import Transcriber, build_transcriber


@dataclass
class AppState:
    settings: Settings
    embedder: Embedder
    transcriber: Transcriber
    registry: SpeakerRegistry

    def vad_config(self) -> VadConfig:
        """Segmenter configuration derived from settings."""
        s = self.settings
        return VadConfig(
            silence_ms=s.silence_ms,
            max_utterance_s=s.max_utterance_s,
            min_utterance_s=s.min_utterance_s,
        )

    def turn_config(self) -> TurnConfig:
        """Speaker-turn splitting configuration derived from settings."""
        s = self.settings
        return TurnConfig(
            min_segment_s=s.turn_min_segment_s,
            window_s=s.turn_window_s,
            hop_s=s.turn_hop_s,
            min_turn_s=s.turn_min_s,
        )


def build_state(
    settings: Settings,
    embedder: Embedder | None = None,
    transcriber: Transcriber | None = None,
) -> AppState:
    """Load models (unless injected) and the speaker registry."""
    embedder = embedder or build_embedder(settings.embedder, settings.data_dir, settings.device)
    transcriber = transcriber or build_transcriber(
        settings.transcriber, settings.whisper_model, settings.device
    )
    threshold = (
        settings.match_threshold
        if settings.match_threshold is not None
        else embedder.default_threshold
    )
    registry = SpeakerRegistry(
        embedder.name, threshold, settings.data_dir / "speakers.json", adapt=settings.adapt
    )
    return AppState(settings, embedder, transcriber, registry)
