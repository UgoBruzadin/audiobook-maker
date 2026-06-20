"""
TTS engine interface and registry.

Pluggable backend design: each engine implements the TTSEngine protocol.
The pipeline doesn't care which engine is active — it just calls synthesize().
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class VoiceConfig:
    """Voice configuration for a speaker."""
    speaker_id: str
    ref_audio: str | None = None       # Path to reference audio for cloning
    ref_text: str | None = None        # Transcript of reference audio
    embedding_path: str | None = None  # Pre-computed embedding (.pt or .npy)
    description: str | None = None     # Text description (for voice design)
    style: str | None = None           # Style/instruct modifier


class TTSEngine(ABC):
    """Abstract TTS engine interface."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Engine identifier."""
        ...

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Output sample rate."""
        ...

    @abstractmethod
    def load(self, **kwargs) -> None:
        """Load model into memory."""
        ...

    @abstractmethod
    def synthesize(
        self,
        text: str,
        voice: VoiceConfig,
        language: str = "en",
        speed: float = 1.0,
    ) -> np.ndarray:
        """
        Synthesize speech from text with a given voice.

        Args:
            text: Text to synthesize (should be pre-chunked, <400 chars).
            voice: Voice configuration (reference audio, embedding, etc.)
            language: Language code.
            speed: Playback speed multiplier.

        Returns:
            Audio waveform as numpy array (mono, at self.sample_rate).
        """
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready."""
        ...


# Engine registry
_ENGINES: dict[str, type[TTSEngine]] = {}


def register_engine(name: str):
    """Decorator to register a TTS engine class."""
    def decorator(cls: type[TTSEngine]):
        _ENGINES[name] = cls
        return cls
    return decorator


def get_engine(name: str, **kwargs) -> TTSEngine:
    """Get and load a TTS engine by name."""
    if name not in _ENGINES:
        available = ", ".join(_ENGINES.keys()) or "(none registered)"
        raise ValueError(f"Unknown TTS engine '{name}'. Available: {available}")
    engine = _ENGINES[name]()
    engine.load(**kwargs)
    return engine


def list_engines() -> list[str]:
    """List available engine names."""
    return list(_ENGINES.keys())
