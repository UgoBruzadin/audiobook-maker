"""TTS synthesis: annotated script + voice map → audio."""

from .engine import TTSEngine, VoiceConfig, get_engine, list_engines, register_engine
from .pipeline import synthesize_script, SynthesisConfig, RenderedEntry, load_voice_map, save_voice_map

# Register backends on import
from . import backends  # noqa: F401

__all__ = [
    "TTSEngine", "VoiceConfig", "get_engine", "list_engines", "register_engine",
    "synthesize_script", "SynthesisConfig", "RenderedEntry",
    "load_voice_map", "save_voice_map",
]
