"""
Synthesis pipeline: annotated script + voice map → chapter audio.

Takes the annotated script (list of {speaker, text, instruct}) and a voice
configuration, then renders audio per entry with appropriate pauses.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from ..annotate.annotator import ScriptEntry
from ..parse.text_processing import chunk_text
from .engine import TTSEngine, VoiceConfig, get_engine


# Pause durations (seconds)
PAUSE_BETWEEN_SPEAKERS = 0.5
PAUSE_SAME_SPEAKER = 0.25
PAUSE_CHAPTER_BREAK = 1.5


@dataclass
class SynthesisConfig:
    engine_name: str = "xtts_v2"
    engine_kwargs: dict = field(default_factory=dict)
    max_chunk_chars: int = 350
    language: str = "en"
    speed: float = 1.0
    output_sample_rate: int | None = None  # None = use engine's native rate


@dataclass(frozen=True)
class RenderedEntry:
    """A single rendered script entry."""
    entry_index: int
    speaker: str
    audio_path: str
    duration: float
    chapter_index: int


def synthesize_script(
    script: list[ScriptEntry],
    voice_map: dict[str, VoiceConfig],
    output_dir: str | Path,
    config: SynthesisConfig | None = None,
) -> list[RenderedEntry]:
    """
    Render an annotated script to audio files.

    Each script entry becomes one WAV file. Pauses are handled during
    assembly (not baked into individual clips).

    Args:
        script: Annotated script entries from the annotation step.
        voice_map: Speaker ID → VoiceConfig mapping.
        output_dir: Directory for output WAV files.
        config: Synthesis configuration.

    Returns:
        List of RenderedEntry with paths to rendered audio.
    """
    if config is None:
        config = SynthesisConfig()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load engine
    engine = get_engine(config.engine_name, **config.engine_kwargs)
    sr = config.output_sample_rate or engine.sample_rate

    rendered = []
    prev_speaker = None

    for i, entry in enumerate(script):
        # Get voice config for this speaker (fall back to NARRATOR config)
        voice = voice_map.get(entry.speaker) or voice_map.get("NARRATOR")
        if voice is None:
            print(f"  WARNING: No voice config for '{entry.speaker}', skipping")
            continue

        # Apply instruct as style override
        if entry.instruct:
            voice = VoiceConfig(
                speaker_id=voice.speaker_id,
                ref_audio=voice.ref_audio,
                ref_text=voice.ref_text,
                embedding_path=voice.embedding_path,
                description=voice.description,
                style=entry.instruct,
            )

        # Chunk text if too long for the engine
        chunks = chunk_text(entry.text, max_chars=config.max_chunk_chars)

        # Synthesize each chunk and concatenate
        audio_parts = []
        for chunk in chunks:
            audio = engine.synthesize(
                text=chunk,
                voice=voice,
                language=config.language,
                speed=config.speed,
            )
            audio_parts.append(audio)
            # Small gap between chunks (same entry)
            audio_parts.append(np.zeros(int(sr * 0.1), dtype=np.float32))

        if not audio_parts:
            continue

        full_audio = np.concatenate(audio_parts)

        # Save
        clip_path = output_dir / f"{i:05d}_{entry.speaker}.wav"
        sf.write(str(clip_path), full_audio, sr)

        rendered.append(RenderedEntry(
            entry_index=i,
            speaker=entry.speaker,
            audio_path=str(clip_path),
            duration=len(full_audio) / sr,
            chapter_index=entry.chapter_index,
        ))

        prev_speaker = entry.speaker

        if (i + 1) % 50 == 0:
            print(f"  Rendered {i + 1}/{len(script)} entries...")

    print(f"Synthesis complete: {len(rendered)} audio clips in {output_dir}")
    return rendered


def load_voice_map(path: str | Path) -> dict[str, VoiceConfig]:
    """
    Load voice map from JSON.

    Expected format:
    {
        "NARRATOR": {"ref_audio": "voices/narrator.wav", ...},
        "ELENA": {"ref_audio": "voices/elena.wav", "description": "..."},
        ...
    }
    """
    with open(path) as f:
        data = json.load(f)

    voice_map = {}
    for speaker_id, config in data.items():
        voice_map[speaker_id.upper()] = VoiceConfig(
            speaker_id=speaker_id.upper(),
            ref_audio=config.get("ref_audio"),
            ref_text=config.get("ref_text"),
            embedding_path=config.get("embedding_path"),
            description=config.get("description"),
            style=config.get("style"),
        )

    return voice_map


def save_voice_map(voice_map: dict[str, VoiceConfig], path: str | Path):
    """Save voice map to JSON."""
    data = {}
    for speaker_id, vc in voice_map.items():
        data[speaker_id] = {
            "ref_audio": vc.ref_audio,
            "ref_text": vc.ref_text,
            "embedding_path": vc.embedding_path,
            "description": vc.description,
            "style": vc.style,
        }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
