"""
Test: Synthesize first N entries of annotated Fourth Wing Chapter One.

Requires GPU. Run via: sbatch run_test.slurm
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, "src")

from audiobook_maker.annotate import load_script
from audiobook_maker.synthesize import synthesize_script, SynthesisConfig
from audiobook_maker.synthesize.engine import VoiceConfig
from audiobook_maker.assemble import assemble_chapters

# Config
N_ENTRIES = 10  # How many entries to synthesize (set higher for full chapter)
PROJECT_DIR = Path("test_project")
VOICE_REF = str(Path("src/audiobook_maker/epubs/../../../tts/voices/Ugo/Ugo1.wav"))  # Use Ugo's voice as narrator for now

# Load annotated script
script = load_script(PROJECT_DIR / "annotated_script.json")
script = script[:N_ENTRIES]  # Limit for testing

print(f"Synthesizing {len(script)} entries...")
for e in script[:5]:
    print(f"  [{e.speaker}] {e.text[:60]}...")

# Voice map: all speakers use same voice for now (can be different later)
voice_map = {
    "NARRATOR": VoiceConfig(speaker_id="NARRATOR", ref_audio=VOICE_REF),
    "MIRA": VoiceConfig(speaker_id="MIRA", ref_audio=VOICE_REF),
    "VIOLET": VoiceConfig(speaker_id="VIOLET", ref_audio=VOICE_REF),
}

# Synthesize
config = SynthesisConfig(
    engine_name="xtts_v2",
    max_chunk_chars=300,
    language="en",
    speed=1.0,
)

rendered = synthesize_script(
    script=script,
    voice_map=voice_map,
    output_dir=PROJECT_DIR / "audio",
    config=config,
)

print(f"\nRendered {len(rendered)} entries")

# Assemble into one chapter
if rendered:
    chapters = assemble_chapters(
        rendered,
        output_dir=PROJECT_DIR / "chapters",
        chapter_titles={0: "Chapter One"},
    )
    print(f"\nFinal chapter audio: {chapters[0].audio_path}")
    print(f"Duration: {chapters[0].duration:.1f}s ({chapters[0].duration/60:.1f} min)")
