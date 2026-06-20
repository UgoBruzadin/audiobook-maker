"""
Test: Synthesize first N entries of annotated Fourth Wing Chapter One.

Requires GPU. Run via: sbatch run_test.slurm

Voice assignments:
- NARRATOR (Violet, first person): Daisy Studious — clear, young female, natural reading
- MIRA (older sister, intense): Brenda Stern — authoritative female
- VIOLET (when speaking dialogue): Sofia Hellen — young, energetic female
"""

import sys
from pathlib import Path

sys.path.insert(0, "src")

from audiobook_maker.annotate import load_script
from audiobook_maker.synthesize import synthesize_script, SynthesisConfig
from audiobook_maker.synthesize.engine import VoiceConfig
from audiobook_maker.assemble import assemble_chapters

# Config
N_ENTRIES = 15  # How many entries to synthesize
PROJECT_DIR = Path("test_project")

# Load annotated script
script = load_script(PROJECT_DIR / "annotated_script.json")
script = script[:N_ENTRIES]

print(f"Synthesizing {len(script)} entries...")
for e in script[:5]:
    print(f"  [{e.speaker}] {e.text[:60]}...")

# Voice map using built-in XTTS v2 speakers
# ref_audio is used as speaker name when it's not a file path
voice_map = {
    "NARRATOR": VoiceConfig(speaker_id="NARRATOR", ref_audio="Daisy Studious"),
    "MIRA": VoiceConfig(speaker_id="MIRA", ref_audio="Brenda Stern"),
    "VIOLET": VoiceConfig(speaker_id="VIOLET", ref_audio="Sofia Hellen"),
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
    print("\nDone! Listen to the output at:")
    print(f"  {chapters[0].audio_path}")
