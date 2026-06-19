# audiobook-maker

Multi-voice audiobook generator from EPUBs with character identification, voice cloning, and style finetuning.

## Features

- **EPUB parsing** — Extract structured chapters with dialogue/narration detection
- **Speaker attribution** — LLM-powered character identification per line (who said what)
- **Voice cloning** — Clone character voices from reference audio (XTTS v2)
- **Voice extraction** — Diarize existing audiobooks to extract per-character voice clips
- **Multi-voice synthesis** — Different voices for narrator and each character
- **Style finetuning** — Tune delivery nuances per character (pace, emotion, tone)
- **Chaptered export** — M4B with chapter markers, MP3, or per-line WAV

## Pipeline

```
EPUB → Parse → Annotate (LLM) → Assign Voices → Synthesize → Export
                                       ↑
                          Extract from existing audiobook
                          (diarize → cluster → clip → clone)
```

## Quick Start

```bash
pixi install

# 1. Parse an EPUB
pixi run python -m audiobook_maker parse book.epub -o ./my_project

# 2. Annotate with speaker attribution
pixi run python -m audiobook_maker annotate ./my_project

# 3. Synthesize with voice map
pixi run python -m audiobook_maker synthesize ./my_project --voice-map voices.json

# 4. Export as M4B
pixi run python -m audiobook_maker export ./my_project --format m4b -o audiobook.m4b
```

## Voice Extraction from Existing Audiobooks

```bash
# Extract character voices from a professional audiobook
pixi run python -m audiobook_maker extract-voices audiobook.mp3 -o ./voices
```

This will:
1. Diarize the audio (pyannote-audio) to identify speaker segments
2. Cluster segments by speaker (resemblyzer embeddings)
3. Select clean 15-30s clips per speaker
4. Save as reference audio for voice cloning

## Project Structure

```
src/audiobook_maker/
├── parse/          # EPUB → structured chapters (ebooklib + BS4)
├── annotate/       # Character/narrator attribution per line (LLM)
├── voices/
│   ├── extract/    # Diarize existing audiobooks → per-character clips
│   ├── clone/      # Generate voice embeddings from clips (XTTS v2)
│   └── finetune/   # Style tuning per character
├── synthesize/     # Annotated script + voice map → audio
├── assemble/       # Chapters → M4B with metadata, pauses, transitions
└── __main__.py     # CLI orchestration
```

## Annotation Format

Each line in the annotated script:

```json
[
  {"speaker": "NARRATOR", "text": "The room fell silent.", "instruct": "Tense, measured narration"},
  {"speaker": "ELENA", "text": "I didn't expect to find you here.", "instruct": "Quiet authority, restrained anger"},
  {"speaker": "NARRATOR", "text": "He turned slowly.", "instruct": "Slow, deliberate pacing"}
]
```

## Tech Stack

| Component | Tool |
|-----------|------|
| EPUB parsing | ebooklib + BeautifulSoup4 |
| Sentence splitting | NLTK sent_tokenize |
| Speaker attribution | LLM (OpenAI-compatible API) |
| TTS engine | XTTS v2 (Coqui TTS) |
| Speaker diarization | pyannote-audio |
| Voice clustering | resemblyzer |
| Audio export | FFmpeg + mutagen |

## Requirements

- Python 3.10
- CUDA GPU (for TTS and diarization)
- FFmpeg (for M4B export)
- OpenAI-compatible LLM server (for annotation)
