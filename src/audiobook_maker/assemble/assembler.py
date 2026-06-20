"""
Audio assembly: stitch per-entry WAVs into chapters with pauses.

Takes rendered entries and assembles them into chapter-level audio files
with appropriate pauses between speakers and at chapter breaks.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from ..synthesize.pipeline import RenderedEntry


# Pause durations (seconds)
PAUSE_BETWEEN_SPEAKERS = 0.5
PAUSE_SAME_SPEAKER = 0.25
PAUSE_CHAPTER_BREAK = 1.5


@dataclass(frozen=True)
class ChapterAudio:
    """Assembled chapter audio."""
    chapter_index: int
    title: str | None
    audio_path: str
    duration: float
    num_entries: int


def assemble_chapters(
    rendered: list[RenderedEntry],
    output_dir: str | Path,
    sample_rate: int = 24000,
    chapter_titles: dict[int, str] | None = None,
    pause_between_speakers: float = PAUSE_BETWEEN_SPEAKERS,
    pause_same_speaker: float = PAUSE_SAME_SPEAKER,
    pause_chapter_break: float = PAUSE_CHAPTER_BREAK,
) -> list[ChapterAudio]:
    """
    Assemble rendered entries into chapter-level audio files.

    Inserts pauses between entries based on speaker changes.

    Args:
        rendered: List of RenderedEntry from synthesis step.
        output_dir: Directory for chapter audio files.
        sample_rate: Sample rate of input files.
        chapter_titles: Optional {chapter_index: title} map.
        pause_between_speakers: Seconds of silence between different speakers.
        pause_same_speaker: Seconds of silence between same speaker entries.
        pause_chapter_break: Seconds of silence at start of each chapter.

    Returns:
        List of ChapterAudio objects.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if chapter_titles is None:
        chapter_titles = {}

    # Group entries by chapter
    chapters: dict[int, list[RenderedEntry]] = {}
    for entry in rendered:
        chapters.setdefault(entry.chapter_index, []).append(entry)

    # Sort entries within each chapter by index
    for ch_entries in chapters.values():
        ch_entries.sort(key=lambda e: e.entry_index)

    results = []
    for ch_idx in sorted(chapters.keys()):
        entries = chapters[ch_idx]
        title = chapter_titles.get(ch_idx)

        audio_segments = []

        # Chapter break pause at start
        audio_segments.append(_silence(pause_chapter_break, sample_rate))

        prev_speaker = None
        for entry in entries:
            # Insert pause based on speaker change
            if prev_speaker is not None:
                if entry.speaker != prev_speaker:
                    audio_segments.append(_silence(pause_between_speakers, sample_rate))
                else:
                    audio_segments.append(_silence(pause_same_speaker, sample_rate))

            # Load entry audio
            audio, sr = sf.read(entry.audio_path)
            if sr != sample_rate:
                # Resample if needed (simple linear interpolation)
                audio = _resample(audio, sr, sample_rate)

            audio_segments.append(audio.astype(np.float32))
            prev_speaker = entry.speaker

        # Concatenate
        chapter_audio = np.concatenate(audio_segments)
        duration = len(chapter_audio) / sample_rate

        # Save
        safe_title = _safe_filename(title or f"chapter_{ch_idx:03d}")
        out_path = output_dir / f"{ch_idx:03d}_{safe_title}.wav"
        sf.write(str(out_path), chapter_audio, sample_rate)

        results.append(ChapterAudio(
            chapter_index=ch_idx,
            title=title,
            audio_path=str(out_path),
            duration=duration,
            num_entries=len(entries),
        ))

        print(f"  Chapter {ch_idx}: {title or '(untitled)'} — "
              f"{duration:.1f}s, {len(entries)} entries")

    total = sum(ch.duration for ch in results)
    print(f"\nAssembly complete: {len(results)} chapters, {total:.0f}s total ({total/60:.1f} min)")
    return results


def _silence(duration: float, sample_rate: int) -> np.ndarray:
    """Generate silence of given duration."""
    return np.zeros(int(duration * sample_rate), dtype=np.float32)


def _resample(audio: np.ndarray, src_sr: int, target_sr: int) -> np.ndarray:
    """Simple resampling via linear interpolation."""
    if src_sr == target_sr:
        return audio
    ratio = target_sr / src_sr
    new_length = int(len(audio) * ratio)
    indices = np.linspace(0, len(audio) - 1, new_length)
    return np.interp(indices, np.arange(len(audio)), audio)


def _safe_filename(name: str, max_len: int = 50) -> str:
    """Convert string to filesystem-safe filename."""
    import re
    safe = re.sub(r'[^\w\s-]', '', name)
    safe = re.sub(r'\s+', '_', safe).strip('_')
    return safe[:max_len]
