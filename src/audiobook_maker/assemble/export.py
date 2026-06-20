"""
Export: convert chapter WAVs to M4B (chaptered audiobook) or MP3.

Uses FFmpeg for encoding and mutagen for metadata.
"""

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .assembler import ChapterAudio


@dataclass(frozen=True)
class ExportResult:
    """Result of an export operation."""
    output_path: str
    format: str
    duration: float
    num_chapters: int


def export_m4b(
    chapters: list[ChapterAudio],
    output_path: str | Path,
    title: str = "Audiobook",
    author: str = "Unknown",
    sample_rate: int = 24000,
) -> ExportResult:
    """
    Export chapters as a single M4B file with chapter markers.

    Requires FFmpeg with AAC encoder.

    Args:
        chapters: List of ChapterAudio from assembly step.
        output_path: Output .m4b file path.
        title: Book title for metadata.
        author: Author for metadata.
        sample_rate: Audio sample rate.

    Returns:
        ExportResult with output path and duration.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create file list and chapter metadata for FFmpeg
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        # Write concat file list
        filelist = tmp / "filelist.txt"
        with open(filelist, "w") as f:
            for ch in chapters:
                # FFmpeg concat demuxer format
                escaped = ch.audio_path.replace("'", "'\\''")
                f.write(f"file '{escaped}'\n")

        # Write chapter metadata file (FFmpeg metadata format)
        metadata_file = tmp / "metadata.txt"
        with open(metadata_file, "w") as f:
            f.write(";FFMETADATA1\n")
            f.write(f"title={title}\n")
            f.write(f"artist={author}\n")
            f.write(f"album={title}\n")
            f.write(f"genre=Audiobook\n\n")

            # Chapter markers
            offset_ms = 0
            for ch in chapters:
                end_ms = offset_ms + int(ch.duration * 1000)
                ch_title = ch.title or f"Chapter {ch.chapter_index + 1}"
                f.write("[CHAPTER]\n")
                f.write("TIMEBASE=1/1000\n")
                f.write(f"START={offset_ms}\n")
                f.write(f"END={end_ms}\n")
                f.write(f"title={ch_title}\n\n")
                offset_ms = end_ms

        # Concatenate to single WAV first, then encode
        concat_wav = tmp / "concat.wav"
        _run_ffmpeg([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(filelist),
            "-c", "copy",
            str(concat_wav),
        ])

        # Encode to M4B (AAC in M4A container with chapters)
        _run_ffmpeg([
            "ffmpeg", "-y",
            "-i", str(concat_wav),
            "-i", str(metadata_file),
            "-map_metadata", "1",
            "-c:a", "aac",
            "-b:a", "128k",
            "-ar", str(sample_rate),
            "-movflags", "+faststart",
            str(output_path),
        ])

    total_duration = sum(ch.duration for ch in chapters)
    print(f"Exported: {output_path} ({total_duration / 60:.1f} min, {len(chapters)} chapters)")

    return ExportResult(
        output_path=str(output_path),
        format="m4b",
        duration=total_duration,
        num_chapters=len(chapters),
    )


def export_mp3(
    chapters: list[ChapterAudio],
    output_path: str | Path,
    title: str = "Audiobook",
    author: str = "Unknown",
    bitrate: str = "192k",
) -> ExportResult:
    """
    Export chapters as a single MP3 file.

    Args:
        chapters: List of ChapterAudio from assembly step.
        output_path: Output .mp3 file path.
        title: Book title for metadata.
        author: Author for metadata.
        bitrate: MP3 bitrate.

    Returns:
        ExportResult with output path and duration.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        # Write file list
        filelist = tmp / "filelist.txt"
        with open(filelist, "w") as f:
            for ch in chapters:
                escaped = ch.audio_path.replace("'", "'\\''")
                f.write(f"file '{escaped}'\n")

        # Concat and encode to MP3
        _run_ffmpeg([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(filelist),
            "-c:a", "libmp3lame",
            "-b:a", bitrate,
            "-metadata", f"title={title}",
            "-metadata", f"artist={author}",
            "-metadata", "genre=Audiobook",
            str(output_path),
        ])

    total_duration = sum(ch.duration for ch in chapters)
    print(f"Exported: {output_path} ({total_duration / 60:.1f} min)")

    return ExportResult(
        output_path=str(output_path),
        format="mp3",
        duration=total_duration,
        num_chapters=len(chapters),
    )


def _run_ffmpeg(cmd: list[str]):
    """Run an FFmpeg command, raising on failure."""
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg failed (exit {result.returncode}):\n"
            f"Command: {' '.join(cmd)}\n"
            f"Stderr: {result.stderr[:500]}"
        )
