"""Tests for assembly module: chapter stitching, pauses, export."""

import numpy as np
import soundfile as sf
from pathlib import Path
from unittest.mock import patch

from audiobook_maker.synthesize.pipeline import RenderedEntry
from audiobook_maker.assemble.assembler import (
    assemble_chapters,
    _silence,
    _resample,
    _safe_filename,
    ChapterAudio,
    PAUSE_BETWEEN_SPEAKERS,
    PAUSE_SAME_SPEAKER,
    PAUSE_CHAPTER_BREAK,
)
from audiobook_maker.assemble.export import export_m4b, export_mp3


def _create_test_wav(path: Path, duration: float = 1.0, sr: int = 24000):
    """Create a test WAV file with a sine tone."""
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * 440 * t) * 0.3
    sf.write(str(path), audio.astype(np.float32), sr)
    return path


class TestSilence:
    def test_correct_length(self):
        s = _silence(0.5, 24000)
        assert len(s) == 12000
        assert np.all(s == 0)

    def test_zero_duration(self):
        s = _silence(0.0, 24000)
        assert len(s) == 0


class TestResample:
    def test_upsample(self):
        audio = np.array([0.0, 1.0, 0.0, -1.0], dtype=np.float32)
        resampled = _resample(audio, 16000, 32000)
        # Should be roughly double length
        assert len(resampled) == 8

    def test_downsample(self):
        audio = np.ones(48000, dtype=np.float32)
        resampled = _resample(audio, 48000, 24000)
        assert len(resampled) == 24000

    def test_same_rate_noop(self):
        audio = np.array([1.0, 2.0, 3.0])
        result = _resample(audio, 24000, 24000)
        np.testing.assert_array_equal(result, audio)


class TestSafeFilename:
    def test_removes_special_chars(self):
        assert _safe_filename("Chapter 1: The Beginning!") == "Chapter_1_The_Beginning"

    def test_truncates(self):
        long = "A" * 100
        assert len(_safe_filename(long, max_len=50)) == 50

    def test_handles_spaces(self):
        assert _safe_filename("hello world") == "hello_world"


class TestAssembleChapters:
    def test_assembles_single_chapter(self, tmp_path):
        # Create test WAV files
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        wav1 = _create_test_wav(audio_dir / "entry_0.wav", duration=1.0)
        wav2 = _create_test_wav(audio_dir / "entry_1.wav", duration=0.5)

        rendered = [
            RenderedEntry(0, "NARRATOR", str(wav1), 1.0, chapter_index=0),
            RenderedEntry(1, "ELENA", str(wav2), 0.5, chapter_index=0),
        ]

        output_dir = tmp_path / "chapters"
        chapters = assemble_chapters(rendered, output_dir)

        assert len(chapters) == 1
        assert chapters[0].chapter_index == 0
        assert Path(chapters[0].audio_path).exists()
        # Duration should be: chapter_break + entry1 + speaker_change_pause + entry2
        expected_min = 1.0 + 0.5 + PAUSE_CHAPTER_BREAK + PAUSE_BETWEEN_SPEAKERS
        assert chapters[0].duration >= expected_min - 0.1

    def test_separates_multiple_chapters(self, tmp_path):
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        wav1 = _create_test_wav(audio_dir / "ch0.wav", duration=1.0)
        wav2 = _create_test_wav(audio_dir / "ch1.wav", duration=1.0)

        rendered = [
            RenderedEntry(0, "NARRATOR", str(wav1), 1.0, chapter_index=0),
            RenderedEntry(1, "NARRATOR", str(wav2), 1.0, chapter_index=1),
        ]

        chapters = assemble_chapters(rendered, tmp_path / "out")
        assert len(chapters) == 2
        assert chapters[0].chapter_index == 0
        assert chapters[1].chapter_index == 1

    def test_uses_chapter_titles(self, tmp_path):
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        wav = _create_test_wav(audio_dir / "entry.wav")

        rendered = [RenderedEntry(0, "NARRATOR", str(wav), 1.0, chapter_index=0)]
        titles = {0: "The Beginning"}

        chapters = assemble_chapters(rendered, tmp_path / "out", chapter_titles=titles)
        assert chapters[0].title == "The Beginning"
        assert "The_Beginning" in chapters[0].audio_path

    def test_same_speaker_gets_shorter_pause(self, tmp_path):
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        wav1 = _create_test_wav(audio_dir / "a.wav", duration=0.5)
        wav2 = _create_test_wav(audio_dir / "b.wav", duration=0.5)

        # Same speaker
        rendered_same = [
            RenderedEntry(0, "NARRATOR", str(wav1), 0.5, chapter_index=0),
            RenderedEntry(1, "NARRATOR", str(wav2), 0.5, chapter_index=0),
        ]
        ch_same = assemble_chapters(rendered_same, tmp_path / "same")

        # Different speaker
        rendered_diff = [
            RenderedEntry(0, "NARRATOR", str(wav1), 0.5, chapter_index=0),
            RenderedEntry(1, "ELENA", str(wav2), 0.5, chapter_index=0),
        ]
        ch_diff = assemble_chapters(rendered_diff, tmp_path / "diff")

        # Different speakers should produce longer audio (more pause)
        assert ch_diff[0].duration > ch_same[0].duration


class TestExport:
    def test_export_m4b(self, tmp_path):
        # Create chapter WAVs
        ch_dir = tmp_path / "chapters"
        ch_dir.mkdir()
        _create_test_wav(ch_dir / "ch0.wav", duration=2.0)
        _create_test_wav(ch_dir / "ch1.wav", duration=1.5)

        chapters = [
            ChapterAudio(0, "Chapter One", str(ch_dir / "ch0.wav"), 2.0, 5),
            ChapterAudio(1, "Chapter Two", str(ch_dir / "ch1.wav"), 1.5, 3),
        ]

        output = tmp_path / "book.m4b"
        result = export_m4b(chapters, output, title="Test Book", author="Author")

        assert Path(result.output_path).exists()
        assert result.format == "m4b"
        assert result.num_chapters == 2
        assert result.duration == 3.5

    def test_export_mp3(self, tmp_path):
        ch_dir = tmp_path / "chapters"
        ch_dir.mkdir()
        _create_test_wav(ch_dir / "ch0.wav", duration=1.0)

        chapters = [
            ChapterAudio(0, "Only Chapter", str(ch_dir / "ch0.wav"), 1.0, 2),
        ]

        output = tmp_path / "book.mp3"
        result = export_mp3(chapters, output, title="Test", author="Me")

        assert Path(result.output_path).exists()
        assert result.format == "mp3"
