"""Tests for voice extraction: diarization result handling, clustering, clip selection."""

import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from audiobook_maker.voices.extract.diarize import SpeakerSegment, DiarizationResult
from audiobook_maker.voices.extract.cluster import (
    compute_speaker_profiles,
    match_speakers_across_files,
    ClusteringResult,
    SpeakerProfile,
)
from audiobook_maker.voices.extract.clip import (
    extract_clips,
    _extract_segment,
    _passes_quality_check,
    _safe_name,
)


class TestSpeakerSegment:
    def test_duration(self):
        seg = SpeakerSegment(speaker="A", start=1.0, end=3.5)
        assert seg.duration == 2.5

    def test_immutable(self):
        seg = SpeakerSegment(speaker="A", start=0.0, end=1.0)
        try:
            seg.speaker = "B"
            assert False, "Should be immutable"
        except AttributeError:
            pass


class TestDiarizationResult:
    def _make_result(self):
        segments = (
            SpeakerSegment("SPEAKER_00", 0.0, 10.0),
            SpeakerSegment("SPEAKER_01", 10.0, 15.0),
            SpeakerSegment("SPEAKER_00", 15.0, 25.0),
            SpeakerSegment("SPEAKER_01", 25.0, 30.0),
            SpeakerSegment("SPEAKER_02", 30.0, 32.0),
        )
        return DiarizationResult(segments=segments, audio_path="/fake/audio.wav", sample_rate=16000)

    def test_speakers(self):
        result = self._make_result()
        assert result.speakers == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"]

    def test_total_duration(self):
        result = self._make_result()
        assert result.total_duration == 32.0

    def test_segments_for_speaker(self):
        result = self._make_result()
        segs = result.segments_for_speaker("SPEAKER_00")
        assert len(segs) == 2
        assert all(s.speaker == "SPEAKER_00" for s in segs)

    def test_speaker_duration(self):
        result = self._make_result()
        assert result.speaker_duration("SPEAKER_00") == 20.0
        assert result.speaker_duration("SPEAKER_01") == 10.0
        assert result.speaker_duration("SPEAKER_02") == 2.0

    def test_empty_result(self):
        result = DiarizationResult(segments=(), audio_path="/fake.wav", sample_rate=16000)
        assert result.speakers == []
        assert result.total_duration == 0.0


class TestQualityCheck:
    def test_passes_normal_audio(self):
        # Sine wave at moderate volume — guaranteed no clipping
        t = np.linspace(0, 1, 16000)
        clip = np.sin(2 * np.pi * 440 * t) * 0.5
        assert _passes_quality_check(clip) is True

    def test_rejects_clipping(self):
        clip = np.ones(16000)  # all at max
        assert _passes_quality_check(clip) is False

    def test_rejects_silence(self):
        clip = np.zeros(16000) + 0.001  # nearly silent
        assert _passes_quality_check(clip) is False

    def test_threshold_edge_cases(self):
        # Just below clipping threshold
        clip = np.ones(16000) * 0.98
        assert _passes_quality_check(clip) is True

        # Just above silence threshold
        clip = np.ones(16000) * 0.02
        assert _passes_quality_check(clip) is True


class TestExtractSegment:
    def test_extracts_correct_range(self):
        audio = np.arange(48000, dtype=np.float64)  # 3 seconds at 16kHz
        seg = SpeakerSegment("A", start=1.0, end=2.0)
        clip = _extract_segment(audio, 16000, seg)
        assert len(clip) == 16000
        # Values should be around 16000-32000 (the second second)
        assert clip[100] != 0  # not at the start of audio

    def test_applies_fade(self):
        audio = np.ones(48000)
        seg = SpeakerSegment("A", start=0.0, end=3.0)
        clip = _extract_segment(audio, 16000, seg)
        # Fade-in: first samples should be less than 1.0
        assert clip[0] < 1.0
        # Fade-out: last samples should be less than 1.0
        assert clip[-1] < 1.0
        # Middle should be untouched
        assert clip[len(clip) // 2] == 1.0

    def test_does_not_mutate_source(self):
        audio = np.ones(48000)
        seg = SpeakerSegment("A", start=0.0, end=1.0)
        _extract_segment(audio, 16000, seg)
        # Source audio should still be all ones
        assert np.all(audio == 1.0)


class TestSafeName:
    def test_replaces_spaces(self):
        assert _safe_name("Speaker 01") == "Speaker_01"

    def test_replaces_slashes(self):
        assert _safe_name("path/to/voice") == "path_to_voice"

    def test_simple_name(self):
        assert _safe_name("SPEAKER_00") == "SPEAKER_00"


class TestMatchSpeakersAcrossFiles:
    def test_single_file(self):
        profile = SpeakerProfile(
            speaker_id="SPEAKER_00",
            embedding=np.array([1.0, 0.0, 0.0]),
            total_duration=100.0,
            num_segments=10,
            best_clips=(),
        )
        clustering = ClusteringResult(profiles=(profile,), audio_path="/fake.wav")
        result = match_speakers_across_files([clustering])
        assert "SPEAKER_00" in result
        assert result["SPEAKER_00"] == [(0, "SPEAKER_00")]

    def test_matches_similar_speakers(self):
        # Two files, same speaker (similar embeddings)
        emb = np.array([1.0, 0.0, 0.0, 0.0])
        emb_similar = np.array([0.95, 0.1, 0.0, 0.0])
        emb_similar = emb_similar / np.linalg.norm(emb_similar)

        p1 = SpeakerProfile("SPEAKER_00", emb, 50.0, 5, ())
        p2 = SpeakerProfile("SPEAKER_00", emb_similar, 40.0, 4, ())

        c1 = ClusteringResult(profiles=(p1,), audio_path="/f1.wav")
        c2 = ClusteringResult(profiles=(p2,), audio_path="/f2.wav")

        result = match_speakers_across_files([c1, c2], similarity_threshold=0.75)
        # Should match — cosine similarity of emb and emb_similar is high
        assert len(result) == 1
        assert len(list(result.values())[0]) == 2

    def test_separates_different_speakers(self):
        # Two files, different speakers (orthogonal embeddings)
        emb1 = np.array([1.0, 0.0, 0.0, 0.0])
        emb2 = np.array([0.0, 1.0, 0.0, 0.0])

        p1 = SpeakerProfile("SPEAKER_00", emb1, 50.0, 5, ())
        p2 = SpeakerProfile("SPEAKER_00", emb2, 40.0, 4, ())

        c1 = ClusteringResult(profiles=(p1,), audio_path="/f1.wav")
        c2 = ClusteringResult(profiles=(p2,), audio_path="/f2.wav")

        result = match_speakers_across_files([c1, c2], similarity_threshold=0.75)
        # Should NOT match — orthogonal embeddings
        assert len(result) == 2

    def test_empty_input(self):
        assert match_speakers_across_files([]) == {}


class TestExtractClips:
    def test_extracts_and_saves_clips(self, tmp_path):
        # Create a fake audio file (sine wave — passes quality check)
        sr = 16000
        t = np.linspace(0, 60, sr * 60)
        audio = np.sin(2 * np.pi * 200 * t) * 0.5  # 60 seconds, clean tone
        audio_path = tmp_path / "test_audio.wav"

        import soundfile as sf
        sf.write(str(audio_path), audio, sr)

        # Create a clustering result with one speaker
        seg1 = SpeakerSegment("NARRATOR", 5.0, 30.0)  # 25s
        seg2 = SpeakerSegment("NARRATOR", 35.0, 55.0)  # 20s
        profile = SpeakerProfile(
            speaker_id="NARRATOR",
            embedding=np.random.randn(256),
            total_duration=45.0,
            num_segments=2,
            best_clips=(seg1, seg2),
        )
        clustering = ClusteringResult(
            profiles=(profile,),
            audio_path=str(audio_path),
        )

        output_dir = tmp_path / "output"
        voices = extract_clips(clustering, output_dir, target_duration=20.0, min_clip_duration=5.0)

        assert len(voices) > 0
        for v in voices:
            assert Path(v.clip_path).exists()
            assert v.speaker_id == "NARRATOR"
            assert v.duration >= 5.0

        # Embedding should be saved
        emb_path = output_dir / "NARRATOR" / "embedding.npy"
        assert emb_path.exists()
        loaded = np.load(emb_path)
        assert loaded.shape == (256,)
