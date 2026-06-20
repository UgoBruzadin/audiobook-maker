"""Extract character voices from existing audiobooks via diarization."""

from .diarize import diarize, DiarizationResult, SpeakerSegment
from .cluster import compute_speaker_profiles, match_speakers_across_files, ClusteringResult, SpeakerProfile
from .clip import extract_clips, ExtractedVoice

__all__ = [
    "diarize", "DiarizationResult", "SpeakerSegment",
    "compute_speaker_profiles", "match_speakers_across_files", "ClusteringResult", "SpeakerProfile",
    "extract_clips", "ExtractedVoice",
]
