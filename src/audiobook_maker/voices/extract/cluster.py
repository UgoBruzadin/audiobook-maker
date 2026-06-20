"""
Speaker clustering: group diarized segments by voice similarity.

Uses resemblyzer to compute speaker embeddings, then clusters them.
Pyannote's diarization already assigns speaker labels, but for audiobooks
with many chapters processed separately, we may need to re-cluster
across files. This module also handles quality filtering.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .diarize import DiarizationResult, SpeakerSegment


@dataclass(frozen=True)
class SpeakerProfile:
    """A speaker's voice profile with embedding and best clips."""
    speaker_id: str
    embedding: np.ndarray  # 256-dim resemblyzer embedding
    total_duration: float
    num_segments: int
    best_clips: tuple[SpeakerSegment, ...]  # sorted by quality/duration


@dataclass(frozen=True)
class ClusteringResult:
    """Clustering output: speaker profiles with embeddings."""
    profiles: tuple[SpeakerProfile, ...]
    audio_path: str

    @property
    def speakers(self) -> list[str]:
        return [p.speaker_id for p in self.profiles]


def compute_speaker_profiles(
    diarization: DiarizationResult,
    min_segment_duration: float = 2.0,
    max_clip_duration: float = 30.0,
    num_best_clips: int = 5,
) -> ClusteringResult:
    """
    Compute voice embeddings for each speaker from diarized segments.

    Filters out short segments, computes embeddings, and selects
    the best clips (longest, cleanest) for voice cloning.

    Args:
        diarization: Output from diarize().
        min_segment_duration: Ignore segments shorter than this (seconds).
        max_clip_duration: Cap clip length for embedding computation.
        num_best_clips: How many reference clips to keep per speaker.

    Returns:
        ClusteringResult with profiles for each speaker.
    """
    from resemblyzer import VoiceEncoder, preprocess_wav

    encoder = VoiceEncoder()

    # Load audio once
    audio, sr = sf.read(diarization.audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # mono

    profiles = []
    for speaker in diarization.speakers:
        segments = diarization.segments_for_speaker(speaker)

        # Filter by minimum duration
        valid_segments = [s for s in segments if s.duration >= min_segment_duration]
        if not valid_segments:
            continue

        # Sort by duration (longest first — best for embedding)
        valid_segments.sort(key=lambda s: s.duration, reverse=True)

        # Compute embeddings from top segments
        embeddings = []
        for seg in valid_segments[:num_best_clips * 2]:  # compute more, keep best
            start_sample = int(seg.start * sr)
            end_sample = int(min(seg.end, seg.start + max_clip_duration) * sr)
            clip = audio[start_sample:end_sample]

            if len(clip) < sr:  # less than 1 second after slicing
                continue

            # Preprocess for resemblyzer (16kHz, normalized)
            processed = preprocess_wav(clip, source_sr=sr)
            if len(processed) < 8000:  # minimum for encoder
                continue

            emb = encoder.embed_utterance(processed)
            embeddings.append(emb)

        if not embeddings:
            continue

        # Average embedding for this speaker
        avg_embedding = np.mean(embeddings, axis=0)
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

        # Select best clips (longest valid segments)
        best = tuple(valid_segments[:num_best_clips])

        profiles.append(SpeakerProfile(
            speaker_id=speaker,
            embedding=avg_embedding,
            total_duration=diarization.speaker_duration(speaker),
            num_segments=len(segments),
            best_clips=best,
        ))

    # Sort profiles by total speaking time (main speaker first)
    profiles.sort(key=lambda p: p.total_duration, reverse=True)

    result = ClusteringResult(
        profiles=tuple(profiles),
        audio_path=diarization.audio_path,
    )

    print(f"  Computed {len(profiles)} speaker profiles")
    for p in profiles:
        print(f"    {p.speaker_id}: {p.total_duration:.1f}s, "
              f"embedding dim={p.embedding.shape[0]}, "
              f"{len(p.best_clips)} best clips")

    return result


def match_speakers_across_files(
    profiles_list: list[ClusteringResult],
    similarity_threshold: float = 0.75,
) -> dict[str, list[tuple[int, str]]]:
    """
    Match speakers across multiple audio files by embedding similarity.

    Useful for multi-file audiobooks where pyannote assigns independent
    labels per file (SPEAKER_00 in file1 might be SPEAKER_01 in file2).

    Args:
        profiles_list: ClusteringResults from multiple files.
        similarity_threshold: Cosine similarity threshold for matching.

    Returns:
        Dict mapping unified speaker ID → [(file_idx, original_speaker_id), ...]
    """
    if len(profiles_list) <= 1:
        if profiles_list:
            return {p.speaker_id: [(0, p.speaker_id)] for p in profiles_list[0].profiles}
        return {}

    # Use first file as reference
    reference = profiles_list[0]
    unified: dict[str, list[tuple[int, str]]] = {
        p.speaker_id: [(0, p.speaker_id)] for p in reference.profiles
    }

    # Match subsequent files against reference embeddings
    ref_embeddings = np.array([p.embedding for p in reference.profiles])
    ref_ids = [p.speaker_id for p in reference.profiles]

    next_id = 0
    for file_idx, clustering in enumerate(profiles_list[1:], start=1):
        for profile in clustering.profiles:
            # Cosine similarity against all reference speakers
            similarities = ref_embeddings @ profile.embedding
            best_idx = int(np.argmax(similarities))
            best_sim = similarities[best_idx]

            if best_sim >= similarity_threshold:
                matched_id = ref_ids[best_idx]
                unified[matched_id].append((file_idx, profile.speaker_id))
            else:
                # New speaker not in reference
                new_id = f"SPEAKER_{next_id:02d}"
                while new_id in unified:
                    next_id += 1
                    new_id = f"SPEAKER_{next_id:02d}"
                unified[new_id] = [(file_idx, profile.speaker_id)]
                # Add to reference for future matching
                ref_embeddings = np.vstack([ref_embeddings, profile.embedding[None, :]])
                ref_ids.append(new_id)
                next_id += 1

    return unified
