"""
Clip extraction: save clean reference audio per speaker.

Selects the best segments for voice cloning and exports them as WAV files.
Quality criteria: long enough, no clipping, reasonable energy.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .diarize import SpeakerSegment
from .cluster import ClusteringResult, SpeakerProfile


@dataclass(frozen=True)
class ExtractedVoice:
    """A saved voice clip ready for cloning."""
    speaker_id: str
    clip_path: str
    duration: float
    embedding_path: str | None = None


def extract_clips(
    clustering: ClusteringResult,
    output_dir: str | Path,
    target_duration: float = 20.0,
    min_clip_duration: float = 5.0,
    max_clips_per_speaker: int = 3,
) -> list[ExtractedVoice]:
    """
    Extract and save clean reference clips for each speaker.

    For each speaker, saves up to max_clips_per_speaker clips that are
    closest to target_duration. Also saves the speaker embedding.

    Args:
        clustering: Output from compute_speaker_profiles().
        output_dir: Directory to save clips.
        target_duration: Ideal clip length for voice cloning (seconds).
        min_clip_duration: Minimum acceptable clip length.
        max_clips_per_speaker: How many clips to save per speaker.

    Returns:
        List of ExtractedVoice objects with paths to saved files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load source audio once
    audio, sr = sf.read(clustering.audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    extracted = []

    for profile in clustering.profiles:
        speaker_dir = output_dir / _safe_name(profile.speaker_id)
        speaker_dir.mkdir(parents=True, exist_ok=True)

        # Select clips closest to target duration
        candidates = [c for c in profile.best_clips if c.duration >= min_clip_duration]
        candidates.sort(key=lambda c: abs(c.duration - target_duration))
        selected = candidates[:max_clips_per_speaker]

        if not selected:
            continue

        # Save embedding
        emb_path = speaker_dir / "embedding.npy"
        np.save(emb_path, profile.embedding)

        # Save audio clips
        for i, seg in enumerate(selected):
            clip = _extract_segment(audio, sr, seg)

            # Quality check: skip if clipping or too quiet
            if not _passes_quality_check(clip):
                continue

            clip_path = speaker_dir / f"clip_{i:02d}.wav"
            sf.write(str(clip_path), clip, sr)

            extracted.append(ExtractedVoice(
                speaker_id=profile.speaker_id,
                clip_path=str(clip_path),
                duration=seg.duration,
                embedding_path=str(emb_path),
            ))

        print(f"  {profile.speaker_id}: saved {len([e for e in extracted if e.speaker_id == profile.speaker_id])} clips → {speaker_dir}")

    print(f"\nExtracted {len(extracted)} voice clips for {len(clustering.profiles)} speakers")
    return extracted


def _extract_segment(audio: np.ndarray, sr: int, segment: SpeakerSegment) -> np.ndarray:
    """Extract audio segment with small fade-in/out to avoid clicks."""
    start = int(segment.start * sr)
    end = int(segment.end * sr)
    clip = audio[start:end].copy()

    # 10ms cosine fade-in/out
    fade_samples = min(int(0.01 * sr), len(clip) // 4)
    if fade_samples > 0:
        fade_in = np.cos(np.linspace(np.pi, 2 * np.pi, fade_samples)) * 0.5 + 0.5
        fade_out = np.cos(np.linspace(0, np.pi, fade_samples)) * 0.5 + 0.5
        clip[:fade_samples] *= fade_in
        clip[-fade_samples:] *= fade_out

    return clip


def _passes_quality_check(clip: np.ndarray, clip_threshold: float = 0.99, silence_threshold: float = 0.01) -> bool:
    """Check clip quality: no clipping, not too quiet."""
    peak = np.max(np.abs(clip))

    # Reject if clipping
    if peak >= clip_threshold:
        return False

    # Reject if too quiet
    rms = np.sqrt(np.mean(clip ** 2))
    if rms < silence_threshold:
        return False

    return True


def _safe_name(speaker_id: str) -> str:
    """Convert speaker ID to filesystem-safe name."""
    return speaker_id.replace(" ", "_").replace("/", "_").replace("\\", "_")
