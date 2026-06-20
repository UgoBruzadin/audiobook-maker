"""
Speaker diarization: identify who speaks when in an audio file.

Uses pyannote-audio for state-of-the-art diarization.
Returns immutable segment data — no mutation downstream.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class SpeakerSegment:
    """A single speaker segment (immutable)."""
    speaker: str
    start: float  # seconds
    end: float    # seconds

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class DiarizationResult:
    """Complete diarization output (immutable)."""
    segments: tuple[SpeakerSegment, ...]
    audio_path: str
    sample_rate: int

    @property
    def speakers(self) -> list[str]:
        return sorted(set(s.speaker for s in self.segments))

    @property
    def total_duration(self) -> float:
        if not self.segments:
            return 0.0
        return max(s.end for s in self.segments)

    def segments_for_speaker(self, speaker: str) -> tuple[SpeakerSegment, ...]:
        return tuple(s for s in self.segments if s.speaker == speaker)

    def speaker_duration(self, speaker: str) -> float:
        return sum(s.duration for s in self.segments_for_speaker(speaker))


def diarize(
    audio_path: str | Path,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    hf_token: str | None = None,
) -> DiarizationResult:
    """
    Run speaker diarization on an audio file.

    Args:
        audio_path: Path to audio file (WAV, MP3, FLAC, etc.)
        num_speakers: Exact number of speakers (if known).
        min_speakers: Minimum expected speakers.
        max_speakers: Maximum expected speakers.
        hf_token: HuggingFace token for pyannote models.

    Returns:
        DiarizationResult with all speaker segments.
    """
    from pyannote.audio import Pipeline

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    pipeline.to(device)

    # Build kwargs for speaker count hints
    kwargs = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers

    print(f"Diarizing: {audio_path.name} ...")
    annotation = pipeline(str(audio_path), **kwargs)

    # Convert to our immutable data structure
    segments = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append(SpeakerSegment(
            speaker=speaker,
            start=turn.start,
            end=turn.end,
        ))

    # Sort by start time
    segments.sort(key=lambda s: s.start)

    # Get sample rate from the audio file
    import soundfile as sf
    info = sf.info(str(audio_path))

    result = DiarizationResult(
        segments=tuple(segments),
        audio_path=str(audio_path),
        sample_rate=info.samplerate,
    )

    print(f"  Found {len(result.speakers)} speakers, "
          f"{len(segments)} segments, "
          f"{result.total_duration:.0f}s total")
    for spk in result.speakers:
        dur = result.speaker_duration(spk)
        n_segs = len(result.segments_for_speaker(spk))
        print(f"    {spk}: {dur:.1f}s across {n_segs} segments")

    return result
