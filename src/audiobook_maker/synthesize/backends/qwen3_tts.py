"""
Qwen3-TTS backend.

Alibaba's Qwen3-TTS model — supports voice cloning from 5-15s reference
audio AND instruct-based delivery control. Native support for style/emotion
directions via the instruct field.

Requires: transformers, qwen3-tts model weights.
"""

import os

import numpy as np
import torch

from ..engine import TTSEngine, VoiceConfig, register_engine


@register_engine("qwen3_tts")
class Qwen3TTSEngine(TTSEngine):
    """Qwen3-TTS engine with voice cloning and instruct-based style control."""

    def __init__(self):
        self._model = None
        self._processor = None
        self._device = None
        self._sample_rate = 24000

    @property
    def name(self) -> str:
        return "qwen3_tts"

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def load(self, model_id: str | None = None, device: str | None = None, **kwargs) -> None:
        """
        Load Qwen3-TTS model.

        Args:
            model_id: HuggingFace model ID or local path.
            device: Device to load on (cuda/cpu).
        """
        from transformers import AutoModelForCausalLM, AutoProcessor

        if model_id is None:
            model_id = os.environ.get("QWEN3_TTS_MODEL", "Qwen/Qwen3-TTS")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if "cuda" in device else torch.float32,
            device_map=device,
            trust_remote_code=True,
        )
        self._device = device

        # Get sample rate from model config if available
        if hasattr(self._model.config, "audio_config"):
            self._sample_rate = getattr(self._model.config.audio_config, "sample_rate", 24000)

    def is_loaded(self) -> bool:
        return self._model is not None

    def synthesize(
        self,
        text: str,
        voice: VoiceConfig,
        language: str = "en",
        speed: float = 1.0,
    ) -> np.ndarray:
        """
        Synthesize with Qwen3-TTS.

        Uses the instruct field from VoiceConfig.style for delivery directions.
        Uses ref_audio + ref_text for voice cloning.
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load() first.")

        # Build the prompt with instruct (style direction)
        messages = self._build_messages(text, voice, language)

        # Process inputs
        inputs = self._processor(
            messages,
            return_tensors="pt",
        ).to(self._device)

        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=4096,
                temperature=0.7,
                top_p=0.9,
            )

        # Decode audio from tokens
        audio = self._processor.decode_audio(outputs, inputs)
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()

        return audio.astype(np.float32).squeeze()

    def _build_messages(self, text: str, voice: VoiceConfig, language: str) -> list[dict]:
        """Build Qwen3-TTS message format with optional clone + instruct."""
        messages = []

        # System message with instruct (if provided)
        instruct = voice.style or ""
        if instruct:
            messages.append({
                "role": "system",
                "content": f"You are a voice actor. Speak with: {instruct}",
            })

        # User message with text to synthesize
        user_content = []

        # Add reference audio for voice cloning
        if voice.ref_audio and os.path.isfile(voice.ref_audio):
            user_content.append({
                "type": "audio",
                "audio_url": f"file://{os.path.abspath(voice.ref_audio)}",
            })
            if voice.ref_text:
                user_content.append({
                    "type": "text",
                    "text": f"[Reference transcript: {voice.ref_text}]",
                })

        # The text to synthesize
        user_content.append({
            "type": "text",
            "text": f"Please speak the following: {text}",
        })

        messages.append({"role": "user", "content": user_content})

        return messages
