"""
XTTS v2 backend.

Coqui's XTTS v2 model — voice cloning from ~30s reference audio.
Best results with chunks of 200-400 characters.
"""

import os

import numpy as np
import torch

from ..engine import TTSEngine, VoiceConfig, register_engine


@register_engine("xtts_v2")
class XTTSv2Engine(TTSEngine):
    """XTTS v2 TTS engine with voice cloning."""

    def __init__(self):
        self._model = None
        self._device = None

    @property
    def name(self) -> str:
        return "xtts_v2"

    @property
    def sample_rate(self) -> int:
        return 24000

    def load(self, model_dir: str | None = None, device: str | None = None, **kwargs) -> None:
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        if model_dir is None:
            model_dir = os.environ.get(
                "XTTS_MODEL_DIR",
                os.path.expanduser("~/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2"),
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        config = XttsConfig()
        config.load_json(os.path.join(model_dir, "config.json"))
        model = Xtts.init_from_config(config)
        model.load_checkpoint(config, checkpoint_dir=model_dir)
        model.to(device)
        model.eval()

        self._model = model
        self._device = device

    def is_loaded(self) -> bool:
        return self._model is not None

    def synthesize(
        self,
        text: str,
        voice: VoiceConfig,
        language: str = "en",
        speed: float = 1.0,
    ) -> np.ndarray:
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load() first.")

        gpt_cond_latent, speaker_embedding = self._get_conditioning(voice)

        out = self._model.inference(
            text,
            language,
            gpt_cond_latent,
            speaker_embedding,
            speed=speed,
            temperature=0.3,
            repetition_penalty=10.0,
            top_k=30,
            top_p=0.85,
        )

        return np.array(out["wav"], dtype=np.float32)

    def _get_conditioning(self, voice: VoiceConfig):
        """Load speaker conditioning from embedding file or reference audio."""
        # Pre-computed embedding
        if voice.embedding_path and os.path.isfile(voice.embedding_path):
            if voice.embedding_path.endswith(".pt"):
                data = torch.load(voice.embedding_path, map_location=self._device)
                return data["gpt_cond_latent"], data["speaker_embedding"]

        # Compute from reference audio
        if voice.ref_audio and os.path.isfile(voice.ref_audio):
            gpt_cond_latent, speaker_embedding = self._model.get_conditioning_latents(
                audio_path=[voice.ref_audio]
            )
            return gpt_cond_latent, speaker_embedding

        raise ValueError(
            f"Voice '{voice.speaker_id}' has no valid reference audio or embedding. "
            f"Provide ref_audio or embedding_path."
        )
