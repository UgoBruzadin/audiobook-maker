"""Tests for synthesis module: engine registry, pipeline logic, voice map I/O."""

import json
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

from audiobook_maker.synthesize.engine import (
    TTSEngine,
    VoiceConfig,
    register_engine,
    get_engine,
    list_engines,
    _ENGINES,
)
from audiobook_maker.synthesize.pipeline import (
    synthesize_script,
    load_voice_map,
    save_voice_map,
    SynthesisConfig,
    PAUSE_BETWEEN_SPEAKERS,
    PAUSE_SAME_SPEAKER,
)
from audiobook_maker.annotate.annotator import ScriptEntry


class TestVoiceConfig:
    def test_frozen(self):
        vc = VoiceConfig(speaker_id="NARRATOR", ref_audio="/path.wav")
        try:
            vc.speaker_id = "OTHER"
            assert False, "Should be frozen"
        except AttributeError:
            pass

    def test_defaults_to_none(self):
        vc = VoiceConfig(speaker_id="TEST")
        assert vc.ref_audio is None
        assert vc.embedding_path is None
        assert vc.style is None


class TestEngineRegistry:
    def test_register_and_retrieve(self):
        # Register a dummy engine
        @register_engine("test_dummy")
        class DummyEngine(TTSEngine):
            @property
            def name(self): return "test_dummy"
            @property
            def sample_rate(self): return 16000
            def load(self, **kwargs): pass
            def synthesize(self, text, voice, language="en", speed=1.0):
                return np.zeros(100)
            def is_loaded(self): return True

        assert "test_dummy" in list_engines()
        engine = get_engine("test_dummy")
        assert engine.name == "test_dummy"
        assert engine.sample_rate == 16000

        # Cleanup
        del _ENGINES["test_dummy"]

    def test_get_unknown_engine_raises(self):
        try:
            get_engine("nonexistent_engine_xyz")
            assert False, "Should raise"
        except ValueError as e:
            assert "nonexistent_engine_xyz" in str(e)

    def test_list_engines_includes_registered(self):
        # xtts_v2 and qwen3_tts should be registered via backends import
        engines = list_engines()
        assert "xtts_v2" in engines
        assert "qwen3_tts" in engines


class TestVoiceMapIO:
    def test_save_and_load_roundtrip(self, tmp_path):
        voice_map = {
            "NARRATOR": VoiceConfig(
                speaker_id="NARRATOR",
                ref_audio="/voices/narrator.wav",
                description="Deep male voice",
            ),
            "ELENA": VoiceConfig(
                speaker_id="ELENA",
                ref_audio="/voices/elena.wav",
                ref_text="Hello, how are you?",
                embedding_path="/voices/elena.npy",
            ),
        }

        path = tmp_path / "voices.json"
        save_voice_map(voice_map, path)

        loaded = load_voice_map(path)
        assert "NARRATOR" in loaded
        assert "ELENA" in loaded
        assert loaded["NARRATOR"].ref_audio == "/voices/narrator.wav"
        assert loaded["ELENA"].ref_text == "Hello, how are you?"
        assert loaded["ELENA"].embedding_path == "/voices/elena.npy"

    def test_uppercases_speaker_ids(self, tmp_path):
        data = {"narrator": {"ref_audio": "/path.wav"}}
        path = tmp_path / "voices.json"
        with open(path, "w") as f:
            json.dump(data, f)

        loaded = load_voice_map(path)
        assert "NARRATOR" in loaded


class TestSynthesizePipeline:
    @patch("audiobook_maker.synthesize.pipeline.get_engine")
    def test_renders_entries(self, mock_get_engine, tmp_path):
        # Mock engine
        mock_engine = MagicMock()
        mock_engine.sample_rate = 24000
        mock_engine.synthesize.return_value = np.zeros(24000, dtype=np.float32)  # 1s silence
        mock_get_engine.return_value = mock_engine

        script = [
            ScriptEntry("NARRATOR", "The door opened.", "calm narration", chapter_index=0),
            ScriptEntry("ELENA", "Hello!", "bright greeting", chapter_index=0),
        ]

        voice_map = {
            "NARRATOR": VoiceConfig(speaker_id="NARRATOR", ref_audio="/fake.wav"),
            "ELENA": VoiceConfig(speaker_id="ELENA", ref_audio="/fake2.wav"),
        }

        config = SynthesisConfig(engine_name="xtts_v2")
        rendered = synthesize_script(script, voice_map, tmp_path / "audio", config)

        assert len(rendered) == 2
        assert rendered[0].speaker == "NARRATOR"
        assert rendered[1].speaker == "ELENA"
        assert Path(rendered[0].audio_path).exists()
        assert Path(rendered[1].audio_path).exists()

        # Engine should have been called with correct voices
        assert mock_engine.synthesize.call_count == 2

    @patch("audiobook_maker.synthesize.pipeline.get_engine")
    def test_applies_instruct_as_style(self, mock_get_engine, tmp_path):
        mock_engine = MagicMock()
        mock_engine.sample_rate = 24000
        mock_engine.synthesize.return_value = np.zeros(24000, dtype=np.float32)
        mock_get_engine.return_value = mock_engine

        script = [
            ScriptEntry("NARRATOR", "Quietly.", "whispered, barely audible", chapter_index=0),
        ]

        voice_map = {
            "NARRATOR": VoiceConfig(speaker_id="NARRATOR", ref_audio="/fake.wav"),
        }

        synthesize_script(script, voice_map, tmp_path / "audio", SynthesisConfig())

        # Check that style was passed through
        call_args = mock_engine.synthesize.call_args
        voice_arg = call_args[1]["voice"] if "voice" in call_args[1] else call_args[0][1]
        assert voice_arg.style == "whispered, barely audible"

    @patch("audiobook_maker.synthesize.pipeline.get_engine")
    def test_falls_back_to_narrator_voice(self, mock_get_engine, tmp_path):
        mock_engine = MagicMock()
        mock_engine.sample_rate = 24000
        mock_engine.synthesize.return_value = np.zeros(24000, dtype=np.float32)
        mock_get_engine.return_value = mock_engine

        # UNKNOWN_CHARACTER not in voice map — should fall back to NARRATOR
        script = [
            ScriptEntry("UNKNOWN_CHARACTER", "Hello.", "", chapter_index=0),
        ]

        voice_map = {
            "NARRATOR": VoiceConfig(speaker_id="NARRATOR", ref_audio="/fake.wav"),
        }

        rendered = synthesize_script(script, voice_map, tmp_path / "audio", SynthesisConfig())
        assert len(rendered) == 1  # should render (using narrator voice)

    @patch("audiobook_maker.synthesize.pipeline.get_engine")
    def test_skips_entry_with_no_voice(self, mock_get_engine, tmp_path):
        mock_engine = MagicMock()
        mock_engine.sample_rate = 24000
        mock_engine.synthesize.return_value = np.zeros(24000, dtype=np.float32)
        mock_get_engine.return_value = mock_engine

        script = [
            ScriptEntry("ORPHAN", "Hello.", "", chapter_index=0),
        ]

        # Empty voice map — no NARRATOR fallback either
        rendered = synthesize_script(script, {}, tmp_path / "audio", SynthesisConfig())
        assert len(rendered) == 0

    @patch("audiobook_maker.synthesize.pipeline.get_engine")
    def test_chunks_long_text(self, mock_get_engine, tmp_path):
        mock_engine = MagicMock()
        mock_engine.sample_rate = 24000
        mock_engine.synthesize.return_value = np.zeros(24000, dtype=np.float32)
        mock_get_engine.return_value = mock_engine

        # Text longer than max_chunk_chars should be split
        long_text = "This is a sentence. " * 30  # ~600 chars
        script = [ScriptEntry("NARRATOR", long_text, "", chapter_index=0)]
        voice_map = {"NARRATOR": VoiceConfig(speaker_id="NARRATOR", ref_audio="/f.wav")}

        config = SynthesisConfig(max_chunk_chars=200)
        synthesize_script(script, voice_map, tmp_path / "audio", config)

        # Engine should have been called multiple times (text was chunked)
        assert mock_engine.synthesize.call_count > 1
