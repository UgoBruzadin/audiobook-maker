"""Tests for annotation module: chunking, JSON parsing, review logic."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import json

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from audiobook_maker.annotate.annotator import (
    _chunk_paragraphs,
    _position_marker,
    _parse_json_response,
    _validate_entries,
    ScriptEntry,
)
from audiobook_maker.annotate.reviewer import (
    _merge_short_narrators,
    _word_count,
    _make_batches,
)
from audiobook_maker.annotate.prompts import build_user_prompt, build_review_prompt


class TestChunkParagraphs:
    def test_groups_small_paragraphs(self):
        paragraphs = [
            {"text": "Short one."},
            {"text": "Short two."},
            {"text": "Short three."},
        ]
        chunks = _chunk_paragraphs(paragraphs, max_chars=100)
        assert len(chunks) == 1
        assert "Short one." in chunks[0]["text"]
        assert "Short three." in chunks[0]["text"]

    def test_splits_when_exceeds_limit(self):
        paragraphs = [
            {"text": "A" * 50},
            {"text": "B" * 50},
            {"text": "C" * 50},
        ]
        chunks = _chunk_paragraphs(paragraphs, max_chars=80)
        assert len(chunks) > 1

    def test_handles_empty_paragraphs(self):
        paragraphs = [
            {"text": ""},
            {"text": "Real content here."},
            {"text": ""},
        ]
        chunks = _chunk_paragraphs(paragraphs, max_chars=100)
        assert len(chunks) == 1
        assert "Real content" in chunks[0]["text"]

    def test_tracks_start_para_index(self):
        paragraphs = [
            {"text": "A" * 50},
            {"text": "B" * 50},
            {"text": "C" * 50},
        ]
        chunks = _chunk_paragraphs(paragraphs, max_chars=60)
        # Each paragraph gets its own chunk
        assert chunks[0]["start_para"] == 0
        assert chunks[1]["start_para"] == 1

    def test_empty_input(self):
        assert _chunk_paragraphs([], max_chars=100) == []


class TestPositionMarker:
    def test_beginning_of_book(self):
        result = _position_marker(0, 10, 0, 3)
        assert "Beginning of book" in result
        assert "Chapter 1/10" in result

    def test_end_of_book(self):
        result = _position_marker(9, 10, 2, 3)
        assert "End of book" in result

    def test_middle(self):
        result = _position_marker(3, 10, 1, 5)
        assert "Beginning" not in result
        assert "End" not in result
        assert "Chapter 4/10" in result
        assert "section 2/5" in result


class TestParseJsonResponse:
    def test_parses_direct_json(self):
        content = json.dumps([
            {"speaker": "NARRATOR", "text": "Hello.", "instruct": "calm"},
        ])
        result = _parse_json_response(content)
        assert len(result) == 1
        assert result[0]["speaker"] == "NARRATOR"

    def test_parses_markdown_fenced(self):
        content = '```json\n[{"speaker": "ELENA", "text": "Hi.", "instruct": "bright"}]\n```'
        result = _parse_json_response(content)
        assert len(result) == 1
        assert result[0]["speaker"] == "ELENA"

    def test_parses_with_surrounding_text(self):
        content = 'Here is the annotated script:\n[{"speaker": "NARRATOR", "text": "He walked.", "instruct": "steady"}]\nDone!'
        result = _parse_json_response(content)
        assert len(result) == 1

    def test_returns_empty_on_garbage(self):
        result = _parse_json_response("This is not JSON at all.")
        assert result == []

    def test_filters_invalid_entries(self):
        content = json.dumps([
            {"speaker": "NARRATOR", "text": "Valid.", "instruct": "calm"},
            {"speaker": "BOB"},  # missing text
            "not a dict",
            {"speaker": "ALICE", "text": "Also valid.", "instruct": ""},
        ])
        result = _parse_json_response(content)
        assert len(result) == 2
        assert result[0]["text"] == "Valid."
        assert result[1]["text"] == "Also valid."


class TestValidateEntries:
    def test_uppercases_speaker(self):
        entries = [{"speaker": "narrator", "text": "Hello.", "instruct": "calm"}]
        result = _validate_entries(entries)
        assert result[0]["speaker"] == "NARRATOR"

    def test_adds_missing_instruct(self):
        entries = [{"speaker": "BOB", "text": "Hello."}]
        result = _validate_entries(entries)
        assert result[0]["instruct"] == ""

    def test_strips_whitespace(self):
        entries = [{"speaker": "  NARRATOR  ", "text": "  Hello.  ", "instruct": "  calm  "}]
        result = _validate_entries(entries)
        assert result[0]["speaker"] == "NARRATOR"
        assert result[0]["text"] == "Hello."
        assert result[0]["instruct"] == "calm"

    def test_rejects_empty_text(self):
        entries = [
            {"speaker": "NARRATOR", "text": "", "instruct": "calm"},
            {"speaker": "NARRATOR", "text": "Real.", "instruct": "calm"},
        ]
        result = _validate_entries(entries)
        assert len(result) == 1


class TestMergeShortNarrators:
    def test_merges_short_consecutive_narrators(self):
        entries = [
            ScriptEntry("NARRATOR", "Short.", "calm", chapter_index=0),
            ScriptEntry("NARRATOR", "Also short.", "steady", chapter_index=0),
            ScriptEntry("NARRATOR", "Third one.", "measured", chapter_index=0),
        ]
        result = _merge_short_narrators(entries, min_chars=100)
        # All three should merge into one (each is <100 chars)
        assert len(result) == 1
        assert "Short." in result[0].text
        assert "Also short." in result[0].text
        assert "Third one." in result[0].text

    def test_does_not_merge_long_entries(self):
        entries = [
            ScriptEntry("NARRATOR", "A" * 150, "calm", chapter_index=0),
            ScriptEntry("NARRATOR", "B" * 150, "steady", chapter_index=0),
        ]
        result = _merge_short_narrators(entries, min_chars=100)
        assert len(result) == 2

    def test_does_not_merge_different_speakers(self):
        entries = [
            ScriptEntry("NARRATOR", "Short.", "calm", chapter_index=0),
            ScriptEntry("ELENA", "Hello!", "bright", chapter_index=0),
            ScriptEntry("NARRATOR", "Brief.", "calm", chapter_index=0),
        ]
        result = _merge_short_narrators(entries, min_chars=100)
        assert len(result) == 3

    def test_does_not_merge_across_chapters(self):
        entries = [
            ScriptEntry("NARRATOR", "End.", "calm", chapter_index=0),
            ScriptEntry("NARRATOR", "Start.", "fresh", chapter_index=1),
        ]
        result = _merge_short_narrators(entries, min_chars=100)
        assert len(result) == 2

    def test_keeps_better_instruct(self):
        entries = [
            ScriptEntry("NARRATOR", "Short.", "ok", chapter_index=0),
            ScriptEntry("NARRATOR", "More.", "much more descriptive instruct", chapter_index=0),
        ]
        result = _merge_short_narrators(entries, min_chars=100)
        assert result[0].instruct == "much more descriptive instruct"

    def test_empty_input(self):
        assert _merge_short_narrators([], min_chars=100) == []


class TestMakeBatches:
    def test_respects_batch_size(self):
        entries = [ScriptEntry("NARRATOR", f"Entry {i}.", "", chapter_index=0) for i in range(60)]
        batches = _make_batches(entries, batch_size=25)
        assert all(len(b) <= 25 for b in batches)
        assert sum(len(b) for b in batches) == 60

    def test_splits_on_chapter_boundary(self):
        entries = [
            ScriptEntry("NARRATOR", "Ch1 entry.", "", chapter_index=0),
            ScriptEntry("NARRATOR", "Ch1 again.", "", chapter_index=0),
            ScriptEntry("NARRATOR", "Ch2 start.", "", chapter_index=1),
        ]
        batches = _make_batches(entries, batch_size=25)
        assert len(batches) == 2
        assert batches[0][-1].chapter_index == 0
        assert batches[1][0].chapter_index == 1


class TestWordCount:
    def test_counts_words(self):
        entries = [
            {"text": "one two three"},
            {"text": "four five"},
        ]
        assert _word_count(entries) == 5

    def test_empty(self):
        assert _word_count([]) == 0


class TestBuildUserPrompt:
    def test_includes_chunk_text(self):
        prompt = build_user_prompt("Hello world.", position="Chapter 1/5")
        assert "Hello world." in prompt

    def test_includes_position(self):
        prompt = build_user_prompt("Text.", position="Chapter 3/10, section 2/4")
        assert "Chapter 3/10" in prompt

    def test_includes_roster(self):
        prompt = build_user_prompt("Text.", roster=["ALICE", "BOB"])
        assert "ALICE" in prompt
        assert "BOB" in prompt

    def test_includes_tail(self):
        tail = [{"speaker": "NARRATOR", "text": "Previous.", "instruct": "calm"}]
        prompt = build_user_prompt("New text.", tail=tail)
        assert "Previous." in prompt

    def test_includes_chapter_title(self):
        prompt = build_user_prompt("Text.", chapter_title="The Dark Forest")
        assert "The Dark Forest" in prompt


class TestBuildReviewPrompt:
    def test_includes_entries(self):
        entries = [{"speaker": "NARRATOR", "text": "Hello.", "instruct": "calm"}]
        prompt = build_review_prompt(entries)
        assert "Hello." in prompt

    def test_includes_roster(self):
        prompt = build_review_prompt([], roster=["ELENA", "MARCUS"])
        assert "ELENA" in prompt
        assert "MARCUS" in prompt
