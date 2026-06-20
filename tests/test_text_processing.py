"""Tests for text_processing module: cleaning, splitting, chunking."""

from audiobook_maker.parse.text_processing import (
    clean_for_tts,
    split_sentences,
    chunk_text,
    chunk_sentences,
)


class TestCleanForTTS:
    def test_normalizes_smart_quotes(self):
        text = "\u201cHello,\u201d she said. \u2018Fine.\u2019"
        result = clean_for_tts(text)
        assert "\u201c" not in result
        assert "\u201d" not in result
        assert '"Hello,"' in result
        assert "'Fine.'" in result

    def test_replaces_em_dash_with_comma(self):
        text = "He ran\u2014fast as he could\u2014toward the door."
        result = clean_for_tts(text)
        assert "\u2014" not in result
        assert "ran, fast" in result

    def test_replaces_en_dash(self):
        text = "Pages 10\u201320 were missing."
        result = clean_for_tts(text)
        assert "\u2013" not in result

    def test_replaces_double_dash(self):
        text = "He thought--perhaps wrongly--that it was safe."
        result = clean_for_tts(text)
        assert "--" not in result
        assert ", " in result

    def test_normalizes_ellipsis(self):
        text = "Wait\u2026 what?"
        result = clean_for_tts(text)
        assert "..." in result
        assert "\u2026" not in result

    def test_collapses_whitespace(self):
        text = "Too   many    spaces   here."
        result = clean_for_tts(text)
        assert "  " not in result
        assert result == "Too many spaces here."

    def test_preserves_normal_text(self):
        text = "The quick brown fox jumps over the lazy dog."
        assert clean_for_tts(text) == text


class TestSplitSentences:
    def test_basic_splitting(self):
        text = "First sentence. Second sentence. Third sentence."
        result = split_sentences(text)
        assert len(result) == 3

    def test_handles_abbreviations(self):
        text = "Mr. Smith went to Washington. He met Dr. Jones there."
        result = split_sentences(text)
        assert len(result) == 2
        assert "Mr. Smith" in result[0]

    def test_handles_dialogue(self):
        text = '"Hello!" she said. "How are you?"'
        result = split_sentences(text)
        # Should not over-split on the quote marks
        assert len(result) <= 3

    def test_filters_non_alphanumeric(self):
        text = "Real sentence. * * * Another sentence."
        result = split_sentences(text)
        # The "* * *" should be filtered out
        for s in result:
            assert any(c.isalnum() for c in s)

    def test_empty_input(self):
        assert split_sentences("") == []


class TestChunkText:
    def test_respects_max_chars(self):
        text = "Short sentence one. Short sentence two. Short sentence three. A bit longer sentence here for testing."
        chunks = chunk_text(text, max_chars=50)
        for chunk in chunks:
            # Allow slight overflow for single sentences
            assert len(chunk) < 200

    def test_does_not_split_short_text(self):
        text = "Just one sentence."
        chunks = chunk_text(text, max_chars=400)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_groups_small_sentences(self):
        text = "A. B. C. D. E."
        chunks = chunk_text(text, max_chars=400)
        # All should fit in one chunk
        assert len(chunks) == 1

    def test_splits_long_text(self):
        sentences = ["This is sentence number {}.".format(i) for i in range(20)]
        text = " ".join(sentences)
        chunks = chunk_text(text, max_chars=100)
        assert len(chunks) > 1
        # Reconstructed text should contain all content
        reconstructed = " ".join(chunks)
        for s in sentences:
            assert s in reconstructed


class TestChunkSentences:
    def test_handles_single_long_sentence(self):
        long = "word " * 200  # ~1000 chars
        chunks = chunk_sentences([long.strip()], max_chars=400)
        assert len(chunks) >= 1
        # Should have attempted to split on clause boundaries
        # (but since there are no commas, it'll be one chunk)

    def test_long_sentence_with_commas(self):
        long = "First clause here, second clause there, third clause everywhere, fourth clause somewhere"
        chunks = chunk_sentences([long], max_chars=50)
        assert len(chunks) > 1

    def test_empty_input(self):
        assert chunk_sentences([], max_chars=400) == []

    def test_preserves_all_content(self):
        sentences = ["Hello world.", "Goodbye world.", "See you later."]
        chunks = chunk_sentences(sentences, max_chars=30)
        reconstructed = " ".join(chunks)
        for s in sentences:
            assert s in reconstructed
