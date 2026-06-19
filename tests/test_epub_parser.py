"""Tests for epub_parser module: HTML processing, filtering, paragraph extraction."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bs4 import BeautifulSoup
from audiobook_maker.parse.epub_parser import (
    _should_skip,
    _extract_title,
    _strip_non_narrative,
    _extract_paragraphs,
    _build_toc_map,
    Paragraph,
)


def _make_item(name: str):
    """Create a mock epub item."""
    item = MagicMock()
    item.get_name.return_value = name
    return item


class TestShouldSkip:
    def test_skips_cover_by_filename(self):
        item = _make_item("cover.xhtml")
        soup = BeautifulSoup("<body><p>Cover image</p></body>", "html.parser")
        assert _should_skip(item, soup) is True

    def test_skips_toc_by_filename(self):
        item = _make_item("toc.xhtml")
        soup = BeautifulSoup("<body><p>Table of contents</p></body>", "html.parser")
        assert _should_skip(item, soup) is True

    def test_skips_copyright_by_filename(self):
        item = _make_item("copyright.xhtml")
        soup = BeautifulSoup("<body><p>All rights reserved blah blah</p></body>", "html.parser")
        assert _should_skip(item, soup) is True

    def test_skips_by_epub_type(self):
        item = _make_item("chapter00.xhtml")
        soup = BeautifulSoup('<body epub:type="titlepage"><p>Title Page Content Here Extra</p></body>', "html.parser")
        assert _should_skip(item, soup) is True

    def test_skips_short_content(self):
        item = _make_item("chapter01.xhtml")
        soup = BeautifulSoup("<body><p>Short.</p></body>", "html.parser")
        assert _should_skip(item, soup) is True

    def test_keeps_normal_chapter(self):
        item = _make_item("chapter01.xhtml")
        long_text = "This is a normal chapter with plenty of content. " * 10
        soup = BeautifulSoup(f"<body><p>{long_text}</p></body>", "html.parser")
        assert _should_skip(item, soup) is False


class TestExtractTitle:
    def test_extracts_h1(self):
        soup = BeautifulSoup("<h1>Chapter One</h1><p>Content here.</p>", "html.parser")
        item = _make_item("ch01.xhtml")
        title = _extract_title(soup, item, {})
        assert title == "Chapter One"

    def test_extracts_h2(self):
        soup = BeautifulSoup("<h2>The Beginning</h2><p>Content.</p>", "html.parser")
        item = _make_item("ch01.xhtml")
        title = _extract_title(soup, item, {})
        assert title == "The Beginning"

    def test_falls_back_to_toc(self):
        soup = BeautifulSoup("<p>No heading here, just content.</p>", "html.parser")
        item = _make_item("ch05.xhtml")
        toc_titles = {"ch05.xhtml": "Chapter Five"}
        title = _extract_title(soup, item, toc_titles)
        assert title == "Chapter Five"

    def test_returns_none_when_no_title(self):
        soup = BeautifulSoup("<p>Just text.</p>", "html.parser")
        item = _make_item("unknown.xhtml")
        title = _extract_title(soup, item, {})
        assert title is None

    def test_heading_is_removed_from_soup(self):
        soup = BeautifulSoup("<h1>Title</h1><p>Body text.</p>", "html.parser")
        item = _make_item("ch01.xhtml")
        _extract_title(soup, item, {})
        assert soup.find("h1") is None


class TestStripNonNarrative:
    def test_removes_aside(self):
        soup = BeautifulSoup("<p>Main text.</p><aside>Footnote content</aside>", "html.parser")
        _strip_non_narrative(soup)
        assert soup.find("aside") is None
        assert "Main text" in soup.get_text()

    def test_removes_sup(self):
        soup = BeautifulSoup("<p>Text with reference<sup>1</sup> here.</p>", "html.parser")
        _strip_non_narrative(soup)
        assert soup.find("sup") is None

    def test_removes_epub_type_footnote(self):
        html = '<p>Main.</p><div epub:type="footnote"><p>Note text</p></div>'
        soup = BeautifulSoup(html, "html.parser")
        _strip_non_narrative(soup)
        assert "Note text" not in soup.get_text()

    def test_removes_footnote_class(self):
        html = '<p>Main.</p><div class="footnote"><p>Note</p></div>'
        soup = BeautifulSoup(html, "html.parser")
        _strip_non_narrative(soup)
        assert "Note" not in soup.get_text()

    def test_removes_images(self):
        soup = BeautifulSoup('<p>Text</p><img src="pic.jpg" alt="A photo"/>', "html.parser")
        _strip_non_narrative(soup)
        assert soup.find("img") is None

    def test_replaces_br_with_newline(self):
        soup = BeautifulSoup("<p>Line one<br/>Line two</p>", "html.parser")
        _strip_non_narrative(soup)
        text = soup.get_text()
        assert "Line one" in text
        assert "Line two" in text


class TestExtractParagraphs:
    def test_basic_extraction(self):
        html = "<p>First paragraph.</p><p>Second paragraph.</p>"
        soup = BeautifulSoup(html, "html.parser")
        paragraphs = _extract_paragraphs(soup)
        assert len(paragraphs) == 2
        assert paragraphs[0].text == "First paragraph."
        assert paragraphs[1].text == "Second paragraph."

    def test_skips_empty_paragraphs(self):
        html = "<p>Real content.</p><p>   </p><p></p><p>More content.</p>"
        soup = BeautifulSoup(html, "html.parser")
        paragraphs = _extract_paragraphs(soup)
        assert len(paragraphs) == 2

    def test_detects_dialogue(self):
        html = '<p>"Hello there," he said.</p><p>She nodded.</p>'
        soup = BeautifulSoup(html, "html.parser")
        paragraphs = _extract_paragraphs(soup)
        assert paragraphs[0].is_dialogue is True
        assert paragraphs[1].is_dialogue is False

    def test_detects_smart_quote_dialogue(self):
        html = '<p>\u201cHello,\u201d she whispered.</p>'
        soup = BeautifulSoup(html, "html.parser")
        paragraphs = _extract_paragraphs(soup)
        # After clean_for_tts normalizes quotes, it should still detect dialogue
        assert paragraphs[0].is_dialogue is True

    def test_detects_blockquote(self):
        html = "<blockquote><p>Dear Sir, I write to inform you...</p></blockquote>"
        soup = BeautifulSoup(html, "html.parser")
        paragraphs = _extract_paragraphs(soup)
        assert len(paragraphs) == 1
        assert paragraphs[0].is_blockquote is True

    def test_applies_tts_cleanup(self):
        html = "<p>He ran\u2014fast\u2014toward the exit.</p>"
        soup = BeautifulSoup(html, "html.parser")
        paragraphs = _extract_paragraphs(soup)
        assert "\u2014" not in paragraphs[0].text
        assert ", " in paragraphs[0].text
