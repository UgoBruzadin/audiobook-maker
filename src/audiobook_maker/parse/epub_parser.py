"""
EPUB → structured chapters.

Extracts chapters in reading order with titles, paragraphs, and metadata.
Handles TOC/spine resolution, front/back matter filtering, footnote stripping.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urldefrag

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

from .text_processing import clean_for_tts


@dataclass
class Paragraph:
    text: str
    is_dialogue: bool = False
    is_emphasis: bool = False
    is_blockquote: bool = False


@dataclass
class Chapter:
    title: str | None
    paragraphs: list[Paragraph] = field(default_factory=list)
    index: int = 0

    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.paragraphs)

    @property
    def word_count(self) -> int:
        return sum(len(p.text.split()) for p in self.paragraphs)


@dataclass
class Book:
    title: str
    author: str
    chapters: list[Chapter] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        return sum(ch.word_count for ch in self.chapters)


# Sections to skip (epub:type values and filename patterns)
SKIP_EPUB_TYPES = {
    "titlepage", "toc", "copyright-page", "colophon", "dedication",
    "acknowledgements", "bibliography", "index", "loi", "lot",
}
SKIP_FILENAME_PATTERNS = ["cover", "title", "copyright", "toc", "nav"]


def parse_epub(path: str | Path) -> Book:
    """Parse an EPUB file into structured Book with chapters."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"EPUB not found: {path}")

    book = epub.read_epub(str(path))

    # Extract metadata
    title = _get_metadata(book, "title") or path.stem
    author = _get_metadata(book, "creator") or "Unknown"

    # Build TOC title map: href_base -> title
    toc_titles = _build_toc_map(book.toc)

    # Process spine items in reading order
    chapters = []
    chapter_idx = 0

    for item_id, _ in book.spine:
        item = book.get_item_with_id(item_id)
        if item is None:
            continue

        soup = BeautifulSoup(item.get_content(), "html.parser")

        if _should_skip(item, soup):
            continue

        # Extract chapter title from heading or TOC
        chapter_title = _extract_title(soup, item, toc_titles)

        # Strip non-narrative elements
        _strip_non_narrative(soup)

        # Extract paragraphs
        paragraphs = _extract_paragraphs(soup)
        if not paragraphs:
            continue

        chapters.append(Chapter(
            title=chapter_title,
            paragraphs=paragraphs,
            index=chapter_idx,
        ))
        chapter_idx += 1

    return Book(title=title, author=author, chapters=chapters)


def _get_metadata(book: epub.EpubBook, field: str) -> str | None:
    values = book.get_metadata("DC", field)
    if values:
        return values[0][0]
    return None


def _build_toc_map(toc) -> dict[str, str]:
    """Flatten TOC into {href_base: title} map."""
    titles = {}
    for entry in toc:
        if isinstance(entry, epub.Link):
            base, _ = urldefrag(entry.href)
            titles[base] = entry.title
        elif isinstance(entry, tuple):
            section, children = entry
            if hasattr(section, "href") and section.href:
                base, _ = urldefrag(section.href)
                titles[base] = section.title
            titles.update(_build_toc_map(children))
    return titles


def _should_skip(item, soup: BeautifulSoup) -> bool:
    """Determine if this spine item is non-narrative (front/back matter)."""
    # Check epub:type
    body = soup.find("body")
    if body:
        epub_type = body.get("epub:type", "")
        if any(t in epub_type for t in SKIP_EPUB_TYPES):
            return True

    # Check filename
    name = item.get_name().lower()
    if any(p in name for p in SKIP_FILENAME_PATTERNS):
        return True

    # Very short content is likely front/back matter
    text = soup.get_text(strip=True)
    if len(text) < 100:
        return True

    return False


def _extract_title(soup: BeautifulSoup, item, toc_titles: dict) -> str | None:
    """Get chapter title from heading tags or TOC map."""
    # Try heading tags first
    heading = soup.find(["h1", "h2", "h3"])
    if heading:
        title = heading.get_text(strip=True)
        heading.decompose()
        if title:
            return title

    # Fall back to TOC
    item_href = item.get_name()
    return toc_titles.get(item_href)


def _strip_non_narrative(soup: BeautifulSoup):
    """Remove footnotes, endnotes, annotations, images."""
    # Footnotes and asides
    for tag in soup.find_all(["aside", "sup"]):
        tag.decompose()

    # epub:type footnotes/endnotes
    for tag in soup.find_all(attrs={"epub:type": True}):
        epub_type = tag.get("epub:type", "")
        if any(t in epub_type for t in ["footnote", "endnote", "noteref", "annotation"]):
            tag.decompose()

    # Footnote-like classes
    for tag in soup.find_all(class_=lambda c: c and any(
        x in c.lower() for x in ["footnote", "endnote", "note"]
    )):
        tag.decompose()

    # Images (keep alt text if meaningful)
    for img in soup.find_all("img"):
        img.decompose()

    # Replace <br/> with newlines (for verse/poetry)
    for br in soup.find_all("br"):
        br.replace_with("\n")


def _extract_paragraphs(soup: BeautifulSoup) -> list[Paragraph]:
    """Extract paragraphs with dialogue/emphasis detection."""
    paragraphs = []

    for p_tag in soup.find_all("p"):
        text = p_tag.get_text(strip=True)
        if not text or not any(c.isalnum() for c in text):
            continue

        text = clean_for_tts(text)

        # Detect dialogue (starts with quote)
        is_dialogue = bool(re.match(r'^["\u201c]', text))

        # Detect emphasis blocks (entire paragraph in italics)
        is_emphasis = (
            p_tag.find("em") is not None
            and p_tag.find("em").get_text(strip=True) == text
        )

        # Detect blockquotes
        is_blockquote = p_tag.parent and p_tag.parent.name == "blockquote"

        paragraphs.append(Paragraph(
            text=text,
            is_dialogue=is_dialogue,
            is_emphasis=is_emphasis,
            is_blockquote=is_blockquote,
        ))

    return paragraphs


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python epub_parser.py <path.epub>")
        sys.exit(1)

    book = parse_epub(sys.argv[1])
    print(f"Title: {book.title}")
    print(f"Author: {book.author}")
    print(f"Chapters: {len(book.chapters)}")
    print(f"Total words: {book.word_count:,}")
    print()
    for ch in book.chapters:
        print(f"  [{ch.index}] {ch.title or '(untitled)'} — {ch.word_count:,} words, {len(ch.paragraphs)} paragraphs")
