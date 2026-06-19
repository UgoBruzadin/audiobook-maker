"""
Text cleaning and segmentation for TTS synthesis.

Handles punctuation normalization, sentence splitting, and chunking
to produce TTS-friendly text segments.
"""

import re

import nltk
from nltk.tokenize import sent_tokenize

# Ensure punkt tokenizer is available
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


def clean_for_tts(text: str) -> str:
    """Normalize text for TTS consumption."""
    # Normalize unicode quotes
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")

    # Replace em-dashes and semicolons with commas (natural pauses)
    text = text.replace("\u2014", ", ")  # em-dash
    text = text.replace("\u2013", ", ")  # en-dash
    text = text.replace("--", ", ")

    # Ellipsis normalization
    text = text.replace("\u2026", "...")

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def split_sentences(text: str) -> list[str]:
    """Split text into sentences using NLTK (handles abbreviations)."""
    sentences = sent_tokenize(text)
    return [s.strip() for s in sentences if any(c.isalnum() for c in s)]


def chunk_text(text: str, max_chars: int = 400) -> list[str]:
    """
    Split text into TTS-sized chunks at sentence boundaries.

    Groups sentences up to max_chars. For XTTS v2, 200-400 chars
    produces the best quality (avoids hallucinations on long input).
    """
    sentences = split_sentences(text)
    return chunk_sentences(sentences, max_chars)


def chunk_sentences(sentences: list[str], max_chars: int = 400) -> list[str]:
    """Group sentences into chunks respecting max_chars limit."""
    chunks = []
    current = []
    current_len = 0

    for s in sentences:
        # If a single sentence exceeds limit, split on clauses
        if len(s) > max_chars:
            if current:
                chunks.append(" ".join(current))
                current, current_len = [], 0
            chunks.extend(_split_long_sentence(s, max_chars))
            continue

        if current_len + len(s) + 1 > max_chars and current:
            chunks.append(" ".join(current))
            current, current_len = [s], len(s)
        else:
            current.append(s)
            current_len += len(s) + 1

    if current:
        chunks.append(" ".join(current))

    return chunks


def _split_long_sentence(sentence: str, max_chars: int) -> list[str]:
    """Split an overly long sentence on clause boundaries (commas, colons)."""
    parts = re.split(r"(?<=[,;:])\s+", sentence)
    chunks = []
    current = ""

    for part in parts:
        if len(current) + len(part) + 1 > max_chars and current:
            chunks.append(current.strip())
            current = part
        else:
            current = f"{current} {part}".strip()

    if current:
        chunks.append(current.strip())

    return chunks
