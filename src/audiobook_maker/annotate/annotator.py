"""
LLM-powered speaker attribution.

Converts prose into an annotated script: each line attributed to NARRATOR
or a named CHARACTER, with a voice direction (instruct) field.

Strategy (inspired by Alexandria):
- Chunk text by paragraphs (max ~3000 chars)
- Pass each chunk to LLM with continuity context (character roster + last 3 entries)
- LLM returns [{speaker, text, instruct}, ...]
- Optional review pass to fix attribution errors
"""

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

from openai import OpenAI

from .prompts import SYSTEM_PROMPT, build_user_prompt


@dataclass
class ScriptEntry:
    speaker: str
    text: str
    instruct: str
    chapter_index: int = 0
    paragraph_index: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AnnotationConfig:
    llm_base_url: str = "http://localhost:8000/v1"
    llm_model: str = "qwen3-30b"
    llm_api_key: str = "not-needed"
    max_chunk_chars: int = 3000
    temperature: float = 0.4
    top_p: float = 0.8
    max_tokens: int = 8000
    max_retries: int = 2


def annotate_book(
    parsed_book: dict,
    config: AnnotationConfig | None = None,
) -> list[ScriptEntry]:
    """
    Annotate a parsed book with speaker attribution.

    Args:
        parsed_book: Dict from parse step (title, author, chapters with paragraphs).
        config: LLM and chunking configuration.

    Returns:
        List of ScriptEntry objects (the full annotated script).
    """
    if config is None:
        config = AnnotationConfig()

    client = OpenAI(base_url=config.llm_base_url, api_key=config.llm_api_key)

    all_entries: list[ScriptEntry] = []
    characters_seen: set[str] = set()

    chapters = parsed_book.get("chapters", [])
    total_chapters = len(chapters)

    for ch_idx, chapter in enumerate(chapters):
        chapter_title = chapter.get("title")
        paragraphs = chapter.get("paragraphs", [])

        # Build text chunks from paragraphs
        chunks = _chunk_paragraphs(paragraphs, config.max_chunk_chars)
        total_chunks = len(chunks)

        for chunk_idx, chunk in enumerate(chunks):
            # Build context for continuity
            position = _position_marker(ch_idx, total_chapters, chunk_idx, total_chunks)
            roster = sorted(characters_seen)
            tail = [e.to_dict() for e in all_entries[-3:]] if all_entries else []

            user_prompt = build_user_prompt(
                chunk_text=chunk["text"],
                chapter_title=chapter_title,
                position=position,
                roster=roster,
                tail=tail,
            )

            # Call LLM
            entries = _call_llm(client, config, user_prompt)

            # Track characters and store entries
            for entry_dict in entries:
                speaker = entry_dict.get("speaker", "NARRATOR").upper()
                if speaker != "NARRATOR":
                    characters_seen.add(speaker)

                all_entries.append(ScriptEntry(
                    speaker=speaker,
                    text=entry_dict.get("text", ""),
                    instruct=entry_dict.get("instruct", ""),
                    chapter_index=ch_idx,
                    paragraph_index=chunk["start_para"],
                ))

        print(f"  Chapter {ch_idx + 1}/{total_chapters}: "
              f"{chapter_title or '(untitled)'} — "
              f"{len([e for e in all_entries if e.chapter_index == ch_idx])} entries")

    print(f"\nAnnotation complete: {len(all_entries)} entries, "
          f"{len(characters_seen)} characters detected")
    print(f"Characters: {', '.join(sorted(characters_seen)) or '(none)'}")

    return all_entries


def _chunk_paragraphs(paragraphs: list[dict], max_chars: int) -> list[dict]:
    """Group paragraphs into chunks up to max_chars."""
    chunks = []
    current_texts = []
    current_len = 0
    start_para = 0

    for i, para in enumerate(paragraphs):
        text = para.get("text", "")
        if not text:
            continue

        if current_len + len(text) + 2 > max_chars and current_texts:
            chunks.append({
                "text": "\n\n".join(current_texts),
                "start_para": start_para,
            })
            current_texts = [text]
            current_len = len(text)
            start_para = i
        else:
            current_texts.append(text)
            current_len += len(text) + 2

    if current_texts:
        chunks.append({
            "text": "\n\n".join(current_texts),
            "start_para": start_para,
        })

    return chunks


def _position_marker(ch_idx: int, total_ch: int, chunk_idx: int, total_chunks: int) -> str:
    """Generate position context string."""
    parts = []
    if ch_idx == 0 and chunk_idx == 0:
        parts.append("(Beginning of book)")
    elif ch_idx == total_ch - 1 and chunk_idx == total_chunks - 1:
        parts.append("(End of book)")

    parts.append(f"Chapter {ch_idx + 1}/{total_ch}, section {chunk_idx + 1}/{total_chunks}")
    return " ".join(parts)


def _call_llm(client: OpenAI, config: AnnotationConfig, user_prompt: str) -> list[dict]:
    """Call LLM and parse JSON response, with retries."""
    for attempt in range(config.max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=config.llm_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=config.max_tokens,
            )

            content = response.choices[0].message.content.strip()
            entries = _parse_json_response(content)

            if entries:
                return entries

        except (json.JSONDecodeError, KeyError, IndexError):
            # Response was malformed — retry
            continue
        except Exception as e:
            if attempt == config.max_retries:
                print(f"  WARNING: LLM call failed after {config.max_retries + 1} attempts: {e}")
                return []
            continue

    return []


def _parse_json_response(content: str) -> list[dict]:
    """Extract JSON array from LLM response (handles markdown fences, etc.)."""
    # Try direct parse
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return _validate_entries(data)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code fence
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            if isinstance(data, list):
                return _validate_entries(data)
        except json.JSONDecodeError:
            pass

    # Try finding array brackets
    match = re.search(r"\[.*\]", content, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, list):
                return _validate_entries(data)
        except json.JSONDecodeError:
            pass

    return []


def _validate_entries(entries: list) -> list[dict]:
    """Validate and clean entries, keeping only well-formed ones."""
    valid = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if "speaker" not in entry or "text" not in entry:
            continue
        text = entry["text"].strip()
        if not text:
            continue
        valid.append({
            "speaker": entry["speaker"].strip().upper(),
            "text": text,
            "instruct": entry.get("instruct", "").strip(),
        })
    return valid


