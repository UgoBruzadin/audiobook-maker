"""
Review pass for annotated scripts.

Fixes common attribution errors, merges short segments, validates
text preservation. Rejects batches with significant text loss.
"""

import json
import re
from dataclasses import asdict

from openai import OpenAI

from .annotator import ScriptEntry, AnnotationConfig, _parse_json_response
from .prompts import REVIEW_SYSTEM_PROMPT, build_review_prompt


# Acceptable word count ratio between input and output
TEXT_LOSS_MIN = 0.95
TEXT_LOSS_MAX = 1.05

BATCH_SIZE = 25


def review_script(
    entries: list[ScriptEntry],
    config: AnnotationConfig | None = None,
) -> list[ScriptEntry]:
    """
    Run a review pass over the annotated script.

    Processes in batches of BATCH_SIZE entries. Validates that text is
    preserved (rejects batches with >5% text loss).

    Args:
        entries: The annotated script entries to review.
        config: LLM configuration.

    Returns:
        Reviewed and corrected entries.
    """
    if config is None:
        config = AnnotationConfig()

    client = OpenAI(base_url=config.llm_base_url, api_key=config.llm_api_key)

    # Collect all known characters
    roster = sorted(set(
        e.speaker for e in entries if e.speaker != "NARRATOR"
    ))

    reviewed: list[ScriptEntry] = []
    batches = _make_batches(entries, BATCH_SIZE)

    for batch_idx, batch in enumerate(batches):
        batch_dicts = [asdict(e) for e in batch]
        original_word_count = _word_count(batch_dicts)

        result = _review_batch(client, config, batch_dicts, roster)

        if result is None:
            # Review failed — keep originals
            reviewed.extend(batch)
            continue

        # Validate text preservation
        result_word_count = _word_count(result)
        ratio = result_word_count / original_word_count if original_word_count > 0 else 1.0

        if ratio < TEXT_LOSS_MIN or ratio > TEXT_LOSS_MAX:
            print(f"  Batch {batch_idx + 1}: rejected (word ratio {ratio:.2f}, "
                  f"expected {TEXT_LOSS_MIN}-{TEXT_LOSS_MAX})")
            reviewed.extend(batch)
        else:
            # Accept reviewed batch
            for entry_dict in result:
                reviewed.append(ScriptEntry(
                    speaker=entry_dict.get("speaker", "NARRATOR").upper(),
                    text=entry_dict.get("text", ""),
                    instruct=entry_dict.get("instruct", ""),
                    chapter_index=batch[0].chapter_index,
                    paragraph_index=batch[0].paragraph_index,
                ))

    # Post-processing: merge short consecutive narrator entries
    reviewed = _merge_short_narrators(reviewed, min_chars=100)

    print(f"Review complete: {len(entries)} → {len(reviewed)} entries")
    return reviewed


def _review_batch(
    client: OpenAI,
    config: AnnotationConfig,
    batch: list[dict],
    roster: list[str],
) -> list[dict] | None:
    """Send batch to LLM for review. Returns corrected entries or None on failure."""
    user_prompt = build_review_prompt(batch, roster)

    for attempt in range(config.max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=config.llm_model,
                messages=[
                    {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,  # Lower temp for review (more conservative)
                top_p=config.top_p,
                max_tokens=config.max_tokens,
            )

            content = response.choices[0].message.content.strip()
            entries = _parse_json_response(content)

            if entries:
                return entries

        except Exception as e:
            if attempt == config.max_retries:
                print(f"  Review batch failed: {e}")
                return None

    return None


def _make_batches(entries: list[ScriptEntry], batch_size: int) -> list[list[ScriptEntry]]:
    """Split entries into batches, respecting chapter boundaries."""
    batches = []
    current_batch = []

    for entry in entries:
        # Start new batch at chapter boundaries or when full
        if (current_batch and
            (len(current_batch) >= batch_size or
             entry.chapter_index != current_batch[-1].chapter_index)):
            batches.append(current_batch)
            current_batch = []
        current_batch.append(entry)

    if current_batch:
        batches.append(current_batch)

    return batches


def _merge_short_narrators(entries: list[ScriptEntry], min_chars: int) -> list[ScriptEntry]:
    """Merge consecutive short NARRATOR entries."""
    if not entries:
        return entries

    merged = [entries[0]]

    for entry in entries[1:]:
        prev = merged[-1]

        # Merge if both narrator, same chapter, and previous is short
        if (entry.speaker == "NARRATOR" and
            prev.speaker == "NARRATOR" and
            entry.chapter_index == prev.chapter_index and
            len(prev.text) < min_chars):
            prev.text = f"{prev.text} {entry.text}"
            # Keep the more descriptive instruct
            if len(entry.instruct) > len(prev.instruct):
                prev.instruct = entry.instruct
        else:
            merged.append(entry)

    return merged


def _word_count(entries: list[dict]) -> int:
    """Count total words across entries."""
    return sum(len(e.get("text", "").split()) for e in entries)
