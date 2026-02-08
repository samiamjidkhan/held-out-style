"""Step 1: Extract style principles from raw corpus using judge model."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from .corpus import Chunk, load_and_chunk
from .inference import InferenceBackend, GenerationConfig
from .rubric import StyleConfig

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """\
Analyze the following passages by {author}. Identify {num_principles} specific, \
actionable writing principles that define this author's distinctive style. \
Focus on: sentence structure, word choice, rhythm, humor mechanics, \
emotional register, and rhetorical devices. Be concrete and specific, \
not generic.

For each principle, provide a short title and a detailed description of the pattern.

{passages}

Respond in this exact JSON format:
{{
  "principles": [
    {{
      "title": "Short principle title",
      "description": "Detailed description of the stylistic pattern, with examples from the text."
    }}
  ]
}}"""


def extract_principles(
    style: StyleConfig,
    backend: InferenceBackend,
    chunks_per_call: int = 5,
    num_principles: int = 8,
    output_dir: str = "data/principles",
) -> list[dict[str, str]]:
    """Extract style principles from corpus using the judge model.

    Returns a list of {"title": ..., "description": ...} dicts.
    """
    corpus_dir = Path(style.corpus_dir)
    chunks = load_and_chunk(corpus_dir)

    if not chunks:
        raise ValueError(f"No chunks extracted from corpus: {corpus_dir}")

    logger.info(f"Loaded {len(chunks)} chunks from {corpus_dir}")

    # Sample chunks for extraction (use multiple calls if many chunks)
    random.shuffle(chunks)
    selected = chunks[:chunks_per_call * 3]  # Use up to 3x for multiple extraction rounds

    all_principles: list[dict[str, str]] = []

    for batch_start in range(0, len(selected), chunks_per_call):
        batch = selected[batch_start:batch_start + chunks_per_call]
        passages = _format_passages(batch)

        prompt = EXTRACTION_PROMPT.format(
            author=style.name.replace("_", " ").title(),
            num_principles=num_principles,
            passages=passages,
        )

        config = GenerationConfig(temperature=0.4, max_tokens=2048)
        response = backend.generate(prompt, config)

        try:
            parsed = _parse_principles(response)
            all_principles.extend(parsed)
            logger.info(f"Extracted {len(parsed)} principles from batch starting at {batch_start}")
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse principles from batch: {e}")
            continue

    # Deduplicate and consolidate
    principles = _deduplicate_principles(all_principles, backend, num_principles)

    # Save
    output_path = Path(output_dir) / f"{style.name}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"style": style.name, "principles": principles}, f, indent=2)

    logger.info(f"Saved {len(principles)} principles to {output_path}")
    return principles


def _format_passages(chunks: list[Chunk]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"--- Passage {i} (from {chunk.source_file}) ---\n{chunk.text}")
    return "\n\n".join(parts)


def _parse_principles(response: str) -> list[dict[str, str]]:
    """Parse JSON principles from model response, handling markdown code blocks."""
    text = response.strip()
    if "```" in text:
        # Extract from code block
        start = text.index("```") + 3
        if text[start:start + 4] == "json":
            start += 4
        end = text.index("```", start)
        text = text[start:end].strip()

    data = json.loads(text)
    return data["principles"]


def _deduplicate_principles(
    principles: list[dict[str, str]],
    backend: InferenceBackend,
    target_count: int,
) -> list[dict[str, str]]:
    """Use the judge model to consolidate duplicate/overlapping principles."""
    if len(principles) <= target_count:
        return principles

    principles_text = "\n".join(
        f"{i+1}. {p['title']}: {p['description']}" for i, p in enumerate(principles)
    )

    prompt = f"""\
The following style principles were extracted from multiple passages and may contain \
duplicates or overlapping concepts. Consolidate them into exactly {target_count} \
distinct, non-overlapping principles. Merge similar ones and keep the most specific version.

{principles_text}

Respond in this exact JSON format:
{{
  "principles": [
    {{
      "title": "Short principle title",
      "description": "Consolidated description."
    }}
  ]
}}"""

    config = GenerationConfig(temperature=0.2, max_tokens=2048)
    response = backend.generate(prompt, config)

    try:
        return _parse_principles(response)[:target_count]
    except (json.JSONDecodeError, KeyError):
        logger.warning("Failed to consolidate principles, returning first N")
        return principles[:target_count]


def load_principles(style_name: str, output_dir: str = "data/principles") -> list[dict[str, str]]:
    """Load previously extracted principles from disk."""
    path = Path(output_dir) / f"{style_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Principles not found: {path}. Run extract-style first.")
    with open(path) as f:
        data = json.load(f)
    return data["principles"]
