"""LLM-as-judge scoring engine for rubric-based evaluation."""

from __future__ import annotations

import json
import logging
import re

from .inference import InferenceBackend, GenerationConfig
from .rubric import Dimension

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """\
Score the following response on the dimension "{dim_name}":
{dim_description}

Scoring anchors:
1: {anchor_1}
3: {anchor_3}
5: {anchor_5}

Response to evaluate:
---
{response}
---

Provide your evaluation in this exact JSON format:
{{
  "score": <integer 1-5>,
  "justification": "<brief justification for the score>"
}}"""


def score_response(
    response: str,
    dimension: Dimension,
    backend: InferenceBackend,
) -> dict | None:
    """Score a single response on a single dimension.

    Returns {"score": int, "justification": str} or None on failure.
    """
    prompt = JUDGE_PROMPT.format(
        dim_name=dimension.name,
        dim_description=dimension.description,
        anchor_1=dimension.anchors.get(1, "Low quality"),
        anchor_3=dimension.anchors.get(3, "Medium quality"),
        anchor_5=dimension.anchors.get(5, "High quality"),
        response=response,
    )

    config = GenerationConfig(temperature=0.1, max_tokens=256)

    try:
        raw = backend.generate(prompt, config)
        return _parse_judge_response(raw)
    except Exception as e:
        logger.warning(f"Judge scoring failed for dimension {dimension.name}: {e}")
        return None


def score_response_all_dimensions(
    response: str,
    dimensions: list[Dimension],
    backend: InferenceBackend,
) -> dict[str, dict]:
    """Score a response across all dimensions.

    Returns {dimension_name: {"score": int, "justification": str}}.
    """
    results = {}
    for dim in dimensions:
        result = score_response(response, dim, backend)
        if result is not None:
            results[dim.name] = result
        else:
            results[dim.name] = {"score": None, "justification": "Scoring failed"}
    return results


def _parse_judge_response(response: str) -> dict:
    """Parse judge JSON response, handling markdown code blocks and edge cases."""
    text = response.strip()

    # Try extracting from code block first
    if "```" in text:
        start = text.index("```") + 3
        if text[start:start + 4] == "json":
            start += 4
        end = text.index("```", start)
        text = text[start:end].strip()

    # Try direct JSON parse
    try:
        data = json.loads(text)
        score = int(data["score"])
        score = max(1, min(5, score))
        return {"score": score, "justification": data.get("justification", "")}
    except (json.JSONDecodeError, KeyError, ValueError):
        pass

    # Fallback: regex extraction
    score_match = re.search(r'"score"\s*:\s*(\d)', text)
    if score_match:
        score = int(score_match.group(1))
        score = max(1, min(5, score))
        just_match = re.search(r'"justification"\s*:\s*"([^"]*)"', text)
        justification = just_match.group(1) if just_match else ""
        return {"score": score, "justification": justification}

    # Last resort: look for any digit 1-5
    digit_match = re.search(r'\b([1-5])\b', text)
    if digit_match:
        return {"score": int(digit_match.group(1)), "justification": text[:200]}

    raise ValueError(f"Could not parse judge response: {text[:200]}")
