"""Steps 2-3: Generate diverse prompts and preference pairs for DPO training."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from .inference import InferenceBackend, GenerationConfig
from .rubric import StyleConfig, Dimension
from .judge import score_response

logger = logging.getLogger(__name__)

# --- Step 2: Prompt Generation ---

PROMPT_GEN_TEMPLATE = """\
You are generating diverse conversational prompts that would benefit from \
a response written in a specific style.

Style: {style_description}

Key style principles:
{principles_text}

Training dimensions this style is evaluated on:
{dimensions_text}

Generate {batch_size} diverse conversational prompts. Include a mix of:
- Open-ended questions
- Topic-specific requests
- Opinion prompts
- Storytelling requests
- Explanation requests

Each prompt should be 1-2 sentences. The prompts should be ones where the \
target style would be natural and effective.

Respond in this exact JSON format:
{{
  "prompts": ["prompt1", "prompt2", ...]
}}"""


def generate_prompts(
    style: StyleConfig,
    principles: list[dict[str, str]],
    backend: InferenceBackend,
    num_prompts: int = 300,
    batch_size: int = 25,
) -> list[str]:
    """Generate diverse conversational prompts suited to the target style."""
    principles_text = "\n".join(f"- {p['title']}: {p['description']}" for p in principles)
    dimensions_text = "\n".join(
        f"- {d.name}: {d.description}" for d in style.train_rubric.dimensions
    )

    all_prompts: list[str] = []
    num_batches = (num_prompts + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        prompt = PROMPT_GEN_TEMPLATE.format(
            style_description=style.description,
            principles_text=principles_text,
            dimensions_text=dimensions_text,
            batch_size=batch_size,
        )

        config = GenerationConfig(temperature=0.9, max_tokens=2048)
        response = backend.generate(prompt, config)

        try:
            parsed = _parse_prompts(response)
            all_prompts.extend(parsed)
            logger.info(f"Batch {batch_idx + 1}/{num_batches}: generated {len(parsed)} prompts")
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse prompt batch {batch_idx + 1}: {e}")

    # Deduplicate
    seen = set()
    unique = []
    for p in all_prompts:
        normalized = p.strip().lower()
        if normalized not in seen:
            seen.add(normalized)
            unique.append(p.strip())

    result = unique[:num_prompts]
    logger.info(f"Generated {len(result)} unique prompts for {style.name}")
    return result


# --- Step 3: Preference Pair Generation ---

CHOSEN_TEMPLATE = """\
Respond to the following prompt in the style described below. \
Follow the style principles closely. Your response should be natural \
and authentic to this voice.

Style: {style_description}

Style principles to follow:
{principles_text}

Prompt: {prompt}

Respond directly in character. Do not explain the style or mention that \
you are following any principles."""

REJECTED_TEMPLATE = """\
Respond to the following prompt in a helpful but generic way. \
Use a neutral, standard tone without any distinctive stylistic features.

Prompt: {prompt}"""

THINKING_PREFILL = "\n<think>\nI want to respond authentically in this style. Key principles:\n{principles}\n</think>\n"


def generate_preference_pairs(
    style: StyleConfig,
    principles: list[dict[str, str]],
    prompts: list[str],
    teacher_backend: InferenceBackend,
    student_backend: InferenceBackend | None = None,
    min_score_gap: float = 1.5,
    chosen_temperature: float = 0.8,
    rejected_temperature: float = 0.3,
    batch_size: int = 32,
    use_prefill_thinking: bool = False,
) -> list[dict[str, str]]:
    """Generate and filter preference pairs (prompt, chosen, rejected) for DPO.

    Uses batched generation for much faster processing on GPU.
    Teacher/student split: teacher (70B) generates chosen, student (8B) generates rejected.
    """
    principles_text = "\n".join(f"- {p['title']}: {p['description']}" for p in principles)

    # Use teacher for both if no student provided (backwards compatibility)
    if student_backend is None:
        student_backend = teacher_backend

    # Step 1: Generate chosen responses with TEACHER model (70B) + optional thinking prefill
    logger.info(f"Generating {len(prompts)} chosen responses with teacher model...")
    chosen_prompts = []
    for p in prompts:
        base = CHOSEN_TEMPLATE.format(
            style_description=style.description,
            principles_text=principles_text,
            prompt=p,
        )
        if use_prefill_thinking:
            base += THINKING_PREFILL.format(principles=principles_text)
        chosen_prompts.append(base)

    chosen_config = GenerationConfig(temperature=chosen_temperature, max_tokens=1024)
    chosen_responses = _batch_generate(teacher_backend, chosen_prompts, chosen_config, batch_size, "chosen")

    # Strip thinking tokens if used
    if use_prefill_thinking:
        chosen_responses = [
            r.split("</think>")[1].strip() if "</think>" in r else r
            for r in chosen_responses
        ]

    # Step 2: Generate rejected responses with STUDENT model (8B base)
    logger.info(f"Generating {len(prompts)} rejected responses with student model...")
    rejected_prompts = [REJECTED_TEMPLATE.format(prompt=p) for p in prompts]
    rejected_config = GenerationConfig(temperature=rejected_temperature, max_tokens=1024)
    rejected_responses = _batch_generate(student_backend, rejected_prompts, rejected_config, batch_size, "rejected")

    # Step 3: Score all responses using teacher (judge) model
    logger.info("Scoring all responses with judge model...")
    dimensions = style.train_rubric.dimensions

    # Score chosen responses
    chosen_scores_list = _batch_score_all(chosen_responses, dimensions, teacher_backend, batch_size)

    # Score rejected responses
    rejected_scores_list = _batch_score_all(rejected_responses, dimensions, teacher_backend, batch_size)

    # Step 4: Filter pairs by score gap
    logger.info("Filtering pairs by score gap...")
    pairs: list[dict[str, str]] = []
    skipped = 0

    for i, user_prompt in enumerate(prompts):
        chosen_scores = chosen_scores_list[i]
        rejected_scores = rejected_scores_list[i]

        if not chosen_scores or not rejected_scores:
            skipped += 1
            continue

        chosen_avg = sum(chosen_scores.values()) / len(chosen_scores)
        rejected_avg = sum(rejected_scores.values()) / len(rejected_scores)
        gap = chosen_avg - rejected_avg

        if gap >= min_score_gap:
            pairs.append({
                "prompt": user_prompt,
                "chosen": chosen_responses[i],
                "rejected": rejected_responses[i],
                "chosen_avg_score": chosen_avg,
                "rejected_avg_score": rejected_avg,
                "score_gap": gap,
            })
        else:
            skipped += 1

    logger.info(f"Generated {len(pairs)} preference pairs ({skipped} filtered out)")
    return pairs


def _batch_generate(
    backend: InferenceBackend,
    prompts: list[str],
    config: GenerationConfig,
    batch_size: int,
    label: str = "",
) -> list[str]:
    """Generate responses in batches."""
    results = []
    num_batches = (len(prompts) + batch_size - 1) // batch_size

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        batch_num = i // batch_size + 1
        logger.info(f"  {label} batch {batch_num}/{num_batches} ({len(batch)} prompts)")
        batch_results = backend.generate_batch(batch, config)
        results.extend(batch_results)

    return results


def _batch_score_all(
    responses: list[str],
    dimensions: list,
    backend: InferenceBackend,
    batch_size: int,
) -> list[dict[str, float]]:
    """Score all responses on all dimensions using batched calls."""
    import re
    from .judge import JUDGE_PROMPT

    # For each dimension, score all responses in batches
    all_scores = [{} for _ in responses]

    for dim in dimensions:
        logger.info(f"  Scoring dimension: {dim.name}")

        # Build scoring prompts for all responses
        score_prompts = []
        for response in responses:
            prompt = JUDGE_PROMPT.format(
                dim_name=dim.name,
                dim_description=dim.description,
                anchor_1=dim.anchors.get(1, "Low quality"),
                anchor_3=dim.anchors.get(3, "Medium quality"),
                anchor_5=dim.anchors.get(5, "High quality"),
                response=response,
            )
            score_prompts.append(prompt)

        # Batch generate scores
        config = GenerationConfig(temperature=0.1, max_tokens=128)
        score_responses = _batch_generate(backend, score_prompts, config, batch_size, f"  {dim.name}")

        # Parse scores
        for i, score_text in enumerate(score_responses):
            try:
                # Try JSON parse first
                if '"score"' in score_text:
                    match = re.search(r'"score"\s*:\s*(\d)', score_text)
                    if match:
                        all_scores[i][dim.name] = float(match.group(1))
                        continue
                # Fallback: look for any digit 1-5
                match = re.search(r'\b([1-5])\b', score_text)
                if match:
                    all_scores[i][dim.name] = float(match.group(1))
            except Exception:
                pass

    return all_scores


def _score_all_dimensions(
    response: str,
    dimensions: list[Dimension],
    backend: InferenceBackend,
) -> dict[str, float]:
    """Score a response on all dimensions. Returns {dimension_name: score}."""
    scores = {}
    for dim in dimensions:
        result = score_response(response, dim, backend)
        if result is not None:
            scores[dim.name] = result["score"]
    return scores


def save_dataset(
    pairs: list[dict[str, str]],
    style_name: str,
    output_dir: str = "data/datasets",
) -> Path:
    """Save preference pairs in DPO-compatible format."""
    out_path = Path(output_dir) / style_name
    out_path.mkdir(parents=True, exist_ok=True)

    # Save full dataset with metadata
    full_path = out_path / "preference_pairs.json"
    with open(full_path, "w") as f:
        json.dump(pairs, f, indent=2)

    # Save DPO-format (just prompt/chosen/rejected)
    dpo_path = out_path / "dpo_dataset.jsonl"
    with open(dpo_path, "w") as f:
        for pair in pairs:
            record = {
                "prompt": pair["prompt"],
                "chosen": pair["chosen"],
                "rejected": pair["rejected"],
            }
            f.write(json.dumps(record) + "\n")

    # Save SFT-format (just prompt/response from chosen)
    sft_path = out_path / "sft_dataset.jsonl"
    with open(sft_path, "w") as f:
        for pair in pairs:
            record = {
                "prompt": pair["prompt"],
                "response": pair["chosen"],
            }
            f.write(json.dumps(record) + "\n")

    logger.info(f"Saved datasets to {out_path}")
    return out_path


def _parse_prompts(response: str) -> list[str]:
    text = response.strip()
    if "```" in text:
        start = text.index("```") + 3
        if text[start:start + 4] == "json":
            start += 4
        end = text.index("```", start)
        text = text[start:end].strip()

    data = json.loads(text)
    return data["prompts"]
