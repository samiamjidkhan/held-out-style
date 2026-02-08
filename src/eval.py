"""Evaluation orchestration and reporting (GPU version).

Memory-efficient design: generates all responses first, unloads variant models,
then loads 70B judge for scoring. This allows evaluation on GPUs that can't
hold both models simultaneously.
"""

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .inference import InferenceBackend, GenerationConfig, create_backend
from .judge import score_response_all_dimensions
from .rubric import StyleConfig, Dimension

logger = logging.getLogger(__name__)


def _clear_gpu_memory():
    """Aggressively clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_eval_config(config_path: str = "configs/eval.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_training_config(config_path: str = "configs/training.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class HFPeftBackend(InferenceBackend):
    """HuggingFace backend with optional PEFT adapter for evaluation."""

    def __init__(
        self,
        base_model: str,
        adapter_path: str | None = None,
        default_config: GenerationConfig | None = None,
        system_prompt: str | None = None,
    ):
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.default_config = default_config or GenerationConfig()
        self.system_prompt = system_prompt
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return

        logger.info(f"Loading base model: {self.base_model}")
        self._model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        if self.adapter_path:
            logger.info(f"Loading adapter: {self.adapter_path}")
            self._model = PeftModel.from_pretrained(self._model, self.adapter_path)

        self._tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def generate(self, prompt: str, config: GenerationConfig | None = None) -> str:
        self._load()
        config = config or self.default_config

        # Build messages with optional system prompt
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        inputs = self._tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self._model.device)

        outputs = self._model.generate(
            inputs,
            max_new_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            do_sample=config.temperature > 0,
            pad_token_id=self._tokenizer.pad_token_id,
        )

        response = self._tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        return response.strip()

    def generate_batch(self, prompts: list[str], config: GenerationConfig | None = None) -> list[str]:
        return [self.generate(p, config) for p in prompts]

    def shutdown(self):
        """Release GPU memory by unloading the model."""
        if self._model is not None:
            logger.info(f"Shutting down model: {self.base_model}")
            del self._model
            self._model = None
            self._tokenizer = None
            _clear_gpu_memory()


def run_evaluation(
    style: StyleConfig,
    eval_config_path: str = "configs/eval.yaml",
    training_config_path: str = "configs/training.yaml",
) -> dict[str, Any]:
    """Run full evaluation for a style across all model variants.

    Memory-efficient two-phase approach:
    1. Generate all responses (one variant at a time, unload after each)
    2. Load 70B judge once and score all responses

    This allows evaluation on GPUs that can't hold both 8B variant and 70B judge.
    """
    eval_cfg = load_eval_config(eval_config_path)
    train_cfg = load_training_config(training_config_path)

    # Collect all dimensions (train + eval)
    all_dimensions = style.train_rubric.dimensions + style.eval_rubric.dimensions
    train_dim_names = set(d.name for d in style.train_rubric.dimensions)
    eval_dim_names = set(d.name for d in style.eval_rubric.dimensions)

    gen_config = GenerationConfig(
        temperature=eval_cfg["generation"]["temperature"],
        max_tokens=eval_cfg["generation"]["max_tokens"],
    )

    results: dict[str, Any] = {
        "style": style.name,
        "eval_prompts": style.eval_prompts,
        "variants": {},
    }

    # ========== PHASE 1: Generate all responses ==========
    # Store responses temporarily: {variant_name: [(prompt, response), ...]}
    all_responses: dict[str, list[tuple[str, str]]] = {}

    logger.info("=" * 60)
    logger.info("PHASE 1: Generating responses for all variants")
    logger.info("=" * 60)

    for variant in eval_cfg["eval_variants"]:
        variant_name = variant["name"]
        logger.info(f"\nGenerating responses for variant: {variant_name}")

        # Create model backend for this variant
        adapter_path = None
        if variant.get("adapter_path"):
            adapter_path = variant["adapter_path"].format(style=style.name)

        # Get system prompt if specified (for baseline comparison)
        system_prompt = variant.get("system_prompt")
        if system_prompt:
            logger.info(f"  Using system prompt: {system_prompt[:50]}...")

        model_backend = HFPeftBackend(
            base_model=train_cfg["base_model"],
            adapter_path=adapter_path,
            default_config=gen_config,
            system_prompt=system_prompt,
        )

        # Generate all responses for this variant
        variant_responses = []
        for i, prompt in enumerate(style.eval_prompts):
            logger.info(f"  Generating {i+1}/{len(style.eval_prompts)}")
            response = model_backend.generate(prompt, gen_config)
            variant_responses.append((prompt, response))

        all_responses[variant_name] = variant_responses

        # Explicitly unload model to free GPU memory
        logger.info(f"  Unloading {variant_name} model...")
        model_backend.shutdown()
        del model_backend
        _clear_gpu_memory()

    logger.info("\nAll responses generated. GPU memory cleared.")

    # ========== PHASE 2: Score all responses with judge ==========
    logger.info("=" * 60)
    logger.info("PHASE 2: Loading 70B judge and scoring all responses")
    logger.info("=" * 60)

    logger.info("Loading judge model...")
    judge_backend = create_backend(eval_cfg["judge_model"])

    for variant_name, variant_responses in all_responses.items():
        logger.info(f"\nScoring variant: {variant_name}")

        all_scores: dict[str, list[dict]] = {d.name: [] for d in all_dimensions}
        prompt_results: list[dict] = []

        for i, (prompt, response) in enumerate(variant_responses):
            logger.info(f"  Scoring {i+1}/{len(variant_responses)}")

            # Score on all dimensions
            dim_scores = score_response_all_dimensions(response, all_dimensions, judge_backend)

            prompt_result = {
                "prompt": prompt,
                "response": response,
                "scores": dim_scores,
            }
            prompt_results.append(prompt_result)

            for dim_name, score_data in dim_scores.items():
                all_scores[dim_name].append(score_data)

        # Aggregate scores per dimension
        dimension_scores = {}
        for dim_name, scores in all_scores.items():
            valid = [s["score"] for s in scores if s["score"] is not None]
            dimension_scores[dim_name] = {
                "mean": sum(valid) / len(valid) if valid else None,
                "scores": valid,
                "n": len(valid),
            }

        variant_results = {
            "prompt_results": prompt_results,
            "dimension_scores": dimension_scores,
        }

        # Split results into train/eval rubric scores
        variant_results["train_rubric_avg"] = _avg_scores(dimension_scores, train_dim_names)
        variant_results["eval_rubric_avg"] = _avg_scores(dimension_scores, eval_dim_names)

        results["variants"][variant_name] = variant_results

    # Shutdown judge backend
    if hasattr(judge_backend, 'shutdown'):
        judge_backend.shutdown()
    del judge_backend
    _clear_gpu_memory()

    # Compute deltas and generalization gap
    results["analysis"] = _compute_analysis(results["variants"])

    # Save results
    output_dir = Path(eval_cfg["output_dir"]) / style.name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "eval_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nEvaluation results saved to {output_path}")
    return results


def _evaluate_variant(
    model_backend: InferenceBackend,
    judge_backend: InferenceBackend,
    prompts: list[str],
    dimensions: list[Dimension],
    gen_config: GenerationConfig,
) -> dict[str, Any]:
    """Evaluate a single model variant on all prompts and dimensions."""
    all_scores: dict[str, list[dict]] = {d.name: [] for d in dimensions}
    prompt_results: list[dict] = []

    for i, prompt in enumerate(prompts):
        logger.info(f"  Prompt {i+1}/{len(prompts)}")
        # Generate response
        response = model_backend.generate(prompt, gen_config)

        # Score on all dimensions
        dim_scores = score_response_all_dimensions(response, dimensions, judge_backend)

        prompt_result = {
            "prompt": prompt,
            "response": response,
            "scores": dim_scores,
        }
        prompt_results.append(prompt_result)

        for dim_name, score_data in dim_scores.items():
            all_scores[dim_name].append(score_data)

    # Aggregate scores per dimension
    dimension_scores = {}
    for dim_name, scores in all_scores.items():
        valid = [s["score"] for s in scores if s["score"] is not None]
        dimension_scores[dim_name] = {
            "mean": sum(valid) / len(valid) if valid else None,
            "scores": valid,
            "n": len(valid),
        }

    return {
        "prompt_results": prompt_results,
        "dimension_scores": dimension_scores,
    }


def _avg_scores(dimension_scores: dict, dim_names: set[str]) -> float | None:
    """Compute average score across a subset of dimensions."""
    means = [
        dimension_scores[name]["mean"]
        for name in dim_names
        if name in dimension_scores and dimension_scores[name]["mean"] is not None
    ]
    return sum(means) / len(means) if means else None


def _compute_analysis(variants: dict[str, Any]) -> dict[str, Any]:
    """Compute deltas and generalization gap across variants."""
    analysis: dict[str, Any] = {}

    base = variants.get("base")
    if base is None:
        return analysis

    base_train = base.get("train_rubric_avg")
    base_eval = base.get("eval_rubric_avg")

    for name, variant in variants.items():
        if name == "base":
            continue

        train_avg = variant.get("train_rubric_avg")
        eval_avg = variant.get("eval_rubric_avg")

        delta_train = (train_avg - base_train) if (train_avg and base_train) else None
        delta_eval = (eval_avg - base_eval) if (eval_avg and base_eval) else None
        gen_gap = abs(delta_train - delta_eval) if (delta_train is not None and delta_eval is not None) else None

        analysis[name] = {
            "delta_train_rubric": delta_train,
            "delta_eval_rubric": delta_eval,
            "generalization_gap": gen_gap,
        }

    return analysis


def format_report(results: dict[str, Any]) -> str:
    """Format evaluation results into a readable report string."""
    lines = [
        f"Style Evaluation Report: {results['style']}",
        "=" * 60,
        "",
    ]

    for variant_name, variant_data in results["variants"].items():
        lines.append(f"--- {variant_name.upper()} ---")

        train_avg = variant_data.get("train_rubric_avg")
        eval_avg = variant_data.get("eval_rubric_avg")

        lines.append(f"  Train rubric avg: {train_avg:.2f}" if train_avg else "  Train rubric avg: N/A")
        lines.append(f"  Eval rubric avg:  {eval_avg:.2f}" if eval_avg else "  Eval rubric avg:  N/A")

        lines.append("  Dimension scores:")
        for dim_name, dim_data in variant_data["dimension_scores"].items():
            mean = dim_data["mean"]
            n = dim_data["n"]
            score_str = f"{mean:.2f}" if mean is not None else "N/A"
            lines.append(f"    {dim_name}: {score_str} (n={n})")
        lines.append("")

    if "analysis" in results and results["analysis"]:
        lines.append("--- ANALYSIS ---")
        for variant_name, analysis in results["analysis"].items():
            lines.append(f"  {variant_name}:")
            dt = analysis.get("delta_train_rubric")
            de = analysis.get("delta_eval_rubric")
            gg = analysis.get("generalization_gap")
            lines.append(f"    Delta train rubric: {dt:+.2f}" if dt is not None else "    Delta train rubric: N/A")
            lines.append(f"    Delta eval rubric:  {de:+.2f}" if de is not None else "    Delta eval rubric:  N/A")
            lines.append(f"    Generalization gap: {gg:.2f}" if gg is not None else "    Generalization gap: N/A")
        lines.append("")

    return "\n".join(lines)
