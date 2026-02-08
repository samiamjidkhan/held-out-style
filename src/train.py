"""LoRA fine-tuning with TRL: DPO → SFT pipeline (GPU version)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

logger = logging.getLogger(__name__)


def load_training_config(config_path: str = "configs/training.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _load_dpo_dataset(style_name: str, dataset_dir: str | None = None) -> Dataset:
    """Load DPO dataset in the format expected by TRL."""
    dataset_dir = dataset_dir or f"data/datasets/{style_name}"
    dpo_path = Path(dataset_dir) / "dpo_dataset.jsonl"
    if not dpo_path.exists():
        raise FileNotFoundError(f"DPO dataset not found: {dpo_path}")

    records = [json.loads(line) for line in dpo_path.read_text().strip().split("\n")]

    # TRL DPOTrainer expects: prompt, chosen, rejected
    return Dataset.from_list(records)


def _load_sft_dataset(style_name: str, dataset_dir: str | None = None) -> Dataset:
    """Load SFT dataset (introspection data)."""
    dataset_dir = dataset_dir or f"data/datasets/{style_name}"
    sft_path = Path(dataset_dir) / "introspection_dataset.jsonl"

    # Fall back to basic SFT dataset if introspection not available
    if not sft_path.exists():
        sft_path = Path(dataset_dir) / "sft_dataset.jsonl"

    if not sft_path.exists():
        raise FileNotFoundError(f"SFT dataset not found: {sft_path}")

    records = [json.loads(line) for line in sft_path.read_text().strip().split("\n")]

    # Convert to messages format for SFTTrainer
    formatted = []
    for r in records:
        formatted.append({
            "messages": [
                {"role": "user", "content": r["prompt"]},
                {"role": "assistant", "content": r["response"]},
            ]
        })

    return Dataset.from_list(formatted)


def run_dpo(
    style_name: str,
    dataset_dir: str | None = None,
    config_path: str = "configs/training.yaml",
) -> str:
    """Run DPO training phase (FIRST stage - learns style from preferences).

    Returns the path to the DPO adapter directory.
    """
    cfg = load_training_config(config_path)
    dpo_cfg = cfg["dpo"]
    lora_cfg = cfg["lora"]

    # Output directory
    adapter_dir = Path(cfg["output_dir"]) / style_name / "dpo"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # Load model and tokenizer
    logger.info(f"Loading base model: {cfg['base_model']}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        torch_dtype=torch.bfloat16 if dpo_cfg.get("bf16", True) else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply LoRA
    lora_config = LoraConfig(
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    train_dataset = _load_dpo_dataset(style_name, dataset_dir)
    logger.info(f"Loaded {len(train_dataset)} DPO pairs")

    # DPO Config (TRL 0.27+)
    training_args = DPOConfig(
        output_dir=str(adapter_dir),
        num_train_epochs=dpo_cfg["num_epochs"],
        per_device_train_batch_size=dpo_cfg["per_device_batch_size"],
        gradient_accumulation_steps=dpo_cfg["gradient_accumulation_steps"],
        learning_rate=dpo_cfg["learning_rate"],
        warmup_ratio=dpo_cfg.get("warmup_ratio", 0.1),
        bf16=dpo_cfg.get("bf16", True),
        logging_steps=1,
        save_strategy="epoch",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        report_to="none",
        beta=dpo_cfg["beta"],
        max_length=dpo_cfg["max_seq_length"],
        max_prompt_length=dpo_cfg["max_seq_length"] // 2,
    )

    # DPO Trainer
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    # Train
    logger.info("Starting DPO training...")
    trainer.train()

    # Save adapter
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    logger.info(f"DPO adapter saved to {adapter_dir}")
    return str(adapter_dir)


def run_sft(
    style_name: str,
    dpo_checkpoint: str | None = None,
    dataset_dir: str | None = None,
    config_path: str = "configs/training.yaml",
) -> str:
    """Run SFT training phase (SECOND stage - internalizes style via introspection).

    Uses the DPO checkpoint as base model.
    Returns the path to the SFT adapter directory.
    """
    cfg = load_training_config(config_path)
    sft_cfg = cfg["sft"]
    lora_cfg = cfg["lora"]

    # Use DPO checkpoint as base
    dpo_checkpoint = dpo_checkpoint or str(Path(cfg["output_dir"]) / style_name / "dpo")

    # Output directory
    adapter_dir = Path(cfg["output_dir"]) / style_name / "sft"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # Load base model with DPO adapter
    logger.info(f"Loading base model: {cfg['base_model']}")
    logger.info(f"With DPO adapter: {dpo_checkpoint}")

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        torch_dtype=torch.bfloat16 if sft_cfg.get("bf16", True) else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, dpo_checkpoint)

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    train_dataset = _load_sft_dataset(style_name, dataset_dir)
    logger.info(f"Loaded {len(train_dataset)} SFT samples")

    # SFT Config (TRL 0.27+)
    training_args = SFTConfig(
        output_dir=str(adapter_dir),
        num_train_epochs=sft_cfg["num_epochs"],
        per_device_train_batch_size=sft_cfg["per_device_batch_size"],
        gradient_accumulation_steps=sft_cfg["gradient_accumulation_steps"],
        learning_rate=sft_cfg["learning_rate"],
        warmup_ratio=sft_cfg.get("warmup_ratio", 0.1),
        bf16=sft_cfg.get("bf16", True),
        logging_steps=1,
        save_strategy="epoch",
        gradient_checkpointing=True,
        report_to="none",
        max_length=sft_cfg["max_seq_length"],
    )

    # SFT Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    # Train
    logger.info("Starting SFT training...")
    trainer.train()

    # Save adapter
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    logger.info(f"SFT adapter saved to {adapter_dir}")
    return str(adapter_dir)


def merge_adapters(
    style_name: str,
    config_path: str = "configs/training.yaml",
) -> str:
    """Merge DPO and SFT adapters with weighted combination.

    Returns path to merged adapter.
    """
    cfg = load_training_config(config_path)
    merge_cfg = cfg.get("merge", {"dpo_weight": 1.0, "sft_weight": 0.25})

    dpo_path = Path(cfg["output_dir"]) / style_name / "dpo"
    sft_path = Path(cfg["output_dir"]) / style_name / "sft"
    merged_path = Path(cfg["output_dir"]) / style_name / "merged"
    merged_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading base model: {cfg['base_model']}")
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load DPO adapter
    logger.info(f"Loading DPO adapter from {dpo_path}")
    model = PeftModel.from_pretrained(base_model, str(dpo_path), adapter_name="dpo")

    # Load SFT adapter
    logger.info(f"Loading SFT adapter from {sft_path}")
    model.load_adapter(str(sft_path), adapter_name="sft")

    # Weighted merge
    dpo_weight = merge_cfg["dpo_weight"]
    sft_weight = merge_cfg["sft_weight"]
    logger.info(f"Merging adapters: {dpo_weight}×DPO + {sft_weight}×SFT")

    model.add_weighted_adapter(
        adapters=["dpo", "sft"],
        weights=[dpo_weight, sft_weight],
        adapter_name="merged",
        combination_type="linear",
    )
    model.set_adapter("merged")

    # Save merged adapter
    model.save_pretrained(str(merged_path))

    # Also save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=True)
    tokenizer.save_pretrained(str(merged_path))

    logger.info(f"Merged adapter saved to {merged_path}")
    return str(merged_path)
