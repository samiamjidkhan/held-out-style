"""CLI entry point for held-out-style pipeline."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

app = typer.Typer(name="held-out-style", help="Character style training via synthetic pipeline with held-out evaluation")
console = Console()


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def load_pipeline_config(path: str = "configs/pipeline.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@app.command()
def extract_style(
    style: str = typer.Argument(..., help="Style name (e.g. norm_macdonald)"),
    config: str = typer.Option("configs/pipeline.yaml", help="Pipeline config path"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Step 1: Extract style principles from corpus."""
    setup_logging(verbose)
    from src.rubric import load_style
    from src.extract import extract_principles
    from src.inference import create_backend

    pipeline_cfg = load_pipeline_config(config)
    style_cfg = load_style(style)
    backend = create_backend(pipeline_cfg["judge_model"])

    ext_cfg = pipeline_cfg["extraction"]
    principles = extract_principles(
        style=style_cfg,
        backend=backend,
        chunks_per_call=ext_cfg["chunks_per_call"],
        num_principles=ext_cfg["num_principles"],
    )

    console.print(f"\n[bold green]Extracted {len(principles)} principles for {style}:[/bold green]")
    for i, p in enumerate(principles, 1):
        console.print(f"  {i}. [bold]{p['title']}[/bold]: {p['description']}")


@app.command()
def generate_data(
    style: str = typer.Argument(..., help="Style name"),
    config: str = typer.Option("configs/pipeline.yaml", help="Pipeline config path"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Steps 2-3: Generate prompts and preference pairs."""
    setup_logging(verbose)
    from src.rubric import load_style
    from src.extract import load_principles
    from src.generate import generate_prompts, generate_preference_pairs, save_dataset
    from src.inference import create_backend

    pipeline_cfg = load_pipeline_config(config)
    style_cfg = load_style(style)
    principles = load_principles(style)

    gen_cfg = pipeline_cfg["generation"]

    # Load teacher model (70B) - same as judge
    teacher_backend = create_backend(pipeline_cfg["judge_model"])
    console.print(f"[bold]Teacher model:[/bold] {pipeline_cfg['judge_model']['model_name']}")

    # Load student model (8B base) if configured
    student_backend = None
    if "student_model" in pipeline_cfg:
        student_backend = create_backend(pipeline_cfg["student_model"])
        console.print(f"[bold]Student model:[/bold] {pipeline_cfg['student_model']['model_name']}")
    else:
        console.print("[yellow]No student_model configured, using teacher for both[/yellow]")

    console.print(f"[bold]Generating prompts for {style}...[/bold]")
    prompts = generate_prompts(
        style=style_cfg,
        principles=principles,
        backend=teacher_backend,
        num_prompts=gen_cfg["num_prompts"],
    )
    console.print(f"  Generated {len(prompts)} prompts")

    console.print(f"[bold]Generating preference pairs...[/bold]")
    pairs = generate_preference_pairs(
        style=style_cfg,
        principles=principles,
        prompts=prompts,
        teacher_backend=teacher_backend,
        student_backend=student_backend,
        min_score_gap=gen_cfg["min_score_gap"],
        chosen_temperature=gen_cfg["chosen_temperature"],
        rejected_temperature=gen_cfg["rejected_temperature"],
        use_prefill_thinking=gen_cfg.get("use_prefill_thinking", False),
    )
    console.print(f"  Generated {len(pairs)} preference pairs")

    if len(pairs) < gen_cfg["min_pairs"]:
        console.print(
            f"[yellow]Warning: Only {len(pairs)} pairs generated "
            f"(target: {gen_cfg['min_pairs']}). Consider lowering min_score_gap.[/yellow]"
        )

    output_path = save_dataset(pairs, style)
    console.print(f"[green]Dataset saved to {output_path}[/green]")


@app.command()
def train(
    style: str = typer.Argument(..., help="Style name"),
    phase: str = typer.Option("both", help="Training phase: dpo, sft, or both"),
    config: str = typer.Option("configs/training.yaml", help="Training config path"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Train LoRA adapter (DPO → SFT pipeline following OpenCharacterTraining)."""
    setup_logging(verbose)
    from src.train import run_sft, run_dpo, merge_adapters

    dpo_path = None

    if phase in ("dpo", "both"):
        console.print(f"[bold]Starting DPO training for {style} (Stage 1: Learn style from preferences)...[/bold]")
        dpo_path = run_dpo(style, config_path=config)
        console.print(f"[green]DPO checkpoint: {dpo_path}[/green]")

    if phase in ("sft", "both"):
        console.print(f"[bold]Starting SFT training for {style} (Stage 2: Internalize via introspection)...[/bold]")
        sft_path = run_sft(style, dpo_checkpoint=dpo_path, config_path=config)
        console.print(f"[green]SFT checkpoint: {sft_path}[/green]")

    if phase == "both":
        console.print(f"[bold]Merging adapters (1.0×DPO + 0.25×SFT)...[/bold]")
        merged_path = merge_adapters(style, config_path=config)
        console.print(f"[green]Merged adapter: {merged_path}[/green]")


@app.command("eval")
def evaluate(
    style: str = typer.Argument(..., help="Style name"),
    eval_config: str = typer.Option("configs/eval.yaml", help="Eval config path"),
    training_config: str = typer.Option("configs/training.yaml", help="Training config path"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Evaluate model variants against train + eval rubrics."""
    setup_logging(verbose)
    from src.rubric import load_style
    from src.eval import run_evaluation, format_report

    style_cfg = load_style(style)

    console.print(f"[bold]Running evaluation for {style}...[/bold]")
    results = run_evaluation(
        style=style_cfg,
        eval_config_path=eval_config,
        training_config_path=training_config,
    )

    report = format_report(results)
    console.print(report)


@app.command()
def run_all(
    style: str = typer.Argument(..., help="Style name"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run the full pipeline: extract → generate → train → eval."""
    setup_logging(verbose)
    from src.rubric import load_style
    from src.extract import extract_principles, load_principles
    from src.generate import generate_prompts, generate_preference_pairs, save_dataset
    from src.train import run_sft, run_dpo
    from src.eval import run_evaluation, format_report
    from src.inference import create_backend

    pipeline_cfg = load_pipeline_config()
    style_cfg = load_style(style)

    # Load teacher model (70B) - same as judge
    teacher_backend = create_backend(pipeline_cfg["judge_model"])
    console.print(f"[bold]Teacher model:[/bold] {pipeline_cfg['judge_model']['model_name']}")

    # Load student model (8B base) if configured
    student_backend = None
    if "student_model" in pipeline_cfg:
        student_backend = create_backend(pipeline_cfg["student_model"])
        console.print(f"[bold]Student model:[/bold] {pipeline_cfg['student_model']['model_name']}")

    # Step 1: Extract
    console.print(f"\n[bold cyan]Step 1: Extracting style principles...[/bold cyan]")
    ext_cfg = pipeline_cfg["extraction"]
    principles = extract_principles(
        style=style_cfg,
        backend=teacher_backend,
        chunks_per_call=ext_cfg["chunks_per_call"],
        num_principles=ext_cfg["num_principles"],
    )
    console.print(f"  Extracted {len(principles)} principles")

    # Step 2-3: Generate
    console.print(f"\n[bold cyan]Step 2-3: Generating training data...[/bold cyan]")
    gen_cfg = pipeline_cfg["generation"]
    prompts = generate_prompts(
        style=style_cfg,
        principles=principles,
        backend=teacher_backend,
        num_prompts=gen_cfg["num_prompts"],
    )
    pairs = generate_preference_pairs(
        style=style_cfg,
        principles=principles,
        prompts=prompts,
        teacher_backend=teacher_backend,
        student_backend=student_backend,
        min_score_gap=gen_cfg["min_score_gap"],
        chosen_temperature=gen_cfg["chosen_temperature"],
        rejected_temperature=gen_cfg["rejected_temperature"],
        use_prefill_thinking=gen_cfg.get("use_prefill_thinking", False),
    )
    save_dataset(pairs, style)
    console.print(f"  Generated {len(pairs)} preference pairs")

    # Clean up vLLM model to free GPU memory for training
    console.print(f"\n[bold cyan]Cleaning up GPU memory before training...[/bold cyan]")
    if hasattr(teacher_backend, 'shutdown'):
        teacher_backend.shutdown()
    if student_backend is not None and hasattr(student_backend, 'shutdown'):
        student_backend.shutdown()
    del teacher_backend
    if student_backend is not None:
        del student_backend
    import gc
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    console.print("  GPU memory cleared")

    # Step 4: Train (DPO first, then SFT on DPO checkpoint)
    console.print(f"\n[bold cyan]Step 4: Training (DPO → SFT)...[/bold cyan]")
    from src.train import merge_adapters
    dpo_path = run_dpo(style)
    console.print(f"  DPO: {dpo_path}")
    sft_path = run_sft(style, dpo_checkpoint=dpo_path)
    console.print(f"  SFT: {sft_path}")
    merged_path = merge_adapters(style)
    console.print(f"  Merged: {merged_path}")

    # Step 5: Eval
    console.print(f"\n[bold cyan]Step 5: Evaluating...[/bold cyan]")
    results = run_evaluation(style=style_cfg)
    report = format_report(results)
    console.print(report)


@app.command()
def list_styles():
    """List available style configurations."""
    from src.rubric import list_styles as _list_styles

    styles = _list_styles()
    if not styles:
        console.print("[yellow]No style configs found in configs/styles/[/yellow]")
        return

    table = Table(title="Available Styles")
    table.add_column("Name", style="bold")
    table.add_column("Description")
    table.add_column("Train Dims")
    table.add_column("Eval Dims")

    from src.rubric import load_style

    for name in styles:
        cfg = load_style(name)
        table.add_row(
            name,
            cfg.description,
            str(len(cfg.train_rubric.dimensions)),
            str(len(cfg.eval_rubric.dimensions)),
        )

    console.print(table)


if __name__ == "__main__":
    app()
