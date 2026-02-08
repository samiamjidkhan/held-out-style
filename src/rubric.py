"""Rubric loading, schema, and scoring dimension helpers."""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Dimension:
    name: str
    description: str
    anchors: dict[int, str]  # score -> anchor text

    def format_for_prompt(self) -> str:
        lines = [f'Dimension: "{self.name}"', f"Description: {self.description}", "", "Scoring anchors:"]
        for score in sorted(self.anchors):
            lines.append(f"  {score}: {self.anchors[score]}")
        return "\n".join(lines)


@dataclass
class Rubric:
    dimensions: list[Dimension]

    def dimension_names(self) -> list[str]:
        return [d.name for d in self.dimensions]


@dataclass
class StyleConfig:
    name: str
    description: str
    corpus_dir: str
    train_rubric: Rubric
    eval_rubric: Rubric
    eval_prompts: list[str]

    @classmethod
    def from_yaml(cls, path: str | Path) -> StyleConfig:
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)

        style = data["style"]
        train = _parse_rubric(data["train_rubric"])
        eval_ = _parse_rubric(data["eval_rubric"])

        return cls(
            name=style["name"],
            description=style["description"],
            corpus_dir=style["corpus_dir"],
            train_rubric=train,
            eval_rubric=eval_,
            eval_prompts=data.get("eval_prompts", []),
        )


def _parse_rubric(data: dict[str, Any]) -> Rubric:
    dims = []
    for d in data["dimensions"]:
        anchors = {int(k): v for k, v in d["anchors"].items()}
        dims.append(Dimension(name=d["name"], description=d["description"], anchors=anchors))
    return Rubric(dimensions=dims)


def load_style(style_name: str, config_dir: str = "configs/styles") -> StyleConfig:
    """Load a style config by name from the config directory."""
    path = Path(config_dir) / f"{style_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Style config not found: {path}")
    return StyleConfig.from_yaml(path)


def list_styles(config_dir: str = "configs/styles") -> list[str]:
    """List available style names."""
    return sorted(p.stem for p in Path(config_dir).glob("*.yaml"))
