"""Eval run configuration: Pydantic models + YAML loading."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator


class MetricSpec(BaseModel):
    """A metric selection with optional constructor params."""

    name: str
    params: dict = Field(default_factory=dict)


class EvalConfig(BaseModel):
    """Everything needed to run one evaluation."""

    dataset: Path
    adapter: str
    metrics: list[MetricSpec]
    judge_provider: str = "anthropic"
    judge_model: str | None = None
    concurrency: int = 4
    html: bool = True
    judge_validation: Path | None = None  # embed this validation report, if present
    confirm_above: int = 200  # LLM-judged runs larger than this need assume_yes
    assume_yes: bool = False
    output_dir: Path = Path("ragcheck_output")
    cache_path: Path = Path(".ragcheck_cache.sqlite")
    run_name: str | None = None

    @field_validator("metrics", mode="before")
    @classmethod
    def _coerce_metric_specs(cls, value: object) -> object:
        """Allow bare metric names in YAML alongside {name, params} mappings."""
        if isinstance(value, list):
            return [{"name": v} if isinstance(v, str) else v for v in value]
        return value


def load_config(path: str | Path) -> EvalConfig:
    """Load an EvalConfig from YAML, resolving paths relative to the file."""
    path = Path(path)
    data = yaml.safe_load(path.read_text())
    config = EvalConfig.model_validate(data)
    base = path.parent
    for field in ("dataset", "output_dir", "cache_path"):
        value: Path = getattr(config, field)
        if not value.is_absolute():
            setattr(config, field, base / value)
    return config
