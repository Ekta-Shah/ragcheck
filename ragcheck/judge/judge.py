"""LLM-as-judge core with prompt versioning and cached judgments."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import yaml
from pydantic import BaseModel

from ragcheck.cache import JudgmentCache, make_key
from ragcheck.llm import LLMClient


class JudgePrompt(BaseModel):
    """A versioned judge prompt template loaded from markdown+frontmatter."""

    name: str
    version: str
    description: str = ""
    template: str

    def render(self, **fields: str) -> str:
        """Fill the ``{placeholder}`` fields of the template."""
        return self.template.format(**fields)


def load_prompt(name: str) -> JudgePrompt:
    """Load a prompt from ``ragcheck/judge/prompts/<name>.md``.

    The file must start with a YAML frontmatter block declaring at least
    ``version``; the body is the prompt template.
    """
    source = resources.files("ragcheck.judge").joinpath("prompts").joinpath(f"{name}.md")
    return _parse_prompt(name, source.read_text())


def load_prompt_file(path: str | Path) -> JudgePrompt:
    """Load a prompt from an arbitrary filesystem path (for user overrides)."""
    path = Path(path)
    return _parse_prompt(path.stem, path.read_text())


def _parse_prompt(name: str, raw: str) -> JudgePrompt:
    if not raw.startswith("---"):
        raise ValueError(f"Judge prompt {name!r} is missing YAML frontmatter")
    _, frontmatter, body = raw.split("---", 2)
    meta = yaml.safe_load(frontmatter)
    return JudgePrompt(
        name=name,
        version=str(meta["version"]),
        description=meta.get("description", ""),
        template=body.strip(),
    )


class Judge:
    """Runs judge prompts against an LLM, caching every judgment.

    Every call is keyed on (metric_name, prompt_version, *key_parts) so
    re-running an eval on unchanged data is free. The judge model and
    prompt version are recorded by metrics into their MetricResult.
    """

    def __init__(self, llm: LLMClient, cache: JudgmentCache | None = None) -> None:
        """Create a judge around ``llm``; pass ``cache`` to persist judgments."""
        self.llm = llm
        self.cache = cache

    @property
    def model(self) -> str:
        """The judge model identifier, if the client exposes one."""
        return getattr(self.llm, "model", "unknown")

    def ask(
        self,
        prompt: JudgePrompt,
        *,
        metric_name: str,
        key_parts: tuple[str, ...],
        max_tokens: int = 1024,
        **fields: str,
    ) -> str:
        """Render ``prompt`` with ``fields`` and return the (possibly cached) verdict."""
        key = make_key(metric_name, prompt.version, prompt.name, *key_parts)
        if self.cache is not None:
            cached = self.cache.get(key)
            if cached is not None:
                return cached
        text = self.llm.complete(prompt.render(**fields), max_tokens=max_tokens).text
        if self.cache is not None:
            self.cache.set(key, text)
        return text
