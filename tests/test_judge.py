from ragcheck.cache import JudgmentCache
from ragcheck.judge.judge import Judge, load_prompt
from tests.conftest import MockLLMClient


def test_load_prompt_parses_frontmatter():
    prompt = load_prompt("faithfulness_verify")
    assert prompt.version == "v1"
    assert "{claim}" in prompt.template
    assert not prompt.template.startswith("---")


def test_prompt_render():
    prompt = load_prompt("faithfulness_verify")
    rendered = prompt.render(context="CTX", claim="CLAIM")
    assert "CTX" in rendered and "CLAIM" in rendered


def test_judge_uses_cache(tmp_path):
    llm = MockLLMClient(lambda prompt: "SUPPORTED")
    cache = JudgmentCache(tmp_path / "c.sqlite")
    judge = Judge(llm, cache)
    prompt = load_prompt("faithfulness_verify")

    kwargs = dict(metric_name="m", key_parts=("q", "c", "a"), context="c", claim="a")
    assert judge.ask(prompt, **kwargs) == "SUPPORTED"
    assert judge.ask(prompt, **kwargs) == "SUPPORTED"
    assert len(llm.calls) == 1  # second call served from cache

    # different key parts miss the cache
    judge.ask(prompt, metric_name="m", key_parts=("q2", "c", "a"), context="c", claim="a")
    assert len(llm.calls) == 2
