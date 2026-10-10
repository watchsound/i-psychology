"""Phase 2 R3 / R5 — reader prompt variants.

Covers:
- ``get_reader_prompt`` returns distinct bodies for v1, v2, and v3 and
  raises on unknown.
- ``LLMReader`` validates ``prompt_variant`` at construction time, not
  per-call.
- The v1, v2, and v3 prompts actually reach the LLM provider (different
  system messages observed).
- The v3 precision clauses from R5 (proper-noun 'where' bias; integer
  'how many' bias; single-word labels valid) are present in the prompt
  body so a future refactor can't silently drop them.
"""

from __future__ import annotations

import pytest

from psychology_base.eval.adapters import Fragment
from psychology_base.eval.benchmarks.reader import (
    LLMReader,
    get_reader_prompt,
)
from psychology_base.llm import LLMCache, LLMConfig, LLMRouter, LocalOllamaConfig
from psychology_base.llm.config import ProviderConfig
from psychology_base.llm.providers import MockProvider


def _router(tmp_path, capture: list) -> LLMRouter:
    cfg = LLMConfig(
        active_provider="mock",
        providers={
            "mock": ProviderConfig(
                name="mock",
                simple_model="mock-simple",
                advanced_model="mock-advanced",
            )
        },
        local_ollama=LocalOllamaConfig(enabled=False),
    )

    def responder(messages, tier):
        capture.append(messages)
        return "stub answer"

    mock = MockProvider(name="mock", responder=responder)
    return LLMRouter(
        cfg,
        providers={"mock": mock},
        cache=LLMCache(cache_dir=tmp_path / "cache"),
    )


def test_get_reader_prompt_v1_and_v2_are_distinct():
    v1 = get_reader_prompt("v1")
    v2 = get_reader_prompt("v2")
    assert v1 != v2
    assert "not stated" in v1
    assert "not stated" in v2
    # The v2 rebalance adds explicit scan-all and extraction templates.
    assert "scan all fragments" in v2.lower()
    assert "when" in v2.lower() and "where" in v2.lower()


def test_get_reader_prompt_v3_distinct_and_contains_precision_clauses():
    v2 = get_reader_prompt("v2")
    v3 = get_reader_prompt("v3")
    assert v2 != v3
    assert "not stated" in v3
    body = v3.lower()
    # R5 precision clause 1 — 'where' biases toward proper-noun place name,
    # with an explicit negative example to block the "home country" miss.
    assert "sweden" in body and "home country" in body
    # R5 precision clause 2 — 'how many' biases toward integer, not frequency.
    assert "integer" in body and "frequency phrase" in body
    # R5 precision clause 3 — single-word labels like "Single" are valid.
    assert "single" in body and "complete answer" in body
    # v2's imprecise-match win is preserved.
    assert "closest literal mention" in body


def test_get_reader_prompt_unknown_variant_raises():
    with pytest.raises(ValueError, match="unknown reader prompt variant"):
        get_reader_prompt("not-a-variant")


def test_llm_reader_validates_prompt_variant_at_construction(tmp_path):
    router = _router(tmp_path, capture=[])
    # Valid variants construct cleanly.
    LLMReader(router=router, prompt_variant="v1")
    LLMReader(router=router, prompt_variant="v2")
    LLMReader(router=router, prompt_variant="v3")
    # Invalid variant fails fast at __post_init__, not at .answer() time.
    with pytest.raises(ValueError, match="unknown reader prompt variant"):
        LLMReader(router=router, prompt_variant="not-a-variant")


def test_llm_reader_sends_v1_prompt_by_default(tmp_path):
    capture: list = []
    router = _router(tmp_path, capture)
    reader = LLMReader(router=router)
    reader.answer(
        question="where did you move from?",
        recall_content="",
        fragments=(),
    )
    assert len(capture) == 1
    system_msg = capture[0][0]
    assert system_msg.role == "system"
    assert system_msg.content == get_reader_prompt("v1")


def test_llm_reader_sends_v2_prompt_when_selected(tmp_path):
    capture: list = []
    router = _router(tmp_path, capture)
    reader = LLMReader(router=router, prompt_variant="v2")
    reader.answer(
        question="where did you move from?",
        recall_content="",
        fragments=(),
    )
    assert len(capture) == 1
    system_msg = capture[0][0]
    assert system_msg.role == "system"
    assert system_msg.content == get_reader_prompt("v2")
    assert "scan all fragments" in system_msg.content.lower()


def test_llm_reader_sends_v3_prompt_when_selected(tmp_path):
    capture: list = []
    router = _router(tmp_path, capture)
    reader = LLMReader(router=router, prompt_variant="v3")
    reader.answer(
        question="where did you move from?",
        recall_content="",
        fragments=(),
    )
    assert len(capture) == 1
    system_msg = capture[0][0]
    assert system_msg.role == "system"
    assert system_msg.content == get_reader_prompt("v3")
    # R5 proper-noun bias is actually reaching the provider.
    assert "sweden" in system_msg.content.lower()
