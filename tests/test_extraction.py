"""Phase 1.5 build step 2 gate — extraction prompt-contract tests.

Per docs/phases/phase-1.5-build-plan.md §5 step 2:
  - Pipe-separated-text output parses into FactOperation list
  - ADD / UPDATE / NOOP line shapes handled
  - Malformed lines dropped without crashing
  - MockProvider wiring: extraction LLM call is routed, parsed, returned

Real-data gate (10-40 facts on conv-26 session_1, entities include
Jennifer+David, no empty values) runs in step 6 — not here. These tests
cover the deterministic parse-contract that step-6 depends on.
"""

from __future__ import annotations

from datetime import datetime, timezone

from psychology_base.companion.extraction import (
    MAX_FACTS_PER_WINDOW,
    extract_facts,
    parse_extraction_output,
)
from psychology_base.companion.facts import FactOp
from psychology_base.eval.adapters import Fragment
from psychology_base.llm import LLMCache, LLMConfig, LLMRouter, LocalOllamaConfig
from psychology_base.llm.config import ProviderConfig
from psychology_base.llm.providers import MockProvider


def _frag(fid: str, content: str, speaker: str = "Jennifer") -> Fragment:
    return Fragment(
        id=fid,
        content=content,
        timestamp=datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc),
        tags=(f"speaker:{speaker}",),
    )


def test_parse_add_line():
    out = "ADD | Jennifer | owns | Westerly 28 | 0.9"
    ops = parse_extraction_output(out, tick_count=3, source_fragment_ids=("frag-0001",))
    assert len(ops) == 1
    op = ops[0]
    assert op.op == FactOp.ADD
    assert op.fact.entity == "Jennifer"
    assert op.fact.relation == "owns"
    assert op.fact.value == "Westerly 28"
    assert op.fact.confidence == 0.9
    assert op.fact.source_fragment_ids == ("frag-0001",)
    assert op.fact.created_tick == 3
    assert op.fact.updated_tick == 3


def test_parse_update_line():
    out = "UPDATE | fact-0007 | Jennifer | owns | Westerly 28 | 0.95"
    ops = parse_extraction_output(out, tick_count=5, source_fragment_ids=("frag-0010",))
    assert len(ops) == 1
    op = ops[0]
    assert op.op == FactOp.UPDATE
    assert op.target_fact_id == "fact-0007"
    assert op.fact.id == "fact-0007"
    assert op.fact.value == "Westerly 28"
    assert op.fact.confidence == 0.95


def test_parse_noop_line():
    out = "NOOP | fact-0002"
    ops = parse_extraction_output(out, tick_count=1, source_fragment_ids=())
    assert len(ops) == 1
    assert ops[0].op == FactOp.NOOP
    assert ops[0].target_fact_id == "fact-0002"
    assert ops[0].fact is None


def test_parse_multiline_mixed_ops():
    out = "\n".join(
        [
            "ADD | Jennifer | owns | Westerly 28 | 0.9",
            "NOOP | fact-0003",
            "UPDATE | fact-0005 | David | lives_in | Hull | 0.8",
            "ADD | Jennifer | born_in | 1980 | 0.6",
        ]
    )
    ops = parse_extraction_output(out, tick_count=0, source_fragment_ids=())
    assert [o.op for o in ops] == [FactOp.ADD, FactOp.NOOP, FactOp.UPDATE, FactOp.ADD]


def test_parse_drops_malformed_lines():
    out = "\n".join(
        [
            "ADD | Jennifer | owns | Westerly 28 | 0.9",
            "something else entirely",
            "ADD | | relation | value | 0.5",            # empty entity
            "ADD | e | r | v | 1.5",                     # confidence out of range
            "UPDATE | fact-0001 | e | r | v | abc",      # non-float confidence
            "ADD | e | r | v",                           # too few fields
            "",                                           # blank
            "# comment",                                  # comment
            "NOOP | fact-0004",
        ]
    )
    ops = parse_extraction_output(out, tick_count=0, source_fragment_ids=())
    assert len(ops) == 2
    assert ops[0].op == FactOp.ADD
    assert ops[0].fact.entity == "Jennifer"
    assert ops[1].op == FactOp.NOOP


def test_parse_respects_max_facts_cap():
    out = "\n".join(
        f"ADD | e{i} | r | v | 0.5" for i in range(MAX_FACTS_PER_WINDOW + 5)
    )
    ops = parse_extraction_output(out, tick_count=0, source_fragment_ids=())
    assert len(ops) == MAX_FACTS_PER_WINDOW


def test_parse_rejects_empty_value():
    out = "ADD | Jennifer | owns |  | 0.9"
    ops = parse_extraction_output(out, tick_count=0, source_fragment_ids=())
    assert ops == []


def test_parse_rejects_empty_target_on_update():
    out = "UPDATE |  | e | r | v | 0.5"
    ops = parse_extraction_output(out, tick_count=0, source_fragment_ids=())
    assert ops == []


def _router_with_scripted_response(tmp_path, response_text: str) -> tuple[LLMRouter, MockProvider]:
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
    mock = MockProvider(name="mock", responder=lambda msgs, tier: response_text)
    router = LLMRouter(
        cfg,
        providers={"mock": mock},
        cache=LLMCache(cache_dir=tmp_path / "cache"),
    )
    return router, mock


def test_extract_facts_wires_through_router(tmp_path):
    router, mock = _router_with_scripted_response(
        tmp_path,
        "ADD | Jennifer | owns_boat_model | Westerly 28 | 0.9\n"
        "ADD | David | lives_in | Hull | 0.8\n",
    )
    window = [
        _frag("frag-0001", "I bought a Westerly 28 last summer", speaker="Jennifer"),
        _frag("frag-0002", "I'm up in Hull these days", speaker="David"),
    ]
    ops = extract_facts(window, prior_facts=[], router=router, tick_count=4)

    assert len(ops) == 2
    assert {op.fact.entity for op in ops} == {"Jennifer", "David"}
    assert ops[0].fact.source_fragment_ids == ("frag-0001", "frag-0002")
    assert ops[0].fact.created_tick == 4

    # Prompt was dispatched: one chat call, SIMPLE tier
    assert len(mock.calls) == 1
    msgs, tier = mock.calls[0]
    assert tier.value == "simple"
    assert any(m.role == "system" for m in msgs)
    assert any(m.role == "user" for m in msgs)


def test_extract_facts_empty_window_skips_llm(tmp_path):
    router, mock = _router_with_scripted_response(tmp_path, "irrelevant")
    ops = extract_facts([], prior_facts=[], router=router, tick_count=0)
    assert ops == []
    assert mock.calls == []


def test_extract_facts_prior_filter_prefers_entity_matches(tmp_path):
    """When prior facts mention entities present in the window, those priors are
    passed to the prompt; otherwise fall back to most-recently-updated."""
    from psychology_base.companion.facts import Fact

    def make_prior(entity: str, updated: int) -> Fact:
        return Fact(
            id=f"fact-{updated:04d}",
            entity=entity,
            relation="r",
            value="v",
            confidence=0.8,
            source_fragment_ids=("frag-x",),
            created_tick=updated,
            updated_tick=updated,
        )

    captured_user_prompts: list[str] = []

    def responder(msgs, tier):
        for m in msgs:
            if m.role == "user":
                captured_user_prompts.append(m.content)
        return "NOOP | fact-0007"

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
    mock = MockProvider(name="mock", responder=responder)
    router = LLMRouter(
        cfg,
        providers={"mock": mock},
        cache=LLMCache(cache_dir=tmp_path / "cache"),
    )

    priors = [
        make_prior("Jennifer", 1),
        make_prior("Zoe", 10),  # most recent but no window match
    ]
    window = [_frag("frag-0001", "Jennifer bought a boat", speaker="Jennifer")]
    extract_facts(window, prior_facts=priors, router=router, tick_count=0)

    user_prompt = captured_user_prompts[0]
    assert "Jennifer" in user_prompt
    # Entity-matching Jennifer prior wins over most-recent Zoe
    assert "Zoe" not in user_prompt
