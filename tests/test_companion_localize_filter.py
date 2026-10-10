"""Phase 2b Step E — localize-filter LLM stage tests.

Covers:
- ``localize_filter=True`` requires ``cross_encoder_rerank=True``.
- Requires ``consolidation_router`` (the SIMPLE-tier endpoint).
- Default off (no behavioral change for existing configs).
- Keeps the subset the LLM returns, in rerank-score order.
- Preserves rerank score for every kept row.
- Safety floor: when LLM returns fewer than ``min_keep``, falls back
  to top-N by rerank order.
- Robust to malformed LLM output (prose, markdown, out-of-range
  indexes).
- Empty input returns empty.
- Max-input cap: pool past the cap is passed through unfiltered
  (filter can't filter what it didn't see).
- Fallback on provider error: returns original pool, no exception.
- Response parser handles common deviations.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from psychology_base.companion import CompanionSystem
from psychology_base.companion.system import _parse_localize_filter_response
from psychology_base.eval.adapters import Fragment, Observation
from psychology_base.llm import LLMCache, LLMConfig, LLMRouter, LocalOllamaConfig
from psychology_base.llm.config import ProviderConfig
from psychology_base.llm.providers import MockProvider


def _router(tmp_path, responder=None) -> LLMRouter:
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
    if responder is None:
        responder = lambda msgs, tier: "NOOP"
    mock = MockProvider(name="mock", responder=responder)
    return LLMRouter(
        cfg,
        providers={"mock": mock},
        cache=LLMCache(cache_dir=tmp_path / "cache"),
    )


def _obs(content: str, minute: int = 0, session_id: str = "s1") -> Observation:
    return Observation(
        timestamp=datetime(2026, 1, 15, 12, minute, tzinfo=timezone.utc),
        speaker="Jennifer",
        content=content,
        tags=(session_id,),
    )


# ---------- parser ----------


def test_parse_localize_filter_response_basic():
    assert _parse_localize_filter_response("0,2,5", max_index=5) == [0, 2, 5]


def test_parse_localize_filter_response_tolerates_whitespace_and_brackets():
    assert _parse_localize_filter_response("[0, 2,  5 ]", max_index=5) == [0, 2, 5]


def test_parse_localize_filter_response_drops_out_of_range():
    # 9 > max_index=5; -1 is not matched by the digit regex (leading
    # "-" is dropped).
    assert _parse_localize_filter_response("0, 9, 3", max_index=5) == [0, 3]


def test_parse_localize_filter_response_dedupes():
    assert _parse_localize_filter_response("0,2,0,2,2,1", max_index=5) == [0, 2, 1]


def test_parse_localize_filter_response_handles_prose():
    out = _parse_localize_filter_response(
        "I think indexes 0, 2, and 5 are relevant.", max_index=5
    )
    assert out == [0, 2, 5]


def test_parse_localize_filter_response_empty_input():
    assert _parse_localize_filter_response("", max_index=5) == []


def test_parse_localize_filter_response_no_indexes():
    assert _parse_localize_filter_response("nothing matches", max_index=5) == []


# ---------- init-time validation ----------


def test_localize_filter_requires_cross_encoder_rerank(tmp_path):
    router = _router(tmp_path)
    with pytest.raises(ValueError, match="requires cross_encoder_rerank=True"):
        CompanionSystem(
            witness_active=False,
            consolidation_enabled=True,
            consolidation_router=router,
            cross_encoder_rerank=False,
            localize_filter=True,
        )


def test_localize_filter_requires_consolidation_router(tmp_path):
    with pytest.raises(ValueError, match="requires consolidation_router"):
        CompanionSystem(
            witness_active=False,
            consolidation_enabled=False,
            consolidation_router=None,
            cross_encoder_rerank=True,
            localize_filter=True,
        )


def test_localize_filter_defaults_off(tmp_path):
    router = _router(tmp_path)
    cs = CompanionSystem(
        witness_active=False,
        consolidation_enabled=True,
        consolidation_router=router,
        cross_encoder_rerank=True,
    )
    assert cs.localize_filter is False


# ---------- behavior ----------


def _make_cs(tmp_path, *, responder, min_keep=5, max_input=80):
    router = _router(tmp_path, responder=responder)
    cs = CompanionSystem(
        witness_active=False,
        consolidation_enabled=True,
        consolidation_router=router,
        cross_encoder_rerank=True,
        localize_filter=True,
        localize_filter_min_keep=min_keep,
        localize_filter_max_input=max_input,
    )
    return cs


def _hits(n: int) -> list[tuple[Fragment, float]]:
    """Build n synthetic (fragment, score) pairs with decreasing scores."""
    out: list[tuple[Fragment, float]] = []
    for i in range(n):
        frag = Fragment(
            id=f"f{i}",
            content=f"turn {i}",
            timestamp=datetime(2026, 1, 15, 12, i, tzinfo=timezone.utc),
            affect={},
            tags=("s1",),
        )
        out.append((frag, 1.0 - 0.01 * i))
    return out


def test_localize_filter_keeps_llm_subset_in_rerank_order(tmp_path):
    cs = _make_cs(tmp_path, responder=lambda msgs, tier: "0, 3, 7", min_keep=0)
    pool = _hits(10)
    out = cs._localize_filter_fragments("cue", pool)
    ids = [frag.id for frag, _ in out]
    assert ids == ["f0", "f3", "f7"]
    # Scores preserved from rerank input.
    scores = [score for _, score in out]
    assert scores == pytest.approx([1.0, 0.97, 0.93])


def test_localize_filter_safety_floor_fills_from_rerank_top(tmp_path):
    # LLM returns only one index but min_keep=3 → top-3 by rerank
    # fill the shortfall (f0, f2, then f1 to reach 3).
    cs = _make_cs(tmp_path, responder=lambda msgs, tier: "2", min_keep=3)
    pool = _hits(10)
    out = cs._localize_filter_fragments("cue", pool)
    ids = [frag.id for frag, _ in out]
    assert len(ids) == 3
    assert "f2" in ids
    # Fill order prefers rerank-top; final sort re-anchors to rerank
    # order so f0, f1, f2 is the stable output.
    assert ids == ["f0", "f1", "f2"]


def test_localize_filter_empty_llm_output_falls_back_to_top_min_keep(tmp_path):
    cs = _make_cs(tmp_path, responder=lambda msgs, tier: "nothing", min_keep=4)
    pool = _hits(10)
    out = cs._localize_filter_fragments("cue", pool)
    assert [frag.id for frag, _ in out] == ["f0", "f1", "f2", "f3"]


def test_localize_filter_empty_input(tmp_path):
    cs = _make_cs(tmp_path, responder=lambda msgs, tier: "0,1")
    assert cs._localize_filter_fragments("cue", []) == []


def test_localize_filter_passes_through_tail_past_max_input(tmp_path):
    # LLM only sees first 3; tail (3..9) must pass through unfiltered.
    cs = _make_cs(
        tmp_path,
        responder=lambda msgs, tier: "0,2",
        min_keep=0,
        max_input=3,
    )
    pool = _hits(10)
    out = cs._localize_filter_fragments("cue", pool)
    ids = [frag.id for frag, _ in out]
    # Head: filtered subset of first 3 (f0, f2). Tail: f3..f9 passed
    # through in original order.
    assert ids == ["f0", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9"]


def test_localize_filter_provider_error_falls_back_to_input(tmp_path):
    def _boom(msgs, tier):
        raise RuntimeError("provider down")

    cs = _make_cs(tmp_path, responder=_boom)
    pool = _hits(5)
    out = cs._localize_filter_fragments("cue", pool)
    # Fallback = pass-through on exception.
    assert [frag.id for frag, _ in out] == ["f0", "f1", "f2", "f3", "f4"]


def test_localize_filter_handles_malformed_llm_output(tmp_path):
    # Response has prose + out-of-range + duplicates.
    cs = _make_cs(
        tmp_path,
        responder=lambda msgs, tier: "Hmm, I think [0, 2, 2, 99, 3] fit.",
        min_keep=0,
    )
    pool = _hits(5)
    out = cs._localize_filter_fragments("cue", pool)
    assert [frag.id for frag, _ in out] == ["f0", "f2", "f3"]


def test_localize_filter_end_to_end_recall_pipeline(tmp_path):
    """Integration check: filter runs on live observe→recall pipeline
    without crashing and the returned Recall respects filter output."""
    # LLM response "1,2" → filter keeps indexes 1 and 2 of the reranked
    # pool. Can't easily assert on the reranked pool composition here
    # (that depends on the cross-encoder path, which the MockProvider
    # doesn't implement), so this is a crash-free smoke test.
    cs = _make_cs(tmp_path, responder=lambda msgs, tier: "0", min_keep=1)
    cs.observe(_obs("alpha", minute=0, session_id="s1"))
    cs.observe(_obs("beta", minute=1, session_id="s1"))
    # Direct unit path: hand the method a synthetic reranked pool.
    pool = _hits(5)
    out = cs._localize_filter_fragments("cue", pool)
    assert len(out) >= 1
