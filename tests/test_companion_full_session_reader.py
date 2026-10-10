"""Phase 2b Step F — full-session reader context tests (ADDITIVE).

Covers the ADDITIVE variant of Step F (2026-10-07): the method keeps
every reranked top-K fragment AND unions in every un-reranked turn of
the top-1 winning session at the tail (chronologically ordered,
score=0.0 for session-only turns).

The earlier REPLACE variant (replacing fragment_hits with the top-1
session's turn set) regressed by 0.047 F1 across 3 instances — top-1
session was often not the right session, so REPLACE dropped correct
cross-session evidence. These tests pin the ADDITIVE invariants.

Covers:
- ``full_session_reader=True`` requires ``cross_encoder_rerank=True``.
- Mutex with ``within_session_rerank``.
- Default off (behavior unchanged for existing configs).
- Winning session = session_id of the top-1 reranked fragment.
- Reranked top-K is PRESERVED (no evidence loss), AND winning-session
  turns not in reranked input are UNIONED at the tail.
- Order: reranked hits first (preserving cross-encoder rank), then
  session-only extras in chronological order.
- Reranked fragments keep their rerank score; session-only extras get 0.0.
- Cap: ``full_session_reader_cap_turns`` bounds the TOTAL pool size;
  session-only extras are trimmed first (reranked hits always survive).
- Empty input returns empty.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from psychology_base.companion import CompanionSystem
from psychology_base.eval.adapters import Observation
from psychology_base.llm import LLMCache, LLMConfig, LLMRouter, LocalOllamaConfig
from psychology_base.llm.config import ProviderConfig
from psychology_base.llm.providers import MockProvider


def _router(tmp_path) -> LLMRouter:
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
    mock = MockProvider(name="mock", responder=lambda msgs, tier: "NOOP")
    return LLMRouter(
        cfg,
        providers={"mock": mock},
        cache=LLMCache(cache_dir=tmp_path / "cache"),
    )


def _obs(content: str, minute: int = 0, session_id: str = "session_1") -> Observation:
    return Observation(
        timestamp=datetime(2026, 1, 15, 12, minute, tzinfo=timezone.utc),
        speaker="Jennifer",
        content=content,
        tags=(session_id,),
    )


def test_full_session_reader_requires_cross_encoder_rerank(tmp_path):
    router = _router(tmp_path)
    with pytest.raises(ValueError, match="requires cross_encoder_rerank=True"):
        CompanionSystem(
            witness_active=False,
            consolidation_enabled=True,
            consolidation_router=router,
            cross_encoder_rerank=False,
            full_session_reader=True,
        )


def test_full_session_reader_mutex_with_within_session_rerank(tmp_path):
    router = _router(tmp_path)
    with pytest.raises(ValueError, match="mutually exclusive"):
        CompanionSystem(
            witness_active=False,
            consolidation_enabled=True,
            consolidation_router=router,
            cross_encoder_rerank=True,
            within_session_rerank=True,
            full_session_reader=True,
        )


def test_full_session_reader_defaults_off(tmp_path):
    router = _router(tmp_path)
    cs = CompanionSystem(
        witness_active=False,
        consolidation_enabled=True,
        consolidation_router=router,
        cross_encoder_rerank=True,
    )
    assert cs.full_session_reader is False
    assert cs.full_session_reader_cap_turns == 300


def test_full_session_reader_preserves_reranked_and_adds_session_extras(tmp_path):
    """ADDITIVE core win condition: every reranked hit survives, AND
    un-reranked turns of the winning session are unioned at the tail.
    No evidence loss from reranked top-K.
    """
    router = _router(tmp_path)
    cs = CompanionSystem(
        witness_active=False,
        consolidation_enabled=True,
        consolidation_router=router,
        cross_encoder_rerank=True,
        full_session_reader=True,
    )
    # Session sA has 3 turns, sB has 2 turns.
    cs.observe(_obs("sA turn 1 Sweden", minute=0, session_id="sA"))
    cs.observe(_obs("sA turn 2 filler", minute=1, session_id="sA"))
    cs.observe(_obs("sA turn 3 gold", minute=2, session_id="sA"))
    cs.observe(_obs("sB turn 1 Boston", minute=3, session_id="sB"))
    cs.observe(_obs("sB turn 2 filler", minute=4, session_id="sB"))

    frags = cs.companion.store.all()
    by_content = {f.content: f for f in frags}

    # Reranked hits: two turns from sA + one from sB (top-1 is sA).
    reranked = [
        (by_content["sA turn 1 Sweden"], 0.9),
        (by_content["sB turn 1 Boston"], 0.8),
        (by_content["sA turn 3 gold"], 0.7),
    ]

    out = cs._full_session_fragments(reranked)
    contents = [frag.content for frag, _ in out]

    # Every reranked hit survives (including the cross-session sB one).
    assert "sA turn 1 Sweden" in contents
    assert "sB turn 1 Boston" in contents
    assert "sA turn 3 gold" in contents
    # Winning session (sA) extra turn gets pulled in.
    assert "sA turn 2 filler" in contents
    # sB non-reranked turn stays out (not winning session).
    assert "sB turn 2 filler" not in contents
    # Total: 3 reranked + 1 session extra.
    assert len(out) == 4


def test_full_session_reader_orders_reranked_first_then_session_chronological(tmp_path):
    """Order invariant: reranked hits preserve the cross-encoder's rank
    order up front (not re-sorted), then session-only extras follow in
    chronological order at the tail."""
    router = _router(tmp_path)
    cs = CompanionSystem(
        witness_active=False,
        consolidation_enabled=True,
        consolidation_router=router,
        cross_encoder_rerank=True,
        full_session_reader=True,
    )
    cs.observe(_obs("earliest turn", minute=0, session_id="sX"))
    cs.observe(_obs("middle turn A", minute=1, session_id="sX"))
    cs.observe(_obs("middle turn B", minute=2, session_id="sX"))
    cs.observe(_obs("latest turn", minute=3, session_id="sX"))

    frags = cs.companion.store.all()
    by_content = {f.content: f for f in frags}

    # Reranked order deliberately scrambled vs chronology.
    reranked = [
        (by_content["latest turn"], 0.9),
        (by_content["earliest turn"], 0.5),
    ]

    out = cs._full_session_fragments(reranked)
    contents = [frag.content for frag, _ in out]

    # First two preserve rerank order.
    assert contents[:2] == ["latest turn", "earliest turn"]
    # Tail = session-only extras in chronological order.
    assert contents[2:] == ["middle turn A", "middle turn B"]


def test_full_session_reader_reuses_rerank_scores_where_available(tmp_path):
    """Reranked hits carry their rerank score; session-only extras
    (not in reranked input) get 0.0."""
    router = _router(tmp_path)
    cs = CompanionSystem(
        witness_active=False,
        consolidation_enabled=True,
        consolidation_router=router,
        cross_encoder_rerank=True,
        full_session_reader=True,
    )
    cs.observe(_obs("sA turn 1", minute=0, session_id="sA"))
    cs.observe(_obs("sA turn 2 (not in reranked)", minute=1, session_id="sA"))

    frags = cs.companion.store.all()
    by_content = {f.content: f for f in frags}
    reranked = [(by_content["sA turn 1"], 0.9)]

    out = cs._full_session_fragments(reranked)
    scores = {frag.content: score for frag, score in out}
    assert scores["sA turn 1"] == pytest.approx(0.9)
    assert scores["sA turn 2 (not in reranked)"] == pytest.approx(0.0)


def test_full_session_reader_caps_total_pool(tmp_path):
    """Cap bounds the TOTAL pool. Reranked hits always survive; the
    session-only extras budget is ``cap - len(reranked_hits)``, clamped
    to zero. Caller sets the cap tight to exercise the branch."""
    router = _router(tmp_path)
    cs = CompanionSystem(
        witness_active=False,
        consolidation_enabled=True,
        consolidation_router=router,
        cross_encoder_rerank=True,
        full_session_reader=True,
        full_session_reader_cap_turns=2,
    )
    cs.observe(_obs("sA t1", minute=0, session_id="sA"))
    cs.observe(_obs("sA t2", minute=1, session_id="sA"))
    cs.observe(_obs("sA t3", minute=2, session_id="sA"))
    cs.observe(_obs("sA t4", minute=3, session_id="sA"))

    frags = cs.companion.store.all()
    by_content = {f.content: f for f in frags}
    # One reranked hit → budget for session extras = 2 - 1 = 1.
    reranked = [(by_content["sA t1"], 0.9)]
    out = cs._full_session_fragments(reranked)
    assert len(out) == 2
    contents = [frag.content for frag, _ in out]
    assert contents[0] == "sA t1"  # reranked hit stays first
    # The 1 session-extra slot is filled by the earliest non-reranked
    # chronological turn.
    assert contents[1] == "sA t2"


def test_full_session_reader_cap_zero_budget_still_keeps_reranked(tmp_path):
    """When ``cap <= len(reranked_hits)``, session-extra budget is 0 and
    no extras are added — but every reranked hit still survives."""
    router = _router(tmp_path)
    cs = CompanionSystem(
        witness_active=False,
        consolidation_enabled=True,
        consolidation_router=router,
        cross_encoder_rerank=True,
        full_session_reader=True,
        full_session_reader_cap_turns=2,
    )
    cs.observe(_obs("sA t1", minute=0, session_id="sA"))
    cs.observe(_obs("sA t2", minute=1, session_id="sA"))
    cs.observe(_obs("sA t3", minute=2, session_id="sA"))

    frags = cs.companion.store.all()
    by_content = {f.content: f for f in frags}
    # Two reranked hits → budget = 0.
    reranked = [
        (by_content["sA t1"], 0.9),
        (by_content["sA t2"], 0.7),
    ]
    out = cs._full_session_fragments(reranked)
    assert len(out) == 2
    assert [frag.content for frag, _ in out] == ["sA t1", "sA t2"]


def test_full_session_reader_empty_input_returns_empty(tmp_path):
    router = _router(tmp_path)
    cs = CompanionSystem(
        witness_active=False,
        consolidation_enabled=True,
        consolidation_router=router,
        cross_encoder_rerank=True,
        full_session_reader=True,
    )
    out = cs._full_session_fragments([])
    assert out == []


def test_full_session_reader_untagged_winning_fragment(tmp_path):
    """If the top-1 reranked fragment has no session tag, the winning
    "session" is the untagged bucket. Reranked hits survive AND untagged
    non-reranked turns are unioned. Tagged non-reranked turns stay out."""
    router = _router(tmp_path)
    cs = CompanionSystem(
        witness_active=False,
        consolidation_enabled=True,
        consolidation_router=router,
        cross_encoder_rerank=True,
        full_session_reader=True,
    )
    cs.observe(
        Observation(
            timestamp=datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc),
            speaker="Jennifer",
            content="untagged turn A",
            tags=(),
        )
    )
    cs.observe(
        Observation(
            timestamp=datetime(2026, 1, 15, 12, 1, tzinfo=timezone.utc),
            speaker="Jennifer",
            content="untagged turn B",
            tags=(),
        )
    )
    cs.observe(_obs("tagged turn", minute=2, session_id="sA"))

    frags = cs.companion.store.all()
    by_content = {f.content: f for f in frags}
    # Top-1 is untagged → winning "session" = empty-tag bucket.
    # Reranked includes "tagged turn" too (which survives); session
    # extras adds "untagged turn B".
    reranked = [
        (by_content["untagged turn A"], 0.9),
        (by_content["tagged turn"], 0.7),
    ]
    out = cs._full_session_fragments(reranked)
    contents = {frag.content for frag, _ in out}
    assert contents == {"untagged turn A", "tagged turn", "untagged turn B"}
