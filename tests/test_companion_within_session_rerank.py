"""Phase 2 R8-c — within-session two-stage rerank tests.

Covers:
- ``within_session_rerank=True`` requires ``cross_encoder_rerank=True``.
- ``within_session_rerank`` is mutually exclusive with
  ``cross_encoder_rerank_all``.
- Session grouping uses ``Fragment.tags[0]`` and picks top-M sessions by
  max-cosine score across the cosine top-K pool.
- The reranker is called on ALL turns of the top-M sessions, not just the
  turns that appeared in the cosine input pool — this is the whole point
  of R8-c (lifting the gold turn that cosine buried past the top-150
  within its session).
- Fragments with no session_id tag are passed through as an "unsessioned"
  bucket and can still compete.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from psychology_base.companion import CompanionSystem
from psychology_base.eval.adapters import Observation
from psychology_base.llm import LLMCache, LLMConfig, LLMRouter, LocalOllamaConfig
from psychology_base.llm.config import ProviderConfig
from psychology_base.llm.providers import MockProvider


def _router(tmp_path, response_text: str = "NOOP") -> LLMRouter:
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


def test_within_session_rerank_requires_cross_encoder_rerank(tmp_path):
    router = _router(tmp_path)
    with pytest.raises(ValueError, match="requires cross_encoder_rerank=True"):
        CompanionSystem(
            witness_active=False,
            consolidation_enabled=True,
            consolidation_router=router,
            cross_encoder_rerank=False,
            within_session_rerank=True,
        )


def test_within_session_rerank_mutex_with_rerank_all(tmp_path):
    router = _router(tmp_path)
    with pytest.raises(ValueError, match="mutually exclusive"):
        CompanionSystem(
            witness_active=False,
            consolidation_enabled=True,
            consolidation_router=router,
            cross_encoder_rerank=True,
            cross_encoder_rerank_all=True,
            within_session_rerank=True,
        )


def test_within_session_rerank_defaults_off(tmp_path):
    router = _router(tmp_path)
    cs = CompanionSystem(
        witness_active=False,
        consolidation_enabled=True,
        consolidation_router=router,
        cross_encoder_rerank=True,
    )
    assert cs.within_session_rerank is False
    assert cs.within_session_top_m_sessions == 3  # default


def test_within_session_rerank_picks_top_m_sessions_by_max_cosine(tmp_path):
    """Session scoring: max cosine of any fragment in the session wins.
    Top-M sessions are the pool for cross-encoder rerank; everything else
    is excluded even if it ranked in cosine top-150.
    """
    router = _router(tmp_path)
    cs = CompanionSystem(
        witness_active=False,
        consolidation_enabled=True,
        consolidation_router=router,
        cross_encoder_rerank=True,
        within_session_rerank=True,
        within_session_top_m_sessions=2,
    )

    # 3 sessions × 2 turns = 6 fragments total.
    cs.observe(_obs("session A turn 1 Sweden", minute=0, session_id="sA"))
    cs.observe(_obs("session A turn 2 filler", minute=1, session_id="sA"))
    cs.observe(_obs("session B turn 1 Boston", minute=2, session_id="sB"))
    cs.observe(_obs("session B turn 2 filler", minute=3, session_id="sB"))
    cs.observe(_obs("session C turn 1 Mars", minute=4, session_id="sC"))
    cs.observe(_obs("session C turn 2 filler", minute=5, session_id="sC"))

    frags = cs.companion.store.all()
    by_content = {f.content: f for f in frags}

    # Cosine ranked only ONE turn per session into the top-K (sA's turn 1,
    # sB's turn 1, sC's turn 1). Scores decrease across sessions. The
    # session-max picker should pick top-M=2 → {sA, sB}, dropping sC.
    cosine_hits = [
        (by_content["session A turn 1 Sweden"], 0.9),
        (by_content["session B turn 1 Boston"], 0.7),
        (by_content["session C turn 1 Mars"], 0.5),
    ]

    # Mock reranker that returns the pool unchanged in original order,
    # score = 1.0 - idx*0.01, so we can assert which fragments entered
    # the pool.
    captured_pool: list[list[str]] = []

    def fake_rerank(query, documents, *, top_n=None):
        captured_pool.append(list(documents))
        return [(i, 1.0 - i * 0.01) for i in range(len(documents))]

    cs._consolidation_router.rerank = fake_rerank  # type: ignore[assignment]

    reranked = cs._within_session_rerank_fragments("cue", cosine_hits)

    # Pool should be ALL 4 turns from sA + sB (both turns of each),
    # NOT the 2 cosine-ranked turns, NOT sC's turns.
    assert len(captured_pool) == 1
    pool_contents = set(captured_pool[0])
    assert pool_contents == {
        "session A turn 1 Sweden",
        "session A turn 2 filler",
        "session B turn 1 Boston",
        "session B turn 2 filler",
    }, f"got {pool_contents}"

    # The "session A turn 2 filler" is the critical fragment: it was NOT
    # in the cosine input (cosine_hits), but IS in the rerank pool. This
    # is the R8-c win condition — within-session reach widens past
    # cosine's top-K.
    reranked_contents = {frag.content for frag, _ in reranked}
    assert "session A turn 2 filler" in reranked_contents


def test_within_session_rerank_handles_untagged_fragments(tmp_path):
    """Fragments with empty tags bucket together as "unsessioned" and
    can still compete for one of the top-M slots.
    """
    router = _router(tmp_path)
    cs = CompanionSystem(
        witness_active=False,
        consolidation_enabled=True,
        consolidation_router=router,
        cross_encoder_rerank=True,
        within_session_rerank=True,
        within_session_top_m_sessions=1,  # only 1 session wins
    )

    cs.observe(
        Observation(
            timestamp=datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc),
            speaker="Jennifer",
            content="untagged turn with Sweden",
            tags=(),  # no session_id
        )
    )
    cs.observe(_obs("tagged turn in sA", minute=1, session_id="sA"))

    frags = cs.companion.store.all()
    by_content = {f.content: f for f in frags}

    # Cosine ranks untagged turn higher than tagged turn — the unsessioned
    # bucket should win the only top-M=1 slot.
    cosine_hits = [
        (by_content["untagged turn with Sweden"], 0.95),
        (by_content["tagged turn in sA"], 0.5),
    ]

    captured_pool: list[list[str]] = []

    def fake_rerank(query, documents, *, top_n=None):
        captured_pool.append(list(documents))
        return [(i, 1.0 - i * 0.01) for i in range(len(documents))]

    cs._consolidation_router.rerank = fake_rerank  # type: ignore[assignment]

    cs._within_session_rerank_fragments("cue", cosine_hits)

    assert captured_pool == [["untagged turn with Sweden"]]


def test_within_session_rerank_empty_cosine_hits_returns_empty(tmp_path):
    router = _router(tmp_path)
    cs = CompanionSystem(
        witness_active=False,
        consolidation_enabled=True,
        consolidation_router=router,
        cross_encoder_rerank=True,
        within_session_rerank=True,
    )
    out = cs._within_session_rerank_fragments("cue", [])
    assert out == []
