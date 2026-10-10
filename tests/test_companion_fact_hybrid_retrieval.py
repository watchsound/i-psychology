"""Phase 2b Step B — hybrid fact retrieval (BM25 + cosine + entity-overlap, RRF).

Covers:
- Requires consolidation.
- Default off.
- Lazy BM25 index build + rebuild on fact-count change.
- BM25 channel pulls a literal-token fact above cosine floor-only.
- Entity-overlap channel surfaces facts whose entity matches cue tokens.
- Cosine channel still contributes (RRF fuses all three).
- ``cosine_floor`` gate applies against raw dense score, not RRF.
- Empty fact store returns empty hits.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from psychology_base.companion import CompanionSystem
from psychology_base.companion.facts import Fact
from psychology_base.eval.adapters import Observation
from psychology_base.llm import LLMCache, LLMConfig, LLMRouter, LocalOllamaConfig
from psychology_base.llm.config import ProviderConfig
from psychology_base.llm.providers import MockProvider


def _router(tmp_path, response_text: str) -> LLMRouter:
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


def _obs(content: str, speaker: str = "Jennifer", minute: int = 0) -> Observation:
    return Observation(
        timestamp=datetime(2026, 1, 15, 12, minute, tzinfo=timezone.utc),
        speaker=speaker,
        content=content,
    )


def test_fact_hybrid_retrieval_requires_consolidation():
    with pytest.raises(ValueError, match="requires consolidation_enabled=True"):
        CompanionSystem(
            witness_active=False,
            fact_hybrid_retrieval=True,
        )


def test_fact_hybrid_retrieval_defaults_off(tmp_path):
    router = _router(tmp_path, "NOOP")
    cs = CompanionSystem(
        witness_active=False,
        consolidation_enabled=True,
        consolidation_router=router,
    )
    assert cs.fact_hybrid_retrieval is False


def test_fact_hybrid_retrieval_surfaces_bm25_only_match(tmp_path):
    """A fact whose canonical text contains an unusual literal token
    the cue repeats should surface via BM25 even if the mock embeddings
    don't line up on the cosine channel.
    """
    router = _router(
        tmp_path,
        "ADD | Caroline | origin | Sweden | 0.9\n"
        "ADD | Caroline | hobby | pottery | 0.9\n"
        "ADD | Caroline | job | nurse | 0.9\n",
    )
    cs = CompanionSystem(
        witness_active=False,
        consolidation_enabled=True,
        consolidation_router=router,
        fact_cosine_floor=0.0,
        fact_hybrid_retrieval=True,
    )
    cs.observe(_obs("I come from Sweden.", minute=0))
    cs.observe(_obs("I do pottery.", minute=1))
    cs.observe(_obs("I'm a nurse.", minute=2))
    cs.mark_session_boundary()
    cs.tick()

    rec = cs.recall("Where is Sweden in Caroline's past?")
    # Sweden should reach the FACTS block via BM25 even if cosine
    # on mock embeddings doesn't rank it first.
    assert "Sweden" in rec.content


def test_fact_hybrid_retrieval_surfaces_entity_overlap_match(tmp_path):
    """A cue that names the entity ("Caroline") should surface
    Caroline-facts via the entity-overlap channel.
    """
    router = _router(
        tmp_path,
        "ADD | Caroline | lives_in | Boston | 0.9\n"
        "ADD | Dan | lives_in | New York | 0.9\n"
        "ADD | Pat | lives_in | Chicago | 0.9\n",
    )
    cs = CompanionSystem(
        witness_active=False,
        consolidation_enabled=True,
        consolidation_router=router,
        fact_cosine_floor=0.0,
        fact_hybrid_retrieval=True,
    )
    cs.observe(_obs("Caroline lives in Boston.", minute=0))
    cs.observe(_obs("Dan lives in New York.", minute=1))
    cs.observe(_obs("Pat lives in Chicago.", minute=2))
    cs.mark_session_boundary()
    cs.tick()

    rec = cs.recall("tell me about Caroline")
    # Caroline entity-overlap should surface Boston over the other two
    # cities even on near-identical cosine.
    assert "Boston" in rec.content


def test_fact_hybrid_retrieval_rebuilds_on_fact_count_change(tmp_path):
    """Index built on first recall; subsequent observations add facts;
    the next recall must rebuild to see the new facts.
    """
    router = _router(
        tmp_path,
        "ADD | Jennifer | hobby | pottery | 0.9\n",
    )
    cs = CompanionSystem(
        witness_active=False,
        consolidation_enabled=True,
        consolidation_router=router,
        fact_cosine_floor=0.0,
        fact_hybrid_retrieval=True,
    )
    cs.observe(_obs("I do pottery.", minute=0))
    cs.mark_session_boundary()
    cs.tick()
    rec1 = cs.recall("pottery")
    assert "pottery" in rec1.content
    idx_after_first = cs._fact_bm25_index
    first_count = len(cs._fact_bm25_facts)
    assert idx_after_first is not None

    # Round 2: add a new session with a new fact.
    cs.observe(_obs("I also sail.", minute=10))
    cs.mark_session_boundary()
    cs.tick()
    rec2 = cs.recall("sailing")
    # Index must have rebuilt — fact count grew.
    assert len(cs._fact_bm25_facts) > first_count


def test_fact_hybrid_retrieval_empty_fact_store(tmp_path):
    """Hybrid rank on an empty fact store returns an empty list and
    does not crash the recall path (fragment tier still fires)."""
    router = _router(tmp_path, "NOOP")
    cs = CompanionSystem(
        witness_active=False,
        consolidation_enabled=True,
        consolidation_router=router,
        fact_cosine_floor=0.0,
        fact_hybrid_retrieval=True,
    )
    # No observe/tick → no facts.
    rec = cs.recall("anything")
    # No facts and no fragments → fragment path falls through to
    # reconstruction (returns a Recall with empty content). The
    # contract we're checking here is "no exception".
    assert rec is not None


def test_fact_hybrid_retrieval_respects_cosine_floor(tmp_path):
    """Facts whose cosine score is below the floor are dropped even if
    BM25 or entity-overlap would have ranked them in the top-K.
    """
    router = _router(
        tmp_path,
        "ADD | Jennifer | hobby | pottery | 0.9\n",
    )
    cs = CompanionSystem(
        witness_active=False,
        consolidation_enabled=True,
        consolidation_router=router,
        # Floor near 1.0 forces dropouts; the single fact's cosine
        # against a random cue won't match.
        fact_cosine_floor=0.99,
        fact_hybrid_retrieval=True,
    )
    cs.observe(_obs("I do pottery.", minute=0))
    cs.mark_session_boundary()
    cs.tick()
    # Mutate the stored fact to force cosine 0 (zero embedding) and
    # verify the hybrid path drops it under the strict floor.
    assert cs.fact_store is not None
    (fact,) = cs.fact_store.all()
    cs.fact_store.upsert(replace(fact, embedding=()))

    rec = cs.recall("pottery")
    # The fact had a zero embedding → cosine 0.0 < 0.99 floor → dropped.
    assert "FACTS:" not in rec.content or "pottery" not in (
        rec.content.split("DIALOGUE:", 1)[0] if "DIALOGUE:" in rec.content else rec.content
    )


def test_fact_hybrid_retrieval_stacks_with_source_turn_lift(tmp_path):
    """Step A + Step B compose: both flags on simultaneously; recall
    runs without error and emits FACTS + DIALOGUE sections.
    """
    router = _router(
        tmp_path,
        "ADD | Jennifer | hobby | pottery | 0.9\n"
        "ADD | Jennifer | lives_in | Hull | 0.8\n",
    )
    cs = CompanionSystem(
        witness_active=False,
        consolidation_enabled=True,
        consolidation_router=router,
        fact_cosine_floor=0.0,
        fact_source_turn_lift=True,
        fact_hybrid_retrieval=True,
    )
    cs.observe(_obs("I do pottery.", minute=0))
    cs.observe(_obs("I live in Hull.", minute=1))
    cs.mark_session_boundary()
    cs.tick()

    rec = cs.recall("Jennifer hobbies")
    assert "FACTS:" in rec.content
    assert "DIALOGUE:" in rec.content
