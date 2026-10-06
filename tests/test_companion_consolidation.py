"""Phase 1.5 build step 4 integration — CompanionSystem × Consolidator.

Per [phase-1.5-build-plan.md](../docs/phases/phase-1.5-build-plan.md) §5
step 4:

- ``consolidation_enabled=True`` requires a router; absence raises.
- ``observe`` feeds both fragment store AND consolidator window.
- ``mark_session_boundary`` + ``tick`` triggers extraction on the window.
- ``recall`` dispatches to the fact path when ≥ ``MIN_FACTS_FOR_FACT_PATH``
  facts clear the cosine floor; backs off to Phase-1 fragment
  reconstruction otherwise.
- ``inspect`` reports fact count + consolidation stats when enabled.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from psychology_base.companion import CompanionSystem
from psychology_base.eval.adapters import Observation
from psychology_base.llm import LLMCache, LLMConfig, LLMRouter, LocalOllamaConfig
from psychology_base.llm.config import ProviderConfig
from psychology_base.llm.providers import MockProvider


def _router(tmp_path, response_text: str) -> tuple[LLMRouter, MockProvider]:
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


def _obs(content: str, speaker: str = "Jennifer", minute: int = 0) -> Observation:
    return Observation(
        timestamp=datetime(2026, 1, 15, 12, minute, tzinfo=timezone.utc),
        speaker=speaker,
        content=content,
    )


def test_requires_router_when_enabled():
    with pytest.raises(ValueError, match="consolidation_router"):
        CompanionSystem(consolidation_enabled=True)


def test_default_off_leaves_system_phase_1_shaped(tmp_path):
    cs = CompanionSystem(witness_active=False)
    cs.observe(_obs("hello"))
    rec = cs.recall("hello")
    assert cs.fact_store is None
    assert cs.consolidator is None
    assert rec.metadata["path"] == "fragment"
    assert rec.metadata["fact_count"] == 0
    info = cs.inspect()
    assert info["consolidation_enabled"] is False
    assert "fact_count" not in info


def test_boundary_then_tick_extracts_facts(tmp_path):
    router, mock = _router(
        tmp_path,
        "ADD | Jennifer | owns | Westerly 28 | 0.9\n"
        "ADD | Jennifer | lives_in | Hull | 0.8\n",
    )
    cs = CompanionSystem(
        witness_active=False,
        consolidation_enabled=True,
        consolidation_router=router,
    )
    cs.observe(_obs("I bought a Westerly 28", minute=0))
    cs.observe(_obs("I live in Hull", minute=1))

    assert len(cs.fact_store) == 0  # no boundary yet

    cs.mark_session_boundary()
    cs.tick()  # sleep tick runs maybe_consolidate

    assert len(cs.fact_store) == 2
    assert cs.heartbeat.session_boundaries == 1
    assert cs.consolidator.stats.adds == 2
    assert len(mock.embed_calls) >= 2


def test_recall_uses_fact_path_when_three_facts_match(tmp_path):
    router, _ = _router(
        tmp_path,
        "ADD | Jennifer | owns | Westerly 28 | 0.9\n"
        "ADD | Jennifer | lives_in | Hull | 0.8\n"
        "ADD | David | works_at | boatyard | 0.7\n",
    )
    cs = CompanionSystem(
        witness_active=False,
        consolidation_enabled=True,
        consolidation_router=router,
        fact_cosine_floor=0.0,  # mock embeddings are random hashes;
        # floor=0 so top-K hits deterministically meet the threshold
    )
    cs.observe(_obs("I bought a Westerly 28", minute=0))
    cs.observe(_obs("I live in Hull", minute=1))
    cs.observe(_obs("David works at the boatyard", speaker="David", minute=2))
    cs.mark_session_boundary()
    cs.tick()

    rec = cs.recall("tell me about Jennifer")
    # Option 5.C: when consolidation is on, recall is always hierarchical
    # (facts + cosine-ranked raw fragments both in the reader context).
    # The old "fact-only" path was retired because its empty-handed
    # back-off to Phase-1 Jaccard reconstruction surrendered the
    # extraction-coverage gap to a weaker substrate.
    assert rec.metadata["path"] == "hierarchical"
    assert rec.metadata["fact_count"] == 3
    assert len(rec.metadata["facts_used"]) == 3
    assert "FACTS:" in rec.content
    assert "DIALOGUE:" in rec.content
    assert "Westerly 28" in rec.content
    assert "Hull" in rec.content
    assert "boatyard" in rec.content


def test_recall_backs_off_when_fewer_than_three_facts_pass(tmp_path):
    router, _ = _router(
        tmp_path,
        "ADD | Jennifer | owns | Westerly 28 | 0.9\n"
        "ADD | Jennifer | lives_in | Hull | 0.8\n",
    )
    cs = CompanionSystem(
        witness_active=False,
        consolidation_enabled=True,
        consolidation_router=router,
        fact_cosine_floor=0.0,
    )
    cs.observe(_obs("I bought a Westerly 28", minute=0))
    cs.observe(_obs("I live in Hull", minute=1))
    cs.mark_session_boundary()
    cs.tick()

    assert len(cs.fact_store) == 2
    rec = cs.recall("where does Jennifer live?")
    # Option 5.C: even when fewer than MIN_FACTS_FOR_FACT_PATH facts
    # clear the floor, the hierarchical path still composes the raw
    # fragment tier for the reader. Only an entirely empty store would
    # return None and fall back to Phase-1 Jaccard reconstruction.
    assert rec.metadata["path"] == "hierarchical"
    assert rec.metadata["fact_count"] == 2
    assert "DIALOGUE:" in rec.content


def test_recall_backs_off_when_fact_store_empty(tmp_path):
    router, _ = _router(tmp_path, "ADD | e | r | v | 0.5\n")
    cs = CompanionSystem(
        witness_active=False,
        consolidation_enabled=True,
        consolidation_router=router,
    )
    cs.observe(_obs("hello"))
    # No boundary, no tick → consolidator window pending, no facts.
    # Still get the hierarchical path because the fragment tier has
    # content (a single turn), the fact tier is empty.
    rec = cs.recall("hello")
    assert rec.metadata["path"] == "hierarchical"
    assert rec.metadata["fact_count"] == 0
    assert "DIALOGUE:" in rec.content
    assert "FACTS:" not in rec.content


def test_recall_fact_path_resolves_fragment_back_pointers(tmp_path):
    router, _ = _router(
        tmp_path,
        "ADD | Jennifer | owns | Westerly 28 | 0.9\n"
        "ADD | Jennifer | lives_in | Hull | 0.8\n"
        "ADD | Jennifer | sails_in | Solent | 0.7\n",
    )
    cs = CompanionSystem(
        witness_active=False,
        consolidation_enabled=True,
        consolidation_router=router,
        fact_cosine_floor=0.0,
    )
    cs.observe(_obs("I own a Westerly 28", minute=0))
    cs.observe(_obs("I live in Hull", minute=1))
    cs.observe(_obs("I sail in the Solent", minute=2))
    cs.mark_session_boundary()
    cs.tick()

    rec = cs.recall("Jennifer")
    # Each ADD's source_fragment_ids is the full window → the three
    # observed fragments resolve back for the Phase-1 reader.
    assert len(rec.fragments_used) == 3
    frag_contents = [f.content for f in rec.fragments_used]
    assert "I own a Westerly 28" in frag_contents


def test_inspect_reports_consolidation_stats(tmp_path):
    router, _ = _router(
        tmp_path,
        "ADD | Jennifer | owns | Westerly 28 | 0.9\n",
    )
    cs = CompanionSystem(
        witness_active=False,
        consolidation_enabled=True,
        consolidation_router=router,
    )
    cs.observe(_obs("I own a Westerly 28"))
    cs.mark_session_boundary()
    cs.tick()

    info = cs.inspect()
    assert info["consolidation_enabled"] is True
    assert info["fact_count"] == 1
    assert info["consolidation_stats"]["adds"] == 1
    assert info["consolidation_stats"]["updates"] == 0
    assert info["session_boundaries"] == 1


def test_tick_without_boundary_does_not_consolidate_small_window(tmp_path):
    """tick() runs maybe_consolidate every call, but the Consolidator
    skips extraction when there's no boundary and the window is under
    the fallback threshold. Guarding the discipline that extraction is
    cheap (zero LLM calls) when nothing changed."""
    router, mock = _router(tmp_path, "ADD | e | r | v | 0.5\n")
    cs = CompanionSystem(
        witness_active=False,
        consolidation_enabled=True,
        consolidation_router=router,
    )
    cs.observe(_obs("one message"))
    cs.tick()  # no boundary, window=1 < fallback=20 → no extraction
    assert len(mock.calls) == 0
    assert cs.consolidator.stats.adds == 0
