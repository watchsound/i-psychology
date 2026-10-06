"""Phase 1.5 build step 2+3 integration — Consolidator orchestration.

Per docs/phases/phase-1.5-build-plan.md §5:
  - Session-boundary trigger consolidates the pending window on next
    tick (not on observe — sleep work happens on sleep)
  - Fallback window (N observations) triggers when no boundary signal
  - UPDATE with unknown target_fact_id drops safely (counted)
  - NOOP refreshes updated_tick on the prior fact
  - Facts are embedded on write; empty window → no LLM call
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from psychology_base.companion.consolidation import (
    DEFAULT_FALLBACK_WINDOW,
    Consolidator,
)
from psychology_base.companion.facts import Fact, FactStore
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


def test_boundary_triggers_consolidation(tmp_path):
    router, mock = _router(
        tmp_path,
        "ADD | Jennifer | owns | Westerly 28 | 0.9\n"
        "ADD | David | lives_in | Hull | 0.7\n",
    )
    store = FactStore()
    c = Consolidator(router=router, store=store)

    c.observe(_frag("frag-0001", "I bought a Westerly 28"))
    c.observe(_frag("frag-0002", "I live in Hull", speaker="David"))

    # No boundary yet → no consolidation
    stats = c.maybe_consolidate(tick_count=1)
    assert stats.total() == 0
    assert len(store) == 0

    # Mark boundary → next tick consolidates
    c.mark_session_boundary()
    stats = c.maybe_consolidate(tick_count=2)
    assert stats.adds == 2
    assert len(store) == 2
    assert len(mock.embed_calls) == 2


def test_fallback_window_triggers_without_boundary(tmp_path):
    router, _ = _router(
        tmp_path,
        "ADD | user | said | hello | 0.5\n",
    )
    store = FactStore()
    c = Consolidator(router=router, store=store, fallback_window=3)
    for i in range(2):
        c.observe(_frag(f"frag-000{i}", f"msg {i}"))
    assert c.maybe_consolidate(tick_count=1).total() == 0

    c.observe(_frag("frag-0002", "msg 2"))
    stats = c.maybe_consolidate(tick_count=2)
    assert stats.adds == 1


def test_fallback_window_default_matches_build_plan():
    assert DEFAULT_FALLBACK_WINDOW == 20


def test_update_drops_when_target_absent(tmp_path):
    router, _ = _router(
        tmp_path,
        "UPDATE | fact-9999 | Jennifer | owns | Westerly 28 | 0.9\n",
    )
    store = FactStore()
    c = Consolidator(router=router, store=store)
    c.observe(_frag("frag-0001", "Jennifer bought a Westerly"))
    c.mark_session_boundary()
    stats = c.maybe_consolidate(tick_count=1)
    assert stats.updates == 0
    assert stats.dropped == 1
    assert len(store) == 0


def test_noop_refreshes_updated_tick(tmp_path):
    router, _ = _router(
        tmp_path,
        "NOOP | fact-0001\n",
    )
    store = FactStore()
    seed = store.add(
        Fact(
            id="",
            entity="Jennifer",
            relation="owns",
            value="boat",
            confidence=0.9,
            source_fragment_ids=("frag-seed",),
            created_tick=0,
            updated_tick=0,
        )
    )
    assert seed.id == "fact-0001"

    c = Consolidator(router=router, store=store)
    c.observe(_frag("frag-0002", "Jennifer still has the boat"))
    c.mark_session_boundary()
    stats = c.maybe_consolidate(tick_count=5)
    assert stats.noops == 1
    assert stats.updates == 0
    refreshed = store.by_id("fact-0001")
    assert refreshed.updated_tick == 5
    assert refreshed.value == "boat"  # unchanged


def test_update_replaces_fact_in_place(tmp_path):
    router, _ = _router(
        tmp_path,
        "UPDATE | fact-0001 | Jennifer | owns | Westerly 28 | 0.95\n",
    )
    store = FactStore()
    store.add(
        Fact(
            id="",
            entity="Jennifer",
            relation="owns",
            value="sailboat",
            confidence=0.7,
            source_fragment_ids=("frag-seed",),
            created_tick=0,
            updated_tick=0,
        )
    )
    c = Consolidator(router=router, store=store)
    c.observe(_frag("frag-0002", "specifically a Westerly 28"))
    c.mark_session_boundary()
    stats = c.maybe_consolidate(tick_count=3)
    assert stats.updates == 1
    assert len(store) == 1
    refreshed = store.by_id("fact-0001")
    assert refreshed.value == "Westerly 28"
    assert refreshed.confidence == 0.95
    assert refreshed.updated_tick == 3
    assert refreshed.embedding != ()  # embedded on upsert


def test_no_pending_fragments_no_llm_call(tmp_path):
    router, mock = _router(tmp_path, "ADD | e | r | v | 0.5\n")
    store = FactStore()
    c = Consolidator(router=router, store=store)
    c.mark_session_boundary()
    stats = c.maybe_consolidate(tick_count=1)
    assert stats.total() == 0
    assert len(mock.calls) == 0


def test_logs_write_to_disk_when_log_dir_set(tmp_path):
    router, _ = _router(
        tmp_path,
        "ADD | Jennifer | owns | Westerly 28 | 0.9\n",
    )
    store = FactStore()
    log_dir = tmp_path / "fact-logs"
    c = Consolidator(router=router, store=store, log_dir=log_dir, instance_id="conv-26")
    c.observe(_frag("frag-0001", "I bought a Westerly"))
    c.mark_session_boundary()
    c.maybe_consolidate(tick_count=1)

    log_path = log_dir / "conv-26.jsonl"
    assert log_path.exists()
    with log_path.open(encoding="utf-8") as fh:
        lines = fh.readlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["op"] == "ADD"
    assert entry["fact"]["entity"] == "Jennifer"


def test_stats_accumulate_across_calls(tmp_path):
    """A long dialogue consolidates many windows; cumulative stats on the
    Consolidator should sum each pass."""
    router, _ = _router(tmp_path, "ADD | e | r | v | 0.5\n")
    store = FactStore()
    c = Consolidator(router=router, store=store, fallback_window=1)
    c.observe(_frag("frag-0001", "a"))
    c.maybe_consolidate(tick_count=1)
    c.observe(_frag("frag-0002", "b"))
    c.maybe_consolidate(tick_count=2)
    assert c.stats.adds == 2
    assert len(store) == 2
