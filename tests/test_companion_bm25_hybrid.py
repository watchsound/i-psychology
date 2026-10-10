"""Phase 2 R2 — BM25 + semantic hybrid recall tests.

Covers:
- ``_bm25_tokenize`` basic shape (lowercase, split, digits preserved).
- ``bm25_hybrid=True`` builds a BM25 index on first hybrid recall and
  invalidates on new ``observe``.
- RRF fusion ranks a proper-noun-match fragment ahead of a semantically-
  closer but token-mismatched fragment under adversarial cue phrasing.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from psychology_base.companion import CompanionSystem
from psychology_base.companion.system import _bm25_tokenize
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


def _obs(content: str, minute: int = 0) -> Observation:
    return Observation(
        timestamp=datetime(2026, 1, 15, 12, minute, tzinfo=timezone.utc),
        speaker="Jennifer",
        content=content,
    )


def test_bm25_tokenize_lowercases_and_preserves_digits():
    toks = _bm25_tokenize("Moved from Sweden on 19 October 2023!")
    assert toks == ["moved", "from", "sweden", "on", "19", "october", "2023"]


def test_bm25_tokenize_handles_punctuation_and_underscores():
    toks = _bm25_tokenize("single_parent, Charlotte's Web.")
    assert toks == ["single", "parent", "charlotte", "s", "web"]


def test_bm25_hybrid_defaults_off_and_leaves_bm25_index_unbuilt(tmp_path):
    router = _router(tmp_path)
    cs = CompanionSystem(
        witness_active=False,
        consolidation_enabled=True,
        consolidation_router=router,
    )
    cs.observe(_obs("hello world"))
    cs.recall("hello")
    assert cs.bm25_hybrid is False
    assert cs._bm25_index is None


def test_bm25_hybrid_builds_index_lazily_on_first_recall(tmp_path):
    router = _router(tmp_path)
    cs = CompanionSystem(
        witness_active=False,
        consolidation_enabled=True,
        consolidation_router=router,
        bm25_hybrid=True,
    )
    cs.observe(_obs("I moved from Sweden four years ago", minute=0))
    cs.observe(_obs("That was a big transition for me", minute=1))
    assert cs._bm25_index is None  # not built until recall
    cs.recall("where did you move from")
    assert cs._bm25_index is not None


def test_bm25_hybrid_index_invalidates_on_new_observe(tmp_path):
    router = _router(tmp_path)
    cs = CompanionSystem(
        witness_active=False,
        consolidation_enabled=True,
        consolidation_router=router,
        bm25_hybrid=True,
    )
    cs.observe(_obs("I moved from Sweden", minute=0))
    cs.recall("where did you move")
    assert cs._bm25_index is not None
    cs.observe(_obs("I live in Hull now", minute=1))
    assert cs._bm25_index is None  # invalidated


def test_bm25_hybrid_ranks_proper_noun_match_above_cosine_distractor(tmp_path):
    """Core RRF behavior — a fragment that literal-token-matches the cue
    on a proper noun ("Sweden") should rank in the top of the hybrid
    pool even if MockProvider's hash-embedding cosine happens to rank
    other fragments higher. We can only assert ranking relative to the
    hash-embedding baseline, but the Sweden fragment should still appear
    in a top-K pool of 2 when it is the only exact token match.
    """
    router = _router(tmp_path)
    cs = CompanionSystem(
        witness_active=False,
        consolidation_enabled=True,
        consolidation_router=router,
        bm25_hybrid=True,
    )
    cs.observe(_obs("My home country is Sweden", minute=0))
    cs.observe(_obs("Transition was hard but worth it", minute=1))
    cs.observe(_obs("Four years in Hull now", minute=2))

    # Direct hybrid rank call — top-2 pool
    cue = "where was your origin Sweden"
    cue_emb = tuple(router.embed(cue))
    pool = cs._hybrid_rank_fragments(cue, cue_emb, top_k=2)
    assert len(pool) == 2
    contents = [frag.content for frag, _ in pool]
    assert any("Sweden" in c for c in contents), (
        f"Sweden fragment should be in top-2 hybrid pool, got {contents}"
    )
