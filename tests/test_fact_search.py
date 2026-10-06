"""Phase 1.5 build step 3 gate — fact embedding + cosine search.

Per docs/phases/phase-1.5-build-plan.md §5 step 3:
  - search(cue_embedding, k=10) returns facts ordered by cosine desc
  - cosine_floor filters out thin matches (returned list may be shorter than k)
  - facts without embeddings are skipped (cold-start safety)
  - embed_text_for yields the canonical "entity relation value" string
  - cosine(a, a) == 1.0, cosine(a, zero) == 0.0, dim-mismatch raises

The empirical Q0-vs-Q1 overlap-≤-0.5 assertion on real conv-26 is in
step 6 (E1 run), not here — these are deterministic contract tests over
scripted vectors.
"""

from __future__ import annotations

import math

import pytest

from psychology_base.companion.facts import (
    DEFAULT_COSINE_FLOOR,
    DEFAULT_TOP_K,
    Fact,
    FactStore,
    cosine,
    embed_text_for,
)


def _fact(
    entity: str,
    relation: str,
    value: str,
    *,
    embedding: tuple[float, ...] = (),
    fid: str = "",
    confidence: float = 0.9,
) -> Fact:
    return Fact(
        id=fid,
        entity=entity,
        relation=relation,
        value=value,
        confidence=confidence,
        source_fragment_ids=("frag-0001",),
        created_tick=0,
        updated_tick=0,
        embedding=embedding,
    )


def test_embed_text_for_canonical_string():
    f = _fact("Jennifer", "owns_boat_model", "Westerly 28")
    assert embed_text_for(f) == "Jennifer owns_boat_model Westerly 28"


def test_cosine_identity_and_zero():
    v = (1.0, 2.0, 2.0)
    assert cosine(v, v) == pytest.approx(1.0)
    assert cosine(v, (0.0, 0.0, 0.0)) == 0.0
    assert cosine((), v) == 0.0
    assert cosine(v, ()) == 0.0


def test_cosine_orthogonal_axes():
    assert cosine((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)) == pytest.approx(0.0)


def test_cosine_dim_mismatch_raises():
    with pytest.raises(ValueError):
        cosine((1.0, 0.0), (1.0, 0.0, 0.0))


def test_search_orders_by_cosine_desc():
    store = FactStore()
    store.add(_fact("A", "r", "1", embedding=(1.0, 0.0, 0.0)))
    store.add(_fact("B", "r", "2", embedding=(0.0, 1.0, 0.0)))
    store.add(_fact("C", "r", "3", embedding=(0.9, 0.1, 0.0)))
    results = store.search((1.0, 0.0, 0.0), top_k=3, cosine_floor=0.0)
    assert [f.value for f, _ in results] == ["1", "3", "2"]


def test_search_applies_cosine_floor():
    store = FactStore()
    store.add(_fact("A", "r", "1", embedding=(1.0, 0.0)))
    store.add(_fact("B", "r", "2", embedding=(0.1, 0.9)))  # cosine≈0.11 w/ (1,0)
    results = store.search((1.0, 0.0), top_k=2, cosine_floor=0.25)
    assert len(results) == 1
    assert results[0][0].value == "1"


def test_search_skips_facts_without_embedding():
    store = FactStore()
    store.add(_fact("A", "r", "1"))  # empty embedding
    store.add(_fact("B", "r", "2", embedding=(1.0, 0.0)))
    results = store.search((1.0, 0.0), top_k=10, cosine_floor=0.0)
    assert len(results) == 1
    assert results[0][0].value == "2"


def test_search_empty_cue_returns_empty():
    store = FactStore()
    store.add(_fact("A", "r", "1", embedding=(1.0, 0.0)))
    assert store.search((), top_k=5) == []


def test_search_top_k_zero_returns_empty():
    store = FactStore()
    store.add(_fact("A", "r", "1", embedding=(1.0, 0.0)))
    assert store.search((1.0, 0.0), top_k=0) == []


def test_default_cosine_floor_and_top_k_constants():
    assert DEFAULT_COSINE_FLOOR == 0.25
    assert DEFAULT_TOP_K == 10


def test_search_ties_broken_by_insertion_order():
    """When two facts have identical cosine, insertion order wins (Python's
    sort is stable). Guarding against a surprise behavioral change if we
    swap sort implementations."""
    store = FactStore()
    store.add(_fact("A", "r", "first", embedding=(1.0, 0.0)))
    store.add(_fact("A", "r", "second", embedding=(1.0, 0.0)))
    results = store.search((1.0, 0.0), top_k=2, cosine_floor=0.0)
    assert [f.value for f, _ in results] == ["first", "second"]


def test_cosine_normalisation_independent_of_magnitude():
    assert cosine((2.0, 0.0), (3.0, 0.0)) == pytest.approx(1.0)
    assert cosine((1.0, 1.0), (10.0, 10.0)) == pytest.approx(1.0)
