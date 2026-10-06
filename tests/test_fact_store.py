"""Phase 1.5 build step 1 gate — FactStore CRUD plumbing.

Per docs/phases/phase-1.5-build-plan.md §5 step 1:
  - Fact round-trips in-memory
  - insert, upsert replaces by id
  - list_by_entity filters correctly
  - all() preserves insertion order
"""

from __future__ import annotations

import pytest

from psychology_base.companion.facts import (
    DEFAULT_FACT_CAP,
    FACT_ID_PREFIX,
    Fact,
    FactOp,
    FactOperation,
    FactStore,
)


def _mk(
    entity: str,
    relation: str,
    value: str,
    *,
    fact_id: str = "",
    confidence: float = 0.9,
    sources: tuple[str, ...] = ("frag-0001",),
    created: int = 0,
    updated: int = 0,
) -> Fact:
    return Fact(
        id=fact_id,
        entity=entity,
        relation=relation,
        value=value,
        confidence=confidence,
        source_fragment_ids=sources,
        created_tick=created,
        updated_tick=updated,
    )


def test_add_assigns_monotonic_id_when_blank():
    store = FactStore()
    f1 = store.add(_mk("Jennifer", "owns", "Westerly 28"))
    f2 = store.add(_mk("Jennifer", "lives_in", "Hull"))
    assert f1.id == f"{FACT_ID_PREFIX}0001"
    assert f2.id == f"{FACT_ID_PREFIX}0002"


def test_add_rejects_duplicate_id():
    store = FactStore()
    store.add(_mk("Jennifer", "owns", "Westerly 28", fact_id="fact-0007"))
    with pytest.raises(ValueError):
        store.add(_mk("Jennifer", "owns", "sailboat", fact_id="fact-0007"))


def test_upsert_replaces_by_id():
    store = FactStore()
    original = store.add(_mk("Jennifer", "owns_boat_model", "sailboat"))
    revised = _mk(
        "Jennifer",
        "owns_boat_model",
        "Westerly 28",
        fact_id=original.id,
        updated=5,
    )
    returned = store.upsert(revised)
    assert returned.value == "Westerly 28"
    assert returned.updated_tick == 5
    assert len(store) == 1
    assert store.by_id(original.id).value == "Westerly 28"


def test_upsert_adds_when_id_absent():
    store = FactStore()
    new = store.upsert(_mk("David", "age", "42", fact_id="fact-0099"))
    assert new.id == "fact-0099"
    assert store.by_id("fact-0099") is not None


def test_upsert_with_blank_id_delegates_to_add():
    store = FactStore()
    f = store.upsert(_mk("Jennifer", "owns", "Westerly 28"))
    assert f.id == "fact-0001"


def test_by_id_returns_none_for_missing():
    store = FactStore()
    assert store.by_id("fact-9999") is None


def test_by_entity_filters_case_sensitively():
    store = FactStore()
    store.add(_mk("Jennifer", "owns", "boat"))
    store.add(_mk("David", "owns", "car"))
    store.add(_mk("Jennifer", "lives_in", "Hull"))
    jen = store.by_entity("Jennifer")
    assert len(jen) == 2
    assert [f.relation for f in jen] == ["owns", "lives_in"]
    assert store.by_entity("jennifer") == []  # case-sensitive by decision


def test_all_preserves_insertion_order():
    store = FactStore()
    store.add(_mk("A", "r", "1"))
    store.add(_mk("B", "r", "2"))
    store.add(_mk("A", "r", "3"))
    assert [f.value for f in store.all()] == ["1", "2", "3"]


def test_fact_cap_default():
    store = FactStore()
    assert store.fact_cap == DEFAULT_FACT_CAP == 512


def test_fact_operation_wraps_fact_and_op():
    op = FactOperation(op=FactOp.ADD, fact=_mk("Jennifer", "owns", "Westerly 28"))
    assert op.op == FactOp.ADD
    assert op.fact is not None
    assert op.target_fact_id is None

    noop = FactOperation(op=FactOp.NOOP, target_fact_id="fact-0001")
    assert noop.op == FactOp.NOOP
    assert noop.fact is None
    assert noop.target_fact_id == "fact-0001"


def test_fact_embedding_defaults_empty():
    f = _mk("Jennifer", "owns", "Westerly 28")
    assert f.embedding == ()
