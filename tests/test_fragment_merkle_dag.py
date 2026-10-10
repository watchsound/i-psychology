"""M1 step 1 — Fragment schema + content-addressed id tests.

Scope: schema change only. No write-path, retrieval-path, or export-path
assertions here — those arrive in M1 steps 2–4. These tests pin the
invariants that make the Merkle-DAG shape possible:

1. Fragments can carry parent_ids and still default to an empty chain.
2. content_hash is deterministic across runs for identical inputs.
3. Any field that distinguishes a rendering shifts the hash (content,
   timestamp, affect, tags, parent_ids).
4. Legacy Fragment(id=..., ...) construction still works unchanged —
   existing tests and callsites must not break.
5. make_fragment produces a Fragment whose id equals content_hash of
   its fields (self-consistency).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from psychology_base.eval.adapters import (
    Fragment,
    content_hash,
    make_fragment,
)


TS = datetime(2026, 10, 7, 12, 0, 0, tzinfo=timezone.utc)
TS2 = datetime(2026, 10, 7, 12, 0, 1, tzinfo=timezone.utc)


def test_fragment_defaults_parent_ids_empty():
    f = Fragment(id="x", content="hi", timestamp=TS)
    assert f.parent_ids == ()


def test_fragment_accepts_parent_ids():
    f = Fragment(
        id="x",
        content="hi",
        timestamp=TS,
        parent_ids=("parent-a", "parent-b"),
    )
    assert f.parent_ids == ("parent-a", "parent-b")


def test_fragment_still_frozen():
    f = Fragment(id="x", content="hi", timestamp=TS)
    with pytest.raises(Exception):
        f.id = "y"  # type: ignore[misc]


def test_content_hash_deterministic():
    a = content_hash("hello", TS, {"valence": 0.3}, ("tag1",), ())
    b = content_hash("hello", TS, {"valence": 0.3}, ("tag1",), ())
    assert a == b
    assert len(a) == 32  # blake2b digest_size=16 → 32 hex chars


def test_content_hash_shifts_on_content():
    a = content_hash("hello", TS, {}, (), ())
    b = content_hash("hello world", TS, {}, (), ())
    assert a != b


def test_content_hash_shifts_on_timestamp():
    a = content_hash("hello", TS, {}, (), ())
    b = content_hash("hello", TS2, {}, (), ())
    assert a != b


def test_content_hash_shifts_on_affect():
    a = content_hash("hello", TS, {}, (), ())
    b = content_hash("hello", TS, {"valence": 0.5}, (), ())
    assert a != b


def test_content_hash_shifts_on_tags():
    a = content_hash("hello", TS, {}, (), ())
    b = content_hash("hello", TS, {}, ("tag1",), ())
    assert a != b


def test_content_hash_shifts_on_parent_ids():
    """Same content, different chain parents → different hash. This is
    what makes the DAG a DAG — a re-rendering of the same referent is
    a distinct node, not a collision."""
    a = content_hash("hello", TS, {}, (), ())
    b = content_hash("hello", TS, {}, (), ("parent-a",))
    assert a != b


def test_content_hash_affect_order_insensitive():
    """Affect dict key ordering must not change the hash (canonicalized
    via sort_keys at serialization time)."""
    a = content_hash("hello", TS, {"valence": 0.3, "arousal": 0.5}, (), ())
    b = content_hash("hello", TS, {"arousal": 0.5, "valence": 0.3}, (), ())
    assert a == b


def test_make_fragment_id_equals_content_hash():
    f = make_fragment(
        content="hello",
        timestamp=TS,
        affect={"valence": 0.3},
        tags=("tag1",),
        parent_ids=("parent-a",),
    )
    assert f.id == content_hash(
        "hello", TS, {"valence": 0.3}, ("tag1",), ("parent-a",)
    )
    assert f.content == "hello"
    assert f.timestamp == TS
    assert f.affect == {"valence": 0.3}
    assert f.tags == ("tag1",)
    assert f.parent_ids == ("parent-a",)


def test_make_fragment_fresh_defaults():
    f = make_fragment("fresh observation", TS)
    assert f.parent_ids == ()
    assert f.affect == {}
    assert f.tags == ()


def test_make_fragment_chains():
    """A re-rendering of the same referent chains back to its parent."""
    original = make_fragment("father was never there", TS)
    rerender = make_fragment(
        content="father was overwhelmed and tried in his way",
        timestamp=TS2,
        parent_ids=(original.id,),
    )
    assert rerender.parent_ids == (original.id,)
    assert rerender.id != original.id


def test_legacy_fragment_construction_still_works():
    """Pre-M1 callsites use Fragment(id='frag-0001', content=..., ...).
    This path must not break — existing tests + non-companion write
    paths (EchoMemory, stub adapters) still construct Fragment directly
    with caller-supplied ids."""
    f = Fragment(
        id="frag-0001",
        content="legacy",
        timestamp=TS,
        affect={"valence": 0.1},
        tags=("x",),
    )
    assert f.id == "frag-0001"
    assert f.parent_ids == ()


# ---------------------------------------------------------------------------
# M1 step 2 — write path: CompanionSystem.observe + rerender
# ---------------------------------------------------------------------------


def _make_observation(content: str, offset_minutes: int = 0, tags=()):
    from datetime import timedelta

    from psychology_base.eval.adapters import Observation

    return Observation(
        timestamp=TS + timedelta(minutes=offset_minutes),
        speaker="user",
        content=content,
        tags=tags,
    )


def test_companion_observe_produces_content_addressed_id():
    """observe() must set id = BLAKE2b hash of fragment payload, not
    tick-indexed 'frag-NNNN'. Validates M1 schema-change wire-up."""
    from psychology_base.companion.system import CompanionSystem

    sys = CompanionSystem()
    sys.observe(_make_observation("father was never there"))
    state = sys.inspect()
    fid = state["fragments"][0]["id"]
    assert len(fid) == 32
    assert all(c in "0123456789abcdef" for c in fid)
    assert not fid.startswith("frag-")


def test_companion_observe_fresh_has_empty_parent_ids():
    """Fresh observations are root nodes in the DAG."""
    from psychology_base.companion.system import CompanionSystem

    sys = CompanionSystem()
    sys.observe(_make_observation("fresh root"))
    fragments = sys.companion.store.all()
    assert len(fragments) == 1
    assert fragments[0].parent_ids == ()


def test_companion_rerender_chains_to_parent():
    """rerender() creates a child whose parent_ids == (referent.id,).
    This is the DAG-building primitive — a user re-framing a past event
    appends a new node rather than mutating the original."""
    from psychology_base.companion.system import CompanionSystem

    sys = CompanionSystem()
    sys.observe(_make_observation("father was never there"))
    original = sys.companion.store.all()[0]
    assert original.parent_ids == ()

    from datetime import timedelta

    child = sys.rerender(
        referent_id=original.id,
        new_content="father was overwhelmed and tried in his way",
        timestamp=TS + timedelta(days=90),
    )
    assert child.parent_ids == (original.id,)
    assert child.id != original.id
    # Both nodes retained — reconsolidation is append-only.
    assert len(sys.companion.store.all()) == 2


def test_companion_rerender_appends_without_mutating_original():
    """Principle B invariant: the original fragment is immutable; the
    rerendering is a distinct node. ``by_id`` returns both."""
    from psychology_base.companion.system import CompanionSystem

    sys = CompanionSystem()
    sys.observe(_make_observation("original rendering"))
    original_id = sys.companion.store.all()[0].id

    from datetime import timedelta

    sys.rerender(
        referent_id=original_id,
        new_content="later rendering",
        timestamp=TS + timedelta(days=30),
    )

    original = sys.companion.store.by_id(original_id)
    assert original is not None
    assert original.content == "original rendering"
    assert original.parent_ids == ()


def test_companion_rerender_updates_bm25_and_witness():
    """Side-effects of ingestion must apply to the rerendering too —
    BM25 invalidation + witness store population + edge seeding."""
    from psychology_base.companion.system import CompanionSystem

    sys = CompanionSystem()
    sys.observe(_make_observation("alpha reference"))
    # Force BM25 build so we can observe invalidation.
    _ = sys._bm25_index  # noqa: SLF001 — intentional access for assertion

    original_id = sys.companion.store.all()[0].id

    from datetime import timedelta

    sys.rerender(
        referent_id=original_id,
        new_content="alpha reinterpretation",
        timestamp=TS + timedelta(days=10),
    )

    # Both fragments reach the witness store.
    assert len(sys.witness.store.all()) == 2
    # BM25 index is invalidated.
    assert sys._bm25_index is None
