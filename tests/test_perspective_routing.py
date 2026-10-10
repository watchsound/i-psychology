"""Unit tests for the Phase 9+ perspective substrate consolidation.

Covers:
- :class:`ScopeSpec` + :func:`filter_fragments_by_scope` byte-identity
  vs legacy ``system._filter_fragments_by_scope``.
- :class:`PerspectiveSpec` composition (scope + copresence),
  order-independence, pass-through semantics.
- Scope-asymmetric regression pin: fragments with ``subject:<X>`` tag
  but no ``entity:*`` tag of any kind are DROPPED under a two-sided
  scope — this is the Phase 5 Track 1 FANToM +0.791 mechanism, exposed
  as a first-class primitive.
- Frozen hashable records compose into dict keys / set membership
  without surprises.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from psychology_base.companion.copresence import CopresenceScope
from psychology_base.companion.perspective import (
    PerspectiveSpec,
    ScopeSpec,
    filter_fragments_by_copresence,
    filter_fragments_by_scope,
)
from psychology_base.companion.system import _filter_fragments_by_scope
from psychology_base.eval.adapters import make_fragment


def _frag(content: str, tags: tuple[str, ...]):
    return make_fragment(
        content=content,
        timestamp=datetime(2026, 1, 1, 12, 0, 0),
        tags=tags,
    )


# ---------------------------------------------------------------------------
# ScopeSpec — byte-identity vs legacy filter + FANToM mechanism pin
# ---------------------------------------------------------------------------


def test_scope_spec_byte_identity_vs_legacy_filter() -> None:
    frags = [
        _frag("a", ("subject:user", "entity:mother")),
        _frag("b", ("subject:user", "entity:father")),
        _frag("c", ("subject:friend", "entity:mother")),
        _frag("d", ()),  # scope-less — pass-through
    ]
    spec = ScopeSpec(subject="user", entity="mother")
    assert spec.apply(frags) == _filter_fragments_by_scope(
        frags, ("user", "mother")
    )


def test_scope_spec_scopeless_fragments_pass_through() -> None:
    frags = [
        _frag("a", ()),
        _frag("b", ("tag:foo",)),  # non-scope tags only
    ]
    spec = ScopeSpec(subject="user", entity="mother")
    assert spec.apply(frags) == frags


def test_scope_spec_subject_only_fragments_dropped_under_two_sided_scope() -> (
    None
):
    """FANToM Track 1 +0.791 mechanism — DO NOT regress.

    FANToM observations are tagged with ``subject:<speaker>`` only
    (no ``entity:*``). The scorer calls
    ``recall(scope=("subject", target_person))`` which under the
    two-sided ScopeSpec AND-contract requires ``entity:<target_person>``
    as well. Fragments with ``subject:<speaker>`` and no entity tag
    carry ``has_any_scope=True`` → fall into the AND-check → fail on
    the missing entity tag → are DROPPED. Empty recall pool on
    ``BELIEF_INACCESSIBLE`` → agent says "I don't know" → matches gold.

    If a future refactor "cleans up" this asymmetry (e.g., auto-passing
    subject-only fragments when the scope's entity cannot bind), the
    +0.791 Phase 5 Track 1 architectural delta vanishes. This test
    pins the mechanism.
    """
    frags = [
        _frag("alice-utterance", ("subject:alice",)),  # subject-only
        _frag("bob-utterance", ("subject:bob",)),  # subject-only
    ]
    # FANToM-shaped scope call: literal "subject" string in position 0.
    spec = ScopeSpec(subject="subject", entity="alice")
    assert spec.apply(frags) == []  # empty — mechanism preserved


def test_scope_spec_entity_mismatch_dropped() -> None:
    frags = [
        _frag("a", ("subject:user", "entity:mother")),
        _frag("b", ("subject:user", "entity:father")),
    ]
    spec = ScopeSpec(subject="user", entity="mother")
    kept = spec.apply(frags)
    assert len(kept) == 1
    assert kept[0].content == "a"


def test_scope_spec_subject_mismatch_dropped() -> None:
    frags = [
        _frag("a", ("subject:user", "entity:mother")),
        _frag("b", ("subject:friend", "entity:mother")),
    ]
    spec = ScopeSpec(subject="user", entity="mother")
    kept = spec.apply(frags)
    assert len(kept) == 1
    assert kept[0].content == "a"


def test_scope_spec_is_frozen_and_hashable() -> None:
    a = ScopeSpec(subject="user", entity="mother")
    b = ScopeSpec(subject="user", entity="mother")
    assert a == b
    assert hash(a) == hash(b)
    seen = {a, b}
    assert len(seen) == 1


# ---------------------------------------------------------------------------
# PerspectiveSpec — composition, pass-through, metadata
# ---------------------------------------------------------------------------


def test_perspective_spec_empty_is_passthrough() -> None:
    frags = [
        _frag("a", ("subject:user", "entity:mother")),
        _frag("b", ("subject:friend",)),
        _frag("c", ()),
    ]
    assert PerspectiveSpec().apply(frags) == frags
    assert PerspectiveSpec().is_empty() is True


def test_perspective_spec_scope_only_matches_scope_spec_apply() -> None:
    frags = [
        _frag("a", ("subject:user", "entity:mother")),
        _frag("b", ("subject:user", "entity:father")),
    ]
    scope = ScopeSpec(subject="user", entity="mother")
    assert PerspectiveSpec(scope=scope).apply(frags) == scope.apply(frags)


def test_perspective_spec_copresence_only_matches_filter_fragments_by_copresence() -> (
    None
):
    frags = [
        _frag("a", ("witness:Alice",)),
        _frag("b", ("witness:Bob",)),
        _frag("c", ("witness:Alice", "witness:Bob")),
    ]
    cs = CopresenceScope(chain=("Alice",))
    assert PerspectiveSpec(copresence=cs).apply(
        frags
    ) == filter_fragments_by_copresence(frags, cs)


def test_perspective_spec_composition_is_set_intersection() -> None:
    """Both scope and copresence active → fragment must satisfy BOTH."""
    frags = [
        _frag("scope-ok-cop-ok", ("subject:u", "entity:m", "witness:Alice")),
        _frag("scope-ok-cop-no", ("subject:u", "entity:m", "witness:Bob")),
        _frag("scope-no-cop-ok", ("subject:u", "entity:f", "witness:Alice")),
        _frag("scope-off-cop-off", ("witness:Carol",)),  # scope-less: passes scope
    ]
    spec = PerspectiveSpec(
        scope=ScopeSpec(subject="u", entity="m"),
        copresence=CopresenceScope(chain=("Alice",)),
    )
    kept = spec.apply(frags)
    kept_contents = {f.content for f in kept}
    # scope-ok-cop-ok: both pass → keep
    # scope-ok-cop-no: fails copresence → drop
    # scope-no-cop-ok: fails scope → drop
    # scope-off-cop-off: scope-less passes scope; witness:Carol not
    # matching Alice chain → drop
    assert kept_contents == {"scope-ok-cop-ok"}


def test_perspective_spec_order_independent_on_commuting_inputs() -> None:
    """Scope and copresence are idempotent predicates on tag presence.

    For fragments where tags are drawn independently, applying scope
    first then copresence is set-equal to applying copresence first then
    scope. Fixed order is for determinism; semantics commute.
    """
    frags = [
        _frag("a", ("subject:u", "entity:m", "witness:A")),
        _frag("b", ("subject:u", "entity:m", "witness:B")),
        _frag("c", ("subject:u", "entity:f", "witness:A")),
        _frag("d", ("witness:A",)),  # scope-less
        _frag("e", ("subject:u", "entity:m")),  # witness-less
    ]
    scope = ScopeSpec(subject="u", entity="m")
    cs = CopresenceScope(chain=("A",))

    # scope → copresence
    pool_a = scope.apply(frags)
    pool_a = filter_fragments_by_copresence(pool_a, cs)

    # copresence → scope
    pool_b = filter_fragments_by_copresence(frags, cs)
    pool_b = scope.apply(pool_b)

    assert set(id(f) for f in pool_a) == set(id(f) for f in pool_b)


def test_perspective_spec_is_frozen_and_hashable() -> None:
    a = PerspectiveSpec(
        scope=ScopeSpec(subject="u", entity="m"),
        copresence=CopresenceScope(chain=("A",)),
    )
    b = PerspectiveSpec(
        scope=ScopeSpec(subject="u", entity="m"),
        copresence=CopresenceScope(chain=("A",)),
    )
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1


def test_perspective_spec_is_empty_detects_active_subspecs() -> None:
    assert PerspectiveSpec().is_empty() is True
    assert (
        PerspectiveSpec(scope=ScopeSpec(subject="u", entity="m")).is_empty()
        is False
    )
    assert (
        PerspectiveSpec(copresence=CopresenceScope(chain=("A",))).is_empty()
        is False
    )
    assert (
        PerspectiveSpec(
            scope=ScopeSpec(subject="u", entity="m"),
            copresence=CopresenceScope(chain=("A",)),
        ).is_empty()
        is False
    )


# ---------------------------------------------------------------------------
# filter_fragments_by_scope (module-level function) — compat shim
# ---------------------------------------------------------------------------


def test_module_level_filter_fragments_by_scope_matches_system_shim() -> None:
    """``perspective.routing.filter_fragments_by_scope`` is the
    canonical primitive; ``system._filter_fragments_by_scope`` delegates
    to it. Both must produce the same list for every input."""
    frags = [
        _frag("a", ("subject:user", "entity:mother")),
        _frag("b", ("subject:friend", "entity:mother")),
        _frag("c", ()),
    ]
    assert filter_fragments_by_scope(frags, ("user", "mother")) == (
        _filter_fragments_by_scope(frags, ("user", "mother"))
    )
