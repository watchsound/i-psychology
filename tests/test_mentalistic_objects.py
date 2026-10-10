"""Unit tests for the Phase 9+ MentalisticObject protocol.

Covers:
- ``MentalisticObject`` is ``@runtime_checkable`` — ``isinstance(obj,
  MentalisticObject)`` resolves at runtime without manual registration.
- ``TraitProfile`` conforms: carries ``PERSPECTIVE`` + ``reconstruct``
  classmethod + the backward-compat ``get(dim) == get_estimate(dim)``
  identity.
- ``TraitProfile.reconstruct(system).get_estimate(dim)`` is
  byte-identical to ``TraitProfile(system).get(dim)`` on a fixture.
- Passing ``perspective=`` to ``reconstruct()`` overrides the user_id
  when the perspective's ScopeSpec names a different subject.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from psychology_base.applications.emotional.trait_profile import (
    LIKERT_MAX,
    OceanDim,
    TraitProfile,
)
from psychology_base.companion.perspective import (
    MentalisticObject,
    PerspectiveSpec,
    ScopeSpec,
)
from psychology_base.companion.system import CompanionSystem
from psychology_base.eval.adapters import Observation


def _ts(seconds: int = 0) -> datetime:
    return datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc) + timedelta(
        seconds=seconds
    )


def _seed_openness_fragment(cs: CompanionSystem, user_id: str = "user") -> None:
    """Place one high-openness explicit-cue fragment in the companion
    store for ``user_id``. The fragment carries ``subject:<user_id>``
    plus ``trait:openness:explicit`` and ``trait:openness:score:5.0`` —
    the exact tag shape ``TraitProfile._candidate_fragments`` picks up.
    """
    cs.observe(
        Observation(
            timestamp=_ts(0),
            speaker="user",
            content="I love trying novel ideas.",
            subject_id=user_id,
            tags=(
                "trait:openness:explicit",
                "trait:openness:score:5.0",
            ),
        )
    )


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_trait_profile_is_runtime_mentalistic_object() -> None:
    """``@runtime_checkable`` means ``isinstance`` resolves without
    manual registration. ``TraitProfile`` instances carry the
    ``PERSPECTIVE`` attribute + ``reconstruct`` classmethod, so the
    protocol's structural check passes at runtime."""
    cs = CompanionSystem(top_k=4, witness_active=False)
    profile = TraitProfile(system=cs)
    assert isinstance(profile, MentalisticObject)


def test_trait_profile_class_has_perspective_class_attribute() -> None:
    assert isinstance(TraitProfile.PERSPECTIVE, PerspectiveSpec)
    assert TraitProfile.PERSPECTIVE.scope == ScopeSpec(
        subject="user", entity="user"
    )


# ---------------------------------------------------------------------------
# reconstruct() <-> get() backward-compat identity
# ---------------------------------------------------------------------------


def test_reconstruct_matches_direct_instantiation_on_openness_fixture() -> None:
    """``TraitProfile.reconstruct(system).get_estimate(dim)`` returns
    the identical TraitEstimate as ``TraitProfile(system).get(dim)``
    on the same fragment store."""
    cs = CompanionSystem(top_k=4, witness_active=False)
    _seed_openness_fragment(cs)

    direct = TraitProfile(system=cs).get(OceanDim.OPENNESS)
    reconstructed = TraitProfile.reconstruct(cs).get_estimate(
        OceanDim.OPENNESS
    )

    assert direct == reconstructed
    assert reconstructed.value == pytest.approx(LIKERT_MAX)


def test_get_and_get_estimate_are_the_same_method_result() -> None:
    cs = CompanionSystem(top_k=4, witness_active=False)
    _seed_openness_fragment(cs)
    profile = TraitProfile(system=cs)
    for dim in OceanDim:
        assert profile.get(dim) == profile.get_estimate(dim)


# ---------------------------------------------------------------------------
# Overriding PERSPECTIVE at reconstruct() time
# ---------------------------------------------------------------------------


def test_reconstruct_perspective_subject_overrides_default_user_id() -> None:
    """When ``reconstruct(perspective=PerspectiveSpec(scope=ScopeSpec(
    subject='alice', entity='alice')))`` is passed, the resulting
    TraitProfile walks alice's fragments, not user's."""
    cs = CompanionSystem(top_k=4, witness_active=False)
    _seed_openness_fragment(cs, user_id="user")
    _seed_openness_fragment(cs, user_id="alice")

    alice_profile = TraitProfile.reconstruct(
        cs,
        perspective=PerspectiveSpec(
            scope=ScopeSpec(subject="alice", entity="alice")
        ),
    )
    assert alice_profile.user_id == "alice"

    default_profile = TraitProfile.reconstruct(cs)
    assert default_profile.user_id == "user"


def test_reconstruct_without_perspective_defaults_to_user_id() -> None:
    cs = CompanionSystem(top_k=4, witness_active=False)
    _seed_openness_fragment(cs)
    profile = TraitProfile.reconstruct(cs, user_id="user")
    assert profile.user_id == "user"
