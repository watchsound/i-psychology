"""Unit tests for the Phase 9+ unified ``recall(perspective=...)`` API.

Covers:
- ``perspective=`` kwarg produces byte-identical fragment pool to the
  legacy ``scope=``/``copresence=`` kwargs on the same inputs.
- Mutex: passing ``perspective`` AND any legacy kwarg raises ValueError.
- Metadata round-trip: ``perspective`` key carries the structured spec;
  legacy ``scope``/``copresence`` keys continue to populate (via the
  internal shim) for back-compat on anything reading them today.
- Witness-pool filtering under perspective (Step 2.1b): when a
  perspective narrows the main pool, the Witness pool is narrowed by
  the same predicate — pain-signals and recall share one scope.
- Affect-weight orthogonality (Step 2.1c): perspective filtering and
  ``affect_weight`` compose; a perspective-filtered recall still carries
  affect-weighted reconstruction.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from psychology_base.companion.copresence import CopresenceScope
from psychology_base.companion.perspective import (
    PerspectiveSpec,
    ScopeSpec,
)
from psychology_base.companion.system import CompanionSystem
from psychology_base.eval.adapters import Observation


def _ts(seconds: int = 0) -> datetime:
    return datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc) + timedelta(
        seconds=seconds
    )


def _make_system_with_two_perspective_fragments() -> CompanionSystem:
    """System pre-loaded with user→mother + user→father fragments."""
    cs = CompanionSystem(top_k=4, witness_active=False)
    cs.observe(
        Observation(
            timestamp=_ts(0),
            speaker="user",
            content="mother called today",
            subject_id="user",
            entity_id="mother",
        )
    )
    cs.observe(
        Observation(
            timestamp=_ts(1),
            speaker="user",
            content="father arrived",
            subject_id="user",
            entity_id="father",
        )
    )
    return cs


# ---------------------------------------------------------------------------
# Mutex contract
# ---------------------------------------------------------------------------


def test_recall_rejects_perspective_and_scope_together() -> None:
    cs = _make_system_with_two_perspective_fragments()
    with pytest.raises(ValueError, match="cannot mix"):
        cs.recall(
            cue="hi",
            perspective=PerspectiveSpec(
                scope=ScopeSpec(subject="user", entity="mother")
            ),
            scope=("user", "mother"),
        )


def test_recall_rejects_perspective_and_copresence_together() -> None:
    cs = _make_system_with_two_perspective_fragments()
    with pytest.raises(ValueError, match="cannot mix"):
        cs.recall(
            cue="hi",
            perspective=PerspectiveSpec(
                copresence=CopresenceScope(chain=("Alice",))
            ),
            copresence=CopresenceScope(chain=("Alice",)),
        )


# ---------------------------------------------------------------------------
# Byte-identity vs legacy kwargs
# ---------------------------------------------------------------------------


def test_perspective_scope_matches_legacy_scope_kwarg_byte_identically() -> None:
    """``perspective=PerspectiveSpec(scope=ScopeSpec(X, Y))`` must
    recall exactly the same fragment pool as legacy ``scope=(X, Y)``."""
    cs_a = _make_system_with_two_perspective_fragments()
    cs_b = _make_system_with_two_perspective_fragments()

    r_perspective = cs_a.recall(
        cue="who called",
        perspective=PerspectiveSpec(
            scope=ScopeSpec(subject="user", entity="mother")
        ),
    )
    r_legacy = cs_b.recall(cue="who called", scope=("user", "mother"))

    assert [f.id for f in r_perspective.fragments_used] == [
        f.id for f in r_legacy.fragments_used
    ]


def test_perspective_copresence_matches_legacy_copresence_kwarg_byte_identically() -> (
    None
):
    cs_a = CompanionSystem(top_k=4, witness_active=False)
    cs_b = CompanionSystem(top_k=4, witness_active=False)
    for cs in (cs_a, cs_b):
        cs.observe(
            Observation(
                timestamp=_ts(0),
                speaker="narrator",
                content="alice saw the lettuce",
                tags=("witness:Alice",),
            )
        )
        cs.observe(
            Observation(
                timestamp=_ts(1),
                speaker="narrator",
                content="bob saw the pepper",
                tags=("witness:Bob",),
            )
        )

    r_perspective = cs_a.recall(
        cue="what did alice see",
        perspective=PerspectiveSpec(
            copresence=CopresenceScope(chain=("Alice",))
        ),
    )
    r_legacy = cs_b.recall(
        cue="what did alice see", copresence=CopresenceScope(chain=("Alice",))
    )

    assert [f.id for f in r_perspective.fragments_used] == [
        f.id for f in r_legacy.fragments_used
    ]


def test_perspective_empty_matches_no_kwargs_byte_identically() -> None:
    """Bare ``PerspectiveSpec()`` must be indistinguishable from no
    perspective kwarg at all — both are pass-through."""
    cs_a = _make_system_with_two_perspective_fragments()
    cs_b = _make_system_with_two_perspective_fragments()

    r_empty = cs_a.recall(cue="family", perspective=PerspectiveSpec())
    r_none = cs_b.recall(cue="family")

    assert [f.id for f in r_empty.fragments_used] == [
        f.id for f in r_none.fragments_used
    ]


# ---------------------------------------------------------------------------
# Metadata round-trip
# ---------------------------------------------------------------------------


def test_recall_metadata_carries_perspective_structured_spec() -> None:
    cs = _make_system_with_two_perspective_fragments()
    spec = PerspectiveSpec(scope=ScopeSpec(subject="user", entity="mother"))
    r = cs.recall(cue="family", perspective=spec)
    assert r.metadata["perspective"] == spec
    # Legacy keys continue to populate for back-compat consumers.
    assert r.metadata["scope"] == ("user", "mother")
    assert r.metadata["copresence"] is None


def test_recall_metadata_legacy_kwargs_populate_perspective_key() -> None:
    """When legacy ``scope=`` is used, the internal shim constructs a
    PerspectiveSpec; metadata should carry both the legacy ``scope`` key
    AND the structured ``perspective`` object."""
    cs = _make_system_with_two_perspective_fragments()
    r = cs.recall(cue="family", scope=("user", "mother"))
    assert r.metadata["scope"] == ("user", "mother")
    perspective = r.metadata["perspective"]
    assert perspective is not None
    assert perspective.scope == ScopeSpec(subject="user", entity="mother")
    assert perspective.copresence is None


def test_recall_metadata_passthrough_when_no_perspective_kwargs() -> None:
    cs = _make_system_with_two_perspective_fragments()
    r = cs.recall(cue="family")
    assert r.metadata["scope"] is None
    assert r.metadata["copresence"] is None
    assert r.metadata["perspective"] is None


# ---------------------------------------------------------------------------
# Witness-pool filtering (Step 2.1b)
# ---------------------------------------------------------------------------


def test_witness_pool_filtered_under_perspective_matches_legacy_scope() -> None:
    """Pain-signals and main-pool recall share one perspective. Under
    the Phase 9+ consolidation, passing ``perspective=`` filters the
    Witness store identically to how legacy ``scope=`` did. Verified
    by exercising the recall path with Witness active and asserting
    pain_signal is deterministic across both API shapes."""
    cs_a = CompanionSystem(top_k=4, witness_active=True)
    cs_b = CompanionSystem(top_k=4, witness_active=True)
    for cs in (cs_a, cs_b):
        cs.observe(
            Observation(
                timestamp=_ts(0),
                speaker="user",
                content="mother called today",
                subject_id="user",
                entity_id="mother",
            )
        )

    r_perspective = cs_a.recall(
        cue="family",
        perspective=PerspectiveSpec(
            scope=ScopeSpec(subject="user", entity="mother")
        ),
    )
    r_legacy = cs_b.recall(cue="family", scope=("user", "mother"))

    assert r_perspective.metadata["witness_pain"] == r_legacy.metadata[
        "witness_pain"
    ]


# ---------------------------------------------------------------------------
# Affect-weight orthogonality (Step 2.1c)
# ---------------------------------------------------------------------------


def test_affect_weight_composes_with_perspective() -> None:
    """``perspective=`` and ``affect_weight`` are orthogonal: a
    perspective-filtered recall still applies affect weighting at
    reconstruction. This test asserts the recall completes without
    error and both the perspective filter AND affect weight are
    reflected in metadata."""
    cs = _make_system_with_two_perspective_fragments()
    r = cs.recall(
        cue="family",
        context={"affect": {"valence": -0.6, "arousal": 0.4}},
        perspective=PerspectiveSpec(
            scope=ScopeSpec(subject="user", entity="mother")
        ),
        affect_weight=0.7,
    )
    assert r.metadata["affect_weight"] == 0.7
    assert r.metadata["cue_affect"] is not None
    assert r.metadata["perspective"] is not None
