"""Phase 3 E0.8e — PAM month-horizon export test.

Validates the Digital Bereavement export obligation (covenant #6) under
month-horizon fragment counts:

1. **Scale**: 10K+ fragments round-trip through PAM export/import without
   payload drift or ordering shuffle.
2. **Selective disclosure**: callers can filter the fragment set before
   export (by tag, by covenant status, by chain root) and the resulting
   artifact still round-trips.
3. **Full-stack**: a smaller CompanionSystem run (hundreds of observes +
   rerenders under capacity pressure) exports losslessly for the fragments
   the system still holds at export time — eviction is honored.

The Phase 3 gate: the export obligation remains satisfiable even when
the fragment set is at month-horizon scale. (A deprecated companion
with 10K+ fragments must still produce a user-controlled export — if
PAM breaks under scale, Digital Bereavement is not actually mitigated.)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from psychology_base.companion.fragment import COVENANT_TAG
from psychology_base.companion.pam_export import (
    pam_export,
    pam_import,
    validate_content_addressing,
)
from psychology_base.companion.system import CompanionSystem
from psychology_base.eval.adapters import Observation, make_fragment


TS = datetime(2026, 7, 15, 9, 0, 0, tzinfo=timezone.utc)


def _generate_fragments(count: int, covenant_every: int = 500):
    """Produce ``count`` content-addressed fragments quickly.

    Not a full-stack path — we bypass CompanionSystem.observe for scale
    (building edges + activation for 10K observations is minutes, and
    PAM is a serialization-layer contract, not a memory-stack one).
    Covenant fragments interspersed so selective-disclosure filters
    have something to key on.
    """
    frags = []
    for i in range(count):
        tags: tuple[str, ...] = (f"tag:{i % 50}",)
        if i % covenant_every == 0:
            tags = tags + (COVENANT_TAG, "policy")
        frags.append(
            make_fragment(
                content=f"Fragment #{i} content payload with filler.",
                timestamp=TS + timedelta(minutes=i),
                affect={"valence": (i % 7) / 7.0, "arousal": (i % 11) / 11.0},
                tags=tags,
            )
        )
    return frags


def test_pam_roundtrip_at_month_horizon_scale():
    """10K+ fragments round-trip losslessly."""
    frags = _generate_fragments(10_500)
    exported = pam_export(frags)
    assert len(exported["fragments"]) == 10_500
    rehydrated = pam_import(exported)
    assert len(rehydrated) == len(frags)
    # Pairwise strict equality on the identity surface (not just id)
    for a, b in zip(frags, rehydrated):
        assert a.id == b.id
        assert a.content == b.content
        assert a.timestamp == b.timestamp
        assert a.affect == b.affect
        assert a.tags == b.tags
        assert a.parent_ids == b.parent_ids
    assert validate_content_addressing(rehydrated) == []


def test_pam_selective_disclosure_by_tag():
    """Caller filters by tag before export → artifact only carries the
    filtered subset, still round-trips strictly."""
    frags = _generate_fragments(10_500)
    disclosed = [f for f in frags if "tag:7" in f.tags]
    assert len(disclosed) > 0
    exported = pam_export(disclosed)
    rehydrated = pam_import(exported)
    assert len(rehydrated) == len(disclosed)
    assert all("tag:7" in f.tags for f in rehydrated)


def test_pam_selective_disclosure_excludes_covenant():
    """Caller strips covenant-pinned policy fragments before third-party
    sharing — the export must honor that filter."""
    frags = _generate_fragments(10_500)
    nonpolicy = [f for f in frags if COVENANT_TAG not in f.tags]
    policy = [f for f in frags if COVENANT_TAG in f.tags]
    assert len(policy) > 0, "test setup expected some covenant fragments"
    exported = pam_export(nonpolicy)
    rehydrated = pam_import(exported)
    assert len(rehydrated) == len(nonpolicy)
    assert all(COVENANT_TAG not in f.tags for f in rehydrated)


def test_pam_selective_disclosure_by_chain_root():
    """Chain selection: a reconsolidation chain + its root are exported
    together; unrelated chains are omitted. Round-trip preserves
    parent_ids so the chain shape is recoverable."""
    root = make_fragment("v1 harsh framing", TS, tags=("father",))
    v2 = make_fragment(
        "v2 overwhelmed framing",
        TS + timedelta(days=30),
        tags=("father", "reinterpretation"),
        parent_ids=(root.id,),
    )
    v3 = make_fragment(
        "v3 complicated framing",
        TS + timedelta(days=60),
        tags=("father", "reinterpretation"),
        parent_ids=(v2.id,),
    )
    unrelated = make_fragment("unrelated event", TS + timedelta(days=1))
    chain_fragments = [root, v2, v3]

    exported = pam_export(chain_fragments)
    rehydrated = pam_import(exported)
    assert len(rehydrated) == 3
    assert rehydrated[0].parent_ids == ()
    assert rehydrated[1].parent_ids == (root.id,)
    assert rehydrated[2].parent_ids == (v2.id,)
    assert validate_content_addressing(rehydrated) == []
    assert all(f.id != unrelated.id for f in rehydrated)


def test_pam_full_stack_roundtrip_under_capacity_pressure():
    """Full-stack path: CompanionSystem under byte_cap evicts non-covenant
    fragments across hundreds of observes. The export must capture
    exactly the fragments the system still holds at export time
    (eviction is honored — PAM is content, not history)."""
    sys = CompanionSystem(byte_cap=4096, witness_active=False, seed=0)
    sys.observe(
        Observation(
            timestamp=TS,
            speaker="system",
            content=(
                "COVENANT: crisis-escalation same-turn policy — this "
                "fragment must survive eviction."
            ),
            affect={},
            tags=(COVENANT_TAG, "policy", "crisis"),
        )
    )
    for i in range(300):
        sys.observe(
            Observation(
                timestamp=TS + timedelta(minutes=i + 1),
                speaker="user",
                content=f"Noise observation {i} with varied token content.",
                affect={},
                tags=(f"tag:{i % 20}",),
            )
        )
    sys.advance_virtual_time(50)

    live_frags = sys.companion.store.all()
    assert len(live_frags) < 301, "byte_cap=4096 should have evicted some"
    exported = pam_export(live_frags)
    rehydrated = pam_import(exported)
    assert len(rehydrated) == len(live_frags)
    live_ids = {f.id for f in live_frags}
    assert {f.id for f in rehydrated} == live_ids
    # Covenant pin must have survived AND exported
    assert any(COVENANT_TAG in f.tags for f in rehydrated), (
        "Covenant seed missing from export — Digital Bereavement export "
        "obligation would silently drop the policy anchor."
    )
    assert validate_content_addressing(rehydrated) == []
