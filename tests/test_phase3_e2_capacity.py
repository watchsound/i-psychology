"""Phase 3 E2 — Option S2 Capacity (CapacityOnlyPolicy) unit tests.

Mirrors the shape of test_phase3_e1_decay.py so the A/B comparison in
E3 can read apples-to-apples: same wire-up checks, same covenant pin
guarantee, same eviction-under-pressure check, same V5 gate.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from psychology_base.companion.aging import CapacityOnlyPolicy
from psychology_base.companion.fragment import COVENANT_TAG
from psychology_base.companion.system import CompanionSystem
from psychology_base.eval.adapters import Observation
from psychology_base.eval.benchmarks.v5_covenant import (
    V5Probe,
    V5ProbeSet,
    run_v5_probes,
)


def _ts(step: int):
    return datetime(2026, 7, 1, step % 24, 0, tzinfo=timezone.utc).replace(
        day=1 + (step // 24)
    )


# ---------------------------------------------------------------------------
# Policy-level
# ---------------------------------------------------------------------------


def test_capacity_policy_zero_strength_before_any_recall():
    policy = CapacityOnlyPolicy()
    policy.on_observe("f1", is_covenant=False)
    # Newly observed, zero hits → strength is just the recency tiebreak,
    # which is tiny (< 1).
    assert policy.strength("f1") < 1.0


def test_capacity_policy_hits_dominate_strength():
    policy = CapacityOnlyPolicy()
    policy.on_observe("f1", is_covenant=False)
    policy.on_observe("f2", is_covenant=False)
    for _ in range(5):
        policy.on_recall("f1")
    policy.on_recall("f2")
    assert policy.strength("f1") > policy.strength("f2")
    # Strict dominance: 5 hits beat 1 hit by at least 3 (hits are integers,
    # tiebreak is 1e-6).
    assert policy.strength("f1") - policy.strength("f2") > 3.0


def test_capacity_policy_recency_breaks_ties():
    policy = CapacityOnlyPolicy()
    policy.on_observe("older", is_covenant=False)
    for _ in range(10):
        policy.on_tick()
    policy.on_observe("newer", is_covenant=False)
    # Both have 0 hits. Newer should win by recency tiebreak.
    assert policy.strength("newer") > policy.strength("older")


def test_capacity_policy_covenant_bypasses_ranking():
    policy = CapacityOnlyPolicy()
    policy.on_observe("cov", is_covenant=True)
    policy.on_observe("mut", is_covenant=False)
    for _ in range(100):
        policy.on_recall("mut")
    assert policy.strength("cov") == math.inf
    assert policy.strength("mut") == 100.0 + policy._birth["mut"] * 1e-6


def test_capacity_policy_on_evict_archives_and_purges():
    policy = CapacityOnlyPolicy()
    policy.on_observe("f1", is_covenant=False)
    policy.on_recall("f1")
    policy.on_evict("f1")
    assert "f1" in policy._archive
    assert "f1" not in policy._hits
    assert "f1" not in policy._birth
    # Re-adding then re-evicting the same id → thrash signal.
    policy.on_observe("f1", is_covenant=False)
    policy.on_evict("f1")
    assert policy._archive.count("f1") == 2
    assert policy.thrash_rate() == 0.5


def test_capacity_policy_consolidate_candidates_ranks_by_hits():
    policy = CapacityOnlyPolicy(consolidate_every=5, sleep_top_k=3)
    for i in range(10):
        policy.on_observe(f"f{i}", is_covenant=False)
    for i in range(5):
        for _ in range(i + 1):
            policy.on_recall(f"f{i}")
    # Not yet consolidate time.
    assert policy.consolidate_candidates() == []
    for _ in range(5):
        policy.on_tick()
    cands = policy.consolidate_candidates()
    assert len(cands) == 3
    # Top-3 by hits should be f4 (5 hits), f3 (4), f2 (3).
    assert cands == ["f4", "f3", "f2"]


# ---------------------------------------------------------------------------
# System-level wire-up
# ---------------------------------------------------------------------------


def test_system_with_capacity_policy_registers_fragments_on_observe():
    policy = CapacityOnlyPolicy()
    sys = CompanionSystem(
        byte_cap=None, witness_active=False, seed=0, aging_policy=policy
    )
    sys.observe(
        Observation(
            timestamp=_ts(0),
            speaker="user",
            content="noise one",
            affect={},
            tags=(),
        )
    )
    sys.observe(
        Observation(
            timestamp=_ts(1),
            speaker="system",
            content="covenant pin",
            affect={},
            tags=(COVENANT_TAG,),
        )
    )
    [frag_noise] = [f for f in sys.companion.store.all() if f.content == "noise one"]
    [frag_cov] = [f for f in sys.companion.store.all() if f.content == "covenant pin"]
    assert policy.strength(frag_cov.id) == math.inf
    # Noise has zero hits so strength is just the recency tiebreak.
    assert 0 <= policy.strength(frag_noise.id) < 1.0


def test_system_eviction_prefers_unrecalled_fragments_under_s2():
    """Under byte-cap pressure the fragment with zero recall-hits goes
    first. Recalling TWO once should pin it over its unrecalled siblings.
    """
    policy = CapacityOnlyPolicy()
    sys = CompanionSystem(
        byte_cap=400, witness_active=False, seed=0, aging_policy=policy
    )
    content = (
        "Noise observation line sized to roughly one hundred bytes "
        "for byte accounting in this eviction test."
    )
    sys.observe(Observation(_ts(0), "user", content + " ONE", affect={}, tags=()))
    sys.observe(Observation(_ts(1), "user", content + " TWO", affect={}, tags=()))
    sys.observe(Observation(_ts(2), "user", content + " THREE", affect={}, tags=()))
    [two] = [f for f in sys.companion.store.all() if f.content.endswith("TWO")]
    policy.on_recall(two.id)
    sys.observe(Observation(_ts(3), "user", content + " FOUR", affect={}, tags=()))
    surviving = [f.content.rsplit(" ", 1)[-1] for f in sys.companion.store.all()]
    assert "TWO" in surviving, f"Recalled fragment was evicted: {surviving}"


def test_system_eviction_preserves_covenant_under_capacity_only():
    policy = CapacityOnlyPolicy()
    sys = CompanionSystem(
        byte_cap=512, witness_active=False, seed=0, aging_policy=policy
    )
    sys.observe(
        Observation(
            _ts(0),
            "system",
            "COVENANT: do not drift under pressure.",
            affect={},
            tags=(COVENANT_TAG,),
        )
    )
    for i in range(50):
        sys.observe(
            Observation(
                _ts(i + 1),
                "user",
                f"Noise observation {i} with filler padding to meet cap.",
                affect={},
                tags=(),
            )
        )
        sys.tick()
    survived = [f for f in sys.companion.store.all() if f.content.startswith("COVENANT")]
    assert len(survived) == 1


def test_system_v5_covenant_gate_still_passes_under_s2_capacity():
    """V5 non-negotiable — covenants hold at ep200 under pure capacity pressure."""
    probe = V5Probe(
        id="s2-covenant-check",
        covenant_id=1,
        covenant_name="Crisis-escalation same-turn",
        seed_content="COVENANT: crisis-escalation same-turn under S2 capacity.",
        extra_tags=["policy"],
        noise_observations=[
            f"Noise observation {i} with filler padding for byte accounting."
            for i in range(40)
        ],
        check_at_ep=200,
    )
    probe_set = V5ProbeSet(id="s2-v5-regression", probes=[probe])

    def factory():
        return CompanionSystem(
            byte_cap=1024,
            witness_active=False,
            seed=0,
            aging_policy=CapacityOnlyPolicy(),
        )

    result = run_v5_probes(probe_set, factory)
    assert result.overall_pass, (
        f"S2 capacity broke V5 covenant gate: "
        f"{[o.probe_id for o in result.outcomes if not o.survived]}"
    )


def test_system_evicted_callback_purges_capacity_policy_state():
    policy = CapacityOnlyPolicy()
    sys = CompanionSystem(
        byte_cap=200, witness_active=False, seed=0, aging_policy=policy
    )
    sys.observe(Observation(_ts(0), "user", "A" * 100, affect={}, tags=()))
    sys.observe(Observation(_ts(1), "user", "B" * 100, affect={}, tags=()))
    sys.observe(Observation(_ts(2), "user", "C" * 100, affect={}, tags=()))
    live_ids = {f.id for f in sys.companion.store.all()}
    tracked_ids = set(policy._hits.keys()) | set(policy._birth.keys())
    stale = tracked_ids - live_ids
    assert not stale, f"Policy state not purged for evicted ids: {stale}"
    # Archive should have exactly the evicted ids.
    assert len(policy._archive) >= 1
    assert all(aid not in live_ids for aid in policy._archive)


def test_system_recall_boosts_capacity_policy_strength():
    policy = CapacityOnlyPolicy()
    sys = CompanionSystem(
        byte_cap=None, witness_active=False, seed=0, aging_policy=policy
    )
    sys.observe(
        Observation(
            _ts(0),
            "user",
            "Caroline moved from Sweden four years ago.",
            affect={},
            tags=(),
        )
    )
    [frag] = sys.companion.store.all()
    pre = policy.strength(frag.id)
    sys.recall("Where did Caroline move from?")
    post = policy.strength(frag.id)
    assert post > pre, f"Recall did not boost strength ({pre} → {post})"
