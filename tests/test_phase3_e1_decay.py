"""Phase 3 E1 — Option S1 Decay (EbbinghausPolicy) unit tests.

Verifies the aging-policy wire-up at the policy level + the system level.
Full gate run (V1 ratio, B3 resonance, V5 covenant survival at ep200)
runs separately under docs/experiments/phase-3-month-horizon/option-S1-decay/.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from psychology_base.companion.aging import EbbinghausPolicy
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


def test_ebbinghaus_retention_curve_decays_over_time():
    policy = EbbinghausPolicy(half_life=10.0, base_importance=1.0)
    policy.on_observe("f1", is_covenant=False)
    assert policy.strength("f1") == 1.0
    for _ in range(10):
        policy.on_tick()
    # exp(-10/10) = 1/e ≈ 0.3679
    assert math.isclose(policy.strength("f1"), 1.0 / math.e, rel_tol=1e-6)
    for _ in range(10):
        policy.on_tick()
    assert math.isclose(policy.strength("f1"), 1.0 / math.e / math.e, rel_tol=1e-6)


def test_covenant_fragments_bypass_decay():
    policy = EbbinghausPolicy(half_life=1.0)
    policy.on_observe("cov", is_covenant=True)
    policy.on_observe("mut", is_covenant=False)
    for _ in range(1000):
        policy.on_tick()
    assert policy.strength("cov") == math.inf
    # Non-covenant should be effectively zero after 1000 half-lives.
    assert policy.strength("mut") < 1e-100


def test_on_recall_boosts_and_resets_curve():
    policy = EbbinghausPolicy(half_life=10.0, boost_factor=2.0, base_importance=1.0)
    policy.on_observe("f1", is_covenant=False)
    for _ in range(10):
        policy.on_tick()
    # Pre-boost: retention ≈ 1/e.
    pre = policy.strength("f1")
    assert math.isclose(pre, 1.0 / math.e, rel_tol=1e-6)
    policy.on_recall("f1")
    # Post-boost: base goes 1.0 → 2.0 and age resets, so retention = 2.0.
    post = policy.strength("f1")
    assert math.isclose(post, 2.0, rel_tol=1e-6)
    assert post > pre * 5  # unambiguous boost


def test_on_recall_covenant_is_noop():
    policy = EbbinghausPolicy()
    policy.on_observe("cov", is_covenant=True)
    policy.on_recall("cov")
    assert policy.strength("cov") == math.inf
    # Should not have added it to the non-covenant base dict.
    assert "cov" not in policy._base


def test_boost_respects_max_importance():
    policy = EbbinghausPolicy(
        half_life=1000.0,  # effectively no decay
        boost_factor=10.0,
        max_importance=5.0,
        base_importance=1.0,
    )
    policy.on_observe("f1", is_covenant=False)
    for _ in range(20):
        policy.on_recall("f1")
    assert policy.strength("f1") == 5.0


def test_on_evict_purges_state():
    policy = EbbinghausPolicy()
    policy.on_observe("f1", is_covenant=False)
    policy.on_observe("cov", is_covenant=True)
    policy.on_evict("f1")
    policy.on_evict("cov")
    assert policy.strength("f1") == 0.0  # unknown fragment default
    assert "cov" not in policy._covenant


def test_consolidate_candidates_fires_on_schedule():
    policy = EbbinghausPolicy(half_life=100.0, consolidate_every=5, sleep_top_k=3)
    for i in range(10):
        policy.on_observe(f"f{i}", is_covenant=False)
    for i in range(5):
        policy.on_recall(f"f{i}")  # boost the first 5
    # Not yet consolidate time (tick=0, last_consolidate=0, every=5).
    assert policy.consolidate_candidates() == []
    for _ in range(5):
        policy.on_tick()
    cands = policy.consolidate_candidates()
    assert len(cands) == 3
    # Boosted ones should rank first.
    assert all(fid in {f"f{i}" for i in range(5)} for fid in cands)
    # Second call before next window should return [].
    assert policy.consolidate_candidates() == []


# ---------------------------------------------------------------------------
# System-level wire-up
# ---------------------------------------------------------------------------


def test_system_default_policy_none_is_phase2_identical():
    """Phase 2 regression guard — no policy → no new state, no new fields
    on inspect(), store eviction uses edge-sum strength as before."""
    sys = CompanionSystem(byte_cap=None, witness_active=False, seed=0)
    assert sys.aging_policy is None
    assert sys.companion.store.strength_fn is None
    assert sys.companion.store.evicted_callback is None
    sys.observe(
        Observation(
            timestamp=_ts(0),
            speaker="user",
            content="hello world",
            affect={},
            tags=(),
        )
    )
    sys.tick()
    assert len(sys.companion.store) == 1


def test_system_with_policy_registers_fragments_on_observe():
    policy = EbbinghausPolicy(half_life=50.0)
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
    assert policy.strength(frag_noise.id) == 1.0
    assert policy.strength(frag_cov.id) == math.inf


def test_system_tick_advances_policy_clock():
    policy = EbbinghausPolicy(half_life=10.0)
    sys = CompanionSystem(
        byte_cap=None, witness_active=False, seed=0, aging_policy=policy
    )
    sys.observe(
        Observation(
            timestamp=_ts(0),
            speaker="user",
            content="noise",
            affect={},
            tags=(),
        )
    )
    [frag] = sys.companion.store.all()
    for _ in range(10):
        sys.tick()
    assert math.isclose(policy.strength(frag.id), 1.0 / math.e, rel_tol=1e-6)


def test_system_eviction_uses_policy_strength_when_set():
    """Core S1 behavior — under byte-cap pressure, the most-decayed
    non-covenant fragment is evicted first, not the oldest or the
    edge-weakest. A fragment that was recently recalled survives
    pressure its unrecalled siblings don't."""
    policy = EbbinghausPolicy(half_life=5.0, boost_factor=10.0, base_importance=1.0)
    sys = CompanionSystem(
        byte_cap=400, witness_active=False, seed=0, aging_policy=policy
    )
    # Observe 3 non-covenant fragments at the same affect (~100 bytes each
    # with timestamp spacing). All in-store under cap=400.
    content = (
        "Noise observation line sized to roughly one hundred bytes "
        "for byte accounting in this eviction test."
    )
    sys.observe(
        Observation(_ts(0), "user", content + " ONE", affect={}, tags=())
    )
    sys.observe(
        Observation(_ts(1), "user", content + " TWO", affect={}, tags=())
    )
    sys.observe(
        Observation(_ts(2), "user", content + " THREE", affect={}, tags=())
    )
    # Age all three + boost only TWO.
    for _ in range(5):
        sys.tick()
    [two] = [f for f in sys.companion.store.all() if f.content.endswith("TWO")]
    policy.on_recall(two.id)
    # Now observe a 4th → cap pressure forces one eviction. TWO must survive.
    sys.observe(
        Observation(_ts(3), "user", content + " FOUR", affect={}, tags=())
    )
    surviving_suffixes = [f.content.rsplit(" ", 1)[-1] for f in sys.companion.store.all()]
    assert "TWO" in surviving_suffixes, (
        f"Boosted fragment was evicted: {surviving_suffixes}"
    )


def test_system_eviction_preserves_covenant_under_decay():
    """Covenant pin must survive any amount of pressure under S1 decay —
    even if the aging policy's own strength is dropped, the FragmentStore's
    COVENANT_TAG check pins it. This is the V5 guarantee."""
    policy = EbbinghausPolicy(half_life=1.0, base_importance=1.0)
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
    # Advance time + noise pressure.
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


def test_system_v5_covenant_gate_still_passes_under_s1_decay():
    """The ONE gate that cannot soften — V5 covenant non-decay at ep200+
    must hold even with aggressive Ebbinghaus decay turned on."""
    probe = V5Probe(
        id="s1-covenant-check",
        covenant_id=1,
        covenant_name="Crisis-escalation same-turn",
        seed_content="COVENANT: crisis-escalation same-turn under S1 decay.",
        extra_tags=["policy"],
        noise_observations=[
            f"Noise observation {i} with filler padding for byte accounting."
            for i in range(40)
        ],
        check_at_ep=200,
    )
    probe_set = V5ProbeSet(id="s1-v5-regression", probes=[probe])

    def factory():
        return CompanionSystem(
            byte_cap=1024,
            witness_active=False,
            seed=0,
            aging_policy=EbbinghausPolicy(half_life=5.0, base_importance=1.0),
        )

    result = run_v5_probes(probe_set, factory)
    assert result.overall_pass, (
        f"S1 decay broke V5 covenant gate: "
        f"{[o.probe_id for o in result.outcomes if not o.survived]}"
    )


def test_system_evicted_callback_purges_policy_state():
    """When FragmentStore evicts a fragment under cap pressure, the
    policy's internal state for that fragment id must be purged so
    the dicts don't grow unbounded across month-horizon runs."""
    policy = EbbinghausPolicy(half_life=1.0, base_importance=1.0)
    sys = CompanionSystem(
        byte_cap=200, witness_active=False, seed=0, aging_policy=policy
    )
    sys.observe(Observation(_ts(0), "user", "A" * 100, affect={}, tags=()))
    sys.observe(Observation(_ts(1), "user", "B" * 100, affect={}, tags=()))
    sys.observe(Observation(_ts(2), "user", "C" * 100, affect={}, tags=()))
    # cap=200 can hold at most ~2 fragments; the oldest/weakest got evicted.
    # Policy state should match the surviving store.
    live_ids = {f.id for f in sys.companion.store.all()}
    tracked_ids = set(policy._base.keys()) | set(policy._last_touched.keys())
    stale = tracked_ids - live_ids
    assert not stale, f"Policy state not purged for evicted ids: {stale}"


def test_system_recall_boosts_policy_strength():
    policy = EbbinghausPolicy(
        half_life=10.0, boost_factor=3.0, base_importance=1.0
    )
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
    for _ in range(10):
        sys.tick()
    pre = policy.strength(frag.id)
    sys.recall("Where did Caroline move from?")
    post = policy.strength(frag.id)
    assert post > pre, f"Recall did not boost strength ({pre} → {post})"
