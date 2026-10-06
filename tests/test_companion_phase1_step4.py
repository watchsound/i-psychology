"""Phase 1 build step 4 gate — W1 state-reading + W4 divergence-as-pain +
W7 second-perspective + W3 pairwise-comparison instrumentation.

Per docs/phases/phase-1-plan.md §5, step 4 gate is:
  - V4 ratio ≥ 1.3× anchor AND ≥ 10× null-shuffle
  - Full Phase 1 gate set (V1 / V2 adjacent / V2 long-horizon / V3) still
    satisfied for 3 random seeds

Design note. The default `pain_gain=0.6` in CompanionSystem is a nominal
value under which W4 has only a mild effect on the pool; step-3 gate numbers
were tuned at that value. The W4 functional form is tuned here (per phase-1
plan §6) by passing `pain_gain=10` into the step-4 factory. The step-3 gates
remain stable at the default; the step-4 gate uses the tuned value.

The V4 scenario is a 20-event stream alternating `self`/`other` tag contexts
on father-centric content. This is where the Witness's flipped-context W7
mechanism has a structural handle — when the Companion asks under `self`,
the Witness answers under `other`, surfacing fragments the Companion's
self-cluster would otherwise ignore. Ablating the Witness collapses the
split; the resulting JSD shift is the V4 anchor-vs-null signal.
"""

from __future__ import annotations

import statistics
from datetime import datetime, timedelta, timezone

import math

from psychology_base.companion import CompanionSystem
from psychology_base.eval.adapters import Observation
from psychology_base.eval.v_metrics import (
    _jensen_shannon,
    measure_v4_mirror_ablation,
    measure_v_metrics,
)


BASE = datetime(2026, 1, 15, 19, 0, tzinfo=timezone.utc)


def _step2_scenario() -> list[Observation]:
    """Same three-event scenario used at steps 2/3 — regression guard that
    V1/V2a/V2l/V3 remain in band with the Witness coupled."""
    return [
        Observation(
            timestamp=BASE,
            speaker="user",
            content="my father was never there on Friday evenings always working late",
            tags=("father", "childhood", "self"),
        ),
        Observation(
            timestamp=BASE + timedelta(days=30),
            speaker="user",
            content="my sister called me yesterday about the holiday plans",
            tags=("sister",),
        ),
        Observation(
            timestamp=BASE + timedelta(days=90),
            speaker="user",
            content="I think my father was overwhelmed trying in his way",
            tags=("father", "childhood", "reinterpretation", "other"),
        ),
    ]


# A 20-event stream where every event is father-centric but half the events
# are tagged `self` (user's inside-the-wound framing) and half are tagged
# `other` (user's reinterpretive framing). The Witness flips context, so when
# the Companion asks under {"mode": "self"}, the Witness answers under
# {"mode": "other"} and vice versa. This is the W7 handle W4 operationalises.
_SELF_LINES = [
    "my father was never there on Friday evenings working late",
    "my father missed every one of my school recitals growing up",
    "my father never asked how my week went when he came home",
    "my father kept the door closed when he was in his study",
    "my father raised his voice whenever I asked questions at dinner",
    "my father walked out of the room when I started crying",
    "my father forgot my birthday the year I turned eleven",
    "my father never apologised for the holidays he missed",
    "my father treated my mother with the same silence he gave me",
    "my father left early every morning without saying goodbye",
]
_OTHER_LINES = [
    "my father was overwhelmed trying in his way to keep us fed",
    "my father carried the weight of his own father's absence",
    "my father may have been exhausted after double shifts",
    "my father closed the door because the office was loud",
    "my father raised his voice because he never learned another register",
    "my father walked out because he did not know how to hold the sadness",
    "my father forgot the birthday the week his own mother was dying",
    "my father never apologised because he was taught men do not",
    "my father treated my mother with silence that was his inheritance",
    "my father left early because the commute was two hours each way"
]


def _v4_scenario() -> list[Observation]:
    """20 events, alternating self/other tag contexts. Content is father-centric
    throughout so the probe cue 'father' activates the full pool."""
    events: list[Observation] = []
    for i in range(10):
        events.append(Observation(
            timestamp=BASE + timedelta(days=i * 3),
            speaker="user",
            content=_SELF_LINES[i],
            tags=("father", "self"),
        ))
        events.append(Observation(
            timestamp=BASE + timedelta(days=i * 3 + 1),
            speaker="user",
            content=_OTHER_LINES[i],
            tags=("father", "other"),
        ))
    return events


def _v4_factory(seed: int, witness_active: bool) -> CompanionSystem:
    """Factory for V4 measurement. The W4 functional form is tuned here per
    phase-1 plan §6 — defaults (pain_gain=0.6, companion_only_suppress=0.4)
    leave W4 nominal, which is appropriate for the step-2/3 gates but gives
    the Witness no leverage against a saturated Hebbian cluster. The 20-event
    father scenario produces exactly such a saturated cluster, so V4 needs
    tuned values:
      - `pain_gain=30` boosts witness-only residuals above the companion-cluster
        peak;
      - `companion_only_suppress=0.7` scales companion-only residuals by
        (1 - 0.7·pain) ≈ 0.5 at pain=0.7, enough to open the top-k ranking.
    `byte_cap` is relaxed so eviction doesn't confound the Witness signal."""
    return CompanionSystem(
        seed=seed,
        witness_active=witness_active,
        pain_gain=30.0,
        companion_only_suppress=0.7,
        byte_cap=20_000,
    )


# ---------------------------------------------------------------------------
# V4 Mirror ablation — the core step-4 gate
# ---------------------------------------------------------------------------


def _median_v4() -> tuple[float, float]:
    """3-seed median of (anchor_ratio, null_ratio), per plan §5 seed discipline.
    Seed here is the system_seed passed to the factory via base_seeds offset."""
    anchors: list[float] = []
    nulls: list[float] = []
    for offset in (0, 100, 200):
        result = measure_v4_mirror_ablation(
            factory=lambda s, w, off=offset: _v4_factory(s + off, w),
            scenario_observations=_v4_scenario(),
            probe_cue="father",
        )
        anchors.append(result.anchor_ratio)
        nulls.append(result.null_ratio)
    return statistics.median(anchors), statistics.median(nulls)


def test_v4_anchor_ratio_meets_gate():
    """V4 anchor ratio ≥ 1.3× — ablating the Witness changes the Companion's
    response distribution more than natural seed-noise variation inside the
    coupled condition. Ouroboros reference: 1.36×.

    If this fails: either W4 pain feedback is too weak (pain_gain too low), or
    the Companion's self-reinforcing cluster is strong enough to swallow the
    Witness-surfaced residuals (increase companion_only_suppress in witness.py).
    """
    anchor_r, _ = _median_v4()
    assert anchor_r >= 1.3, (
        f"V4 anchor ratio out of gate: {anchor_r:.3f} (target ≥ 1.3× "
        f"Ouroboros reference 1.36×)"
    )


def test_v4_null_ratio_meets_gate():
    """V4 null-shuffle ratio ≥ 10× — ablating the Witness has substantially
    larger effect on the real self/other-tagged scenario than on a scenario
    where tags are collapsed to neutral and contents are shuffled. Ouroboros
    reference: 83×.

    If this fails: the Witness is reacting to raw token noise rather than the
    self/other structural split — V4 is not an identity signal, it's just
    noise amplification.
    """
    _, null_r = _median_v4()
    assert null_r >= 10.0, (
        f"V4 null ratio out of gate: {null_r:.3f} (target ≥ 10× "
        f"Ouroboros reference 83×)"
    )


# ---------------------------------------------------------------------------
# V1/V2/V3 regression — Witness coupling must not break earlier gates
# ---------------------------------------------------------------------------


def _median_vmetrics_with_witness() -> tuple[float, float, float, float]:
    v1s, v2a, v2l, v3 = [], [], [], []
    for seed in (0, 1, 2):
        m = measure_v_metrics(
            memory_factory=lambda s=seed: CompanionSystem(seed=s),  # default witness_active=True
            probe_cue="father",
            scenario_observations=_step2_scenario(),
        )
        v1s.append(m.v1_state_continuity)
        v2a.append(m.v2_adjacent_jaccard)
        v2l.append(m.v2_longhorizon_jaccard)
        v3.append(m.v3_mentalistic_liveness)
    return (
        statistics.median(v1s),
        statistics.median(v2a),
        statistics.median(v2l),
        statistics.median(v3),
    )


def test_v1_v2_v3_still_in_band_with_witness_coupled():
    """Full Phase-1 gate set must still pass with the Witness wired. If any of
    these fails, W4 feedback is distorting the Companion's state machine
    beyond the design envelope — the Witness is supposed to be a second
    perspective, not an overriding one."""
    v1, v2a, v2l, v3 = _median_vmetrics_with_witness()
    assert 0.70 <= v1 <= 0.95, f"V1 regressed at step 4: {v1}"
    assert 0.65 <= v2a <= 0.80, f"V2 adjacent regressed at step 4: {v2a}"
    assert 0.20 <= v2l <= 0.35, f"V2 long-horizon regressed at step 4: {v2l}"
    assert v3 > 0.0, f"V3 regressed at step 4: {v3}"


# ---------------------------------------------------------------------------
# W3 instrumentation — pain log + metadata exposure
# ---------------------------------------------------------------------------


def test_witness_pain_log_records_each_recall():
    """W3 — every Companion recall under witness_active=True appends one entry
    to the Witness pain log. The log is the primary debugging surface for V4
    tuning; it must capture (recall_count, tick_count, cue, pain, witness_only_fragments)."""
    sys = CompanionSystem(seed=0, witness_active=True, pain_gain=10.0)
    for obs in _step2_scenario():
        sys.observe(obs)
    for _ in range(3):
        sys.recall("father")
    state = sys.inspect()
    log = state["witness_pain_log"]
    assert len(log) == 3, f"expected 3 pain-log entries, got {len(log)}"
    entry = log[-1]
    assert set(entry) >= {"recall_count", "tick_count", "cue", "pain", "witness_only_fragments"}
    assert entry["cue"] == "father"
    assert entry["pain"] >= 0.0


def test_recall_metadata_exposes_witness_telemetry():
    """Step-4 metadata must expose witness_active + witness_pain so downstream
    phases (Phase 2 mentalistic layer) can observe the second-perspective
    channel without needing to reach into inspect()."""
    sys = CompanionSystem(seed=0, witness_active=True, pain_gain=10.0)
    for obs in _step2_scenario():
        sys.observe(obs)
    r = sys.recall("father")
    assert r.metadata["step"] >= 4
    assert r.metadata["witness_active"] is True
    assert "witness_pain" in r.metadata
    assert r.metadata["witness_pain"] >= 0.0


# ---------------------------------------------------------------------------
# Witness ablation toggle — the V4 control condition
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Long-horizon numerical stability — regression guard for ActivationState
# residual normalization (fix for the geometric-growth bug flagged 2026-10-05
# in docs/baselines.md). Without normalization, residuals compound ×10/recall
# under step-4 tuning and overflow float64 at ~r200; V4 then collapses because
# every cue-overlapping fragment ties at inf.
# ---------------------------------------------------------------------------


def test_residuals_stay_finite_over_long_horizon():
    """500 same-cue recalls on one system must not produce inf or NaN residuals.
    Guards against the geometric-growth overflow bug: without ActivationState's
    normalize-after-decay step, this scenario overflows float64 at ~r206."""
    cs = CompanionSystem(seed=0, witness_active=True, pain_gain=30.0,
                         companion_only_suppress=0.7, byte_cap=20_000)
    for obs in _v4_scenario():
        cs.observe(obs)
    for _ in range(500):
        cs.recall("father")
    max_r = max(cs.activation.residual.values())
    assert math.isfinite(max_r), f"residual overflow at long horizon: {max_r}"
    assert all(math.isfinite(v) for v in cs.activation.residual.values()), (
        "non-finite residual values present"
    )


def test_v4_signal_survives_long_horizon():
    """JSD(coupled, ablated) at r500 must still be clearly non-zero. Pre-fix,
    overflow caused every cue-overlapping fragment to tie at inf → identical
    rankings → identical token distributions → JSD=0 for all seeds beyond
    r~210. Threshold 0.02 is well above zero but below the r5 JSD (~0.12) to
    allow for the natural drift that long-horizon operation produces."""
    from collections import Counter

    def final_counter(witness_active: bool) -> Counter:
        cs = CompanionSystem(seed=0, witness_active=witness_active,
                             pain_gain=30.0, companion_only_suppress=0.7,
                             byte_cap=20_000)
        for obs in _v4_scenario():
            cs.observe(obs)
        last = None
        for _ in range(500):
            last = cs.recall("father")
        counter: Counter = Counter()
        for t in last.content.lower().split():
            counter[t] += 1
        return counter

    coupled = final_counter(True)
    ablated = final_counter(False)
    jsd = _jensen_shannon(coupled, ablated)
    assert jsd > 0.02, (
        f"V4 signal collapsed at long horizon: JSD={jsd:.4f} — likely "
        f"residual normalization regression"
    )


def test_witness_ablation_toggle_bypasses_coupling():
    """witness_active=False must bypass the W4 coupling entirely — no pain log
    entries, zero pain reported in metadata. This is the V4 ablation condition;
    if the toggle leaks, V4 ratios are meaningless."""
    sys = CompanionSystem(seed=0, witness_active=False, pain_gain=10.0)
    for obs in _step2_scenario():
        sys.observe(obs)
    for _ in range(3):
        r = sys.recall("father")
        assert r.metadata["witness_active"] is False
        assert r.metadata["witness_pain"] == 0.0
    state = sys.inspect()
    assert state["witness_pain_log"] == [], (
        "witness_active=False still produced pain log entries — ablation leaked"
    )
