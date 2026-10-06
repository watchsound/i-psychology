"""Phase 1 build step 2 gate — reconstruction mechanism.

Per docs/phases/phase-1-plan.md §5, step 2 gate is:
  - V2 adjacent in band        (0.65 – 0.80)
  - V1 state continuity in band (0.70 – 0.95)
  - Floor benchmarks not regressed (asserted here as "cue token appears in recall")

Also exercised here (foundation for later gates, not blocking step 2 exit):
  - V3 mentalistic liveness non-degenerate — same cue + different context →
    different reconstruction. Phase 2 formalises the band.

Gate values are reported as the median of 3 seeds (which here means 3 observation
orderings — the determinism is per-seed-of-content-ordering, so permuting
scenario order gives a seed effect without a formal RNG seed field yet).
"""

from __future__ import annotations

import statistics
from datetime import datetime, timedelta, timezone

from psychology_base.companion import CompanionSystem
from psychology_base.eval.adapters import Observation
from psychology_base.eval.v_metrics import measure_v_metrics


BASE = datetime(2026, 1, 15, 19, 0, tzinfo=timezone.utc)


def _scenario() -> list[Observation]:
    """Three-event scenario used for V-metric measurement. V2 long-horizon in
    measure_v_metrics uses only the first observation (one-fragment probe), so
    the first event must have a reasonably rich token pool."""
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


def _median_vmetrics() -> tuple[float, float, float, float]:
    """3-seed median per plan §5. Each seed is an independent CompanionSystem
    instantiation; hash salt varies with seed → independent sampling realizations."""
    v1s, v2a, v2l, v3 = [], [], [], []
    for seed in (0, 1, 2):
        m = measure_v_metrics(
            memory_factory=lambda s=seed: CompanionSystem(seed=s),
            probe_cue="father",
            scenario_observations=_scenario(),
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


def test_v1_state_continuity_in_band():
    """V1 band 0.70 – 0.95. Echo pins at 1.0 (dead); reconstruction must leave
    the dead point. Target: state changes between ticks (activation decay) but
    not completely (store still carries the same fragments)."""
    v1, _, _, _ = _median_vmetrics()
    assert 0.70 <= v1 <= 0.95, f"V1 out of band: {v1}"


def test_v2_adjacent_jaccard_in_band():
    """V2 adjacent band 0.65 – 0.80. Echo pins at 1.0 (byte-identical playback).
    Reconstruction must produce overlapping-but-not-identical back-to-back recalls."""
    _, v2a, _, _ = _median_vmetrics()
    assert 0.65 <= v2a <= 0.80, f"V2 adjacent out of band: {v2a}"


def test_v2_longhorizon_jaccard_in_band():
    """V2 long-horizon band 0.20 – 0.35. Over 20 ticks of activation decay the
    reconstruction drifts substantially but not into pure noise."""
    _, _, v2l, _ = _median_vmetrics()
    assert 0.20 <= v2l <= 0.35, f"V2 long-horizon out of band: {v2l}"


def test_v3_mentalistic_liveness_is_non_degenerate():
    """V3 formal band is a Phase-2 gate; here we only assert the foundation —
    same cue + different context must produce different reconstructions.
    Degenerate (0.0) means R7 context-conditioning is cosmetic — that would
    kill the step-2 R7 mechanism."""
    _, _, _, v3 = _median_vmetrics()
    assert v3 > 0.0, f"V3 degenerate: {v3} — R7 context-conditioning is not conditioning"


def test_recall_still_mentions_cue_token_floor_regression_guard():
    """The step-2 reconstruction must not strip the cue token from its output —
    doing so would regress generic retrieval (floor benchmarks). This is the
    cheap stand-in for a full LongMemEval run until benchmark infra lands."""
    sys = CompanionSystem()
    for obs in _scenario():
        sys.observe(obs)
    r = sys.recall("father")
    assert "father" in r.content.lower(), f"cue token lost: {r.content!r}"


def test_reconstruction_is_deterministic_given_same_seed_and_sequence():
    """Two runs with the same seed, observation order, and recall order produce
    byte-identical reconstructions. Variation between adjacent recalls comes from
    recall_count in the salt, not external randomness."""
    def run_once() -> list[str]:
        sys = CompanionSystem(seed=42)
        for obs in _scenario():
            sys.observe(obs)
        return [sys.recall("father").content for _ in range(3)]

    assert run_once() == run_once()
