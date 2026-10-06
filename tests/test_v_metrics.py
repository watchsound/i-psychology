from __future__ import annotations

from datetime import datetime, timedelta, timezone

from psychology_base.eval.adapters import Observation
from psychology_base.eval.baselines import EchoMemory
from psychology_base.eval.v_metrics import measure_v_metrics


def _scenario() -> list[Observation]:
    base = datetime(2026, 1, 15, 19, 0, tzinfo=timezone.utc)
    return [
        Observation(
            timestamp=base,
            speaker="user",
            content="My father was never there on Friday evenings.",
            tags=("father", "childhood"),
        ),
        Observation(
            timestamp=base + timedelta(days=90),
            speaker="user",
            content="I think my father was overwhelmed, trying in his way.",
            tags=("father", "childhood", "reinterpretation"),
        ),
    ]


def test_echo_memory_v_metrics_hit_expected_baseline_floor():
    """EchoMemory is intentionally conscious-dead; every V-metric should land
    OUTSIDE its target band — confirming the metrics are discriminative.

    Phase-1-locked target bands (from docs/sota-goal.md):
        V1 state continuity:        0.70 – 0.95   (Echo expected ≈ 1.0)
        V2 adjacent-tick Jaccard:   0.65 – 0.80   (Echo expected ≈ 1.0)
        V2 long-horizon Jaccard:    0.20 – 0.35   (Echo expected ≈ 1.0)
        V3 mentalistic liveness:    0.10 – 0.30   (Echo expected ≈ 0.0)
    """
    m = measure_v_metrics(
        memory_factory=EchoMemory,
        probe_cue="father",
        scenario_observations=_scenario(),
    )

    # Each metric is outside its target band — baseline is conscious-dead by design.
    assert m.v1_state_continuity > 0.95, m
    assert m.v2_adjacent_jaccard > 0.80, m
    assert m.v2_longhorizon_jaccard > 0.35, m
    assert m.v3_mentalistic_liveness < 0.10, m

    # V4 and V5 are correctly flagged N/A for a baseline that has neither mechanism.
    assert m.v4_mirror_ablation is None
    assert m.v5_rule_internalization is None
    assert "v4" in m.notes
    assert "v5" in m.notes


def test_v_metrics_are_deterministic_across_runs():
    """Running the same measurement twice must produce byte-identical numbers.
    Phase 0 baseline reproducibility gate."""
    a = measure_v_metrics(EchoMemory, "father", _scenario())
    b = measure_v_metrics(EchoMemory, "father", _scenario())
    assert a == b
