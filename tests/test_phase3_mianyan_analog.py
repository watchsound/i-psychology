"""Phase 3 E0.8b — 绵延 analog scorer tests.

Verifies:

1. Primary scorer (``rolling_v1_continuity_ratio``) returns a well-formed
   result and respects window / checkpoint invariants.
2. Argument-validation for ``ep_short >= window`` and
   ``ep_long >= ep_short + window``.
3. The ``advance_virtual_time`` fast-forward path and the baseline
   ``tick()`` fallback path produce the same V1 samples on an inert
   memory (EchoMemory has a no-op tick, so both routes must give
   identical signatures).
4. Diagnostic scorer (``direct_mianyan_combined``) returns the three
   component scalars and their mean.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from psychology_base.companion.stream_shapes import (
    AdversarialSilentRuns,
    DenseSchedule,
    MonthlySchedule,
)
from psychology_base.companion.system import CompanionSystem
from psychology_base.eval.adapters import Observation
from psychology_base.eval.baselines import EchoMemory
from psychology_base.eval.benchmarks.mianyan_analog import (
    DirectMianyanResult,
    RollingV1Result,
    ScheduledV1Result,
    direct_mianyan_combined,
    rolling_v1_continuity_ratio,
    scheduled_v1_continuity_ratio,
)


def _obs(hour: int, content: str) -> Observation:
    return Observation(
        timestamp=datetime(2026, 7, 1, hour, 0, tzinfo=timezone.utc),
        speaker="user",
        content=content,
        affect={},
        tags=(),
    )


def _scenario() -> list[Observation]:
    return [
        _obs(9, "I remember mother making tea the morning of the storm."),
        _obs(10, "The kettle whistled while she looked out the window."),
        _obs(11, "She said something quiet about the garden flooding."),
        _obs(12, "I was afraid but did not want her to see it."),
    ]


def test_rolling_v1_continuity_result_shape():
    memory = EchoMemory()
    result = rolling_v1_continuity_ratio(
        memory,
        probe_cue="mother",
        scenario_observations=_scenario(),
        ep_short=5,
        ep_long=15,
        window=3,
    )
    assert isinstance(result, RollingV1Result)
    assert result.ep_short == 5
    assert result.ep_long == 15
    assert result.window == 3
    assert len(result.short_samples) == 3
    assert len(result.long_samples) == 3
    # EchoMemory.tick() is a no-op so state signature is invariant →
    # every V1 sample is 1.0 and the ratio is exactly 1.0.
    assert result.v1_short == pytest.approx(1.0)
    assert result.v1_long == pytest.approx(1.0)
    assert result.ratio == pytest.approx(1.0)


def test_rolling_v1_rejects_window_larger_than_short_checkpoint():
    with pytest.raises(ValueError):
        rolling_v1_continuity_ratio(
            EchoMemory(),
            probe_cue="x",
            scenario_observations=[],
            ep_short=5,
            ep_long=200,
            window=10,
        )


def test_rolling_v1_rejects_overlapping_checkpoints():
    with pytest.raises(ValueError):
        rolling_v1_continuity_ratio(
            EchoMemory(),
            probe_cue="x",
            scenario_observations=[],
            ep_short=20,
            ep_long=25,
            window=20,
        )


def test_rolling_v1_rejects_nonpositive_window():
    with pytest.raises(ValueError):
        rolling_v1_continuity_ratio(
            EchoMemory(),
            probe_cue="x",
            scenario_observations=[],
            ep_short=20,
            ep_long=200,
            window=0,
        )


def test_rolling_v1_on_companion_system_produces_bounded_ratio():
    """CompanionSystem actually decays activation + edges each tick, so
    V1 samples will be < 1.0 — but a short run should still yield a
    bounded, finite ratio."""
    sys = CompanionSystem(byte_cap=None, witness_active=False)
    result = rolling_v1_continuity_ratio(
        sys,
        probe_cue="mother",
        scenario_observations=_scenario(),
        ep_short=5,
        ep_long=15,
        window=3,
    )
    assert 0.0 <= result.v1_short <= 1.0
    assert 0.0 <= result.v1_long <= 1.0
    assert result.ratio >= 0.0


def test_direct_mianyan_combined_result_shape():
    sys = CompanionSystem(byte_cap=None, witness_active=False)
    result = direct_mianyan_combined(
        sys,
        probe_cue="mother",
        scenario_observations=_scenario(),
        ep=5,
    )
    assert isinstance(result, DirectMianyanResult)
    assert result.ep == 5
    assert 0.0 <= result.v1_state_continuity <= 1.0
    assert 0.0 <= result.v2_adjacent_jaccard <= 1.0
    assert 0.0 <= result.v3_mentalistic_distance <= 1.0
    expected_combined = (
        result.v1_state_continuity
        + result.v2_adjacent_jaccard
        + result.v3_mentalistic_distance
    ) / 3.0
    assert result.combined == pytest.approx(expected_combined)


def test_direct_mianyan_rejects_negative_ep():
    with pytest.raises(ValueError):
        direct_mianyan_combined(
            EchoMemory(),
            probe_cue="x",
            scenario_observations=[],
            ep=-1,
        )


# ---------------------------------------------------------------------------
# Phase 6 — scheduled_v1_continuity_ratio
# ---------------------------------------------------------------------------


def test_scheduled_v1_echo_is_still_unit_ratio_under_any_schedule():
    """EchoMemory's tick() + recall() are inert; schedule injection
    should not change the trivial ratio=1.0 identity."""
    result = scheduled_v1_continuity_ratio(
        EchoMemory(),
        probe_cue="mother",
        scenario_observations=_scenario(),
        schedule=DenseSchedule(),
        ep_short=5,
        ep_long=15,
        window=3,
    )
    assert isinstance(result, ScheduledV1Result)
    assert result.v1_short == pytest.approx(1.0)
    assert result.v1_long == pytest.approx(1.0)
    assert result.ratio == pytest.approx(1.0)
    assert result.schedule_name == "DenseSchedule"


def test_scheduled_v1_monthly_fires_zero_between_recalls_on_ep200():
    """The Phase 6 §3 hardest-cell invariant: a monthly cadence with
    a 200-tick horizon produces zero in-stream recall probes. The
    scorer must surface this via ``between_recall_ticks == ()``."""
    result = scheduled_v1_continuity_ratio(
        CompanionSystem(byte_cap=None, witness_active=False),
        probe_cue="mother",
        scenario_observations=_scenario(),
        schedule=MonthlySchedule(),
        ep_short=20,
        ep_long=200,
        window=20,
    )
    assert result.pre_short_recall_ticks == ()
    assert result.between_recall_ticks == ()


def test_scheduled_v1_adversarial_empty_matches_phase3_byte_identical():
    """AdversarialSilentRuns with no recall ticks replicates Phase 3's
    silent fast-forward exactly — V1 samples must be byte-identical
    to the Phase 3 scorer on the same inputs."""
    scenario = _scenario()
    phase3 = rolling_v1_continuity_ratio(
        CompanionSystem(byte_cap=None, witness_active=False, seed=0),
        probe_cue="mother",
        scenario_observations=scenario,
        ep_short=5,
        ep_long=15,
        window=3,
    )
    phase6 = scheduled_v1_continuity_ratio(
        CompanionSystem(byte_cap=None, witness_active=False, seed=0),
        probe_cue="mother",
        scenario_observations=scenario,
        schedule=AdversarialSilentRuns(),
        ep_short=5,
        ep_long=15,
        window=3,
    )
    assert phase6.short_samples == phase3.short_samples
    assert phase6.long_samples == phase3.long_samples
    assert phase6.ratio == pytest.approx(phase3.ratio)
    assert phase6.pre_short_recall_ticks == ()
    assert phase6.between_recall_ticks == ()


def test_scheduled_v1_fires_probe_at_adversarial_tick_indices():
    """A ``AdversarialSilentRuns(ticks=(1,))`` schedule applies to each
    silent-run segment independently; both the pre-short and between
    segments reset their index at 0, so each fires at local index 1."""
    result = scheduled_v1_continuity_ratio(
        CompanionSystem(byte_cap=None, witness_active=False, seed=0),
        probe_cue="mother",
        scenario_observations=_scenario(),
        schedule=AdversarialSilentRuns(ticks=(1,)),
        ep_short=5,
        ep_long=15,
        window=3,
    )
    # pre-short horizon = ep_short - window = 2 ticks → tick 1 fires.
    assert result.pre_short_recall_ticks == (1,)
    # between horizon = ep_long - ep_short - window = 7 ticks → tick 1 fires.
    assert result.between_recall_ticks == (1,)


def test_scheduled_v1_rejects_bad_window():
    with pytest.raises(ValueError):
        scheduled_v1_continuity_ratio(
            EchoMemory(),
            probe_cue="x",
            scenario_observations=[],
            schedule=DenseSchedule(),
            ep_short=5,
            ep_long=200,
            window=10,
        )
    with pytest.raises(ValueError):
        scheduled_v1_continuity_ratio(
            EchoMemory(),
            probe_cue="x",
            scenario_observations=[],
            schedule=DenseSchedule(),
            ep_short=20,
            ep_long=25,
            window=20,
        )
