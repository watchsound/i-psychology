"""Unit tests for Phase 6 recall-cadence stream shapes.

The schedule predicates are pure math (period, offset, membership), so
these tests exercise them directly rather than through a full
``CompanionSystem`` run. :func:`walk_schedule` is tested against a
minimal fake system to confirm the ``tick()`` + ``probe_fn`` ordering
and the ``recall_ticks_fired`` ledger match the schedule's own
``recall_ticks`` enumeration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from psychology_base.companion.stream_shapes import (
    AdversarialSilentRuns,
    DenseSchedule,
    MonthlySchedule,
    ScheduleRun,
    WeeklySchedule,
    walk_schedule,
)


# ---------------------------------------------------------------------------
# Dense
# ---------------------------------------------------------------------------


def test_dense_default_fires_every_tick():
    s = DenseSchedule()
    assert s.should_recall(0)
    assert s.should_recall(1)
    assert s.should_recall(99)
    assert s.recall_ticks(5) == (0, 1, 2, 3, 4)


def test_dense_strided_every_two():
    s = DenseSchedule(every=2)
    assert s.recall_ticks(6) == (0, 2, 4)
    assert s.should_recall(0)
    assert not s.should_recall(1)
    assert s.should_recall(2)


def test_dense_rejects_zero_or_negative_every():
    with pytest.raises(ValueError):
        DenseSchedule(every=0)
    with pytest.raises(ValueError):
        DenseSchedule(every=-1)


def test_dense_rejects_negative_tick_index():
    s = DenseSchedule()
    with pytest.raises(ValueError):
        s.should_recall(-1)


def test_dense_horizon_zero_is_empty():
    assert DenseSchedule().recall_ticks(0) == ()


# ---------------------------------------------------------------------------
# Weekly
# ---------------------------------------------------------------------------


def test_weekly_default_period_is_168_ticks():
    s = WeeklySchedule()
    assert s.period == 24 * 7
    # Default offset = period - 1 (end of first week).
    assert s.should_recall(167)
    assert not s.should_recall(166)
    assert not s.should_recall(0)


def test_weekly_fires_at_end_of_each_week():
    s = WeeklySchedule()
    # 500-tick horizon → ticks 167, 335 (not 503 — out of range).
    assert s.recall_ticks(500) == (167, 335)
    assert s.should_recall(335)
    assert not s.should_recall(336)


def test_weekly_offset_fires_at_offset_then_period():
    s = WeeklySchedule(offset=0)
    assert s.should_recall(0)
    assert s.should_recall(168)
    assert not s.should_recall(167)


def test_weekly_horizon_shorter_than_period_fires_zero():
    s = WeeklySchedule()  # default offset = 167
    assert s.recall_ticks(100) == ()


def test_weekly_custom_cadence():
    s = WeeklySchedule(ticks_per_day=1, days_per_week=5, offset=0)
    assert s.period == 5
    assert s.recall_ticks(20) == (0, 5, 10, 15)


def test_weekly_rejects_bad_params():
    with pytest.raises(ValueError):
        WeeklySchedule(ticks_per_day=0)
    with pytest.raises(ValueError):
        WeeklySchedule(days_per_week=0)
    with pytest.raises(ValueError):
        WeeklySchedule(offset=-1)


# ---------------------------------------------------------------------------
# Monthly
# ---------------------------------------------------------------------------


def test_monthly_default_period_is_720_ticks():
    s = MonthlySchedule()
    assert s.period == 24 * 30


def test_monthly_fires_at_720_under_default():
    s = MonthlySchedule()
    assert s.recall_ticks(1000) == (719,)
    assert s.should_recall(719)
    assert not s.should_recall(720)


def test_monthly_ep500_hardest_cell_fires_zero_recalls():
    """Phase 6 §3 names ep500 sparse-monthly as the hardest cell —
    a 500-tick horizon under a 720-tick period fires no recalls at
    all. This test pins that invariant so Phase 6 runners can assert
    on ``ScheduleRun.recall_ticks_fired == ()`` as the signal that
    the "silent horizon longer than cadence" case is actually
    exercised."""
    s = MonthlySchedule()
    assert s.recall_ticks(500) == ()


def test_monthly_offset_zero():
    s = MonthlySchedule(offset=0)
    assert s.should_recall(0)
    assert s.should_recall(720)


# ---------------------------------------------------------------------------
# Adversarial
# ---------------------------------------------------------------------------


def test_adversarial_fires_only_at_named_ticks():
    s = AdversarialSilentRuns(ticks=(5, 50, 150))
    assert s.should_recall(5)
    assert s.should_recall(50)
    assert s.should_recall(150)
    assert not s.should_recall(4)
    assert not s.should_recall(51)
    assert not s.should_recall(149)
    assert not s.should_recall(0)


def test_adversarial_canonicalizes_ticks():
    s = AdversarialSilentRuns(ticks=(150, 5, 50, 5))
    assert s.ticks == (5, 50, 150)


def test_adversarial_rejects_negative_tick():
    with pytest.raises(ValueError):
        AdversarialSilentRuns(ticks=(-1,))


def test_adversarial_recall_ticks_respects_horizon():
    s = AdversarialSilentRuns(ticks=(5, 50, 150, 500))
    assert s.recall_ticks(200) == (5, 50, 150)
    assert s.recall_ticks(1000) == (5, 50, 150, 500)
    assert s.recall_ticks(0) == ()


def test_adversarial_empty_schedule_is_pure_silence():
    s = AdversarialSilentRuns()
    assert s.recall_ticks(100) == ()
    assert not s.should_recall(0)


# ---------------------------------------------------------------------------
# walk_schedule
# ---------------------------------------------------------------------------


@dataclass
class _FakeSystem:
    """Minimal stand-in: counts ``tick()`` calls and records probe hits.

    Avoids spinning up a full :class:`CompanionSystem` for a scheduler
    test. The real integration lives in the Phase 6 runners.
    """

    tick_count: int = 0
    probed_at: list[int] = field(default_factory=list)

    def tick(self) -> None:
        self.tick_count += 1


def test_walk_schedule_advances_full_horizon():
    s = DenseSchedule()
    sys = _FakeSystem()
    run = walk_schedule(sys, s, horizon=10)
    assert sys.tick_count == 10
    assert run.horizon == 10
    assert run.schedule_name == "DenseSchedule"


def test_walk_schedule_fires_probe_on_recall_ticks_only():
    s = AdversarialSilentRuns(ticks=(2, 7))
    sys = _FakeSystem()
    outputs: list[str] = []

    def probe(system, i):
        system.probed_at.append(i)
        outputs.append(f"probe@{i}")
        return f"result-{i}"

    run = walk_schedule(sys, s, horizon=10, probe_fn=probe)
    assert sys.tick_count == 10
    assert sys.probed_at == [2, 7]
    assert run.recall_ticks_fired == (2, 7)
    assert run.probe_results == ("result-2", "result-7")
    assert outputs == ["probe@2", "probe@7"]


def test_walk_schedule_without_probe_fn_still_records_fired_ticks():
    s = WeeklySchedule(ticks_per_day=1, days_per_week=5, offset=0)
    sys = _FakeSystem()
    run = walk_schedule(sys, s, horizon=20)
    assert sys.tick_count == 20
    assert run.recall_ticks_fired == (0, 5, 10, 15)
    assert run.probe_results == ()


def test_walk_schedule_tick_precedes_probe_each_step():
    """When tick i's probe fires, ``system.tick()`` must have been
    called i+1 times before ``probe_fn`` sees it — the schedule
    describes the *post-tick* state."""
    s = AdversarialSilentRuns(ticks=(0, 3))
    sys = _FakeSystem()
    tick_counts_at_probe: list[int] = []

    def probe(system, i):
        tick_counts_at_probe.append(system.tick_count)

    walk_schedule(sys, s, horizon=5, probe_fn=probe)
    # i=0 probe sees tick_count=1; i=3 probe sees tick_count=4.
    assert tick_counts_at_probe == [1, 4]


def test_walk_schedule_horizon_zero_is_no_op():
    s = DenseSchedule()
    sys = _FakeSystem()
    run = walk_schedule(sys, s, horizon=0)
    assert sys.tick_count == 0
    assert run.recall_ticks_fired == ()


def test_walk_schedule_rejects_negative_horizon():
    with pytest.raises(ValueError):
        walk_schedule(_FakeSystem(), DenseSchedule(), horizon=-1)


def test_walk_schedule_monthly_ep500_fires_zero_but_still_ticks_500():
    """The hardest Phase 6 cell: 500-tick horizon under monthly
    cadence advances the full stream silently."""
    s = MonthlySchedule()
    sys = _FakeSystem()
    run = walk_schedule(sys, s, horizon=500, probe_fn=lambda sys_, i: None)
    assert sys.tick_count == 500
    assert run.recall_ticks_fired == ()


# ---------------------------------------------------------------------------
# ScheduleRun container
# ---------------------------------------------------------------------------


def test_schedule_run_notes_default_empty():
    r = ScheduleRun(
        schedule_name="DenseSchedule",
        horizon=10,
        recall_ticks_fired=(0, 1, 2),
    )
    assert r.notes == {}
    assert r.probe_results == ()
