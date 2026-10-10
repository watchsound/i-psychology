"""Phase 6 — recall-cadence stream shapes.

Phase 3's dense-recall stream (every tick in the sampling window fires
a recall against the same anchor cue) washed out the S1 vs S2
differentiation: the probe repeatedly reconsolidated the same anchor
fragments under both policies, so decay never had a silent run long
enough to pick a different victim than selection pressure alone. See
``docs/phases/phase-3-close-out.md`` §"stream-shape confound" and
``docs/phases/phase-6-month-horizon-plan.md`` §2 for the full framing.

This module is the Phase 6 §2 scope anchor: a configurable
recall-cadence schedule injected at session-player / runner level.
``CompanionSystem`` core stays stream-shape-agnostic — nothing here
reaches into the system's internals.

Each schedule is a predicate on **tick index within a horizon**. Index
0 is the first tick after setup; the predicate decides whether that
tick fires a recall probe or passes silently. The companion helper
:func:`walk_schedule` iterates a tick budget, calling ``system.tick()``
every step and ``probe_fn(system, tick_index)`` on recall ticks only.
Caller supplies ``probe_fn``, so this module stays measurement-agnostic
(V1 continuity, B3 affect, V5 covenant — all plug in the same way).

Convention: 1 tick ≈ 1 hour (same as Phase 3's compressed-time
harness). Day = 24 ticks, week = 168, month = 720. Horizons in the
Phase 6 plan are expressed in episodes (ep200 / ep500 / ep1000) where
one episode = one tick, matching the ``rolling_v1_continuity_ratio``
checkpoint convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence


class StreamSchedule(Protocol):
    """A recall-cadence schedule over a tick horizon.

    Implementations decide which 0-based tick indices (counted from the
    start of the horizon, not the system's internal clock) fire a
    recall probe. The schedule holds no reference to any
    ``CompanionSystem`` — it is a pure predicate so the same instance
    is reusable across multiple runs.
    """

    def should_recall(self, tick_index: int) -> bool:
        """Return True if tick ``tick_index`` fires a recall probe."""
        ...

    def recall_ticks(self, horizon: int) -> tuple[int, ...]:
        """List of 0-based tick indices firing a probe over ``horizon``.

        Equivalent to filtering ``range(horizon)`` by
        :meth:`should_recall`; schedules may override for efficiency.
        """
        ...


# ---------------------------------------------------------------------------
# Dense — Phase 3 default
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DenseSchedule:
    """Recall on every tick (Phase 3 default stream shape).

    The baseline against which sparse variants are compared. ``every>1``
    produces a strided-dense variant (every 2nd / 3rd tick) that is
    still "dense" relative to weekly/monthly scales — kept as a knob so
    the dense-schedule sanity re-run in Phase 6 §6 item 3 can thin the
    recall rate without jumping all the way to weekly.
    """

    every: int = 1

    def __post_init__(self) -> None:
        if self.every < 1:
            raise ValueError(f"DenseSchedule.every must be >= 1, got {self.every}")

    def should_recall(self, tick_index: int) -> bool:
        if tick_index < 0:
            raise ValueError(f"tick_index must be >= 0, got {tick_index}")
        return tick_index % self.every == 0

    def recall_ticks(self, horizon: int) -> tuple[int, ...]:
        if horizon < 0:
            raise ValueError(f"horizon must be >= 0, got {horizon}")
        return tuple(range(0, horizon, self.every))


# ---------------------------------------------------------------------------
# Weekly / monthly — fixed-period sparse
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WeeklySchedule:
    """Recall once per compressed week.

    Default period is one tick per hour × 24 hours × 7 days = 168 ticks.
    By convention the probe fires on the *last* tick of each week
    (indices ``period-1``, ``2*period-1``, …) so a horizon that is an
    exact multiple of ``period`` ends on a recall. ``offset`` shifts
    the first recall if the stream should start with a probe
    (``offset=0`` fires at index 0 then every ``period`` ticks after).
    """

    ticks_per_day: int = 24
    days_per_week: int = 7
    offset: int | None = None
    """First recall tick index. ``None`` → ``period-1`` (end of first week)."""

    def __post_init__(self) -> None:
        if self.ticks_per_day < 1:
            raise ValueError(
                f"ticks_per_day must be >= 1, got {self.ticks_per_day}"
            )
        if self.days_per_week < 1:
            raise ValueError(
                f"days_per_week must be >= 1, got {self.days_per_week}"
            )
        if self.offset is not None and self.offset < 0:
            raise ValueError(f"offset must be >= 0, got {self.offset}")

    @property
    def period(self) -> int:
        return self.ticks_per_day * self.days_per_week

    def _first(self) -> int:
        return self.period - 1 if self.offset is None else self.offset

    def should_recall(self, tick_index: int) -> bool:
        if tick_index < 0:
            raise ValueError(f"tick_index must be >= 0, got {tick_index}")
        first = self._first()
        if tick_index < first:
            return False
        return (tick_index - first) % self.period == 0

    def recall_ticks(self, horizon: int) -> tuple[int, ...]:
        if horizon < 0:
            raise ValueError(f"horizon must be >= 0, got {horizon}")
        first = self._first()
        if first >= horizon:
            return ()
        return tuple(range(first, horizon, self.period))


@dataclass(frozen=True)
class MonthlySchedule:
    """Recall once per compressed month (30 days by default).

    Same shape as :class:`WeeklySchedule` with a longer period. At the
    ep1000 horizon this fires once (end of month 1); on ep500 it does
    not fire at all unless ``offset`` is set — the horizon is shorter
    than one month. Phase 6 plan §3 names ``ep500 sparse-monthly`` as
    the hardest cell on purpose: it exercises the architecture under a
    silent run longer than the full horizon.
    """

    ticks_per_day: int = 24
    days_per_month: int = 30
    offset: int | None = None

    def __post_init__(self) -> None:
        if self.ticks_per_day < 1:
            raise ValueError(
                f"ticks_per_day must be >= 1, got {self.ticks_per_day}"
            )
        if self.days_per_month < 1:
            raise ValueError(
                f"days_per_month must be >= 1, got {self.days_per_month}"
            )
        if self.offset is not None and self.offset < 0:
            raise ValueError(f"offset must be >= 0, got {self.offset}")

    @property
    def period(self) -> int:
        return self.ticks_per_day * self.days_per_month

    def _first(self) -> int:
        return self.period - 1 if self.offset is None else self.offset

    def should_recall(self, tick_index: int) -> bool:
        if tick_index < 0:
            raise ValueError(f"tick_index must be >= 0, got {tick_index}")
        first = self._first()
        if tick_index < first:
            return False
        return (tick_index - first) % self.period == 0

    def recall_ticks(self, horizon: int) -> tuple[int, ...]:
        if horizon < 0:
            raise ValueError(f"horizon must be >= 0, got {horizon}")
        first = self._first()
        if first >= horizon:
            return ()
        return tuple(range(first, horizon, self.period))


# ---------------------------------------------------------------------------
# Adversarial — explicit sparse recall ticks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdversarialSilentRuns:
    """Recall only at explicitly-named tick indices; silent otherwise.

    The Phase 6 plan §2 names ``recalls at ep5 / ep50 / ep150, silent
    in between`` — long silent runs between short recall events,
    deliberately shaped to force decay to operate independently of the
    recall boost. If this schedule *also* produces byte-identical
    numbers under S1 vs S2, the Phase 3 stream-shape confound is
    upgraded to "mechanism tie" rather than "stream artefact".

    Ticks are deduplicated and sorted on construction; a tick index
    outside the eventual horizon is harmless (it simply never fires).
    """

    ticks: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if any(t < 0 for t in self.ticks):
            raise ValueError(f"all ticks must be >= 0, got {self.ticks}")
        # Freeze a canonical sorted/dedup representation. object.__setattr__
        # bypass is the pattern for mutating a frozen dataclass inside
        # __post_init__ — same shape as :class:`EbbinghausPolicy` elsewhere
        # doesn't need because it's not frozen.
        object.__setattr__(self, "ticks", tuple(sorted(set(self.ticks))))

    def should_recall(self, tick_index: int) -> bool:
        if tick_index < 0:
            raise ValueError(f"tick_index must be >= 0, got {tick_index}")
        return tick_index in self.ticks

    def recall_ticks(self, horizon: int) -> tuple[int, ...]:
        if horizon < 0:
            raise ValueError(f"horizon must be >= 0, got {horizon}")
        return tuple(t for t in self.ticks if t < horizon)


# ---------------------------------------------------------------------------
# walk_schedule — the Phase 6 runner glue
# ---------------------------------------------------------------------------


@dataclass
class ScheduleRun:
    """Record of one :func:`walk_schedule` traversal.

    Carries the recall-tick indices actually fired (post-horizon
    filter) and any per-recall outputs the probe function returned.
    Lets Phase 6 runners audit "we claimed sparse-monthly on ep500 but
    the schedule actually fired 0 recalls" rather than silently
    producing a dense-equivalent trajectory.
    """

    schedule_name: str
    horizon: int
    recall_ticks_fired: tuple[int, ...]
    probe_results: tuple[Any, ...] = ()
    notes: dict = field(default_factory=dict)


def walk_schedule(
    system: Any,
    schedule: StreamSchedule,
    *,
    horizon: int,
    probe_fn: Callable[[Any, int], Any] | None = None,
) -> ScheduleRun:
    """Advance ``system`` by ``horizon`` ticks under ``schedule``.

    At every tick index ``i`` in ``range(horizon)``:

    - call ``system.tick()`` once (the silent heartbeat beat)
    - if ``schedule.should_recall(i)``, call ``probe_fn(system, i)`` and
      capture its return value

    ``probe_fn`` is optional — a schedule may be walked purely to
    advance the system under a given cadence without measurement
    (useful for seeding Phase 6 §6 item 4 sparse runs before the
    measurement window opens). When ``probe_fn`` is ``None``, recall
    ticks are counted but no measurement is taken.

    The returned :class:`ScheduleRun` records which ticks fired, so
    downstream reporting can assert the schedule-shape invariant
    without re-deriving it. Walking produces no LLM calls and no I/O;
    cost is bounded by ``horizon`` ``system.tick()`` calls plus
    whatever the probe function itself does.
    """
    if horizon < 0:
        raise ValueError(f"horizon must be >= 0, got {horizon}")
    fired: list[int] = []
    results: list[Any] = []
    for i in range(horizon):
        system.tick()
        if schedule.should_recall(i):
            fired.append(i)
            if probe_fn is not None:
                results.append(probe_fn(system, i))
    return ScheduleRun(
        schedule_name=type(schedule).__name__,
        horizon=horizon,
        recall_ticks_fired=tuple(fired),
        probe_results=tuple(results),
    )


__all__ = [
    "StreamSchedule",
    "DenseSchedule",
    "WeeklySchedule",
    "MonthlySchedule",
    "AdversarialSilentRuns",
    "ScheduleRun",
    "walk_schedule",
]
