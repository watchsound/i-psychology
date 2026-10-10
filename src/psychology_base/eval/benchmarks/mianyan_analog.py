"""Phase 3 E0.8b — 绵延 analog scorers.

The month-horizon stress test gate is::

    V1(ep200 20-tick window) / V1(ep20 20-tick window) >= 0.80

V1 (state continuity) is the cosine between a memory system's state
signature before vs. after one sleep tick (see :mod:`eval.v_metrics`).
``ep20`` and ``ep200`` are measurement checkpoints into a compressed
stream — early vs. deep into the month-horizon run. The *ratio* is the
substrate-neutral 绵延 analog: an architecture that preserves its own
dynamics after 200 episodes retains the ratio near 1.0; one that
collapses into fixed-point attractors (Ouroboros's variance-collapse
failure mode) decays the ratio toward 0.

**Primary scorer**: :func:`rolling_v1_continuity_ratio` — substrate-
neutral; the Phase 3 roadmap gate uses this.

**Diagnostic scorer**: :func:`direct_mianyan_combined` — approximate
port of Ouroboros's combined mianyan score (V1 + V2 adjacent + V3
mentalistic). Known-decaying by construction (Ouroboros peaks at ep20,
collapses by ep30). Carried as a *diagnostic* so Phase 3 options that
raise the primary gate can also be inspected against the Ouroboros
reference failure mode. **Not an acceptance gate.**
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from psychology_base.companion.stream_shapes import StreamSchedule, walk_schedule
from psychology_base.eval.adapters import MemorySystem, Observation
from psychology_base.eval.v_metrics import (
    _cosine_bow,
    _jaccard,
    _state_signature,
    _tokens,
)


# ---------------------------------------------------------------------------
# Primary scorer — rolling V1 continuity ratio
# ---------------------------------------------------------------------------


@dataclass
class RollingV1Result:
    """Full breakdown of the rolling-V1 continuity measurement.

    Carries the raw per-tick V1 samples at both checkpoints so later
    phases can plot the collapse curve rather than just the ratio.
    """

    v1_short: float
    """Mean V1 over the ``window`` ticks ending at ``ep_short``."""

    v1_long: float
    """Mean V1 over the ``window`` ticks ending at ``ep_long``."""

    ratio: float
    """``v1_long / v1_short``; the Phase 3 gate quantity (≥ 0.80)."""

    short_samples: tuple[float, ...]
    long_samples: tuple[float, ...]
    ep_short: int
    ep_long: int
    window: int
    notes: dict = field(default_factory=dict)


def _rolling_v1_window(
    memory: MemorySystem, probe_cue: str, window: int
) -> tuple[float, tuple[float, ...]]:
    """Advance ``window`` ticks and record one V1 sample per tick.

    Mutates ``memory`` by advancing ``window`` ticks. Caller is expected
    to use a fresh memory instance per full measurement — the ratio
    only makes sense when both checkpoints are reached by the same
    trajectory.
    """
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    samples: list[float] = []
    for _ in range(window):
        sig_before = _state_signature(memory, probe_cue)
        memory.tick()
        sig_after = _state_signature(memory, probe_cue)
        samples.append(_cosine_bow(sig_before, sig_after))
    mean = sum(samples) / len(samples)
    return mean, tuple(samples)


def rolling_v1_continuity_ratio(
    memory: MemorySystem,
    probe_cue: str,
    scenario_observations: list[Observation],
    *,
    ep_short: int = 20,
    ep_long: int = 200,
    window: int = 20,
) -> RollingV1Result:
    """Primary 绵延-analog scorer: V1(ep_long window)/V1(ep_short window).

    Protocol:

    1. Observe the scenario on the provided (fresh) ``memory``.
    2. Advance ``ep_short − window`` ticks (fast-forward to just before
       the short checkpoint).
    3. Advance ``window`` more ticks, sampling V1 each step → ``v1_short``.
    4. Advance ``ep_long − ep_short`` more ticks (fast-forward to just
       before the long checkpoint).
    5. Advance ``window`` more ticks, sampling V1 each step → ``v1_long``.
    6. Return the ratio.

    ``ep_short`` must be ≥ ``window`` and ``ep_long`` must be
    ≥ ``ep_short + window`` so both measurement windows fit.
    """
    if ep_short < window:
        raise ValueError(
            f"ep_short ({ep_short}) must be >= window ({window})"
        )
    if ep_long < ep_short + window:
        raise ValueError(
            f"ep_long ({ep_long}) must be >= ep_short+window ({ep_short + window})"
        )

    for obs in scenario_observations:
        memory.observe(obs)

    advance_fn = getattr(memory, "advance_virtual_time", None)

    def _fast_forward(n: int) -> None:
        if n <= 0:
            return
        if callable(advance_fn):
            advance_fn(n)
        else:
            for _ in range(n):
                memory.tick()

    _fast_forward(ep_short - window)
    v1_short, short_samples = _rolling_v1_window(memory, probe_cue, window)

    _fast_forward(ep_long - ep_short - window)
    v1_long, long_samples = _rolling_v1_window(memory, probe_cue, window)

    if v1_short <= 0.0:
        # Degenerate: short-horizon V1 is zero means the architecture
        # already drifted out of its own state in the first ep_short
        # ticks. Report ratio = 0.0 (not inf / NaN) so downstream
        # comparisons stay well-ordered.
        ratio = 0.0
        notes = {"degenerate": "v1_short == 0; ratio clamped to 0.0"}
    else:
        ratio = v1_long / v1_short
        notes = {}

    return RollingV1Result(
        v1_short=v1_short,
        v1_long=v1_long,
        ratio=ratio,
        short_samples=short_samples,
        long_samples=long_samples,
        ep_short=ep_short,
        ep_long=ep_long,
        window=window,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Diagnostic scorer — direct Ouroboros mianyan port (approximation)
# ---------------------------------------------------------------------------


@dataclass
class DirectMianyanResult:
    v1_state_continuity: float
    v2_adjacent_jaccard: float
    v3_mentalistic_distance: float
    combined: float
    """Mean of V1, V2, V3 — Ouroboros's combined mianyan scalar.

    Known to peak at ep20 and decay by ep30 in Ouroboros due to variance
    collapse and Hebbian synchronization. Carried as a diagnostic hook
    only; the primary gate is :func:`rolling_v1_continuity_ratio`.
    """
    ep: int


def direct_mianyan_combined(
    memory: MemorySystem,
    probe_cue: str,
    scenario_observations: list[Observation],
    *,
    ep: int = 20,
    mentalistic_contexts: tuple[dict, dict] = (
        {"mode": "self"},
        {"mode": "other"},
    ),
    adjacent_tick_gap: int = 1,
) -> DirectMianyanResult:
    """Approximation of Ouroboros's combined mianyan score at a checkpoint.

    Observes the scenario, fast-forwards to ``ep`` ticks, then samples:

    - V1: cosine of state-signature across one more tick
    - V2 adjacent: Jaccard of two back-to-back recalls (with
      ``adjacent_tick_gap`` ticks between — set to 0 to match
      ``measure_v_metrics`` exactly)
    - V3 mentalistic: 1 − Jaccard of recall under two different contexts

    Combined score = mean of the three. **Diagnostic only.**
    """
    if ep < 0:
        raise ValueError(f"ep must be non-negative, got {ep}")

    for obs in scenario_observations:
        memory.observe(obs)

    advance_fn = getattr(memory, "advance_virtual_time", None)
    if callable(advance_fn):
        advance_fn(ep)
    else:
        for _ in range(ep):
            memory.tick()

    sig_before = _state_signature(memory, probe_cue)
    memory.tick()
    sig_after = _state_signature(memory, probe_cue)
    v1 = _cosine_bow(sig_before, sig_after)

    r_a = memory.recall(probe_cue).content
    for _ in range(adjacent_tick_gap):
        memory.tick()
    r_b = memory.recall(probe_cue).content
    v2_adj = _jaccard(_tokens(r_a), _tokens(r_b))

    r_c1 = memory.recall(probe_cue, mentalistic_contexts[0]).content
    r_c2 = memory.recall(probe_cue, mentalistic_contexts[1]).content
    v3 = 1.0 - _jaccard(_tokens(r_c1), _tokens(r_c2))

    combined = (v1 + v2_adj + v3) / 3.0
    return DirectMianyanResult(
        v1_state_continuity=v1,
        v2_adjacent_jaccard=v2_adj,
        v3_mentalistic_distance=v3,
        combined=combined,
        ep=ep,
    )


# ---------------------------------------------------------------------------
# Phase 6 — schedule-aware rolling V1
# ---------------------------------------------------------------------------


@dataclass
class ScheduledV1Result:
    """Phase 6 variant of :class:`RollingV1Result`.

    Same measurement shape (``v1_short`` / ``v1_long`` / ``ratio`` over
    rolling windows) but with explicit sparse-recall intervals between
    the two measurement windows. The schedule drives probe cadence
    *outside* the measurement windows — inside the windows the Phase 3
    sampling pattern (one V1 sample per tick, every tick) is preserved,
    so the Phase 3 and Phase 6 numbers remain directly comparable.
    """

    v1_short: float
    v1_long: float
    ratio: float
    short_samples: tuple[float, ...]
    long_samples: tuple[float, ...]
    ep_short: int
    ep_long: int
    window: int
    schedule_name: str
    pre_short_recall_ticks: tuple[int, ...]
    between_recall_ticks: tuple[int, ...]
    """0-based tick indices (relative to each silent-run segment) at
    which the schedule fired a probe. Lets Phase 6 reporting assert
    the stream-shape invariant (``sparse-monthly on ep200 → () both``)
    rather than re-derive it."""
    notes: dict = field(default_factory=dict)


def scheduled_v1_continuity_ratio(
    memory: MemorySystem,
    probe_cue: str,
    scenario_observations: list[Observation],
    schedule: StreamSchedule,
    *,
    ep_short: int = 20,
    ep_long: int = 200,
    window: int = 20,
    probe_cue_during_schedule: str | None = None,
    probe_context: dict | None = None,
) -> ScheduledV1Result:
    """Phase 6 §2 — schedule-driven stream between measurement windows.

    Protocol:

    1. Observe the scenario on the provided (fresh) ``memory``.
    2. Walk ``ep_short − window`` ticks under ``schedule`` — the schedule
       decides which ticks within this silent run fire a recall probe
       on ``probe_cue_during_schedule`` (defaults to ``probe_cue``).
    3. Sample V1 for ``window`` ticks (one per tick, as Phase 3) →
       ``v1_short``.
    4. Walk ``ep_long − ep_short − window`` ticks under the same
       schedule.
    5. Sample V1 for ``window`` ticks → ``v1_long``.
    6. Return the ratio.

    The two walked segments reset their tick-index counter at zero so
    the schedule's cadence applies consistently to each. For
    ``MonthlySchedule`` with period 720 ticks, an ``ep_long=200`` long
    segment fires zero probes (silent run shorter than cadence) — this
    is the hardest Phase 6 cell, and the ``between_recall_ticks``
    tuple being empty is the signal that the "silent horizon longer
    than cadence" invariant is actually exercised.

    The measurement windows themselves (steps 3, 5) still sample V1 at
    every tick inside the window — same as Phase 3. Schedule injection
    affects only the silent runs between windows. Phase 3 numbers are
    recovered by passing an empty :class:`AdversarialSilentRuns`
    (zero in-stream probes); any divergence from Phase 3 under that
    schedule points at a stream-shape confound, not a scorer bug.
    """
    if ep_short < window:
        raise ValueError(
            f"ep_short ({ep_short}) must be >= window ({window})"
        )
    if ep_long < ep_short + window:
        raise ValueError(
            f"ep_long ({ep_long}) must be >= ep_short+window ({ep_short + window})"
        )

    for obs in scenario_observations:
        memory.observe(obs)

    cue_for_schedule = (
        probe_cue if probe_cue_during_schedule is None else probe_cue_during_schedule
    )
    ctx = dict(probe_context) if probe_context is not None else {}

    def _probe(system, _i):
        return system.recall(cue_for_schedule, dict(ctx))

    pre_short_horizon = ep_short - window
    pre_short_run = walk_schedule(
        memory, schedule, horizon=pre_short_horizon, probe_fn=_probe
    )

    v1_short, short_samples = _rolling_v1_window(memory, probe_cue, window)

    between_horizon = ep_long - ep_short - window
    between_run = walk_schedule(
        memory, schedule, horizon=between_horizon, probe_fn=_probe
    )

    v1_long, long_samples = _rolling_v1_window(memory, probe_cue, window)

    if v1_short <= 0.0:
        ratio = 0.0
        notes = {"degenerate": "v1_short == 0; ratio clamped to 0.0"}
    else:
        ratio = v1_long / v1_short
        notes = {}

    return ScheduledV1Result(
        v1_short=v1_short,
        v1_long=v1_long,
        ratio=ratio,
        short_samples=short_samples,
        long_samples=long_samples,
        ep_short=ep_short,
        ep_long=ep_long,
        window=window,
        schedule_name=pre_short_run.schedule_name,
        pre_short_recall_ticks=pre_short_run.recall_ticks_fired,
        between_recall_ticks=between_run.recall_ticks_fired,
        notes=notes,
    )


__all__ = [
    "RollingV1Result",
    "DirectMianyanResult",
    "ScheduledV1Result",
    "rolling_v1_continuity_ratio",
    "direct_mianyan_combined",
    "scheduled_v1_continuity_ratio",
]
