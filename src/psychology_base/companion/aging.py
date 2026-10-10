"""Phase 3 — aging policies.

Phase 3 explores four candidate ways to make the memory substrate survive
month-horizon pressure (see ``docs/phases/phase-3-month-horizon-plan.md``
§6). Each option lands here as an ``AgingPolicy`` implementation plus an
off-by-default wire-up on ``CompanionSystem``. With no policy set the
Phase 2 behaviour is byte-identical — the policy is the aging mechanism,
and Phase 2 had none.

**Current shipped option — E1 S1 Decay (``EbbinghausPolicy``).**
Ebbinghaus retention curve ``R(t) = exp(-t/S)`` applied to a per-fragment
importance score. Each ``tick()`` advances every non-covenant fragment's
retention; each ``recall()`` reconsolidates (boosts retention back toward
the ceiling). Covenant-tagged fragments (``COVENANT_TAG``) are immune —
their retention is pinned at ``+inf`` and never decays. Eviction's
strength key reads the policy's retention value when set, replacing the
edge-weight strength used in Phase 2. This leaves Principle D (survival
under scarcity) intact but gives it a decay-curve shape instead of
pure selection pressure.

**E2 S2 Capacity (``CapacityOnlyPolicy``)** — companion option; pure
Principle D with no time-decay curve. Strength is the observed selection
pressure (recall hit count), tie-broken by recency. What survives
eviction across the stream is what mattered. Covenant tag → ``+inf``
(same as S1). An archival trace of evicted ids is kept so thrashing is
observable even without resurrecting the content (per plan kill
criterion). The S3/S4 options (MetabolismPolicy / TieredPolicy) will
land here as additional classes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol


class AgingPolicy(Protocol):
    """Interface Phase 3 aging mechanisms must satisfy.

    All hooks are side-effect only; return values exist only for the
    eviction strength query (``strength``) and the sleep-cycle hook
    (``consolidate_candidates``).
    """

    def on_observe(self, fragment_id: str, *, is_covenant: bool) -> None:
        """Register a newly observed fragment."""

    def on_recall(self, fragment_id: str) -> None:
        """A fragment entered a recall pool; reconsolidate it."""

    def on_tick(self) -> None:
        """Advance the virtual clock by one tick."""

    def on_evict(self, fragment_id: str) -> None:
        """A fragment was evicted; drop its policy state."""

    def strength(self, fragment_id: str) -> float:
        """Current retention / strength. Higher is more pinned.

        Eviction picks the lowest-strength non-covenant fragment, so a
        policy that pushes importance low expresses "evict this one
        first." Covenant-tagged fragments short-circuit this via the
        store's covenant check; policies may still return a finite value
        for them without changing eviction order.
        """
        ...

    def consolidate_candidates(self, top_k: int) -> list[str]:
        """Sleep-cycle hook: top-K fragments worth distilling now.

        Returns a list of fragment ids (highest strength first,
        length ≤ top_k). Caller decides what to do with them — the
        aging policy owns ranking, not the semantic consolidation
        itself. Return an empty list if the policy has nothing to
        suggest (e.g., it's not time yet).
        """
        ...


# ---------------------------------------------------------------------------
# E1 — Option S1: Ebbinghaus decay
# ---------------------------------------------------------------------------


@dataclass
class EbbinghausPolicy:
    """Ebbinghaus retention curve on per-fragment importance.

    State for each non-covenant fragment:

      ``retention(fid, now) = base_importance[fid] *
                              exp(-(now - last_touched[fid]) / half_life)``

    ``on_tick()`` advances ``now`` by one. ``on_recall()`` resets
    ``last_touched[fid] = now`` and multiplies ``base_importance[fid]``
    by ``boost_factor`` (clipped to ``max_importance``) — reconsolidation
    both resets the curve and strengthens the trace, matching the
    SleepGate / YourMemory pattern cited in plan §4.

    Covenant-tagged fragments bypass the curve entirely. Their strength
    is ``+inf`` so even if the FragmentStore's covenant pin were ever
    disabled, the aging policy alone would still protect them.

    ``consolidate_candidates`` fires every ``consolidate_every`` ticks
    and returns the top-K active non-covenant fragments; between firings
    it returns ``[]``. Caller (``CompanionSystem.tick``) uses this to
    drive the batched sleep-cycle distillation.
    """

    half_life: float = 50.0
    """Ebbinghaus S parameter in ticks. retention = base * exp(-age/S).
    Default 50 ticks → ~0.37 retention at age=50, ~0.14 at age=100,
    ~0.02 at age=200. Below the covenant pin and the recall boost but
    well above decimal zero so non-covenant content with any residual
    importance can still rank above pure noise at ep200."""

    base_importance: float = 1.0
    """Initial importance for a freshly-observed non-covenant fragment.
    Phase 2 M1 reconsolidation-significance can override this per
    fragment by passing ``importance=`` to ``on_observe``."""

    boost_factor: float = 2.0
    """Multiplicative boost applied to a fragment's base importance each
    time it enters a recall pool. The curve also resets to age=0, so
    boost both strengthens and refreshes."""

    max_importance: float = 10.0
    """Ceiling on base importance to keep repeatedly-recalled fragments
    from dominating the strength budget. Above this cap, reconsolidation
    still refreshes the curve (age reset) but doesn't grow importance."""

    consolidate_every: int = 24
    """Ticks between sleep-cycle firings. 24 ticks ~= one compressed
    day under the Phase 3 harness convention (1 tick / hour)."""

    sleep_top_k: int = 8
    """Max distill candidates per sleep-cycle firing. The caller may
    pass a smaller cap via ``consolidate_candidates(top_k=...)``."""

    _now: int = 0
    _base: dict[str, float] = field(default_factory=dict)
    _last_touched: dict[str, int] = field(default_factory=dict)
    _covenant: set[str] = field(default_factory=set)
    _last_consolidate_at: int = 0

    def on_observe(
        self,
        fragment_id: str,
        *,
        is_covenant: bool,
        importance: float | None = None,
    ) -> None:
        if is_covenant:
            self._covenant.add(fragment_id)
            # Covenant fragments carry +inf retention — no decay path.
            return
        imp = self.base_importance if importance is None else float(importance)
        self._base[fragment_id] = min(imp, self.max_importance)
        self._last_touched[fragment_id] = self._now

    def on_recall(self, fragment_id: str) -> None:
        if fragment_id in self._covenant:
            return
        cur = self._base.get(fragment_id, self.base_importance)
        self._base[fragment_id] = min(cur * self.boost_factor, self.max_importance)
        self._last_touched[fragment_id] = self._now

    def on_tick(self) -> None:
        self._now += 1

    def on_evict(self, fragment_id: str) -> None:
        self._base.pop(fragment_id, None)
        self._last_touched.pop(fragment_id, None)
        self._covenant.discard(fragment_id)

    def strength(self, fragment_id: str) -> float:
        if fragment_id in self._covenant:
            return math.inf
        base = self._base.get(fragment_id)
        if base is None:
            # Fragment registered outside this policy (e.g., pre-existing
            # before policy was wired). Treat as fully-decayed so it's
            # the first eviction candidate rather than crashing.
            return 0.0
        age = self._now - self._last_touched.get(fragment_id, self._now)
        return base * math.exp(-age / self.half_life)

    def consolidate_candidates(self, top_k: int | None = None) -> list[str]:
        if self._now - self._last_consolidate_at < self.consolidate_every:
            return []
        self._last_consolidate_at = self._now
        cap = self.sleep_top_k if top_k is None else min(top_k, self.sleep_top_k)
        ranked = sorted(
            (fid for fid in self._base if fid not in self._covenant),
            key=lambda fid: -self.strength(fid),
        )
        return ranked[:cap]


# ---------------------------------------------------------------------------
# E2 — Option S2: Capacity-only (pure selection pressure)
# ---------------------------------------------------------------------------


@dataclass
class CapacityOnlyPolicy:
    """Pure selection-pressure aging — no explicit decay curve.

    Principle D ("what survives purge is what mattered") in its most
    direct form. Each fragment's strength is a count of how often it has
    been re-selected (``on_recall``) with a small recency tiebreak. When
    byte-cap eviction fires, the lowest-strength non-covenant fragment
    goes first. There is no half-life, no time-decay, no sleep-cycle
    rank. The stream's own selection pressure is the ranking.

    Covenant-tagged fragments short-circuit to ``+inf`` (same two-way
    safety as S1: both the policy pin and the FragmentStore's
    ``is_covenant`` check must break for a covenant to drift).

    An ``_archive`` list tracks evicted fragment ids in order. The
    policy does not re-ingest archived content — resurrection is a
    Phase 3b concern. The archive exists so thrashing (same id leaving
    + re-entering repeatedly) is measurable from policy state alone,
    which the plan names as an explicit kill criterion for S2.

    No consolidate-cycle rank: ``consolidate_candidates`` returns the
    top-K current fragments by hit count on the same cadence as S1 so
    the A/B comparison is apples-to-apples. If S2 is picked and S1 is
    retired, this hook can go idle.
    """

    consolidate_every: int = 24
    """Ticks between sleep-cycle firings — same default as S1 so A/B
    comparison does not mix cadence differences."""

    sleep_top_k: int = 8
    """Max distill candidates per sleep-cycle firing."""

    recency_tiebreak_scale: float = 1e-6
    """How much the birth tick contributes to strength as a tiebreak.
    Kept tiny so hits strictly dominate — a single recall beats any
    amount of recency. Only resolves ties between zero-hit fragments."""

    _now: int = 0
    _hits: dict[str, int] = field(default_factory=dict)
    _birth: dict[str, int] = field(default_factory=dict)
    _covenant: set[str] = field(default_factory=set)
    _archive: list[str] = field(default_factory=list)
    _last_consolidate_at: int = 0

    def on_observe(
        self,
        fragment_id: str,
        *,
        is_covenant: bool,
    ) -> None:
        if is_covenant:
            self._covenant.add(fragment_id)
            return
        self._hits.setdefault(fragment_id, 0)
        self._birth.setdefault(fragment_id, self._now)

    def on_recall(self, fragment_id: str) -> None:
        if fragment_id in self._covenant:
            return
        self._hits[fragment_id] = self._hits.get(fragment_id, 0) + 1

    def on_tick(self) -> None:
        self._now += 1

    def on_evict(self, fragment_id: str) -> None:
        if fragment_id in self._hits or fragment_id in self._birth:
            self._archive.append(fragment_id)
        self._hits.pop(fragment_id, None)
        self._birth.pop(fragment_id, None)
        self._covenant.discard(fragment_id)

    def strength(self, fragment_id: str) -> float:
        if fragment_id in self._covenant:
            return math.inf
        if fragment_id not in self._hits:
            return 0.0
        hits = self._hits[fragment_id]
        birth = self._birth.get(fragment_id, 0)
        return float(hits) + birth * self.recency_tiebreak_scale

    def consolidate_candidates(self, top_k: int | None = None) -> list[str]:
        if self._now - self._last_consolidate_at < self.consolidate_every:
            return []
        self._last_consolidate_at = self._now
        cap = self.sleep_top_k if top_k is None else min(top_k, self.sleep_top_k)
        ranked = sorted(
            (fid for fid in self._hits if fid not in self._covenant),
            key=lambda fid: -self.strength(fid),
        )
        return ranked[:cap]

    def thrash_rate(self) -> float:
        """Observability hook: fraction of evictions whose id has been
        evicted before. High → policy is churning the same slots. Named
        by plan §5 as a kill criterion ('thrashing')."""
        if not self._archive:
            return 0.0
        seen: set[str] = set()
        repeats = 0
        for fid in self._archive:
            if fid in seen:
                repeats += 1
            else:
                seen.add(fid)
        return repeats / len(self._archive)


__all__ = [
    "AgingPolicy",
    "EbbinghausPolicy",
    "CapacityOnlyPolicy",
]
