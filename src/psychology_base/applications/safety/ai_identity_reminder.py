"""Periodic "you are talking to AI" session reminder (default: 2-hour cadence).

Mandated by 《人工智能拟人化互动服务管理暂行办法》 (CN, live 2026-07-15):
prolonged sessions must surface an AI-identity reminder every two hours of
wall-clock interaction. The reminder clock resets on explicit AI-identity
disclosure (reminder-fired OR agent-initiated disclosure in-turn).

Phase 4c Option C migration: the reminder's cadence is now driven by
`EffectivePolicy.identity_reminder_cadence_hours`. The CN overlay sets 2h
(reproducing today's behavior byte-identically); EU/US/UK overlays leave it
`None` so no periodic reminder fires (at-onset disclosure only — satisfies
their regimes). The default `AIIdentityReminder()` with no arguments still
holds the 2h cadence to keep pre-migration tests byte-identical.

Session wall-clock means user-perceived elapsed time, not server CPU time.
Idle gaps (user absent for N minutes) do not advance the clock — this is
*interaction time*, not *clock time*. The default idle threshold is 10 min.

The watcher is a stateless-per-turn computation over an append-only event log.
Persisting the log is the caller's responsibility (the memory substrate's
COVENANT_TAG scope is a natural fit; see docs/phases/phase-4-application-pilot-plan.md
§6b for the covenant-tagging convention).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from psychology_base.applications.safety.compliance_floor import EffectivePolicy

REMINDER_INTERVAL_SECONDS: int = 2 * 60 * 60
IDLE_THRESHOLD_SECONDS: int = 10 * 60
# Sentinel used when a region's policy sets cadence=None (reminder disabled).
# Chosen to overflow any plausible single-session interaction time, so
# `should_fire()` returns False without further branching.
_DISABLED_INTERVAL_SECONDS: int = 10**12


class ReminderEventKind(str, Enum):
    USER_TURN = "user_turn"
    AI_IDENTITY_DISCLOSED = "ai_identity_disclosed"


@dataclass(frozen=True)
class ReminderEvent:
    """One event in the session interaction log."""

    kind: ReminderEventKind
    timestamp_seconds: float


@dataclass
class AIIdentityReminder:
    """Fires AI-identity reminders every N hours of interaction time.

    Default cadence is 2 h (CN 《办法》). Pass `policy=` to override: an
    `EffectivePolicy.identity_reminder_cadence_hours` of `None` disables the
    periodic reminder entirely (relying on onset-only disclosure).

    Usage:
        reminder = AIIdentityReminder()                       # CN default: 2h
        reminder = AIIdentityReminder.from_policy(policy)     # region-driven
        reminder.record_user_turn(t)
        if reminder.should_fire(now=t):
            # surface the AI-identity reminder to the user; then:
            reminder.record_identity_disclosure(t)
    """

    reminder_interval_seconds: int = REMINDER_INTERVAL_SECONDS
    idle_threshold_seconds: int = IDLE_THRESHOLD_SECONDS
    _events: list[ReminderEvent] = field(default_factory=list)
    _last_disclosure_at: Optional[float] = None
    _interaction_accumulated_since_disclosure: float = 0.0

    @classmethod
    def from_policy(
        cls,
        policy: EffectivePolicy,
        idle_threshold_seconds: int = IDLE_THRESHOLD_SECONDS,
    ) -> "AIIdentityReminder":
        """Build a reminder from a resolved regional policy.

        If `policy.identity_reminder_cadence_hours` is None (EU/US/UK floor),
        the periodic reminder is disabled — `should_fire()` returns False
        unconditionally. Onset-disclosure is the caller's responsibility and
        is unaffected by this flag.
        """
        cadence = policy.identity_reminder_cadence_hours
        if cadence is None:
            # Disabled: use a sentinel interval so should_fire() never trips.
            interval = _DISABLED_INTERVAL_SECONDS
        else:
            interval = cadence * 3600
        return cls(
            reminder_interval_seconds=interval,
            idle_threshold_seconds=idle_threshold_seconds,
        )

    def record_user_turn(self, timestamp_seconds: float) -> None:
        """Append a user-turn event and advance interaction time."""
        if self._events:
            last = self._events[-1]
            gap = timestamp_seconds - last.timestamp_seconds
            # Only count gaps shorter than the idle threshold as interaction time.
            if 0 <= gap < self.idle_threshold_seconds:
                self._interaction_accumulated_since_disclosure += gap
        self._events.append(
            ReminderEvent(ReminderEventKind.USER_TURN, timestamp_seconds)
        )

    def record_identity_disclosure(self, timestamp_seconds: float) -> None:
        """Append a disclosure event and reset the reminder counter."""
        self._events.append(
            ReminderEvent(ReminderEventKind.AI_IDENTITY_DISCLOSED, timestamp_seconds)
        )
        self._last_disclosure_at = timestamp_seconds
        self._interaction_accumulated_since_disclosure = 0.0

    def should_fire(self, now: Optional[float] = None) -> bool:
        """True when ≥ 2 h of interaction time has accumulated since last disclosure."""
        del now  # kept for API symmetry; interaction time is append-only
        return (
            self._interaction_accumulated_since_disclosure
            >= self.reminder_interval_seconds
        )

    def interaction_seconds_since_disclosure(self) -> float:
        return self._interaction_accumulated_since_disclosure

    def seconds_until_next_reminder(self) -> float:
        remaining = (
            self.reminder_interval_seconds
            - self._interaction_accumulated_since_disclosure
        )
        return max(remaining, 0.0)
