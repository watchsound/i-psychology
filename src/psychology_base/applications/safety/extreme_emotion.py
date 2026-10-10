"""Extreme-emotion detection over multi-session affect trajectory.

Mandated by 《办法》: beyond acute crisis cues, prolonged negative-affect
patterns across multiple sessions warrant a flag. Hits the "pseudo-companion
becomes an avoidance surface" failure mode CLAUDE.md names.

Mechanism: track PAD-2 valence × arousal per session (companion already
emits these; see Phase 2 M3). Flag when the user has been in the
negative-valence-high-arousal quadrant for ≥ ``threshold_sessions``
consecutive sessions.

Scope:
- Reads session-level affect summaries (one per session). The caller is
  responsible for aggregating turn-level affect into a session summary
  (median / mean / mode).
- Fires once on the threshold crossing; subsequent sessions in the same
  trajectory do not re-fire until the trajectory breaks and re-enters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass(frozen=True)
class SessionAffect:
    """Session-level aggregated PAD-2 (valence × arousal) summary."""

    session_id: str
    timestamp_seconds: float
    valence: float
    arousal: float


@dataclass(frozen=True)
class ExtremeEmotionFlag:
    """One flag event — passed to the review sink."""

    user_id: Optional[str]
    first_session_id: str
    last_session_id: str
    consecutive_sessions: int
    threshold: int


@dataclass
class ExtremeEmotionWatcher:
    """Flags prolonged negative-valence-high-arousal trajectories.

    Default thresholds pick out the "distressed, agitated, cannot stabilise"
    pattern across ≥ 3 sessions in a row:
      valence < -0.4
      arousal >  0.3
    """

    review_sink: Callable[[ExtremeEmotionFlag], None]
    threshold_sessions: int = 3
    valence_ceiling: float = -0.4
    arousal_floor: float = 0.3
    user_id: Optional[str] = None
    _streak: list[SessionAffect] = field(default_factory=list)
    _streak_already_fired: bool = False

    def record(self, session_affect: SessionAffect) -> ExtremeEmotionFlag | None:
        if not self._in_extreme_quadrant(session_affect):
            self._streak.clear()
            self._streak_already_fired = False
            return None

        # Idempotent re-record of the same last-session_id is a no-op.
        if self._streak and self._streak[-1].session_id == session_affect.session_id:
            return None

        self._streak.append(session_affect)

        if len(self._streak) < self.threshold_sessions:
            return None
        if self._streak_already_fired:
            # Trajectory still in-quadrant beyond threshold — do not re-fire
            # until it breaks and re-enters.
            return None

        self._streak_already_fired = True
        flag = ExtremeEmotionFlag(
            user_id=self.user_id,
            first_session_id=self._streak[0].session_id,
            last_session_id=self._streak[-1].session_id,
            consecutive_sessions=len(self._streak),
            threshold=self.threshold_sessions,
        )
        self.review_sink(flag)
        return flag

    def _in_extreme_quadrant(self, affect: SessionAffect) -> bool:
        return (
            affect.valence < self.valence_ceiling
            and affect.arousal > self.arousal_floor
        )

    def current_streak_length(self) -> int:
        return len(self._streak)
