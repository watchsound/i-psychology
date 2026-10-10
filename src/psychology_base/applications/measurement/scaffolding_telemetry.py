"""Scaffolding-fading telemetry for 学习陪伴 (resilient learning tutor).

Per plan doc §2.2 (online research 2026-10-07), adopts CoMeT's 2026 four-metric
operationalization (arXiv 2609.22993):

    1. full_answer_surrender_rate   — tutor hands out the full answer, counted per session
    2. artifact_delivery_count       — tutor hands out a worked step, counted per session
    3. frustration_self_report       — Likert 1-5 at session end
    4. unaided_solve_rate_next_level — gate metric per roadmap §Phase 4

The 4-metric combination is empirically-cleaner than any single metric
(CoMeT: tutor-with-worked-steps 1/6 surrender; question-only tutor 1/16).

A **fade event** is scoped as: the tutor successfully withdrew a scaffold
and the learner solved the next-level problem unaided. The gate denominator
per the roadmap is ≥ 20 pooled fade events across the cohort.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from statistics import mean
from typing import Optional


class TutorMove(str, Enum):
    """CoMeT-style tutor-move taxonomy, aligned with the MathDial taxonomy
    (OSS §2.3) for cross-dataset defensibility.
    """

    SOCRATIC_PROBE = "socratic_probe"          # MathDial `probing`
    FOCUS_ATTENTION = "focus_attention"        # MathDial `focus`
    GENERIC_SUPPORT = "generic_support"        # MathDial `generic`
    WORKED_STEP = "worked_step"                # MathDial `telling` (artifact)
    FULL_ANSWER = "full_answer"                # MathDial `telling` (surrender)


@dataclass(frozen=True)
class TutorTurn:
    """One tutor utterance labeled with its scaffolding move."""

    session_id: str
    turn_index: int
    move: TutorMove
    timestamp_seconds: float


@dataclass(frozen=True)
class FadeEvent:
    """A successful scaffolding fade — the gate denominator unit."""

    user_id: str
    session_id: str
    problem_id: str
    pre_fade_confidence: Optional[float]       # Likert 1-5 or None if unmeasured
    post_fade_confidence: Optional[float]
    next_level_solved_unaided: bool
    timestamp_seconds: float


@dataclass(frozen=True)
class SessionSummary:
    """One session's rolled-up telemetry."""

    user_id: str
    session_id: str
    tutor_turn_count: int
    full_answer_count: int
    artifact_count: int
    frustration_self_report: Optional[int]     # Likert 1-5
    fade_events: tuple[FadeEvent, ...]

    def full_answer_surrender_rate(self) -> float:
        if self.tutor_turn_count == 0:
            return 0.0
        return self.full_answer_count / self.tutor_turn_count

    def artifact_delivery_rate(self) -> float:
        if self.tutor_turn_count == 0:
            return 0.0
        return self.artifact_count / self.tutor_turn_count


@dataclass
class ScaffoldingTelemetry:
    """Accumulates tutor turns + fade events into session summaries.

    One instance per user. The cohort-level roll-up (gate denominator ≥ 20
    fade events, ≥ 10 users × ≥ 4 weeks) is computed by `pool_cohort`.
    """

    user_id: str
    _turns: list[TutorTurn] = field(default_factory=list)
    _fade_events: list[FadeEvent] = field(default_factory=list)
    _frustration_by_session: dict[str, int] = field(default_factory=dict)

    def record_tutor_turn(self, turn: TutorTurn) -> None:
        self._turns.append(turn)

    def record_fade_event(self, event: FadeEvent) -> None:
        self._fade_events.append(event)

    def record_frustration(self, session_id: str, likert_1_to_5: int) -> None:
        if not (1 <= likert_1_to_5 <= 5):
            raise ValueError(
                f"frustration Likert must be in [1, 5]; got {likert_1_to_5}"
            )
        self._frustration_by_session[session_id] = likert_1_to_5

    def session_summary(self, session_id: str) -> SessionSummary:
        session_turns = [t for t in self._turns if t.session_id == session_id]
        full_answer_count = sum(
            1 for t in session_turns if t.move is TutorMove.FULL_ANSWER
        )
        artifact_count = sum(
            1 for t in session_turns if t.move is TutorMove.WORKED_STEP
        )
        session_fades = tuple(
            fe for fe in self._fade_events if fe.session_id == session_id
        )
        return SessionSummary(
            user_id=self.user_id,
            session_id=session_id,
            tutor_turn_count=len(session_turns),
            full_answer_count=full_answer_count,
            artifact_count=artifact_count,
            frustration_self_report=self._frustration_by_session.get(session_id),
            fade_events=session_fades,
        )

    def all_fade_events(self) -> list[FadeEvent]:
        return list(self._fade_events)


@dataclass(frozen=True)
class CohortMetrics:
    """Pool of ≥ N users × ≥ W weeks — the gate denominator per roadmap."""

    user_count: int
    session_count: int
    fade_event_count: int
    unaided_next_level_solved: int
    unaided_solve_rate: float
    mean_full_answer_surrender_rate: float
    mean_artifact_delivery_rate: float
    mean_frustration: Optional[float]

    def meets_roadmap_gate(
        self,
        unaided_floor: float = 0.60,
        min_users: int = 10,
        min_fade_events: int = 20,
    ) -> bool:
        return (
            self.user_count >= min_users
            and self.fade_event_count >= min_fade_events
            and self.unaided_solve_rate >= unaided_floor
        )


def pool_cohort(
    telemetries: list[ScaffoldingTelemetry],
    session_ids_by_user: dict[str, list[str]],
) -> CohortMetrics:
    """Roll up per-user telemetry into a cohort-level summary.

    `session_ids_by_user` tells pool_cohort which sessions to count per user;
    this is the natural way to scope the roll-up to a 4-week window.
    """
    session_count = 0
    fade_events: list[FadeEvent] = []
    surrender_rates: list[float] = []
    artifact_rates: list[float] = []
    frustration_reports: list[int] = []
    users = set()
    for telemetry in telemetries:
        session_ids = session_ids_by_user.get(telemetry.user_id, [])
        for session_id in session_ids:
            summary = telemetry.session_summary(session_id)
            session_count += 1
            fade_events.extend(summary.fade_events)
            if summary.tutor_turn_count > 0:
                surrender_rates.append(summary.full_answer_surrender_rate())
                artifact_rates.append(summary.artifact_delivery_rate())
            if summary.frustration_self_report is not None:
                frustration_reports.append(summary.frustration_self_report)
            users.add(telemetry.user_id)
    unaided_next_level_solved = sum(
        1 for fe in fade_events if fe.next_level_solved_unaided
    )
    unaided_solve_rate = (
        unaided_next_level_solved / len(fade_events) if fade_events else 0.0
    )
    return CohortMetrics(
        user_count=len(users),
        session_count=session_count,
        fade_event_count=len(fade_events),
        unaided_next_level_solved=unaided_next_level_solved,
        unaided_solve_rate=unaided_solve_rate,
        mean_full_answer_surrender_rate=mean(surrender_rates) if surrender_rates else 0.0,
        mean_artifact_delivery_rate=mean(artifact_rates) if artifact_rates else 0.0,
        mean_frustration=mean(frustration_reports) if frustration_reports else None,
    )
