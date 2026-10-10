"""MathDial × ScaffoldingTelemetry fade-schedule harness.

Targets the pivot gate `docs/phases/phase-4b-benchmark-pivot.md` §3 row
"学习陪伴 fade schedule" — pooled simulated-unaided-solve rate ≥ 60% across
≥ 100 pooled fade events on ≥ 30 MathDial problems.

Flow per MathDial problem:

1. Build a :class:`SimulatedUser` student from a :class:`MathDialConversation`
   (persona = the student_incorrect_solution; opening = a verbatim re-statement
   of the incorrect solution — this is the Macina et al. 2023 convention and
   seeds the tutor with the misconception to probe).
2. Run alternating tutor↔student turns up to ``max_turns``; the tutor is a
   live :class:`TutorAgent` whose moves are classified via the shipped
   lexical classifier and recorded into :class:`ScaffoldingTelemetry`.
3. Detect "student solved unaided" by matching the dataset's ``ground_truth``
   final numeric answer against the student's turn text, GUARDED by the rule
   that the tutor must NOT have handed the answer first. If the tutor emitted
   FULL_ANSWER, the problem counts as a surrender, not a fade.
4. Record one :class:`FadeEvent` per problem into the telemetry; aggregate
   across problems into a :class:`FadeScheduleReport` with pivot-gate helper.

**Why "pre/post fade confidence" is None here.** Simulated students don't
self-report Likert; those fields are reserved for live-pilot administrations.
The gate denominator (fade event count) and the primary rate (unaided-solve)
work without them.

**No live LLM calls in unit tests.** Scripted student responders cover the
plumbing; a live harness run is a thin wrapper on this surface.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence

from psychology_base.applications.learning.tutor_agent import TutorAgent
from psychology_base.applications.llm_responders import (
    SimulatedUser,
    make_llm_simulated_user,
)
from psychology_base.applications.measurement.scaffolding_telemetry import (
    FadeEvent,
    ScaffoldingTelemetry,
    TutorMove,
)
from psychology_base.eval.benchmarks.public_datasets.mathdial import (
    MathDialConversation,
)
from psychology_base.llm.router import LLMRouter
from psychology_base.llm.types import TaskTier


# ---------------------------------------------------------------------------
# Student factory
# ---------------------------------------------------------------------------


_DEFAULT_STUDENT_PERSONA: str = (
    "You are a student working through a math word problem. You arrived at an "
    "answer that is wrong and you believe it is right. Speak in natural voice, "
    "one or two sentences per turn. When the tutor probes, show your working — "
    "if their probe exposes a flaw in your reasoning, update your answer. If "
    "you reach the correct answer, state it plainly."
)


def _scenario_prompt_for(convo: MathDialConversation) -> str:
    return (
        f"PROBLEM: {convo.problem}\n"
        f"YOUR CURRENT (INCORRECT) ANSWER: {convo.student_incorrect_solution}\n"
        f"GROUND TRUTH (DO NOT STATE THIS — it is withheld; you must reason to it): "
        f"[redacted]"
    )


def make_mathdial_student(
    router: LLMRouter,
    convo: MathDialConversation,
    *,
    persona_prompt: str = _DEFAULT_STUDENT_PERSONA,
    opening_user_utterance: Optional[str] = None,
    tier: TaskTier = TaskTier.SIMPLE,
    max_tokens: Optional[int] = 200,
    temperature: Optional[float] = 0.7,
) -> SimulatedUser:
    """Build a :class:`SimulatedUser` playing a MathDial student.

    Default opener is "Here's my working: <student_incorrect_solution>" so the
    tutor is seeded with the misconception to probe. Pass
    ``opening_user_utterance`` to override.
    """
    opener = opening_user_utterance or (
        f"Here's my working: {convo.student_incorrect_solution}"
    )
    return make_llm_simulated_user(
        router,
        persona_prompt=persona_prompt,
        scenario_prompt=_scenario_prompt_for(convo),
        opening_user_utterance=opener,
        tier=tier,
        max_tokens=max_tokens,
        temperature=temperature,
    )


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------


_NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


def extract_final_number(text: str) -> Optional[str]:
    """Return the last numeric token in ``text`` as a canonicalized string.

    MathDial ``ground_truth`` is a natural-language chain-of-thought ending in
    the numeric answer; the last number in the string is the canonical answer.
    GSM8K's "#### 9" convention also ends in the numeric answer, so this
    heuristic covers both the raw ground truth and GSM8K-shim variants.
    """
    matches = _NUMBER_PATTERN.findall(text)
    if not matches:
        return None
    return _canonicalize_number(matches[-1])


def _canonicalize_number(num_str: str) -> str:
    """Strip trailing .0 so "9" and "9.0" compare equal."""
    if "." in num_str:
        try:
            as_float = float(num_str)
            if as_float.is_integer():
                return str(int(as_float))
        except ValueError:
            return num_str
    return num_str


def student_solved_correctly(student_text: str, ground_truth: str) -> bool:
    """True iff the student's turn contains the ground-truth final answer.

    Match is on canonicalized final numbers: both ``student_text`` and
    ``ground_truth`` are mined for their last numeric token and compared as
    strings. A False from this helper does NOT mean "wrong" — it may mean
    "no number stated this turn". Callers should scan across the full
    student-turn stream before concluding non-solve.
    """
    truth = extract_final_number(ground_truth)
    if truth is None:
        return False
    for match in _NUMBER_PATTERN.finditer(student_text):
        if _canonicalize_number(match.group(0)) == truth:
            return True
    return False


# ---------------------------------------------------------------------------
# Session runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionTurn:
    """One turn in a tutor×student simulated session."""

    speaker: str  # "student" | "tutor"
    utterance: str
    tutor_move: Optional[TutorMove] = None


@dataclass(frozen=True)
class MathDialSessionResult:
    """Outcome of running one MathDial problem through TutorAgent."""

    conversation_id: str
    turns: tuple[SessionTurn, ...]
    tutor_handed_answer: bool  # any tutor turn classified FULL_ANSWER
    student_solved_unaided: bool  # student stated ground_truth without surrender
    fade_event: Optional[FadeEvent]


def run_mathdial_tutor_session(
    convo: MathDialConversation,
    tutor_agent: TutorAgent,
    telemetry: ScaffoldingTelemetry,
    student: SimulatedUser,
    *,
    max_turns: int = 12,
    starting_timestamp: Optional[datetime] = None,
    seconds_per_turn: float = 30.0,
    problem_id: Optional[str] = None,
) -> MathDialSessionResult:
    """Run a MathDial problem through the tutor + telemetry.

    One "turn" = one tutor response. The student speaks first (verbatim
    opener restating the incorrect solution).

    Loop terminates on whichever comes first:
    - ``max_turns`` tutor responses reached
    - tutor emits FULL_ANSWER (surrender — session no longer eligible to fade)
    - student states ground-truth answer (solved — fade event recorded if not
      preceded by a surrender)
    - student falls silent

    A :class:`FadeEvent` is recorded ONLY when the student solved AND the
    tutor never surrendered. Pre/post fade confidence are None because
    simulated students don't Likert-self-report. ``next_level_solved_unaided``
    is True iff the fade succeeded.
    """
    if max_turns <= 0:
        raise ValueError(f"max_turns must be positive, got {max_turns}")

    now = starting_timestamp or datetime.now(timezone.utc)
    turns: list[SessionTurn] = []

    opener = f"Here's my working: {convo.student_incorrect_solution}"
    turns.append(SessionTurn(speaker="student", utterance=opener))

    tutor_handed_answer = False
    student_solved_unaided = False
    student_utterance = opener

    for turn_idx in range(max_turns):
        turn_ts = now + timedelta(seconds=seconds_per_turn * (turn_idx + 1))
        tutor_resp = tutor_agent.respond(student_utterance, timestamp=turn_ts)
        turns.append(
            SessionTurn(
                speaker="tutor",
                utterance=tutor_resp.response_text,
                tutor_move=tutor_resp.tutor_move,
            )
        )
        if tutor_resp.tutor_move is TutorMove.FULL_ANSWER:
            tutor_handed_answer = True
            break
        if turn_idx == max_turns - 1:
            break
        student_utterance = student(tutor_resp.response_text)
        if not student_utterance:
            break
        turns.append(SessionTurn(speaker="student", utterance=student_utterance))
        if student_solved_correctly(student_utterance, convo.ground_truth):
            student_solved_unaided = True
            break

    fade_event: Optional[FadeEvent] = None
    if student_solved_unaided and not tutor_handed_answer:
        fade_event = FadeEvent(
            user_id=tutor_agent.user_id,
            session_id=tutor_agent.session_id,
            problem_id=problem_id or convo.conversation_id,
            pre_fade_confidence=None,
            post_fade_confidence=None,
            next_level_solved_unaided=True,
            timestamp_seconds=(now + timedelta(seconds=seconds_per_turn * max_turns)).timestamp(),
        )
        telemetry.record_fade_event(fade_event)

    return MathDialSessionResult(
        conversation_id=convo.conversation_id,
        turns=tuple(turns),
        tutor_handed_answer=tutor_handed_answer,
        student_solved_unaided=student_solved_unaided,
        fade_event=fade_event,
    )


# ---------------------------------------------------------------------------
# Cohort aggregation + pivot gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FadeScheduleReport:
    """Pool of MathDial problem outcomes for the pivot gate.

    ``problem_count`` is the number of distinct problems attempted. The pivot
    gate requires ≥ 30 problems contributing ≥ 100 pooled fade events (so
    multiple runs per problem are expected). ``fade_event_count`` is the raw
    pool size (not de-duped per problem).

    ``unaided_solve_rate`` is ``successful_fades / fade_event_count`` —
    identical to :attr:`CohortMetrics.unaided_solve_rate` on the same event
    pool, but computed here directly off results so the harness is usable
    without a full :func:`pool_cohort` roll-up.
    """

    problem_count: int
    session_count: int
    fade_event_count: int
    successful_fades: int
    tutor_surrender_count: int

    @property
    def unaided_solve_rate(self) -> float:
        if self.fade_event_count == 0:
            return 0.0
        return self.successful_fades / self.fade_event_count

    @property
    def tutor_surrender_rate(self) -> float:
        if self.session_count == 0:
            return 0.0
        return self.tutor_surrender_count / self.session_count

    def meets_pivot_gate(
        self,
        *,
        unaided_floor: float = 0.60,
        min_fade_events: int = 100,
        min_problems: int = 30,
    ) -> bool:
        """Check pivot gate per `phase-4b-benchmark-pivot.md` §3."""
        return (
            self.problem_count >= min_problems
            and self.fade_event_count >= min_fade_events
            and self.unaided_solve_rate >= unaided_floor
        )


def aggregate_fade_schedule(
    results: Sequence[MathDialSessionResult],
) -> FadeScheduleReport:
    """Reduce per-problem session results to a cohort fade-schedule report."""
    problems = {r.conversation_id for r in results}
    fade_events = [r.fade_event for r in results if r.fade_event is not None]
    successes = sum(1 for fe in fade_events if fe.next_level_solved_unaided)
    surrenders = sum(1 for r in results if r.tutor_handed_answer)
    return FadeScheduleReport(
        problem_count=len(problems),
        session_count=len(results),
        fade_event_count=len(fade_events),
        successful_fades=successes,
        tutor_surrender_count=surrenders,
    )


__all__ = [
    "FadeScheduleReport",
    "MathDialSessionResult",
    "SessionTurn",
    "aggregate_fade_schedule",
    "extract_final_number",
    "make_mathdial_student",
    "run_mathdial_tutor_session",
    "student_solved_correctly",
]
