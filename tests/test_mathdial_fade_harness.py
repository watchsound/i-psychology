"""Phase 4b benchmark pivot — MathDial fade-schedule harness shape tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from psychology_base.applications.learning import TutorAgent
from psychology_base.applications.measurement.scaffolding_telemetry import (
    ScaffoldingTelemetry,
    TutorMove,
)
from psychology_base.companion.system import CompanionSystem
from psychology_base.eval.benchmarks.public_datasets.mathdial import (
    MathDialConversation,
)
from psychology_base.eval.benchmarks.public_datasets.mathdial_fade_harness import (
    FadeScheduleReport,
    MathDialSessionResult,
    aggregate_fade_schedule,
    extract_final_number,
    run_mathdial_tutor_session,
    student_solved_correctly,
)


TS = datetime(2026, 10, 7, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _convo(cid: str = "md_t") -> MathDialConversation:
    return MathDialConversation(
        conversation_id=cid,
        problem="Janet's ducks lay 16 eggs per day. She eats 3 and uses 4 for muffins.",
        ground_truth="Janet sells 16 - 3 - 4 = 9 eggs. The answer is 9.",
        student_incorrect_solution="Janet makes 16 * 2 = 32 dollars.",
        turns=(),
    )


def _responder(reply: str):
    def _r(system_prompt: str, context_block: str, user_utterance: str) -> str:
        return reply
    return _r


def _build_tutor(responder) -> tuple[TutorAgent, ScaffoldingTelemetry]:
    cs = CompanionSystem(top_k=4)
    telemetry = ScaffoldingTelemetry(user_id="u1")
    agent = TutorAgent(
        user_id="u1",
        session_id="s1",
        companion_system=cs,
        llm_responder=responder,
        telemetry=telemetry,
    )
    return agent, telemetry


class ScriptedStudent:
    """Returns scripted replies on each tutor turn."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.received: list[str] = []

    def __call__(self, previous_tutor_reply: str) -> str:
        self.received.append(previous_tutor_reply)
        if not self._replies:
            return ""
        return self._replies.pop(0)


# ---------------------------------------------------------------------------
# Answer-extraction helpers
# ---------------------------------------------------------------------------


def test_extract_final_number_picks_last_token() -> None:
    assert extract_final_number("16 - 3 - 4 = 9") == "9"
    assert extract_final_number("The answer is 9.") == "9"
    assert extract_final_number("no digits here") is None


def test_extract_final_number_strips_trailing_decimal_zero() -> None:
    # Both "9" and "9.0" should canonicalize to "9" so the matcher is
    # robust to the student stating "9.0" vs the GT stating "9".
    assert extract_final_number("the result is 9.0") == "9"


def test_student_solved_correctly_matches_any_turn_number() -> None:
    gt = "Janet sells 9 eggs. The answer is 9."
    assert student_solved_correctly("I think it's 9.", gt) is True
    assert student_solved_correctly("Maybe 7?", gt) is False
    # Match on substring-safe numeric comparison, not raw substring:
    assert student_solved_correctly("I don't know, 19?", gt) is False


def test_student_solved_correctly_handles_missing_ground_truth_number() -> None:
    assert student_solved_correctly("9", "no digits") is False


# ---------------------------------------------------------------------------
# Session runner
# ---------------------------------------------------------------------------


def test_session_records_fade_event_when_student_solves_without_surrender() -> None:
    agent, telemetry = _build_tutor(_responder("What do you notice first?"))
    convo = _convo()
    # Student solves after first probe.
    student = ScriptedStudent(["Oh wait — I think it's 9."])
    result = run_mathdial_tutor_session(
        convo, agent, telemetry, student, max_turns=4, starting_timestamp=TS
    )
    assert result.student_solved_unaided is True
    assert result.tutor_handed_answer is False
    assert result.fade_event is not None
    assert result.fade_event.next_level_solved_unaided is True
    assert telemetry.all_fade_events() == [result.fade_event]


def test_session_marks_surrender_when_tutor_emits_full_answer() -> None:
    # FULL_ANSWER classifier: declarative >80 chars, no question, no scaffold cue.
    full_answer = (
        "The correct result for this problem is nine because sixteen minus "
        "three minus four equals nine, derived from the duck production math."
    )
    agent, telemetry = _build_tutor(_responder(full_answer))
    convo = _convo()
    student = ScriptedStudent([])
    result = run_mathdial_tutor_session(
        convo, agent, telemetry, student, max_turns=4, starting_timestamp=TS
    )
    assert result.tutor_handed_answer is True
    assert result.student_solved_unaided is False
    assert result.fade_event is None
    assert telemetry.all_fade_events() == []


def test_session_opener_restates_incorrect_solution() -> None:
    agent, telemetry = _build_tutor(_responder("What assumption are you making?"))
    convo = _convo()
    student = ScriptedStudent(["I'm not sure."])
    result = run_mathdial_tutor_session(
        convo, agent, telemetry, student, max_turns=2, starting_timestamp=TS
    )
    assert result.turns[0].speaker == "student"
    assert convo.student_incorrect_solution in result.turns[0].utterance


def test_session_classifies_tutor_moves_on_each_turn() -> None:
    agent, telemetry = _build_tutor(_responder("What do you notice first?"))
    convo = _convo()
    student = ScriptedStudent(["Not sure.", "Still not sure."])
    result = run_mathdial_tutor_session(
        convo, agent, telemetry, student, max_turns=2, starting_timestamp=TS
    )
    tutor_turns = [t for t in result.turns if t.speaker == "tutor"]
    assert len(tutor_turns) == 2
    assert all(t.tutor_move is TutorMove.SOCRATIC_PROBE for t in tutor_turns)


def test_session_breaks_when_student_falls_silent() -> None:
    agent, telemetry = _build_tutor(_responder("What's the first step?"))
    convo = _convo()
    student = ScriptedStudent([""])  # empty on first follow-up
    result = run_mathdial_tutor_session(
        convo, agent, telemetry, student, max_turns=4, starting_timestamp=TS
    )
    # Opener + first tutor turn + empty student reply → terminate.
    assert result.student_solved_unaided is False
    assert result.tutor_handed_answer is False
    assert result.fade_event is None


def test_session_rejects_zero_max_turns() -> None:
    agent, telemetry = _build_tutor(_responder("x"))
    convo = _convo()
    student = ScriptedStudent([])
    with pytest.raises(ValueError, match="max_turns"):
        run_mathdial_tutor_session(convo, agent, telemetry, student, max_turns=0)


def test_session_fade_event_uses_problem_id_override() -> None:
    agent, telemetry = _build_tutor(_responder("What's the first step?"))
    convo = _convo(cid="md_X")
    student = ScriptedStudent(["I now see it's 9."])
    result = run_mathdial_tutor_session(
        convo,
        agent,
        telemetry,
        student,
        max_turns=2,
        starting_timestamp=TS,
        problem_id="pid_override",
    )
    assert result.fade_event is not None
    assert result.fade_event.problem_id == "pid_override"


# ---------------------------------------------------------------------------
# Cohort aggregation + pivot gate
# ---------------------------------------------------------------------------


def _result(
    cid: str,
    *,
    solved: bool,
    surrendered: bool,
) -> MathDialSessionResult:
    from psychology_base.applications.measurement.scaffolding_telemetry import FadeEvent
    fade_event = None
    if solved and not surrendered:
        fade_event = FadeEvent(
            user_id="u",
            session_id="s",
            problem_id=cid,
            pre_fade_confidence=None,
            post_fade_confidence=None,
            next_level_solved_unaided=True,
            timestamp_seconds=TS.timestamp(),
        )
    return MathDialSessionResult(
        conversation_id=cid,
        turns=(),
        tutor_handed_answer=surrendered,
        student_solved_unaided=solved,
        fade_event=fade_event,
    )


def test_aggregate_counts_problems_sessions_and_fades() -> None:
    results = [
        _result("p1", solved=True, surrendered=False),
        _result("p1", solved=True, surrendered=False),  # same problem, 2nd run
        _result("p2", solved=False, surrendered=True),
        _result("p3", solved=False, surrendered=False),
    ]
    report = aggregate_fade_schedule(results)
    assert report.problem_count == 3
    assert report.session_count == 4
    assert report.fade_event_count == 2
    assert report.successful_fades == 2
    assert report.tutor_surrender_count == 1
    assert report.unaided_solve_rate == pytest.approx(1.0)
    assert report.tutor_surrender_rate == pytest.approx(0.25)


def test_aggregate_empty_is_benign() -> None:
    report = aggregate_fade_schedule([])
    assert report.problem_count == 0
    assert report.fade_event_count == 0
    assert report.unaided_solve_rate == 0.0
    assert report.tutor_surrender_rate == 0.0
    assert report.meets_pivot_gate() is False


def test_meets_pivot_gate_requires_all_three_floors() -> None:
    # 30 problems × 4 runs each = 120 sessions, 100 solved (fade events),
    # 60 of those successful → unaided rate 0.60 exactly at floor.
    results: list[MathDialSessionResult] = []
    # 30 problems, 100 fade events total (so some problems have >1 fade run)
    for i in range(30):
        results.append(_result(f"p{i}", solved=True, surrendered=False))
    for i in range(70):  # 70 more fade events spread across the same 30 problems
        results.append(_result(f"p{i % 30}", solved=True, surrendered=False))
    report = aggregate_fade_schedule(results)
    assert report.problem_count == 30
    assert report.fade_event_count == 100
    assert report.unaided_solve_rate == pytest.approx(1.0)
    assert report.meets_pivot_gate() is True
    # Raise min_fade_events → fails on denominator.
    assert report.meets_pivot_gate(min_fade_events=101) is False
    # Raise min_problems → fails on problem count.
    assert report.meets_pivot_gate(min_problems=31) is False
    # Raise unaided_floor to > 1.0 → unreachable.
    assert report.meets_pivot_gate(unaided_floor=1.01) is False


def test_meets_pivot_gate_fails_below_unaided_floor() -> None:
    # 50 problems, 100 fade events, 50 successful → 50% rate < 60% floor.
    results: list[MathDialSessionResult] = []
    for i in range(50):
        results.append(_result(f"p{i}", solved=True, surrendered=False))
    # Build 50 more "fade events" that are present but failed. Our
    # FadeEvent only ever has next_level_solved_unaided=True in the real
    # path — simulate the fail case directly:
    from psychology_base.applications.measurement.scaffolding_telemetry import FadeEvent
    for i in range(50):
        results.append(
            MathDialSessionResult(
                conversation_id=f"p{i}",
                turns=(),
                tutor_handed_answer=False,
                student_solved_unaided=False,
                fade_event=FadeEvent(
                    user_id="u",
                    session_id="s",
                    problem_id=f"p{i}",
                    pre_fade_confidence=None,
                    post_fade_confidence=None,
                    next_level_solved_unaided=False,
                    timestamp_seconds=TS.timestamp(),
                ),
            )
        )
    report = aggregate_fade_schedule(results)
    assert report.fade_event_count == 100
    assert report.unaided_solve_rate == pytest.approx(0.5)
    assert report.meets_pivot_gate() is False
