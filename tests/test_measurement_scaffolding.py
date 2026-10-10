"""Phase 4a — scaffolding-fading telemetry unit tests.

Verifies the CoMeT 4-metric roll-up + the roadmap gate denominator check.
"""

from __future__ import annotations

import pytest

from psychology_base.applications.measurement import (
    FadeEvent,
    ScaffoldingTelemetry,
    TutorMove,
    TutorTurn,
    pool_cohort,
)


def _turn(session_id: str, idx: int, move: TutorMove) -> TutorTurn:
    return TutorTurn(
        session_id=session_id,
        turn_index=idx,
        move=move,
        timestamp_seconds=float(idx),
    )


def _fade(session_id: str, problem_id: str, solved: bool, user_id: str = "u1") -> FadeEvent:
    return FadeEvent(
        user_id=user_id,
        session_id=session_id,
        problem_id=problem_id,
        pre_fade_confidence=3.0,
        post_fade_confidence=4.0,
        next_level_solved_unaided=solved,
        timestamp_seconds=0.0,
    )


def test_empty_session_summary_zero_rates() -> None:
    t = ScaffoldingTelemetry(user_id="u1")
    summary = t.session_summary("s1")
    assert summary.tutor_turn_count == 0
    assert summary.full_answer_surrender_rate() == 0.0
    assert summary.artifact_delivery_rate() == 0.0
    assert summary.frustration_self_report is None


def test_session_summary_counts_moves() -> None:
    t = ScaffoldingTelemetry(user_id="u1")
    t.record_tutor_turn(_turn("s1", 0, TutorMove.SOCRATIC_PROBE))
    t.record_tutor_turn(_turn("s1", 1, TutorMove.WORKED_STEP))
    t.record_tutor_turn(_turn("s1", 2, TutorMove.FULL_ANSWER))
    t.record_tutor_turn(_turn("s1", 3, TutorMove.SOCRATIC_PROBE))
    summary = t.session_summary("s1")
    assert summary.tutor_turn_count == 4
    assert summary.full_answer_count == 1
    assert summary.artifact_count == 1
    assert summary.full_answer_surrender_rate() == pytest.approx(0.25)
    assert summary.artifact_delivery_rate() == pytest.approx(0.25)


def test_frustration_scoped_per_session() -> None:
    t = ScaffoldingTelemetry(user_id="u1")
    t.record_frustration("s1", 3)
    t.record_frustration("s2", 1)
    assert t.session_summary("s1").frustration_self_report == 3
    assert t.session_summary("s2").frustration_self_report == 1
    assert t.session_summary("s3").frustration_self_report is None


def test_frustration_rejects_out_of_range() -> None:
    t = ScaffoldingTelemetry(user_id="u1")
    with pytest.raises(ValueError):
        t.record_frustration("s1", 0)
    with pytest.raises(ValueError):
        t.record_frustration("s1", 6)


def test_fade_events_scoped_per_session() -> None:
    t = ScaffoldingTelemetry(user_id="u1")
    t.record_fade_event(_fade("s1", "p1", True))
    t.record_fade_event(_fade("s1", "p2", False))
    t.record_fade_event(_fade("s2", "p3", True))
    s1 = t.session_summary("s1")
    s2 = t.session_summary("s2")
    assert len(s1.fade_events) == 2
    assert len(s2.fade_events) == 1


def test_pool_cohort_rolls_up_unaided_rate() -> None:
    telemetries = []
    for user_num in range(10):
        user_id = f"u{user_num}"
        t = ScaffoldingTelemetry(user_id=user_id)
        for sess_num in range(4):
            session_id = f"{user_id}-s{sess_num}"
            t.record_tutor_turn(_turn(session_id, 0, TutorMove.SOCRATIC_PROBE))
            # Alternate solved / not — 10 users × 4 sessions × 1 fade each = 40 fades
            solved = (user_num + sess_num) % 2 == 0
            t.record_fade_event(_fade(session_id, f"p{sess_num}", solved, user_id))
            t.record_frustration(session_id, 2)
        telemetries.append(t)
    session_ids = {
        f"u{i}": [f"u{i}-s{j}" for j in range(4)] for i in range(10)
    }
    cohort = pool_cohort(telemetries, session_ids)
    assert cohort.user_count == 10
    assert cohort.session_count == 40
    assert cohort.fade_event_count == 40
    assert cohort.unaided_solve_rate == pytest.approx(0.5)
    assert cohort.mean_frustration == pytest.approx(2.0)


def test_cohort_meets_roadmap_gate() -> None:
    telemetries = []
    for user_num in range(10):
        user_id = f"u{user_num}"
        t = ScaffoldingTelemetry(user_id=user_id)
        for sess_num in range(3):
            session_id = f"{user_id}-s{sess_num}"
            t.record_tutor_turn(_turn(session_id, 0, TutorMove.SOCRATIC_PROBE))
            # 7/10 users solve all 3 → 21 solved; 3/10 users solve 0 → 0 solved
            # Total: 21/30 ≈ 0.70 unaided solve rate
            solved = user_num < 7
            t.record_fade_event(_fade(session_id, f"p{sess_num}", solved, user_id))
        telemetries.append(t)
    session_ids = {
        f"u{i}": [f"u{i}-s{j}" for j in range(3)] for i in range(10)
    }
    cohort = pool_cohort(telemetries, session_ids)
    assert cohort.fade_event_count == 30
    assert cohort.unaided_solve_rate == pytest.approx(0.70)
    assert cohort.meets_roadmap_gate(unaided_floor=0.60) is True
    assert cohort.meets_roadmap_gate(unaided_floor=0.75) is False


def test_cohort_fails_gate_with_too_few_fade_events() -> None:
    telemetries = []
    for user_num in range(10):
        t = ScaffoldingTelemetry(user_id=f"u{user_num}")
        # Only 1 fade event per user → 10 total, below min_fade_events=20.
        session_id = f"u{user_num}-s0"
        t.record_tutor_turn(_turn(session_id, 0, TutorMove.SOCRATIC_PROBE))
        t.record_fade_event(_fade(session_id, "p0", True, f"u{user_num}"))
        telemetries.append(t)
    session_ids = {f"u{i}": [f"u{i}-s0"] for i in range(10)}
    cohort = pool_cohort(telemetries, session_ids)
    assert cohort.fade_event_count == 10
    assert cohort.unaided_solve_rate == 1.0
    assert cohort.meets_roadmap_gate() is False


def test_cohort_fails_gate_with_too_few_users() -> None:
    telemetries = []
    for user_num in range(5):
        t = ScaffoldingTelemetry(user_id=f"u{user_num}")
        for sess_num in range(5):
            session_id = f"u{user_num}-s{sess_num}"
            t.record_tutor_turn(_turn(session_id, 0, TutorMove.SOCRATIC_PROBE))
            t.record_fade_event(_fade(session_id, f"p{sess_num}", True, f"u{user_num}"))
        telemetries.append(t)
    session_ids = {f"u{i}": [f"u{i}-s{j}" for j in range(5)] for i in range(5)}
    cohort = pool_cohort(telemetries, session_ids)
    assert cohort.fade_event_count == 25
    assert cohort.user_count == 5
    assert cohort.meets_roadmap_gate() is False  # 5 < min_users=10
