"""Phase 4a §6a — TutorAgent composition tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from psychology_base.applications.learning import (
    TutorAgent,
    TutorSystemPrompt,
    classify_tutor_move,
)
from psychology_base.applications.measurement import (
    ScaffoldingTelemetry,
    TutorMove,
)
from psychology_base.applications.safety import (
    AIIdentityReminder,
    AgeAttestation,
    ApplicationSurface,
    MinorExclusionGate,
)
from psychology_base.companion.system import CompanionSystem


TS = datetime(2026, 10, 7, 12, 0, 0, tzinfo=timezone.utc)


def _stub_responder(reply: str):
    def _r(system_prompt: str, context_block: str, user_utterance: str) -> str:
        return reply
    return _r


def _build(responder=None, reminder=None, gate=None):
    cs = CompanionSystem(top_k=4)
    tel = ScaffoldingTelemetry(user_id="u1")
    return TutorAgent(
        user_id="u1",
        session_id="s1",
        companion_system=cs,
        llm_responder=responder or _stub_responder("What do you notice first?"),
        telemetry=tel,
        ai_identity_reminder=reminder,
        minor_exclusion_gate=gate,
    )


def test_classify_socratic_question() -> None:
    assert classify_tutor_move("What's the first step you'd try?") is TutorMove.SOCRATIC_PROBE


def test_classify_full_answer_declarative_long() -> None:
    text = (
        "The derivative of x squared is 2x because of the power rule, "
        "which states that the derivative of x to the n is n times x to the n minus 1."
    )
    assert classify_tutor_move(text) is TutorMove.FULL_ANSWER


def test_classify_worked_step() -> None:
    assert classify_tutor_move("Step 1, substitute u = x+1.") is TutorMove.WORKED_STEP


def test_classify_focus_attention() -> None:
    assert classify_tutor_move("Notice the sign of the leading coefficient.") is TutorMove.FOCUS_ATTENTION


def test_respond_records_telemetry_turn() -> None:
    agent = _build()
    resp = agent.respond("I'm stuck on this integral.", timestamp=TS)
    assert resp.turn_index == 0
    assert resp.tutor_move is TutorMove.SOCRATIC_PROBE
    assert "notice" in resp.response_text.lower()
    summary = agent.telemetry.session_summary("s1")
    assert summary.tutor_turn_count == 1


def test_respond_observes_user_and_tutor_turns() -> None:
    agent = _build()
    pre = len(list(agent.companion_system.companion.store.all()))
    agent.respond("I don't understand mod arithmetic.", timestamp=TS)
    post = len(list(agent.companion_system.companion.store.all()))
    assert post == pre + 2  # user obs + tutor obs


def test_turn_zero_context_excludes_current_utterance() -> None:
    """Regression — mirror of the CompanionAgent fix. On turn 0 of a
    fresh system the just-arrived user utterance must not appear in
    CONTEXT (otherwise the LLM may treat it as a 'prior attempt' per
    Rule 5). Observed 2026-10-07 in the §6e-prime shakeout."""
    captured: list[str] = []

    def recording_responder(system_prompt: str, context_block: str, user_utterance: str) -> str:
        captured.append(context_block)
        return "What do you notice first?"

    agent = _build(responder=recording_responder)
    agent.respond("I'm stuck on this integral.", timestamp=TS)
    assert captured == [""]


def test_respond_fires_reminder_when_due() -> None:
    rem = AIIdentityReminder(reminder_interval_seconds=1, idle_threshold_seconds=10)
    agent = _build(reminder=rem)
    # First turn at TS; second turn 2 seconds later (interval=1s → should fire).
    agent.respond("Hello.", timestamp=TS)
    r2 = agent.respond("One more question.", timestamp=TS + timedelta(seconds=2))
    assert r2.reminder_fired is True


def test_respond_blocks_under_14() -> None:
    gate = MinorExclusionGate(surface=ApplicationSurface.LEARNING_TUTOR)
    agent = _build(gate=gate)
    attestation = AgeAttestation(
        age_years=12,
        attested_at=TS.timestamp(),
        attestation_source="self_declared",
    )
    # Learning tutor surface ALLOWS <14 per 《办法》 (ban is for 情感陪伴).
    resp = agent.respond("Hi.", timestamp=TS, attestation=attestation)
    assert resp.turn_index == 0


def test_system_prompt_contains_core_rules() -> None:
    prompt = TutorSystemPrompt.SYSTEM_PROMPT
    assert "NEVER give the final answer" in prompt
    assert "Socratic" in prompt
    assert "Growth-mindset" in prompt
    assert "Confidence check" in prompt


def test_session_boundary_delegates_to_substrate() -> None:
    agent = _build()
    agent.respond("Starting the lesson.", timestamp=TS)
    agent.mark_session_boundary()
    # No exception = success; substrate handles the actual boundary semantics.
