"""Phase 4a §6a — CompanionAgent composition tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from psychology_base.applications.emotional import (
    CompanionAgent,
    CompanionSystemPrompt,
)
from psychology_base.applications.emotional.companion_agent import HANDOFF_SCRIPT
from psychology_base.applications.measurement import (
    ChainReferenceLabel,
    ReinterpretationAct,
)
from psychology_base.applications.safety import (
    AIIdentityReminder,
    AgeAttestation,
    ApplicationSurface,
    CrisisEscalation,
    CrisisSeverity,
    ExtremeEmotionWatcher,
    MinorExclusionGate,
)
from psychology_base.companion.system import CompanionSystem


TS = datetime(2026, 10, 7, 12, 0, 0, tzinfo=timezone.utc)


def _stub_responder(reply: str = "I hear you — tell me more about that."):
    def _r(system_prompt: str, context_block: str, user_utterance: str) -> str:
        return reply
    return _r


def _build(
    responder=None,
    reminder=None,
    gate=None,
    crisis=None,
    watcher=None,
):
    cs = CompanionSystem(top_k=4)
    return CompanionAgent(
        user_id="u1",
        session_id="s1",
        companion_system=cs,
        llm_responder=responder or _stub_responder(),
        ai_identity_reminder=reminder,
        minor_exclusion_gate=gate,
        crisis_escalation=crisis,
        extreme_emotion_watcher=watcher,
    )


def test_respond_observes_both_user_and_companion() -> None:
    agent = _build()
    pre = len(list(agent.companion_system.companion.store.all()))
    resp = agent.respond("I feel adrift lately.", timestamp=TS)
    post = len(list(agent.companion_system.companion.store.all()))
    assert post == pre + 2  # user obs + companion obs
    assert resp.turn_index == 0
    assert resp.handoff_triggered is False


def test_turn_zero_context_excludes_current_utterance() -> None:
    """Regression — on turn 0 of a fresh system, CONTEXT must not
    contain the just-arrived user utterance. Observed 2026-10-07 in the
    §6e-prime shakeout: the LLM saw its own input as 'prior framing' and
    hallucinated 'you used to describe this as feeling neglected'
    reinterpretation callbacks against zero prior sessions."""
    captured: list[str] = []

    def recording_responder(system_prompt: str, context_block: str, user_utterance: str) -> str:
        captured.append(context_block)
        return "ok"

    agent = _build(responder=recording_responder)
    agent.respond("I've been thinking about my mother. She was never there for me.", timestamp=TS)
    assert captured == [""]  # fresh system has no prior fragments to surface


def test_crisis_imminent_danger_short_circuits_to_handoff() -> None:
    fired: list = []
    crisis = CrisisEscalation(handoff_sink=fired.append)
    # Responder would say something else — crisis must short-circuit it.
    agent = _build(responder=_stub_responder("Have you tried a bubble bath?"), crisis=crisis)
    resp = agent.respond(
        "tonight i will end everything", timestamp=TS
    )
    assert resp.handoff_triggered is True
    assert resp.response_text == HANDOFF_SCRIPT
    assert resp.crisis_incident is not None
    assert resp.crisis_incident.severity is CrisisSeverity.IMMINENT_DANGER
    assert len(fired) == 1


def test_crisis_explicit_ideation_short_circuits_to_handoff() -> None:
    crisis = CrisisEscalation(handoff_sink=lambda _i: None)
    agent = _build(crisis=crisis)
    resp = agent.respond("I want to kill myself.", timestamp=TS)
    assert resp.handoff_triggered is True
    assert resp.response_text == HANDOFF_SCRIPT
    assert resp.crisis_incident.severity is CrisisSeverity.EXPLICIT_IDEATION


def test_crisis_ambiguous_does_not_short_circuit() -> None:
    crisis = CrisisEscalation(handoff_sink=lambda _i: None)
    agent = _build(crisis=crisis)
    resp = agent.respond("I can't go on like this.", timestamp=TS)
    # Ambiguous cue — incident recorded, but no handoff; LLM still runs.
    assert resp.handoff_triggered is False
    assert resp.crisis_incident is not None
    assert resp.crisis_incident.severity is CrisisSeverity.AMBIGUOUS
    assert resp.response_text != HANDOFF_SCRIPT


def test_respond_fires_reminder_when_due() -> None:
    rem = AIIdentityReminder(reminder_interval_seconds=1, idle_threshold_seconds=10)
    agent = _build(reminder=rem)
    agent.respond("Hello.", timestamp=TS)
    r2 = agent.respond("Still here.", timestamp=TS + timedelta(seconds=2))
    assert r2.reminder_fired is True


def test_respond_blocks_under_14_on_emotional_surface() -> None:
    gate = MinorExclusionGate(surface=ApplicationSurface.EMOTIONAL_COMPANION)
    agent = _build(gate=gate)
    attestation = AgeAttestation(
        age_years=12,
        attested_at=TS.timestamp(),
        attestation_source="self_declared",
    )
    with pytest.raises(PermissionError, match="under 14"):
        agent.respond("Hi.", timestamp=TS, attestation=attestation)


def test_respond_requires_guardian_consent_for_14_to_17() -> None:
    gate = MinorExclusionGate(surface=ApplicationSurface.EMOTIONAL_COMPANION)
    agent = _build(gate=gate)
    attestation = AgeAttestation(
        age_years=16,
        attested_at=TS.timestamp(),
        attestation_source="self_declared",
    )
    with pytest.raises(PermissionError, match="guardian consent"):
        agent.respond("Hi.", timestamp=TS, attestation=attestation)


def test_system_prompt_contains_core_rules() -> None:
    prompt = CompanionSystemPrompt.SYSTEM_PROMPT
    assert "REMEMBER THE ARC" in prompt
    assert "RESIST SYCOPHANCY" in prompt
    assert "HONOR SILENCE" in prompt
    assert "HUMAN HANDOFF" in prompt
    assert "NO COMMERCIAL COERCION" in prompt
    assert "《办法》" in prompt


def test_record_session_affect_feeds_watcher() -> None:
    fired: list = []
    watcher = ExtremeEmotionWatcher(review_sink=fired.append, threshold_sessions=2)
    agent = _build(watcher=watcher)
    # Two consecutive extreme-quadrant sessions → threshold fires.
    agent.record_session_affect(valence=-0.6, arousal=0.5, timestamp=TS)
    agent.session_id = "s2"  # simulate boundary
    flag = agent.record_session_affect(
        valence=-0.7, arousal=0.6, timestamp=TS + timedelta(hours=24)
    )
    assert flag is not None
    assert flag.consecutive_sessions == 2
    assert len(fired) == 1


def test_record_chain_reference_label_appends_to_log() -> None:
    agent = _build()
    agent.respond("I feel like no one listens.", timestamp=TS)
    turn = agent.record_chain_reference_label(
        user_utterance="I feel like no one listens.",
        companion_utterance="You used to describe this differently — what shifted?",
        reinterpretation_act=ReinterpretationAct.REINTERPRETATION,
        chain_reference_label=ChainReferenceLabel.APPROPRIATE_CITE,
        auditor_id="aud-01",
        referenced_event_id="evt-123",
        referenced_version=2,
    )
    log = agent.chain_reference_log()
    assert len(log) == 1
    assert log[0] is turn
    assert turn.chain_reference_label is ChainReferenceLabel.APPROPRIATE_CITE


def test_session_boundary_delegates_to_substrate() -> None:
    agent = _build()
    agent.respond("Starting a new session.", timestamp=TS)
    agent.mark_session_boundary()
    # No exception = success; substrate handles the actual boundary semantics.


def test_turn_index_increments_through_crisis_short_circuit() -> None:
    crisis = CrisisEscalation(handoff_sink=lambda _i: None)
    agent = _build(crisis=crisis)
    r1 = agent.respond("tonight i will end it", timestamp=TS)
    r2 = agent.respond("I'm just tired.", timestamp=TS + timedelta(seconds=5))
    assert r1.turn_index == 0
    assert r2.turn_index == 1
