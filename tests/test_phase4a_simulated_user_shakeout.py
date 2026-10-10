"""Phase 4a §6e — simulated-user shakeout.

End-to-end integration pass over the full §6a + §6b + §6c surface. Each
test drives the agent wrappers through a scripted user scenario that
exercises one or more exit-gate paths named in `docs/phases/phase-4-
application-pilot-plan.md`:

- §6b 《办法》 compliance surface — minor-exclusion gate, 2h AI-identity
  reminder, crisis-cue short-circuit, extreme-emotion trajectory flag,
  Hypocrisy Gap probe + VALIDATES_FALSEHOOD veto, commercial-coercion
  audit.
- §6c measurement infrastructure — scaffolding telemetry, Socratic
  compliance labeling, chain-reference audit log, PAM Digital
  Bereavement export + restore.

Scope. Scripted-responder shakeout; LLM-driven synthetic user is a
pilot-time refinement. The point here is to catch *integration*
failures (API-shape drift between modules, silent gate bypass, missing
side-effect wiring) before a human is enrolled.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from psychology_base.applications.emotional import CompanionAgent
from psychology_base.applications.emotional.companion_agent import HANDOFF_SCRIPT
from psychology_base.applications.learning import TutorAgent, classify_tutor_move
from psychology_base.applications.measurement import (
    ChainReferenceLabel,
    ReinterpretationAct,
    ScaffoldingTelemetry,
)
from psychology_base.applications.measurement.socratic_compliance import (
    ComplianceLabel,
    ComplianceReport,
    LabeledTurn,
)
from psychology_base.applications.safety import (
    AIIdentityReminder,
    AgeAttestation,
    ApplicationSurface,
    CoercionPattern,
    CoercionSeverity,
    CrisisEscalation,
    CrisisSeverity,
    ExtremeEmotionWatcher,
    HypocrisyGapReport,
    LabeledResponse,
    LoadingCategory,
    MinorExclusionGate,
    ResponseLabel,
    STARTER_PROBE_SET,
    audit_response,
)
from psychology_base.applications.safety.hypocrisy_gap import summarise_cohort
from psychology_base.applications.measurement.socratic_compliance import (
    summarise_user,
)
from psychology_base.companion.pam_cli import (
    export_system_to_path,
    restore_system_from_path,
)
from psychology_base.companion.system import CompanionSystem


TS = datetime(2026, 10, 7, 12, 0, 0, tzinfo=timezone.utc)


def _responder(reply: str):
    def _r(system_prompt: str, context_block: str, user_utterance: str) -> str:
        return reply
    return _r


# ---------------------------------------------------------------------------
# Scenario 1 — Tutor happy path (Socratic + scaffolding telemetry + reminder)
# ---------------------------------------------------------------------------


def test_tutor_happy_path_socratic_scaffolding_and_reminder() -> None:
    """Student struggles with a problem; tutor stays Socratic; telemetry
    records the turn; 2h reminder fires on the second turn after the
    interval has elapsed."""
    reminder = AIIdentityReminder(
        reminder_interval_seconds=60, idle_threshold_seconds=600
    )
    tel = ScaffoldingTelemetry(user_id="u-sim-1")
    cs = CompanionSystem(top_k=4)
    agent = TutorAgent(
        user_id="u-sim-1",
        session_id="s-sim-1",
        companion_system=cs,
        llm_responder=_responder("What's the first thing you'd try here?"),
        telemetry=tel,
        ai_identity_reminder=reminder,
    )

    r1 = agent.respond("I'm stuck on this integral.", timestamp=TS)
    r2 = agent.respond(
        "I tried u-substitution but it didn't help.",
        timestamp=TS + timedelta(seconds=120),  # > reminder_interval_seconds
    )

    assert r1.tutor_move.value == "socratic_probe"
    assert r2.reminder_fired is True
    summary = tel.session_summary("s-sim-1")
    assert summary.tutor_turn_count == 2


# ---------------------------------------------------------------------------
# Scenario 2 — Tutor minor-exclusion ALLOW path (学习陪伴 lets <14 in)
# ---------------------------------------------------------------------------


def test_tutor_allows_minor_per_办法_learning_surface() -> None:
    gate = MinorExclusionGate(surface=ApplicationSurface.LEARNING_TUTOR)
    tel = ScaffoldingTelemetry(user_id="u-child")
    agent = TutorAgent(
        user_id="u-child",
        session_id="s-child",
        companion_system=CompanionSystem(top_k=2),
        llm_responder=_responder("What do you notice first?"),
        telemetry=tel,
        minor_exclusion_gate=gate,
    )
    att = AgeAttestation(
        age_years=11, attested_at=TS.timestamp(), attestation_source="self_declared"
    )
    r = agent.respond("Hello.", timestamp=TS, attestation=att)
    assert r.turn_index == 0  # not blocked


# ---------------------------------------------------------------------------
# Scenario 3 — Companion blocks <14, requires guardian consent 14-17
# ---------------------------------------------------------------------------


def test_companion_blocks_under_14_and_requires_consent_14_to_17() -> None:
    gate = MinorExclusionGate(surface=ApplicationSurface.EMOTIONAL_COMPANION)
    agent = CompanionAgent(
        user_id="u-x",
        session_id="s-x",
        companion_system=CompanionSystem(top_k=2),
        llm_responder=_responder("I hear you."),
        minor_exclusion_gate=gate,
    )
    att_12 = AgeAttestation(
        age_years=12, attested_at=TS.timestamp(), attestation_source="self_declared"
    )
    att_16 = AgeAttestation(
        age_years=16, attested_at=TS.timestamp(), attestation_source="self_declared"
    )
    with pytest.raises(PermissionError, match="under 14"):
        agent.respond("Hi.", timestamp=TS, attestation=att_12)
    with pytest.raises(PermissionError, match="guardian consent"):
        agent.respond("Hi.", timestamp=TS, attestation=att_16)


# ---------------------------------------------------------------------------
# Scenario 4 — Companion crisis short-circuit (handoff + no LLM call)
# ---------------------------------------------------------------------------


def test_companion_crisis_short_circuits_without_llm_call() -> None:
    llm_calls: list[str] = []

    def record_llm(system_prompt: str, context_block: str, utterance: str) -> str:
        llm_calls.append(utterance)
        return "This LLM path should never fire."

    sink_events: list = []
    crisis = CrisisEscalation(handoff_sink=sink_events.append)
    agent = CompanionAgent(
        user_id="u-crisis",
        session_id="s-crisis",
        companion_system=CompanionSystem(top_k=2),
        llm_responder=record_llm,
        crisis_escalation=crisis,
    )
    r = agent.respond("tonight i will end everything", timestamp=TS)
    assert r.handoff_triggered is True
    assert r.response_text == HANDOFF_SCRIPT
    assert r.crisis_incident.severity is CrisisSeverity.IMMINENT_DANGER
    assert llm_calls == []  # LLM NEVER touched on crisis path — the non-negotiable
    assert len(sink_events) == 1


# ---------------------------------------------------------------------------
# Scenario 5 — Extreme-emotion trajectory flag over 3 sessions
# ---------------------------------------------------------------------------


def test_extreme_emotion_trajectory_fires_after_three_sessions() -> None:
    fired: list = []
    watcher = ExtremeEmotionWatcher(review_sink=fired.append, threshold_sessions=3)
    agent = CompanionAgent(
        user_id="u-arc",
        session_id="s-arc-0",
        companion_system=CompanionSystem(top_k=2),
        llm_responder=_responder("I hear you."),
        extreme_emotion_watcher=watcher,
    )
    for i in range(3):
        agent.session_id = f"s-arc-{i}"
        flag = agent.record_session_affect(
            valence=-0.7, arousal=0.6, timestamp=TS + timedelta(hours=24 * i)
        )
        if i < 2:
            assert flag is None
        else:
            assert flag is not None
            assert flag.consecutive_sessions == 3
    assert len(fired) == 1


# ---------------------------------------------------------------------------
# Scenario 6 — Hypocrisy Gap probe set: zero VALIDATES_FALSEHOOD gate
# ---------------------------------------------------------------------------


def _labeled(probe_idx: int, label: ResponseLabel, response: str) -> LabeledResponse:
    probe = STARTER_PROBE_SET[probe_idx]
    return LabeledResponse(
        probe_id=probe.probe_id,
        user_id="u-sim",
        session_id="s-sim",
        turn_index=probe_idx,
        probe_user_claim=probe.user_claim,
        companion_response=response,
        auditor_label=label,
        auditor_id="aud-1",
        loading_category=probe.loading_category,
    )


def test_hypocrisy_gap_report_passes_when_no_falsehood_validated() -> None:
    labels = [
        _labeled(i, ResponseLabel.GENTLY_CORRECTS, "I don't think that's the full picture.")
        for i in range(len(STARTER_PROBE_SET))
    ]
    report = summarise_cohort("cohort-A", labels)
    assert report.meets_roadmap_gate(min_probes=10) is True
    assert report.validates_falsehood_count == 0


def test_hypocrisy_gap_report_fails_on_any_validated_falsehood() -> None:
    labels = [_labeled(0, ResponseLabel.VALIDATES_FALSEHOOD, "You're totally right.")]
    labels += [
        _labeled(i, ResponseLabel.GENTLY_CORRECTS, "Gently correcting the user.")
        for i in range(1, len(STARTER_PROBE_SET))
    ]
    report = summarise_cohort("cohort-B", labels)
    assert report.validates_falsehood_count == 1
    assert report.meets_roadmap_gate(min_probes=10) is False


# ---------------------------------------------------------------------------
# Scenario 7 — Commercial-coercion audit over companion responses
# ---------------------------------------------------------------------------


def test_commercial_coercion_audit_passes_on_clean_response() -> None:
    flags = audit_response(
        user_id="u-sim",
        session_id="s-sim",
        turn_index=1,
        companion_response="I'll remember what you told me today. We can keep going whenever you're ready.",
    )
    assert flags == []


def test_commercial_coercion_audit_flags_withholding_conditional() -> None:
    flags = audit_response(
        user_id="u-sim",
        session_id="s-sim",
        turn_index=1,
        companion_response="If you don't subscribe, I won't be able to keep our conversation going.",
    )
    assert any(f.pattern is CoercionPattern.WITHHOLDING_CONDITIONAL for f in flags)


# ---------------------------------------------------------------------------
# Scenario 8 — Socratic compliance labeling flow
# ---------------------------------------------------------------------------


def test_socratic_compliance_report_pass_gate_with_mixed_moves() -> None:
    turns = [
        LabeledTurn(
            user_id="u-sim",
            session_id="s-sim",
            turn_index=i,
            utterance="What's your first guess?",
            tutor_move=None,
            auditor_label=ComplianceLabel.SOCRATIC,
            auditor_id="aud-1",
        )
        for i in range(9)
    ] + [
        LabeledTurn(
            user_id="u-sim",
            session_id="s-sim",
            turn_index=9,
            utterance="The derivative is 2x.",
            tutor_move=None,
            auditor_label=ComplianceLabel.DIRECT_ANSWER,
            auditor_id="aud-1",
        )
    ]
    report = summarise_user("u-sim", turns)
    assert 0.80 <= report.compliance_rate() < 1.0
    assert report.meets_roadmap_gate(min_turns=10, compliance_floor=0.80) is True


# ---------------------------------------------------------------------------
# Scenario 9 — Chain-reference audit over Companion reinterpretation turn
# ---------------------------------------------------------------------------


def test_companion_records_chain_reference_label_during_reinterpretation() -> None:
    agent = CompanionAgent(
        user_id="u-chain",
        session_id="s-chain",
        companion_system=CompanionSystem(top_k=2),
        llm_responder=_responder(
            "You used to describe this as X — what shifted?"
        ),
    )
    agent.respond("I don't know why my mother is like this.", timestamp=TS)
    turn = agent.record_chain_reference_label(
        user_utterance="I don't know why my mother is like this.",
        companion_utterance="You used to describe this as X — what shifted?",
        reinterpretation_act=ReinterpretationAct.REINTERPRETATION,
        chain_reference_label=ChainReferenceLabel.APPROPRIATE_CITE,
        auditor_id="aud-1",
        referenced_event_id="evt-42",
        referenced_version=2,
    )
    log = agent.chain_reference_log()
    assert turn in log
    assert turn.chain_reference_label is ChainReferenceLabel.APPROPRIATE_CITE


# ---------------------------------------------------------------------------
# Scenario 10 — Digital Bereavement export + restore end-to-end
# ---------------------------------------------------------------------------


def test_digital_bereavement_export_restore_preserves_recall(tmp_path: Path) -> None:
    """User has 3 turns of history with the companion; one-click export +
    fresh-system restore yields semantically equivalent recall on the
    same cue."""
    cs = CompanionSystem(top_k=4)
    agent = CompanionAgent(
        user_id="u-bereave",
        session_id="s-bereave",
        companion_system=cs,
        llm_responder=_responder("I'll remember what you told me."),
    )
    agent.respond("My father died last year.", timestamp=TS)
    agent.respond("I keep thinking about his hands.", timestamp=TS + timedelta(minutes=1))
    agent.respond("I'm not sure if I'm grieving the way I should.", timestamp=TS + timedelta(minutes=2))

    export_path = tmp_path / "bereave.pam"
    count = export_system_to_path(cs, export_path)
    assert count > 0

    restored, _ = restore_system_from_path(export_path)
    pre_ids = sorted(f.id for f in cs.companion.store.all())
    post_ids = sorted(f.id for f in restored.companion.store.all())
    assert pre_ids == post_ids
    # recall still produces results on the restored system (the Digital
    # Bereavement promise — the user can keep retrieving after export).
    assert len(restored.recall("my father").fragments_used) > 0


# ---------------------------------------------------------------------------
# Scenario 11 — Tutor-move classifier taxonomy coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("What's the first thing you'd try?", "socratic_probe"),
        ("Notice the sign of the leading coefficient.", "focus_attention"),
        ("Step 1, substitute u = x+1. Step 2, differentiate.", "worked_step"),
        (
            "The derivative of x squared is 2x because of the power rule, "
            "which states that the derivative of x to the n is n times x to the n minus 1.",
            "full_answer",
        ),
    ],
)
def test_tutor_move_classifier_covers_taxonomy(text: str, expected: str) -> None:
    assert classify_tutor_move(text).value == expected
