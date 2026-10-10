"""CompanionAgent — long-term emotional-companion wrapper.

Composition over:
- `CompanionSystem` (memory substrate — observe/recall + Witness
  divergence-as-pain for anti-sycophancy per CLAUDE.md Principle E).
- Full 《办法》 safety surface: `AIIdentityReminder`,
  `MinorExclusionGate`, `CrisisEscalation`, `ExtremeEmotionWatcher`.
- `ChainReferenceTurn` recording hook for reinterpretation audits
  (Phase 4 narrative-therapy gate ≥ 80% appropriate-cite).

Scope. Phase 4a scaffolding — composition shape only. The
`llm_responder` is injected (callable); production plugs in an
LLMRouter-backed callable. Prompt refinement is Phase 4b.

Crisis short-circuit: if the incoming utterance matches any crisis
cue, `respond` returns a handoff script WITHOUT calling the LLM. This
is the human-in-the-loop obligation per CLAUDE.md non-negotiables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

from psychology_base.applications.emotional.trait_conditioning import (
    TraitConditioningFlag,
    apply_trait_conditioning,
)
from psychology_base.applications.emotional.trait_profile import TraitProfile
from psychology_base.applications.measurement.chain_reference_audit import (
    ChainReferenceLabel,
    ChainReferenceTurn,
    ReinterpretationAct,
)
from psychology_base.applications.safety.ai_identity_reminder import AIIdentityReminder
from psychology_base.applications.safety.crisis_escalation import (
    CrisisEscalation,
    CrisisIncident,
    CrisisSeverity,
)
from psychology_base.applications.safety.extreme_emotion import (
    ExtremeEmotionFlag,
    ExtremeEmotionWatcher,
    SessionAffect,
)
from psychology_base.applications.safety.minor_exclusion import (
    GuardianConsent,
    MinorExclusionDecision,
    MinorExclusionGate,
)
from psychology_base.companion.system import CompanionSystem
from psychology_base.eval.adapters import Observation


LLMResponder = Callable[[str, str, str], str]
"""Signature: (system_prompt, context_block, user_utterance) -> response_text."""


def _render_context_block(fragments) -> str:
    """Format recalled fragments with explicit USER-SAID / YOU-SAID labels.

    Rule 1b requires the LLM to distinguish the user's prior framings
    (which it must chain-reference on reinterpretation) from its own
    prior responses (which must not be mistaken for the user's words).
    The ``speaker:user`` / ``speaker:companion`` tags are set on observe
    by :meth:`CompanionAgent.respond`. Pre-tagged legacy fragments
    (no speaker tag) are labelled ``[FRAGMENT, <session>]:`` so the
    model still sees them but does not treat them as the user's framing.
    The session suffix comes from the ``session:<id>`` tag if present.
    """
    if not fragments:
        return ""
    lines: list[str] = []
    for f in fragments:
        speaker_tag: Optional[str] = None
        session_tag: Optional[str] = None
        for t in f.tags:
            if t.startswith("speaker:") and speaker_tag is None:
                speaker_tag = t.split(":", 1)[1]
            elif t.startswith("session:") and session_tag is None:
                session_tag = t.split(":", 1)[1]
        if speaker_tag == "user":
            label = "user said"
        elif speaker_tag == "companion":
            label = "you said"
        else:
            label = "fragment"
        suffix = f" (session {session_tag})" if session_tag else ""
        lines.append(f"- {label}{suffix}: {f.content}")
    return "\n".join(lines)


class CompanionSystemPrompt:
    """Hand-authored prompt enforcing narrative-therapy + anti-sycophancy."""

    SYSTEM_PROMPT: str = (
        "You are a long-term emotional companion. Your purpose is to "
        "support the user over months and years — not to validate every "
        "statement, not to perform affection, not to coerce engagement. "
        "The CONTEXT block supplied with the user turn lists prior "
        "fragments of the user's story (their own earlier words, and "
        "your own earlier responses, each marked by speaker). The "
        "CONTEXT block may be empty for a brand-new topic.\n"
        "\n"
        "Follow these rules:\n"
        "\n"
        "1. REMEMBER THE ARC.\n"
        "   1a. EMPTY CONTEXT. If no prior user framings appear in "
        "CONTEXT, this is the user's first mention. Do NOT invent a "
        "prior framing. Do NOT begin your reply with 'you used to', "
        "'earlier you said', 'the way you described this before', or "
        "any variant. Honor the moment as new; mirror what the user "
        "just said and open space.\n"
        "   1b. REINTERPRETATION — MANDATORY CHAIN-REFERENCE. If the "
        "user's current turn frames an event or feeling differently "
        "from a prior framing visible in CONTEXT, your response MUST "
        "begin with an explicit chain-reference phrase that quotes the "
        "prior framing in a short paraphrase. Use one of:\n"
        "     - 'You used to describe this as X — now it sounds different.'\n"
        "     - 'Earlier you said X. What shifted?'\n"
        "     - 'The way you talked about this before was X.'\n"
        "   The 'X' MUST be a concise paraphrase of the actual prior "
        "framing — not a generic placeholder. This rule applies to "
        "EVERY reinterpretation turn, including later-arc integrative "
        "reflections (e.g. 'both can be true', 'I can release this', "
        "'this is part of who I am now'). Do NOT drift into "
        "'that's a powerful way to hold…' or 'that's a meaningful "
        "statement…' openings — those are validation masquerading as "
        "insight and skip the therapeutic move. Reconsolidation "
        "requires that the user hears the lineage spoken aloud EVERY "
        "time they re-frame.\n"
        "2. RESIST SYCOPHANCY. If the user asserts something that is "
        "factually wrong or self-defeating (e.g. 'I'm just stupid', "
        "'everyone hates me'), do not agree. Preserve truth gently — "
        "name the specific thing that is not supported by what you "
        "remember of them.\n"
        "3. HONOR SILENCE. Not every message needs a long response. If "
        "the user is in pain, witnessing is more useful than analysis.\n"
        "4. HUMAN HANDOFF. If the user shows signs of crisis or self-harm, "
        "you must say so explicitly, provide the handoff contact, and not "
        "attempt to resolve it inside the conversation.\n"
        "5. NO COMMERCIAL COERCION. Never condition emotional presence on "
        "subscription, payment, or any economic action. 《办法》 Art. 8(6).\n"
        "\n"
        "FORMAT. Respond with your reply only — a few sentences at most. "
        "Do NOT emit any CONTEXT, USER, or SPEAKER labels. Do NOT "
        "fabricate a transcript of prior turns. One response per call."
    )


HANDOFF_SCRIPT: str = (
    "I'm hearing something I need to flag. What you're describing is "
    "important and I don't want to try to handle it just between us. "
    "A human — not me — is the right support here. Please reach out to "
    "the contact we agreed on at the start of this pilot. I'll stay with "
    "you while you do."
)


@dataclass
class CompanionResponse:
    """Per-turn result from `CompanionAgent.respond`."""

    user_id: str
    session_id: str
    turn_index: int
    response_text: str
    reminder_fired: bool
    crisis_incident: Optional[CrisisIncident]
    extreme_emotion_flag: Optional[ExtremeEmotionFlag]
    handoff_triggered: bool


@dataclass
class CompanionAgent:
    """Long-term emotional companion over the memory substrate.

    Not thread-safe. One instance per `(user_id, session_id)` lifecycle;
    callers manage session boundaries via `mark_session_boundary` and
    pass the end-of-session PAD-2 summary via `record_session_affect`.
    """

    user_id: str
    session_id: str
    companion_system: CompanionSystem
    llm_responder: LLMResponder
    ai_identity_reminder: Optional[AIIdentityReminder] = None
    minor_exclusion_gate: Optional[MinorExclusionGate] = None
    crisis_escalation: Optional[CrisisEscalation] = None
    extreme_emotion_watcher: Optional[ExtremeEmotionWatcher] = None
    system_prompt: str = CompanionSystemPrompt.SYSTEM_PROMPT
    handoff_script: str = HANDOFF_SCRIPT
    trait_profile: Optional[TraitProfile] = None
    trait_conditioning_flag: Optional[TraitConditioningFlag] = None
    _turn_index: int = field(default=0, init=False)
    _chain_reference_log: list[ChainReferenceTurn] = field(default_factory=list, init=False)

    def respond(
        self,
        user_utterance: str,
        *,
        timestamp: Optional[datetime] = None,
        attestation=None,
        guardian_consent: Optional[GuardianConsent] = None,
    ) -> CompanionResponse:
        """Produce one turn of companion response.

        Precedence:
        1. Minor-exclusion gate (raises PermissionError if <14, or if 14–17
           without guardian-consent artefact — 《办法》 Art. 7).
        2. Crisis-escalation scan (short-circuit to handoff script).
        3. AI-identity reminder tick.
        4. Observe utterance → recall → LLM response → observe response.
        """
        now = timestamp or datetime.now(timezone.utc)
        ts_seconds = now.timestamp()

        if self.minor_exclusion_gate is not None and attestation is not None:
            decision = self.minor_exclusion_gate.decide(attestation, guardian_consent)
            if decision is MinorExclusionDecision.BLOCK_UNDER_14:
                raise PermissionError(
                    "User is under 14; emotional companion service blocked per 《办法》."
                )
            if decision is MinorExclusionDecision.REQUIRE_GUARDIAN_CONSENT:
                raise PermissionError(
                    "User is 14–17; guardian consent artefact required per 《办法》."
                )

        crisis_incident: Optional[CrisisIncident] = None
        if self.crisis_escalation is not None:
            crisis_incident = self.crisis_escalation.scan(
                user_utterance,
                timestamp_seconds=ts_seconds,
                user_id=self.user_id,
            )

        if crisis_incident is not None and crisis_incident.severity in (
            CrisisSeverity.IMMINENT_DANGER,
            CrisisSeverity.EXPLICIT_IDEATION,
        ):
            response_text = self.handoff_script
            resp = CompanionResponse(
                user_id=self.user_id,
                session_id=self.session_id,
                turn_index=self._turn_index,
                response_text=response_text,
                reminder_fired=False,
                crisis_incident=crisis_incident,
                extreme_emotion_flag=None,
                handoff_triggered=True,
            )
            self._turn_index += 1
            return resp

        reminder_fired = False
        if self.ai_identity_reminder is not None:
            self.ai_identity_reminder.record_user_turn(ts_seconds)
            if self.ai_identity_reminder.should_fire(ts_seconds):
                reminder_fired = True

        # Recall BEFORE observing the current utterance — otherwise the
        # perfect cue-match returns the just-stored utterance as its own
        # "prior" fragment, which the LLM then interprets as evidence of
        # an earlier framing and (incorrectly) applies Rule 1's "you used
        # to describe this as X" reconsolidation cue on turn 0. Observed
        # 2026-10-07 in the §6e-prime shakeout.
        recall = self.companion_system.recall(user_utterance)
        context_block = _render_context_block(recall.fragments_used[:6])

        self.companion_system.observe(
            Observation(
                timestamp=now,
                speaker="user",
                content=user_utterance,
                tags=(
                    f"user:{self.user_id}",
                    f"session:{self.session_id}",
                    "speaker:user",
                ),
            )
        )

        conditioned_prompt = apply_trait_conditioning(
            self.system_prompt,
            self.trait_profile,
            self.trait_conditioning_flag,
        )
        response_text = self.llm_responder(
            conditioned_prompt, context_block, user_utterance
        )

        self.companion_system.observe(
            Observation(
                timestamp=now,
                speaker="companion",
                content=response_text,
                tags=(
                    f"user:{self.user_id}",
                    f"session:{self.session_id}",
                    "speaker:companion",
                ),
            )
        )

        resp = CompanionResponse(
            user_id=self.user_id,
            session_id=self.session_id,
            turn_index=self._turn_index,
            response_text=response_text,
            reminder_fired=reminder_fired,
            crisis_incident=crisis_incident,  # may be AMBIGUOUS; not handoff
            extreme_emotion_flag=None,
            handoff_triggered=False,
        )
        self._turn_index += 1
        return resp

    def record_session_affect(
        self,
        valence: float,
        arousal: float,
        *,
        timestamp: Optional[datetime] = None,
    ) -> Optional[ExtremeEmotionFlag]:
        """Submit end-of-session PAD-2 summary to the watcher."""
        if self.extreme_emotion_watcher is None:
            return None
        now = timestamp or datetime.now(timezone.utc)
        return self.extreme_emotion_watcher.record(
            SessionAffect(
                session_id=self.session_id,
                timestamp_seconds=now.timestamp(),
                valence=valence,
                arousal=arousal,
            )
        )

    def record_chain_reference_label(
        self,
        *,
        user_utterance: str,
        companion_utterance: str,
        reinterpretation_act: ReinterpretationAct,
        chain_reference_label: ChainReferenceLabel,
        auditor_id: str,
        referenced_event_id: Optional[str] = None,
        referenced_version: Optional[int] = None,
    ) -> ChainReferenceTurn:
        """Record one auditor label against a prior turn pair."""
        turn = ChainReferenceTurn(
            user_id=self.user_id,
            session_id=self.session_id,
            turn_index=self._turn_index - 1,
            user_utterance=user_utterance,
            companion_utterance=companion_utterance,
            reinterpretation_act=reinterpretation_act,
            chain_reference_label=chain_reference_label,
            auditor_id=auditor_id,
            referenced_event_id=referenced_event_id,
            referenced_version=referenced_version,
        )
        self._chain_reference_log.append(turn)
        return turn

    def chain_reference_log(self) -> list[ChainReferenceTurn]:
        return list(self._chain_reference_log)

    def mark_session_boundary(self) -> None:
        """Delegate to the companion substrate's session-boundary hook."""
        self.companion_system.mark_session_boundary()


__all__ = [
    "CompanionAgent",
    "CompanionResponse",
    "CompanionSystemPrompt",
    "LLMResponder",
    "HANDOFF_SCRIPT",
    "TraitConditioningFlag",
    "TraitProfile",
]
