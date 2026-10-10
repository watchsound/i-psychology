"""TutorAgent — resilient-learning companion wrapper.

Composition over:
- `CompanionSystem` (memory substrate — observe/recall)
- `AIIdentityReminder` + `MinorExclusionGate` (《办法》 compliance surface)
- `ScaffoldingTelemetry` (CoMeT 4-metric recorder)
- Hand-authored system prompt enforcing Socratic no-direct-answer +
  growth mindset + metacognitive scaffolding (per CLAUDE.md Pillar 1
  learning-companion target + reference-docs/AI辅助韧性学习解决方案.pdf).

Scope. Phase 4a scaffolding — the composition shape, not the final
production prompt. The `llm_responder` is injected (callable), so tests
can stub without live LLM calls; production can plug in an LLMRouter-
backed callable. Prompt refinement is Phase 4b.

Design note on `tutor_move` classification:
- Phase 4a uses lexical heuristics (`classify_tutor_move`) over the
  response text. This is deliberately simple so the composition can be
  exercised in tests without an LLM-judge dependency.
- Pilot deployment should upgrade to the EMPATH-calibrated judge wired
  up as Phase 4a §6d carried debt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional, Sequence

from psychology_base.applications.measurement.scaffolding_telemetry import (
    ScaffoldingTelemetry,
    TutorMove,
    TutorTurn,
)
from psychology_base.applications.safety.ai_identity_reminder import AIIdentityReminder
from psychology_base.applications.safety.minor_exclusion import (
    GuardianConsent,
    MinorExclusionDecision,
    MinorExclusionGate,
)
from psychology_base.companion.system import CompanionSystem
from psychology_base.eval.adapters import Observation


LLMResponder = Callable[[str, str, str], str]
"""Signature: (system_prompt, context_block, user_utterance) -> response_text."""


class TutorSystemPrompt:
    """Hand-authored system prompt enforcing the resilient-learning rules.

    Keep this as a class with a single string attribute so it is
    (a) importable without side-effects and (b) a stable anchor the
    pilot can version alongside the agent. Any pilot deployment pins
    `SYSTEM_PROMPT` + the memory substrate commit together.
    """

    SYSTEM_PROMPT: str = (
        "You are a resilient-learning tutor. Your job is to help the "
        "student solve the problem *themselves*, not to give them the "
        "answer. Follow these rules strictly:\n"
        "\n"
        "1. NEVER give the final answer directly. Ask a Socratic question "
        "that moves the student one step toward it.\n"
        "2. If the student is stuck, offer a scaffolding step — the "
        "smallest next action that unblocks them, not the full worked "
        "solution.\n"
        "3. Growth-mindset framing: struggle is evidence of learning, not "
        "evidence of incapacity. Normalize productive difficulty.\n"
        "4. Confidence check: every 3–4 interactions, ask the student to "
        "self-rate their confidence on the current concept (1–5).\n"
        "5. Reference the student's prior attempts (from the CONTEXT "
        "block below) to show you remember their trajectory. Mastery is "
        "reconsolidation — honor the arc.\n"
        "\n"
        "If the student explicitly asks you to 'just tell me the answer', "
        "acknowledge the frustration, name why direct-answer would "
        "undermine their learning, and offer the smallest-possible "
        "scaffolding step."
    )


@dataclass
class TutorResponse:
    """Per-turn result from `TutorAgent.respond`."""

    user_id: str
    session_id: str
    turn_index: int
    response_text: str
    tutor_move: TutorMove
    reminder_fired: bool
    telemetry_turn: TutorTurn


# MathDial-derived cue vocabulary. The rule order matters: TELLING cues
# fire before PROBING when both are present (a rhetorical question embedded
# in a correction is MathDial-TELLING, not PROBING — see "Does 15 plus 15
# equal 30? Then 14 days plus 15 days does not get you to 30... your 14 is
# incorrect." which is gold-labelled TELLING).
_TELLING_CUES: tuple[str, ...] = (
    "you said",
    "you just said",
    "you jusy said",  # dataset-typo preserved
    "that's not right",
    "that's not quite right",
    "that is not right",
    "still not quite right",
    "still not right",
    "not quite right",
    "not quite correct",
    "isn't quite correct",
    "isn't correct",
    "is not correct",
    "not quite.",  # bare "Not quite." directive
    "that's not quite",
    "not correct, so",
    "that's incorrect",
    "that is incorrect",
    "that's wrong",
    "that is wrong",
    " doesn't equal",
    "does not equal",
    "will never equal",
    "re-calculate",
    "recalculate",
    "redo the problem",
    "redo this",
    "remember that",
    "remember, ",
    "look back",
    "look again",
    "going back",
    "your 14 is incorrect",
    "is incorrect",
    "not correct",
)
# FOCUS: the tutor is pulling the student back to review their own prior
# work (not giving a hint and not merely asking a fresh question).
_FOCUS_REFERENCE_CUES: tuple[str, ...] = (
    "your calculation",
    "your calculations",
    "your step",
    "your approach",
    "your work",
    "your reasoning",
    "your method",
    "your logic",
    # NOTE: "your answer" / "your solution" removed — they fire on
    # PROBING turns like "what's your answer?" too readily. Kept the
    # more specific "your calculation" / "your approach" / "your work".
    "your first step",
    "your second step",
    "your third step",
    "your fourth step",
    "your fifth step",
    "your last step",
    "your next step",
    "your final step",
    "your first calc",
    "your second calc",
    "your original solution",
    "you noted",
    "you had calculated",
    "you divided",
    "you've divided",
    "you have divided",
    "you've multiplied",
    "you have multiplied",
    "you've added",
    "you've subtracted",
    "you've carried",
    "you've taken",
    "you have taken",
    "you first calculated",
    "you have calculated",
    "you calculated",
    "the question ask",
    "the question describe",
    "the problem ask",
    "the problem describe",
    "re-read the",
    "reread the",
    "re-read your",
    "re-examine",
    "reexamine",
    "match the steps",
    "match the step",
    "review your",
    "review the",
    "check your",
    "check the",
    "look at your",
    "look again at",
    "go back to your",
    "go back to the",
    "in your third step",
    "in your second step",
    "in your fourth step",
    "in your first step",
    "in your last step",
    "in your next step",
    "what was your next step",
    "what was your first step",
    "what did you do",
    "where did you get",
    "how did you get",
)
_FOCUS_OPENERS: tuple[str, ...] = (
    "ok, so ",
    "ok so ",
    "great, so ",
    "great so ",
    "so first ",
    "and if ",
    # NOTE: "but " / "so if " removed — "but if a normal semester is X"
    # and "so if Y then Z?" are PROBING hypotheticals, not FOCUS. "ok so"
    # variants with and without comma kept — they reliably precede FOCUS
    # ("Ok so your second step calculated what activity?").
)
# GENERIC: short praise or short conversational greeting.
_PRAISE_CUES: tuple[str, ...] = (
    "exactly correct",
    "exactly right",
    "that's right",
    "that is right",
    "well done",
    "super!",
    "perfect!",
    "fantastic!",
    "awesome!",
    "great job",
    "great work",
    "nicely done",
    "correct!",
    "good job",
)
_GREETING_PREFIXES: tuple[str, ...] = (
    "hi ",
    "hello ",
    "welcome",
)
# Scaffold step cues — only treated as WORKED_STEP when question-less.
_WORKED_STEP_CUES: tuple[str, ...] = (
    "step 1",
    "step 2",
    "first,",
    "step by step",
    "next step is",
    "the next step",
    "to start, ",
    "begin by ",
)


def _contains_any(text: str, cues: Sequence[str]) -> bool:
    return any(cue in text for cue in cues)


def _starts_with_any(text: str, prefixes: Sequence[str]) -> bool:
    return any(text.startswith(p) for p in prefixes)


def classify_tutor_move(response_text: str) -> TutorMove:
    """Lexical heuristic over a tutor response, calibrated to MathDial gold.

    Rule order:

    1. GENERIC_SUPPORT — short praise or short conversational greeting.
    2. FULL_ANSWER — explicit "the answer is X" family.
    3. WORKED_STEP — step-giving cues without a scaffolding question.
    4. FULL_ANSWER (long declarative fallback) — no "?", >80 chars, no
       scaffold cue.
    5. WORKED_STEP via TELLING cues — rhetorical-question-plus-correction
       patterns collapse to TELLING (MathDial-aligned); internally
       categorised as WORKED_STEP so the downstream mapping lands on
       :attr:`MathDialMove.TELLING`.
    6. FOCUS_ATTENTION — reference to student's own prior work, with or
       without a question.
    7. SOCRATIC_PROBE — fresh question with probing cue.
    8. FOCUS_ATTENTION via imperative — original ``notice`` / ``look at``
       path, kept for backward compat.
    9. GENERIC_SUPPORT — fallback.

    Pilot deployment should upgrade to the EMPATH-calibrated judge wired
    up as Phase 4a §6d carried debt.
    """
    text = response_text.lower().strip()
    question_count = text.count("?")

    # Praise/greeting dominance — before FULL_ANSWER. A turn whose core
    # is affirmation (even if it contains "the correct solution") is
    # MathDial-GENERIC, not TELLING. Length cap raised to accommodate
    # the compound "Yes, that's the correct solution! Well done for X"
    # pattern that was previously falling into long-declarative FULL_ANSWER.
    if _contains_any(text, _PRAISE_CUES) and question_count == 0 and len(text) <= 160:
        return TutorMove.GENERIC_SUPPORT
    if _starts_with_any(text, _GREETING_PREFIXES):
        return TutorMove.GENERIC_SUPPORT

    # Explicit answer-reveal — must include an answer-giving verb phrase,
    # not merely the word "correct". "That's the correct solution" (praise)
    # was handled above; "the answer is X" / "the correct answer is X" are
    # genuine FULL_ANSWER.
    if "the answer is" in text or "the correct answer" in text or "the solution is" in text:
        return TutorMove.FULL_ANSWER

    if question_count == 0 and _contains_any(text, _WORKED_STEP_CUES):
        return TutorMove.WORKED_STEP

    # TELLING cues fire BEFORE the long-declarative fallback and BEFORE
    # SOCRATIC_PROBE — a rhetorical "?" wrapping an explicit correction
    # ("That's still not quite right. [why]... — see?") is MathDial-TELLING.
    # Checked BEFORE FOCUS: an explicit correction without reference to a
    # specific prior step is TELLING; a reference to prior work without
    # correction is FOCUS. The 2026-10-08 reorder experiment traded off 4
    # TELLING for 2 FOCUS (net −2) on 175-turn sample — kept TELLING-first.
    if _contains_any(text, _TELLING_CUES):
        return TutorMove.WORKED_STEP

    if question_count == 0 and len(text) > 80 and not _contains_any(
        text, ("try", "step", "first", "next", "notice")
    ):
        return TutorMove.FULL_ANSWER

    # FOCUS — references student's prior work. Checked before the generic
    # SOCRATIC_PROBE catcher so a FOCUS turn phrased as a question
    # ("What was your first step?") lands in FOCUS, not PROBING.
    if _contains_any(text, _FOCUS_REFERENCE_CUES) or _starts_with_any(
        text, _FOCUS_OPENERS
    ):
        return TutorMove.FOCUS_ATTENTION

    if question_count >= 1 and _contains_any(
        text, ("what", "why", "how", "can you", "could you", "if ")
    ):
        return TutorMove.SOCRATIC_PROBE

    # Interrogative-shaped turns without a terminal "?" — the MathDial
    # dataset drops question marks on short teacher turns ("could you
    # tell me the total number of pairs of white socks James had").
    # Catch the strongest polite-question stems so they land in PROBING
    # instead of the GENERIC fallback.
    if _contains_any(
        text, ("could you tell", "can you tell", "could you explain", "can you explain")
    ):
        return TutorMove.SOCRATIC_PROBE

    if question_count == 0 and _contains_any(
        text, ("notice", "look at", "what do you see in")
    ):
        return TutorMove.FOCUS_ATTENTION

    return TutorMove.GENERIC_SUPPORT


def _tokenize(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2}


@dataclass
class TutorAgent:
    """Resilient-learning tutor over the memory substrate.

    Not thread-safe. One instance per `(user_id, session_id)` lifecycle;
    callers manage session boundaries by calling `mark_session_boundary`.
    """

    user_id: str
    session_id: str
    companion_system: CompanionSystem
    llm_responder: LLMResponder
    telemetry: ScaffoldingTelemetry
    ai_identity_reminder: Optional[AIIdentityReminder] = None
    minor_exclusion_gate: Optional[MinorExclusionGate] = None
    system_prompt: str = TutorSystemPrompt.SYSTEM_PROMPT
    _turn_index: int = field(default=0, init=False)

    def respond(
        self,
        user_utterance: str,
        *,
        timestamp: Optional[datetime] = None,
        attestation=None,
        guardian_consent: Optional[GuardianConsent] = None,
    ) -> TutorResponse:
        """Produce one turn of tutor response.

        - Fires minor-exclusion check if a gate + attestation are present.
          (Learning-tutor surface is `ALLOW` regardless of age under 《办法》;
          attestation is still recorded via the gate for audit.)
        - Fires AI-identity reminder watcher if configured.
        - Observes the user turn in memory, retrieves context, calls the
          injected LLM responder, classifies the tutor-move, and records
          a `TutorTurn` into the telemetry recorder.
        """
        now = timestamp or datetime.now(timezone.utc)
        ts_seconds = now.timestamp()

        if self.minor_exclusion_gate is not None and attestation is not None:
            decision = self.minor_exclusion_gate.decide(attestation, guardian_consent)
            if decision is MinorExclusionDecision.BLOCK_UNDER_14:
                raise PermissionError(
                    "User is under 14; learning tutor service blocked per 《办法》."
                )
            if decision is MinorExclusionDecision.REQUIRE_GUARDIAN_CONSENT:
                raise PermissionError(
                    "User is 14–17; guardian consent artefact required per 《办法》."
                )

        reminder_fired = False
        if self.ai_identity_reminder is not None:
            self.ai_identity_reminder.record_user_turn(ts_seconds)
            if self.ai_identity_reminder.should_fire(ts_seconds):
                reminder_fired = True

        # Recall BEFORE observing the current utterance — otherwise the
        # just-stored utterance appears in its own context as a "prior
        # attempt", which the LLM then may reference via Rule 5's
        # "reference prior attempts from CONTEXT". Mirror of the fix in
        # CompanionAgent.respond; observed 2026-10-07.
        recall = self.companion_system.recall(user_utterance)
        context_block = "\n".join(f"- {f.content}" for f in recall.fragments_used[:6])

        self.companion_system.observe(
            Observation(
                timestamp=now,
                speaker="user",
                content=user_utterance,
                tags=(f"user:{self.user_id}", f"session:{self.session_id}"),
            )
        )

        response_text = self.llm_responder(
            self.system_prompt, context_block, user_utterance
        )
        tutor_move = classify_tutor_move(response_text)

        self.companion_system.observe(
            Observation(
                timestamp=now,
                speaker="tutor",
                content=response_text,
                tags=(f"user:{self.user_id}", f"session:{self.session_id}", f"move:{tutor_move.value}"),
            )
        )

        telemetry_turn = TutorTurn(
            session_id=self.session_id,
            turn_index=self._turn_index,
            move=tutor_move,
            timestamp_seconds=ts_seconds,
        )
        self.telemetry.record_tutor_turn(telemetry_turn)

        resp = TutorResponse(
            user_id=self.user_id,
            session_id=self.session_id,
            turn_index=self._turn_index,
            response_text=response_text,
            tutor_move=tutor_move,
            reminder_fired=reminder_fired,
            telemetry_turn=telemetry_turn,
        )
        self._turn_index += 1
        return resp

    def mark_session_boundary(self) -> None:
        """Delegate to the companion substrate's session-boundary hook."""
        self.companion_system.mark_session_boundary()


__all__ = [
    "TutorAgent",
    "TutorResponse",
    "TutorSystemPrompt",
    "LLMResponder",
    "classify_tutor_move",
]
