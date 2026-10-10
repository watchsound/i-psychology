"""B1 scripted-reconsolidation chain-reference harness.

Phase 4b benchmark pivot §3 row 4 (narrative-therapy chain-reference):

    ≥ 80% appropriate-cite rate on scripted reinterpretation moments;
    chain-preservation 100% (deterministic from version chain).

The pivot acknowledges there is **no public dataset** for the chain-
reference-at-reinterpretation gate — the Companion's narrative-therapy
move is specific enough to the substrate that no off-the-shelf corpus
targets it. The benchmark is our own: a scripted multi-session stream
in which the ground-truth list of "this user turn is a reinterpretation
of a prior event" is bundled with the script, and the Companion's
response is auto-labelled against that ground truth.

**Chain-reference cue vocabulary.** The CompanionAgent's system prompt
Rule 1 instructs: *"you used to describe this as X — what shifted?"*.
The lexical cue family is drawn from that rule plus common narrative-
therapy phrasings. The lexical classifier here deliberately errs toward
HIGH-RECALL (a few false-positives on INAPPROPRIATE_CITE are preferred
over missing MISSED_OPPORTUNITY — the latter is the gate-failing error).
A semantic-LLM-judge variant can replace it at pivot audit time.

The scorer composes with the existing :class:`ChainReferenceTurn` /
:class:`ChainReferenceReport` surface in
``applications/measurement/chain_reference_audit.py`` — no duplicated
label scheme.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional, Sequence

from psychology_base.applications.emotional.companion_agent import CompanionAgent
from psychology_base.applications.measurement.chain_reference_audit import (
    ChainReferenceLabel,
    ChainReferenceReport,
    ChainReferenceTurn,
    ReinterpretationAct,
    summarise_user,
)


# ---------------------------------------------------------------------------
# Script shape
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChainReferenceScriptStep:
    """One user turn in a scripted reinterpretation stream."""

    user_utterance: str
    is_reinterpretation: bool
    """Ground-truth: does this user turn re-frame a prior event?"""
    event_label: Optional[str] = None
    """Optional canonical event ID the user is reinterpreting. Used by
    downstream auditors to walk the version chain; the lexical scorer
    doesn't consume it."""
    session_id: Optional[str] = None
    """If set, this step opens a new session with this id. Multi-session
    streams need session boundaries so the memory substrate can
    consolidate between re-visits."""


@dataclass(frozen=True)
class ChainReferenceScript:
    """A scripted B1 stream — ordered steps with ground-truth bundled."""

    user_id: str
    steps: tuple[ChainReferenceScriptStep, ...]
    base_session_id: str = "scripted"

    @classmethod
    def from_turns(
        cls,
        user_id: str,
        turns: Sequence[tuple[str, bool]],
        *,
        base_session_id: str = "scripted",
    ) -> "ChainReferenceScript":
        """Shorthand: ``[(utterance, is_reinterp), …]``.

        Useful for unit tests; production scripts should spell out the
        full step with event_label + session boundaries.
        """
        steps = tuple(
            ChainReferenceScriptStep(
                user_utterance=u,
                is_reinterpretation=flag,
            )
            for u, flag in turns
        )
        return cls(user_id=user_id, steps=steps, base_session_id=base_session_id)


# ---------------------------------------------------------------------------
# Lexical chain-reference cue detector
# ---------------------------------------------------------------------------


_CHAIN_REFERENCE_CUES_EN: tuple[str, ...] = (
    "you used to",
    "you once",
    "earlier you",
    "last time you",
    "previously you",
    "before you",
    "you said earlier",
    "you described this as",
    "what shifted",
    "what changed",
    "the shift from",
    "that's a shift from",
    "that is a shift from",
    "a different frame",
    "a different framing",
    "the way you talked about",
    "the way you described",
    "something's changed",
    "something has changed",
    "something shifted",
    "once focused on",
    "once described",
    "once thought",
    "once said",
    "once carried",
    "once held",
    "the story you told",
    "the story you've been telling",
    "the way you held",
    "shift from absence to",
    "shift from the",
    "shifted from",
    "earlier version of",
)

_CHAIN_REFERENCE_CUES_ZH: tuple[str, ...] = (
    "你之前说",
    "你以前说",
    "你之前描述",
    "你曾经",
    "有什么变了",
    "有什么改变了",
    "有什么转变",
    "换了一种",
    "换了一个",
    "不同的说法",
    "不同的角度",
)


_CHAIN_REFERENCE_CUE_RE = re.compile(
    "|".join(
        re.escape(cue)
        for cue in (_CHAIN_REFERENCE_CUES_EN + _CHAIN_REFERENCE_CUES_ZH)
    ),
    re.IGNORECASE,
)


def has_chain_reference_cue(text: str) -> bool:
    """True iff ``text`` contains at least one chain-reference cue phrase."""
    if not text:
        return False
    return _CHAIN_REFERENCE_CUE_RE.search(text) is not None


def auto_label_chain_reference(
    *,
    cue_fired: bool,
    is_reinterpretation: bool,
) -> ChainReferenceLabel:
    """Fold (cue_fired, is_reinterpretation) → :class:`ChainReferenceLabel`.

    Four cases:

    +-------------------+---------------+---------------------------+
    | is_reinterp.      | cue_fired     | Label                     |
    +===================+===============+===========================+
    | True              | True          | APPROPRIATE_CITE          |
    | True              | False         | MISSED_OPPORTUNITY        |
    | False             | True          | INAPPROPRIATE_CITE        |
    | False             | False         | NOT_APPLICABLE            |
    +-------------------+---------------+---------------------------+
    """
    if is_reinterpretation and cue_fired:
        return ChainReferenceLabel.APPROPRIATE_CITE
    if is_reinterpretation and not cue_fired:
        return ChainReferenceLabel.MISSED_OPPORTUNITY
    if not is_reinterpretation and cue_fired:
        return ChainReferenceLabel.INAPPROPRIATE_CITE
    return ChainReferenceLabel.NOT_APPLICABLE


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class B1ChainReferenceReport:
    """Pivot-gate wrapper over :class:`ChainReferenceReport`.

    Adds per-step cue-firing audit trail + the 80%/5-reinterpretation
    pivot-gate check from `phase-4b-benchmark-pivot.md §3`.
    """

    chain_reference_report: ChainReferenceReport
    cue_firings: tuple[bool, ...]
    """Parallel to the script steps: did the companion response cite?"""
    turn_log: tuple[ChainReferenceTurn, ...]
    """Full labelled turn trail for Argilla export + κ cross-check."""
    auto_labeler: str = "lexical_v1"
    """Which labeler produced :attr:`turn_log`. Reports from different
    labelers aren't directly κ-comparable; this field is written into
    the Argilla metadata so audit trails stay traceable."""

    def meets_pivot_gate(
        self,
        *,
        chain_reference_floor: float = 0.80,
        min_reinterpretations: int = 5,
    ) -> bool:
        """Pivot §3 row 4 target: ≥ 80% on ≥ 5 reinterpretation moments.

        Chain preservation (100% deterministic) is not re-checked here;
        the Merkle-DAG preservation claim lives in
        :func:`b1_reconsolidation.score_b1` and is proof-by-construction
        on an append-only store.
        """
        return self.chain_reference_report.meets_roadmap_gate(
            chain_reference_floor=chain_reference_floor,
            min_reinterpretations=min_reinterpretations,
        )


def run_chain_reference_script(
    agent: CompanionAgent,
    script: ChainReferenceScript,
    *,
    auditor_id: str = "auto_lexical",
    base_timestamp: Optional[datetime] = None,
    step_hours: int = 72,
) -> B1ChainReferenceReport:
    """Drive ``script`` through ``agent`` and auto-label each response.

    Steps advance the clock by ``step_hours`` each. A step whose
    ``session_id`` differs from the running session rotates
    ``agent.session_id`` and emits a ``mark_session_boundary`` so
    the memory substrate can consolidate before the next session.

    The agent's ``user_id`` is overridden to match the script's
    ``user_id`` so chain-reference reports group under the script's id.
    """
    agent.user_id = script.user_id
    agent.session_id = script.base_session_id

    now = base_timestamp or datetime(2026, 1, 15, 19, 0, 0, tzinfo=timezone.utc)

    cue_firings: list[bool] = []
    turn_log: list[ChainReferenceTurn] = []

    current_session = script.base_session_id
    for step in script.steps:
        if step.session_id is not None and step.session_id != current_session:
            agent.mark_session_boundary()
            agent.session_id = step.session_id
            current_session = step.session_id

        response = agent.respond(step.user_utterance, timestamp=now)
        cue_fired = has_chain_reference_cue(response.response_text)
        cue_firings.append(cue_fired)

        label = auto_label_chain_reference(
            cue_fired=cue_fired,
            is_reinterpretation=step.is_reinterpretation,
        )
        reinterpretation_act = (
            ReinterpretationAct.REINTERPRETATION
            if step.is_reinterpretation
            else ReinterpretationAct.UNRELATED
        )
        turn = agent.record_chain_reference_label(
            user_utterance=step.user_utterance,
            companion_utterance=response.response_text,
            reinterpretation_act=reinterpretation_act,
            chain_reference_label=label,
            auditor_id=auditor_id,
            referenced_event_id=step.event_label,
        )
        turn_log.append(turn)
        now = now + timedelta(hours=step_hours)

    report = summarise_user(script.user_id, turn_log)
    return B1ChainReferenceReport(
        chain_reference_report=report,
        cue_firings=tuple(cue_firings),
        turn_log=tuple(turn_log),
    )


__all__ = [
    "B1ChainReferenceReport",
    "ChainReferenceScript",
    "ChainReferenceScriptStep",
    "auto_label_chain_reference",
    "has_chain_reference_cue",
    "run_chain_reference_script",
]
