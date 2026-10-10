"""ESConv × WAI-SR Bond gate harness — simulated-user alliance measurement.

Targets the pivot gate `docs/phases/phase-4b-benchmark-pivot.md` §3 row
"情感陪伴 WAI-SR Bond" — mean WAI-SR Bond ≥ 3.8 across ≥50 simulated users
spanning ESConv emotion categories.

Flow per simulated user:

1. Build a :class:`SimulatedUser` seeker from an :class:`ESConvConversation`
   (seeker persona = ESConv's situation + emotion_type + problem_type,
   opening = the dataset's first seeker utterance).
2. Run an alternating dialogue of ``max_turns`` turns against a live
   :class:`CompanionAgent`.
3. Administer the 5-item WAI-SR to the seeker — the same closure, kept
   warm with the full dialogue history so Bond reflects the actual
   conversation. The seeker is prompted to Likert-rate each item 1–7 and
   its reply is parsed to an int.
4. Aggregate across users into a :class:`CohortBondReport` with a 95% CI
   (normal approximation; CLT-safe at N ≥ 30).

**Role-reversal reused.** :func:`make_llm_simulated_user` already handles
the chat-template role flip (agent → user, sim → assistant) that
qwen2.5:7b required (see §6e-prime notes). This module layers on top; it
does NOT re-implement the role mapping.

**source="llm_simulated"** is stamped on every WAI-SR response set here —
Phase 5+ calibration against human administrations is the gap this flag
makes auditable.

**No live LLM calls in unit tests.** Tests inject a scripted
:class:`SimulatedUser` so the harness shape is covered without pulling
Ollama into CI.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from statistics import mean
from typing import Callable, Optional, Sequence

from psychology_base.applications.emotional.companion_agent import CompanionAgent
from psychology_base.applications.llm_responders import (
    SimulatedUser,
    make_llm_simulated_user,
)
from psychology_base.applications.measurement.wai_sr import (
    WAI_SR_ITEMS,
    WAISRResponse,
    WAISRResponseSet,
    WAISRScores,
    score_wai_sr,
)
from psychology_base.eval.benchmarks.public_datasets.esconv import (
    ESConvConversation,
    seeker_opening_utterance,
)
from psychology_base.llm.router import LLMRouter
from psychology_base.llm.types import TaskTier


# ---------------------------------------------------------------------------
# Seeker factory
# ---------------------------------------------------------------------------


_DEFAULT_SEEKER_PERSONA: str = (
    "You are a person coming to a long-term emotional companion for support. "
    "Speak naturally, in your own voice, in one or two short sentences per "
    "turn. You are not performing or testing — you are living through this. "
    "If the companion makes an unhelpful move you are allowed to push back; "
    "if it helps you are allowed to say so."
)


def _scenario_prompt_for(convo: ESConvConversation) -> str:
    """Compose the scenario description the seeker will play."""
    parts = [f"Your current emotional state: {convo.emotion_type}."]
    if convo.problem_type:
        parts.append(f"Problem domain: {convo.problem_type}.")
    if convo.situation:
        parts.append(f"What is happening: {convo.situation}")
    return " ".join(parts)


def make_esconv_seeker(
    router: LLMRouter,
    convo: ESConvConversation,
    *,
    persona_prompt: str = _DEFAULT_SEEKER_PERSONA,
    opening_user_utterance: Optional[str] = None,
    tier: TaskTier = TaskTier.SIMPLE,
    max_tokens: Optional[int] = 200,
    temperature: Optional[float] = 0.7,
) -> SimulatedUser:
    """Build a :class:`SimulatedUser` playing the help-seeker of ``convo``.

    If ``opening_user_utterance`` is not provided, the first seeker turn
    from the dataset is used; absent that, the ``situation`` field serves
    as the opener so the companion always has a seed to respond to.
    """
    opener = opening_user_utterance or seeker_opening_utterance(convo) or convo.situation
    if not opener:
        raise ValueError(
            f"ESConv convo {convo.conversation_id!r} has no seeker opener and no situation"
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
# Session runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionTurn:
    """One turn in a simulated-user × companion dialogue."""

    speaker: str  # "seeker" | "companion"
    utterance: str


@dataclass(frozen=True)
class SessionTranscript:
    """Full transcript of a simulated ESConv session."""

    conversation_id: str
    turns: tuple[SessionTurn, ...]
    handoff_triggered: bool


def run_esconv_session(
    convo: ESConvConversation,
    companion_agent: CompanionAgent,
    seeker: SimulatedUser,
    *,
    max_turns: int = 12,
    starting_timestamp: Optional[datetime] = None,
    seconds_per_turn: float = 30.0,
) -> SessionTranscript:
    """Run an alternating seeker↔companion dialogue up to ``max_turns`` turns.

    One "turn" = one companion response. The seeker speaks first (its
    opening is already anchored in the simulated-user system prompt per
    :func:`make_llm_simulated_user` semantics, so the first call to
    ``seeker`` passes an empty string so it emits that opener directly).

    If the companion short-circuits to a crisis handoff mid-session, the
    loop exits early with ``handoff_triggered=True`` on the transcript —
    the WAI-SR administration still runs so a mean Bond can be scored,
    but downstream aggregation can filter these out.
    """
    if max_turns <= 0:
        raise ValueError(f"max_turns must be positive, got {max_turns}")

    now = starting_timestamp or datetime.now(timezone.utc)
    turns: list[SessionTurn] = []

    # Opener: the sim's first line. make_llm_simulated_user anchors the
    # opener in the system prompt and expects the first call to carry the
    # agent's first reply — but there IS no agent reply yet, so we read
    # the opener out of the convo (same source make_esconv_seeker used)
    # and emit it directly as the first seeker turn. The sim closure's
    # history stays empty until the companion speaks.
    opener = seeker_opening_utterance(convo) or convo.situation
    if not opener:
        raise ValueError(
            f"ESConv convo {convo.conversation_id!r} has no opener for the session"
        )
    turns.append(SessionTurn(speaker="seeker", utterance=opener))

    handoff = False
    seeker_utterance = opener
    for turn_idx in range(max_turns):
        turn_ts = now + _timedelta_seconds(seconds_per_turn * (turn_idx + 1))
        resp = companion_agent.respond(seeker_utterance, timestamp=turn_ts)
        turns.append(SessionTurn(speaker="companion", utterance=resp.response_text))
        if resp.handoff_triggered:
            handoff = True
            break
        if turn_idx == max_turns - 1:
            break  # Don't ask seeker for a reply we'll never read.
        seeker_utterance = seeker(resp.response_text)
        if not seeker_utterance:
            break  # Seeker fell silent — downstream will see fewer turns.
        turns.append(SessionTurn(speaker="seeker", utterance=seeker_utterance))

    return SessionTranscript(
        conversation_id=convo.conversation_id,
        turns=tuple(turns),
        handoff_triggered=handoff,
    )


def _timedelta_seconds(seconds: float):
    from datetime import timedelta

    return timedelta(seconds=seconds)


# ---------------------------------------------------------------------------
# WAI-SR administration against a simulated seeker
# ---------------------------------------------------------------------------


_LIKERT_RANGE = range(1, 8)
_INT_PATTERN = re.compile(r"-?\d+")


class WAISRParseError(ValueError):
    """Raised when a seeker reply cannot be parsed to a Likert 1–7 integer."""


def _parse_likert(reply: str) -> int:
    """Extract the first in-range integer from a seeker reply.

    Robust to prose wrappers like "I'd say about a 6, because …" — we take
    the first integer in [1, 7]. If none, raise so the caller surfaces
    the misbehaving seeker rather than silently defaulting.
    """
    for match in _INT_PATTERN.finditer(reply):
        try:
            value = int(match.group(0))
        except ValueError:  # defensive — regex guarantees digits
            continue
        if value in _LIKERT_RANGE:
            return value
    raise WAISRParseError(
        f"Could not parse a Likert 1–7 integer from seeker reply: {reply!r}"
    )


def administer_wai_sr_simulated(
    seeker: SimulatedUser,
    *,
    user_id: str,
    administered_at_seconds: float,
    subject_noun: str = "Companion",
) -> WAISRResponseSet:
    """Administer the 5-item WAI-SR to a simulated seeker.

    Each item wording is passed to the seeker as if the companion asked
    the question; the seeker's reply is parsed to an int in [1, 7].
    ``source="llm_simulated"`` is stamped so cohort aggregates stay
    distinguishable from any later human-administered data.
    """
    responses: list[WAISRResponse] = []
    for item_id, subscale, wording in WAI_SR_ITEMS:
        rendered = wording.format(subject=subject_noun)
        prompt = (
            f"Please rate this statement on a scale from 1 (seldom) to 7 "
            f'(always). Answer with just the number. "{rendered}"'
        )
        reply = seeker(prompt)
        score = _parse_likert(reply)
        responses.append(
            WAISRResponse(item_id=item_id, subscale=subscale, score=score)
        )
    return WAISRResponseSet(
        user_id=user_id,
        administered_at_seconds=administered_at_seconds,
        subject_noun=subject_noun,
        responses=tuple(responses),
        source="llm_simulated",
    )


# ---------------------------------------------------------------------------
# Cohort aggregation + pivot gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CohortBondReport:
    """Cohort-level WAI-SR Bond roll-up for the pivot gate.

    ``bond_scores`` are the per-user Bond subscale means (``WAISRScores.bond``).
    The 95% CI uses the normal-approximation half-width ``1.96 · SE`` —
    valid at N ≥ 30 under CLT; the pivot gate requires N ≥ 50 so this is
    safe by construction. Small-N (<30) runs are allowed for shakeout but
    the report emits a wider nominal CI as a readability nudge, not a
    rigorous small-sample interval.
    """

    bond_scores: tuple[float, ...]
    emotion_type_counts: dict[str, int] = field(default_factory=dict)
    excluded_handoffs: int = 0

    @property
    def n(self) -> int:
        return len(self.bond_scores)

    @property
    def mean_bond(self) -> float:
        return mean(self.bond_scores) if self.bond_scores else 0.0

    @property
    def stderr(self) -> float:
        if self.n < 2:
            return 0.0
        m = self.mean_bond
        var = sum((x - m) ** 2 for x in self.bond_scores) / (self.n - 1)
        return math.sqrt(var / self.n)

    @property
    def ci95_half_width(self) -> float:
        return 1.96 * self.stderr

    def meets_pivot_gate(
        self,
        *,
        bond_floor: float = 3.8,
        min_users: int = 50,
    ) -> bool:
        """Check pivot gate: mean Bond ≥ ``bond_floor`` AND N ≥ ``min_users``.

        Defaults match `docs/phases/phase-4b-benchmark-pivot.md` §3 row
        情感陪伴 WAI-SR Bond.
        """
        return self.n >= min_users and self.mean_bond >= bond_floor


def aggregate_cohort_bond(
    response_sets: Sequence[WAISRResponseSet],
    *,
    emotion_types: Optional[Sequence[str]] = None,
    excluded_handoffs: int = 0,
) -> CohortBondReport:
    """Reduce per-user WAI-SR administrations to a cohort Bond report.

    ``emotion_types`` is optional and parallel to ``response_sets``; when
    provided it populates the per-category count table so a reader can see
    whether the cohort spans the ESConv emotion categories or clumps on
    one.
    """
    if emotion_types is not None and len(emotion_types) != len(response_sets):
        raise ValueError(
            f"emotion_types length ({len(emotion_types)}) must match "
            f"response_sets length ({len(response_sets)})"
        )
    bond_scores = tuple(score_wai_sr(rs).bond for rs in response_sets)
    counts: dict[str, int] = {}
    if emotion_types is not None:
        for et in emotion_types:
            counts[et] = counts.get(et, 0) + 1
    return CohortBondReport(
        bond_scores=bond_scores,
        emotion_type_counts=counts,
        excluded_handoffs=excluded_handoffs,
    )


__all__ = [
    "CohortBondReport",
    "SessionTranscript",
    "SessionTurn",
    "WAISRParseError",
    "administer_wai_sr_simulated",
    "aggregate_cohort_bond",
    "make_esconv_seeker",
    "run_esconv_session",
]
