"""MSC × WAI-SR longitudinal Bond harness — Phase 8 Pillar A.

Targets `docs/phases/phase-8-dataset-replay-expansion-plan.md` §2 pillar
A — mean WAI-SR Bond ≥ 3.89 at **session 5** across ≥ 50 simulated users
spanning MSC personas, measured per the same simulated-user WAI-SR
administration pattern as Phase 4b ESConv Bond.

**What's new over the ESConv harness.** This module runs **5 sessions
back-to-back** against a single :class:`CompanionAgent` whose memory
persists across sessions, with in-world time advancing by MSC's natural
``time_interval_days`` between sessions. WAI-SR is administered twice
per user: once after session 1 (early-bond proxy) and once after session
5 (longitudinal-bond proxy). The pillar gate is the session-5 cohort
Bond; the Δ(session 5 − session 1) is a reported-not-gated longitudinal
signal.

Flow per MSC conversation:

1. Build a persona-grounded :class:`SimulatedUser` seeker from
   ``self_persona(convo)`` (speaker_idx=0) and a scenario prompt that
   anchors the persona bullets in a long-term-companion framing.
2. For each MSC session (1..N):
   a. Set the session's ``starting_timestamp`` to the base timestamp +
      cumulative in-world days from the dataset.
   b. Run ``max_turns`` of alternating seeker↔companion dialogue, carrying
      the seeker's conversational memory **within** the session.
   c. At end-of-session, if ``session.session_id`` ∈ {1, N}, administer
      the 5-item WAI-SR against the seeker.
3. Aggregate per-user Bond scores into a :class:`LongitudinalBondReport`
   with session-1 and session-5 cohort stats + the paired delta.

**Simulated-user state across sessions.** The seeker closure keeps its
full dialogue history across sessions by construction (it's one
:class:`SimulatedUser` closure for the whole conversation). The
companion-side carries memory via its :class:`CompanionSystem` which
persists for the whole 5-session run; MSC's ``time_interval_days`` gaps
let the companion's aging policy decay / consolidate between sessions
exactly the way a multi-week deployment would.

**No live LLM calls in unit tests.** Tests inject a scripted
:class:`SimulatedUser` so the harness shape is covered without pulling
Ollama into CI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from statistics import mean
from typing import Optional, Sequence

from psychology_base.applications.emotional.companion_agent import CompanionAgent
from psychology_base.applications.llm_responders import (
    SimulatedUser,
    make_llm_simulated_user,
)
from psychology_base.applications.measurement.wai_sr import (
    WAISRResponseSet,
    WAISRScores,
    score_wai_sr,
)
from psychology_base.eval.benchmarks.public_datasets.esconv_wai_sr_harness import (
    SessionTranscript,
    SessionTurn,
    administer_wai_sr_simulated,
)
from psychology_base.eval.benchmarks.public_datasets.msc import (
    MSCConversation,
    MSCSession,
    SELF_SPEAKER_IDX,
    self_opening_utterance,
    self_persona,
)
from psychology_base.llm.router import LLMRouter
from psychology_base.llm.types import TaskTier


# ---------------------------------------------------------------------------
# Persona-grounded seeker factory
# ---------------------------------------------------------------------------


_DEFAULT_SEEKER_PERSONA_FRAME: str = (
    "You are a person who has been talking with a long-term companion AI "
    "across multiple sessions over several weeks. Speak naturally, in your "
    "own voice, in one or two short sentences per turn. You remember prior "
    "sessions as a real person would — not every detail, but the shape and "
    "feel of them. If the companion makes an unhelpful move you can push "
    "back; if it helps you can say so."
)


def _scenario_prompt_for(convo: MSCConversation) -> str:
    """Compose the scenario description the simulated seeker will play."""
    bullets = self_persona(convo)
    if not bullets:
        return "You have no specific backstory beyond this conversation."
    joined = " ".join(b.strip() for b in bullets if b.strip())
    return f"Your background and current life — things true about you: {joined}"


def make_msc_seeker(
    router: LLMRouter,
    convo: MSCConversation,
    *,
    persona_prompt: str = _DEFAULT_SEEKER_PERSONA_FRAME,
    opening_user_utterance: Optional[str] = None,
    tier: TaskTier = TaskTier.SIMPLE,
    max_tokens: Optional[int] = 200,
    temperature: Optional[float] = 0.7,
) -> SimulatedUser:
    """Build a :class:`SimulatedUser` playing the self-speaker of ``convo``.

    If ``opening_user_utterance`` is not provided, the first self-speaker
    turn from session 1 is used; absent that, a generic opener is used so
    the companion always has a seed to respond to.
    """
    opener = (
        opening_user_utterance
        or self_opening_utterance(convo, session_id=1)
        or "Hi, how have you been?"
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
# Session runner — one MSC session end-to-end
# ---------------------------------------------------------------------------


def run_msc_session(
    session: MSCSession,
    companion_agent: CompanionAgent,
    seeker: SimulatedUser,
    *,
    max_turns: int = 6,
    starting_timestamp: Optional[datetime] = None,
    seconds_per_turn: float = 30.0,
    opening_utterance: Optional[str] = None,
) -> SessionTranscript:
    """Run alternating seeker↔companion dialogue for one MSC session.

    The seeker opens with ``opening_utterance`` if provided, otherwise with
    the first self-speaker turn of the MSC session (so each session gets
    a dataset-authentic opener). The companion's memory persists across
    calls — this function does not reset the agent.
    """
    if max_turns <= 0:
        raise ValueError(f"max_turns must be positive, got {max_turns}")

    now = starting_timestamp or datetime.now(timezone.utc)
    turns: list[SessionTurn] = []

    opener = opening_utterance or _first_self_utterance(session) or "Hi."
    turns.append(SessionTurn(speaker="seeker", utterance=opener))

    handoff = False
    seeker_utterance = opener
    for turn_idx in range(max_turns):
        turn_ts = now + timedelta(seconds=seconds_per_turn * (turn_idx + 1))
        resp = companion_agent.respond(seeker_utterance, timestamp=turn_ts)
        turns.append(SessionTurn(speaker="companion", utterance=resp.response_text))
        if resp.handoff_triggered:
            handoff = True
            break
        if turn_idx == max_turns - 1:
            break
        seeker_utterance = seeker(resp.response_text)
        if not seeker_utterance:
            break
        turns.append(SessionTurn(speaker="seeker", utterance=seeker_utterance))

    return SessionTranscript(
        conversation_id=f"session-{session.session_id}",
        turns=tuple(turns),
        handoff_triggered=handoff,
    )


def _first_self_utterance(session: MSCSession) -> Optional[str]:
    for turn in session.turns:
        if turn.speaker_idx == SELF_SPEAKER_IDX:
            return turn.utterance
    return None


# ---------------------------------------------------------------------------
# Multi-session runner — one MSC conversation end-to-end
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MSCUserRun:
    """One MSC conversation played out across its sessions.

    ``per_session_transcripts`` is indexed by session_id order (not by
    session_id value, which may be sparse if the dataset skips a session).
    ``wai_sr_by_session_id`` holds the WAI-SR response set for each session
    where it was administered (keyed by session_id value).
    """

    conversation_id: str
    per_session_transcripts: tuple[SessionTranscript, ...]
    wai_sr_by_session_id: dict[int, WAISRResponseSet]
    handoff_triggered: bool


def run_msc_conversation(
    convo: MSCConversation,
    companion_agent: CompanionAgent,
    seeker: SimulatedUser,
    *,
    max_turns_per_session: int = 6,
    base_timestamp: Optional[datetime] = None,
    seconds_per_turn: float = 30.0,
    wai_sr_at_sessions: Optional[Sequence[int]] = None,
    user_id: Optional[str] = None,
    subject_noun: str = "Companion",
) -> MSCUserRun:
    """Play out all sessions of ``convo`` end-to-end against one agent.

    ``wai_sr_at_sessions`` defaults to ``(1, last_session_id)`` so the
    report captures early-bond and longitudinal-bond without probing
    mid-chain. If a crisis handoff fires in any session, the loop exits
    early with ``handoff_triggered=True``; WAI-SR is only administered
    for sessions that completed before the handoff.

    Between-session in-world advance comes from
    ``session.time_interval_days`` — the agent's memory substrate sees
    realistic multi-day gaps so aging / consolidation can run between
    sessions exactly as it would in a multi-week deployment.
    """
    base = base_timestamp or datetime.now(timezone.utc)
    sessions_sorted = tuple(convo.sessions)
    if not sessions_sorted:
        raise ValueError(f"MSC conversation {convo.conversation_id!r} has no sessions")

    if wai_sr_at_sessions is None:
        wai_sr_at_sessions = (
            sessions_sorted[0].session_id,
            sessions_sorted[-1].session_id,
        )
    wai_sr_at = set(wai_sr_at_sessions)

    uid = user_id or f"u-msc-{convo.conversation_id}"

    cumulative_days = 0.0
    per_session: list[SessionTranscript] = []
    wai_sr: dict[int, WAISRResponseSet] = {}
    handoff = False
    for session in sessions_sorted:
        cumulative_days += session.time_interval_days
        session_start = base + timedelta(days=cumulative_days)
        transcript = run_msc_session(
            session,
            companion_agent=companion_agent,
            seeker=seeker,
            max_turns=max_turns_per_session,
            starting_timestamp=session_start,
            seconds_per_turn=seconds_per_turn,
        )
        per_session.append(transcript)
        if transcript.handoff_triggered:
            handoff = True
            break
        if session.session_id in wai_sr_at:
            admin_ts = session_start + timedelta(
                seconds=seconds_per_turn * max_turns_per_session + 60.0
            )
            response_set = administer_wai_sr_simulated(
                seeker,
                user_id=uid,
                administered_at_seconds=admin_ts.timestamp(),
                subject_noun=subject_noun,
            )
            wai_sr[session.session_id] = response_set

    return MSCUserRun(
        conversation_id=convo.conversation_id,
        per_session_transcripts=tuple(per_session),
        wai_sr_by_session_id=wai_sr,
        handoff_triggered=handoff,
    )


# ---------------------------------------------------------------------------
# Cohort aggregation + pillar gate
# ---------------------------------------------------------------------------


_WOEBOT_BOND_FLOOR: float = 3.89
_PILLAR_MIN_USERS: int = 50


@dataclass(frozen=True)
class LongitudinalBondReport:
    """Cohort-level longitudinal WAI-SR Bond roll-up for Phase 8 Pillar A.

    ``session_1_bond_scores`` and ``session_5_bond_scores`` are per-user
    Bond subscale means at the two measured points. Users whose session-5
    administration failed (handoff mid-chain, parse failure, etc.) are
    counted in ``excluded_users`` and NOT included in the session-5
    aggregate — session-1 scores for those users ARE kept so the early
    cohort is still fully reported.

    The 95% CI uses the normal-approximation half-width ``1.96 · SE`` —
    valid at N ≥ 30 under CLT; the pillar gate requires N ≥ 50 so this is
    safe by construction. Small-N (<30) runs are allowed for shakeout but
    read the ``n_session_5`` field, not the CI, for honesty.
    """

    session_1_bond_scores: tuple[float, ...]
    session_5_bond_scores: tuple[float, ...]
    paired_deltas: tuple[float, ...]  # session_5 − session_1 per user with both
    excluded_users: int = 0
    excluded_handoffs: int = 0
    persona_counts: dict[str, int] = field(default_factory=dict)

    @property
    def n_session_1(self) -> int:
        return len(self.session_1_bond_scores)

    @property
    def n_session_5(self) -> int:
        return len(self.session_5_bond_scores)

    @property
    def n_paired(self) -> int:
        return len(self.paired_deltas)

    @property
    def mean_session_1_bond(self) -> float:
        return mean(self.session_1_bond_scores) if self.session_1_bond_scores else 0.0

    @property
    def mean_session_5_bond(self) -> float:
        return mean(self.session_5_bond_scores) if self.session_5_bond_scores else 0.0

    @property
    def mean_paired_delta(self) -> float:
        return mean(self.paired_deltas) if self.paired_deltas else 0.0

    @property
    def session_5_stderr(self) -> float:
        return _stderr(self.session_5_bond_scores)

    @property
    def session_5_ci95_half_width(self) -> float:
        return 1.96 * self.session_5_stderr

    @property
    def paired_delta_stderr(self) -> float:
        return _stderr(self.paired_deltas)

    def meets_pillar_gate(
        self,
        *,
        bond_floor: float = _WOEBOT_BOND_FLOOR,
        min_users: int = _PILLAR_MIN_USERS,
    ) -> bool:
        """Phase 8 Pillar A gate: session-5 cohort Bond ≥ ``bond_floor`` at N ≥ ``min_users``.

        Defaults match `docs/phases/phase-8-dataset-replay-expansion-plan.md`
        §2 pillar A (Woebot aggregate floor 3.89, N ≥ 50).
        """
        return (
            self.n_session_5 >= min_users
            and self.mean_session_5_bond >= bond_floor
        )


def _stderr(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    m = mean(values)
    var = sum((x - m) ** 2 for x in values) / (n - 1)
    return math.sqrt(var / n)


def aggregate_longitudinal_bond(
    user_runs: Sequence[MSCUserRun],
    *,
    persona_labels: Optional[Sequence[str]] = None,
    session_1_id: int = 1,
    session_5_id: int = 5,
) -> LongitudinalBondReport:
    """Reduce per-user :class:`MSCUserRun`s to a longitudinal Bond report.

    ``session_1_id`` and ``session_5_id`` are the dataset's session ids for
    the early-bond and longitudinal-bond measurement points; defaults are
    1 and 5 per MSC's canonical 5-session shape, but can be overridden if
    the shim slices sessions differently.

    ``persona_labels`` is optional and parallel to ``user_runs``; when
    provided it populates ``persona_counts`` for the readability of the
    emitted report (e.g., distribution over coarse persona-topic buckets).
    """
    if persona_labels is not None and len(persona_labels) != len(user_runs):
        raise ValueError(
            f"persona_labels length ({len(persona_labels)}) must match "
            f"user_runs length ({len(user_runs)})"
        )

    s1_scores: list[float] = []
    s5_scores: list[float] = []
    deltas: list[float] = []
    excluded = 0
    handoffs = 0

    for run in user_runs:
        if run.handoff_triggered:
            handoffs += 1
        s1_rs = run.wai_sr_by_session_id.get(session_1_id)
        s5_rs = run.wai_sr_by_session_id.get(session_5_id)
        s1_bond = score_wai_sr(s1_rs).bond if s1_rs is not None else None
        s5_bond = score_wai_sr(s5_rs).bond if s5_rs is not None else None
        if s1_bond is not None:
            s1_scores.append(s1_bond)
        if s5_bond is not None:
            s5_scores.append(s5_bond)
        else:
            excluded += 1
        if s1_bond is not None and s5_bond is not None:
            deltas.append(s5_bond - s1_bond)

    counts: dict[str, int] = {}
    if persona_labels is not None:
        for label in persona_labels:
            counts[label] = counts.get(label, 0) + 1

    return LongitudinalBondReport(
        session_1_bond_scores=tuple(s1_scores),
        session_5_bond_scores=tuple(s5_scores),
        paired_deltas=tuple(deltas),
        excluded_users=excluded,
        excluded_handoffs=handoffs,
        persona_counts=counts,
    )


__all__ = [
    "LongitudinalBondReport",
    "MSCUserRun",
    "administer_wai_sr_simulated",  # re-exported for convenience of runners
    "aggregate_longitudinal_bond",
    "make_msc_seeker",
    "run_msc_conversation",
    "run_msc_session",
]
