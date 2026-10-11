"""Counseling-depth scorer — Phase 8 Pillar F.

One harness, three sub-gates:

- **Bond** (ExTES + AnnoMI). Simulated client seeker plays the dataset's
  opening client turn, Companion continues the dialogue for N live turns,
  WAI-SR 5-item Bond is administered against the simulated seeker. Pattern
  reused byte-identically from :mod:`msc_bond_longitudinal` and
  :mod:`esconv_wai_sr_harness`; this scorer only differs in the seeker
  factory (ExTES has no persona bullets, AnnoMI has a topic + MI-quality
  hint).

- **EMPATH-κ** (ExTES only — AnnoMI has no gold counselor response to compare
  against). The empathy judge (`_lexical_empathy_score`, scripted) scores
  both the Companion's live reply and the dataset's reference
  ``counselor_response`` under the same rubric; the inter-rater κ is
  reframed as the per-dialog agreement between the two scores. Lexical
  scoring is a stand-in for the stronger LLM-judge κ this would warrant
  on a well-recruited pilot — carried-debt item documented in the close-out.

- **MI-fidelity** (AnnoMI only — the reflection/open-question MITI counts
  only exist as gold labels on AnnoMI). The Companion plays N live turns
  as the counselor; each live reply is lexically labeled (reflection /
  question / open-question / other) with a rule set documented in
  :func:`_classify_companion_turn`. The Companion's rate is compared
  against the AnnoMI *human-baseline* rate on the same dialogue.

**Gate shape** (Pillar F partial PASS per Phase 8 plan § and this task's
pass-criteria):

- ExTES Bond ≥ 3.89 (Woebot floor)
- ExTES EMPATH-κ ≥ 0.50 (reframed as per-dialog Cohen's κ between the
  lexical empathy judge's companion-side and reference-side binary labels
  across the per-dialog sample; floor is a pragmatic halfway between
  "random" (0) and "near-perfect" (0.75))
- AnnoMI MI-fidelity: Companion reflection rate ≥ 0.30 absolute OR
  ≥ 50 % of AnnoMI human-baseline reflection rate on matched dialogues.
  Same 0.30 floor applies to open-question rate as a secondary signal.
  The 0.30 floor is grounded in AnnoMI's own high-quality MI dialogues
  (reflection rate ~0.40 median per Wu et al. 2023 Table 2).

The scorer produces a :class:`CounselingDepthReport` that gates on all
three for a full PASS; the CLI runners split into per-corpus reports
when running one at a time (``phase8_f_extes_run.py`` and
``phase8_f_annomi_run.py``).

**No live LLM calls in unit tests.** Scripted responder + scripted
seeker injection surface is identical to Pillar A/B.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from statistics import mean
from typing import Callable, Iterable, Optional, Sequence

from psychology_base.applications.emotional.companion_agent import CompanionAgent
from psychology_base.applications.llm_responders import (
    SimulatedUser,
    make_llm_simulated_user,
)
from psychology_base.applications.measurement.wai_sr import score_wai_sr
from psychology_base.eval.benchmarks.public_datasets.annomi import (
    AnnoMIDialogue,
    therapist_open_question_rate,
    therapist_reflection_rate,
)
from psychology_base.eval.benchmarks.public_datasets.esconv_wai_sr_harness import (
    SessionTranscript,
    SessionTurn,
    administer_wai_sr_simulated,
)
from psychology_base.eval.benchmarks.public_datasets.extes import ExTESExample
from psychology_base.llm.router import LLMRouter
from psychology_base.llm.types import TaskTier


# ---------------------------------------------------------------------------
# Gate floors
# ---------------------------------------------------------------------------


_WOEBOT_BOND_FLOOR: float = 3.89
_EMPATH_KAPPA_FLOOR: float = 0.50
_MI_FIDELITY_FLOOR: float = 0.30
_RATE_MATCHED_EMPATHY_ABS_FLOOR: float = 0.30
_RATE_MATCHED_EMPATHY_BASELINE_FRACTION: float = 0.50


def _rate_matched_empathy_pass(
    companion_rate: float,
    reference_rate: float,
    *,
    absolute_floor: float = _RATE_MATCHED_EMPATHY_ABS_FLOOR,
    baseline_fraction: float = _RATE_MATCHED_EMPATHY_BASELINE_FRACTION,
) -> bool:
    """Rate-matched sibling to κ-across-conversants Pillar F gate.

    Companion passes iff its empathic-turn rate clears the absolute floor
    OR clears ``baseline_fraction`` of the reference-counselor rate on
    this corpus. Mirrors the AnnoMI ``meets_mi_fidelity_gate`` shape so
    both gates have the same conceptual contract: *does Companion emit at
    a rate consistent with high-quality counseling baselines*, rather
    than *does Companion agree per-dialog with the dataset reference*.

    Documented across §10.6 / §11.3 / §12.6 of the Pillar F close-out as
    the methodological rework the κ-ceiling signature kept flagging.
    This is NOT a κ gate replacement — both gates ship side-by-side and
    the rollup-of-record remains the κ gate.
    """
    if companion_rate >= absolute_floor:
        return True
    if reference_rate > 0.0 and companion_rate >= baseline_fraction * reference_rate:
        return True
    return False


# ---------------------------------------------------------------------------
# Seeker factories (corpus-specific)
# ---------------------------------------------------------------------------


_EXTES_SEEKER_PERSONA: str = (
    "You are a person coming to an emotional-support companion. You are "
    "living through something hard right now. Speak naturally, one or "
    "two short sentences at a time. If the companion helps you, you may "
    "say so; if it says something that lands wrong, you may push back."
)


def make_extes_seeker(
    router: LLMRouter,
    example: ExTESExample,
    *,
    persona_prompt: str = _EXTES_SEEKER_PERSONA,
    tier: TaskTier = TaskTier.SIMPLE,
    max_tokens: Optional[int] = 200,
    temperature: Optional[float] = 0.7,
) -> SimulatedUser:
    """Build a :class:`SimulatedUser` whose first line is the ExTES client utterance."""
    opener = example.client_utterance.strip()
    if not opener:
        raise ValueError(f"ExTES example {example.example_id!r} has no client utterance")
    scenario = (
        "You are describing your current emotional situation to the "
        "companion. Stay in character; do not switch to a counselor role."
    )
    return make_llm_simulated_user(
        router,
        persona_prompt=persona_prompt,
        scenario_prompt=scenario,
        opening_user_utterance=opener,
        tier=tier,
        max_tokens=max_tokens,
        temperature=temperature,
    )


_ANNOMI_SEEKER_PERSONA: str = (
    "You are a person in a counselling conversation with a therapist. You "
    "are ambivalent about change — reasons for change coexist with reasons "
    "for staying the same. Speak naturally, one or two sentences per turn, "
    "in your own voice. Push back if the therapist is directive; open up "
    "if they listen."
)


def make_annomi_seeker(
    router: LLMRouter,
    dialogue: AnnoMIDialogue,
    *,
    persona_prompt: str = _ANNOMI_SEEKER_PERSONA,
    tier: TaskTier = TaskTier.SIMPLE,
    max_tokens: Optional[int] = 200,
    temperature: Optional[float] = 0.7,
) -> SimulatedUser:
    """Build a :class:`SimulatedUser` seeded by the AnnoMI first client turn."""
    opener_turn = dialogue.first_client_turn()
    if opener_turn is None:
        raise ValueError(
            f"AnnoMI dialogue {dialogue.transcript_id!r} has no client turn"
        )
    opener = opener_turn.utterance_text.strip()
    if not opener:
        raise ValueError(
            f"AnnoMI dialogue {dialogue.transcript_id!r} has empty first client turn"
        )
    scenario = (
        f"Topic you're discussing: {dialogue.topic or 'personal change'}. "
        f"Stay in character as the client."
    )
    return make_llm_simulated_user(
        router,
        persona_prompt=persona_prompt,
        scenario_prompt=scenario,
        opening_user_utterance=opener,
        tier=tier,
        max_tokens=max_tokens,
        temperature=temperature,
    )


# ---------------------------------------------------------------------------
# Session runner (shared between ExTES + AnnoMI)
# ---------------------------------------------------------------------------


def run_counseling_session(
    *,
    conversation_id: str,
    opening_utterance: str,
    companion_agent: CompanionAgent,
    seeker: SimulatedUser,
    max_turns: int = 6,
    starting_timestamp: Optional[datetime] = None,
    seconds_per_turn: float = 30.0,
) -> SessionTranscript:
    """Run an alternating seeker↔companion dialogue — same contract as MSC.

    One "turn" = one companion response. The seeker opens with
    ``opening_utterance`` (the dataset's first client utterance); on each
    subsequent turn the seeker closure drives the next line from
    history+scenario. If the companion short-circuits to a crisis handoff
    the loop exits early.
    """
    if max_turns <= 0:
        raise ValueError(f"max_turns must be positive, got {max_turns}")

    now = starting_timestamp or datetime.now(timezone.utc)
    turns: list[SessionTurn] = []
    opener = opening_utterance.strip()
    if not opener:
        raise ValueError(f"Session {conversation_id!r} opener is empty")
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
        conversation_id=conversation_id,
        turns=tuple(turns),
        handoff_triggered=handoff,
    )


# ---------------------------------------------------------------------------
# Lexical empathy scorer (EMPATH-κ)
# ---------------------------------------------------------------------------


# Three cue sets, used additively. The score is a 0..1 fraction of cues
# matched (clipped). Lexical-only is weak compared to a judge-LLM; the
# close-out flags this as carried debt. The cue list is the same shape
# as :mod:`mathdial_empath_scorer` lexical-track so a drift here is
# auditable against that baseline.
_EMPATHY_RECOGNITION_CUES = (
    "feel",
    "feeling",
    "feels",
    "felt",
    "sound",
    "sounds",
    "seem",
    "seems",
    "understand",
    "hear",
    "see what you mean",
    "makes sense",
    "that's hard",
    "that is hard",
    "difficult",
    "painful",
    "overwhelming",
    "frustrating",
    "scary",
    "lonely",
)

_EMPATHY_PRESENCE_CUES = (
    "i'm here",
    "i am here",
    "i hear you",
    "with you",
    "beside you",
    "not alone",
    "we can",
    "together",
    "take your time",
    "there's no rush",
    "no judgement",
    "no judgment",
)

_EMPATHY_VALIDATION_CUES = (
    "makes sense",
    "that makes sense",
    "it makes sense",
    "no wonder",
    "of course",
    "anyone would",
    "it's okay",
    "it is okay",
    "you have every right",
    "that's valid",
    "that is valid",
    "important",
    "you matter",
)


def _lexical_empathy_score(reply: str) -> float:
    """Weak empathy proxy — fraction of empathy cues present, clipped [0, 1].

    Returns a float where 0 ≈ "no empathy signal" and 1 ≈ "saturated
    across all three families (recognition + presence + validation)".
    Three families contribute equally; within a family we count presence
    of at least one cue as 1/3 of the family's weight (so hitting 3+ cues
    in one family doesn't over-weight it).
    """
    if not reply:
        return 0.0
    low = reply.lower()
    def _family_hit(cues: tuple[str, ...]) -> int:
        for c in cues:
            if c in low:
                return 1
        return 0
    hits = (
        _family_hit(_EMPATHY_RECOGNITION_CUES)
        + _family_hit(_EMPATHY_PRESENCE_CUES)
        + _family_hit(_EMPATHY_VALIDATION_CUES)
    )
    return hits / 3.0


def _binarise_empathy(score: float, threshold: float = 1.0 / 3.0) -> int:
    """1 if score ≥ threshold (at least one empathy family present), else 0."""
    return 1 if score >= threshold else 0


def _cohens_kappa_binary(a: Sequence[int], b: Sequence[int]) -> float:
    """Cohen's κ for two binary rater sequences.

    Returns κ ∈ [-1, 1]. If both raters have zero variance (all 0 or all
    1) the formula diverges — we return 1.0 if the sequences agree
    exactly in that case, else 0.0. N must match.
    """
    if len(a) != len(b):
        raise ValueError(f"kappa inputs length mismatch: {len(a)} vs {len(b)}")
    n = len(a)
    if n == 0:
        return 0.0
    po = sum(1 for i in range(n) if a[i] == b[i]) / n
    pa_pos = sum(a) / n
    pb_pos = sum(b) / n
    pe = pa_pos * pb_pos + (1 - pa_pos) * (1 - pb_pos)
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1.0 - pe)


# ---------------------------------------------------------------------------
# MI-fidelity classifier (reflection / open-question / other)
# ---------------------------------------------------------------------------


_REFLECTION_PATTERNS = (
    re.compile(r"\bit sounds like\b", re.I),
    re.compile(r"\bit seems (?:like |that )?", re.I),
    re.compile(r"\byou(?:'re| are) saying\b", re.I),
    re.compile(r"\byou feel\b", re.I),
    re.compile(r"\byou felt\b", re.I),
    re.compile(r"\byou're feeling\b", re.I),
    re.compile(r"\bso you\b", re.I),
    re.compile(r"\bwhat I hear\b", re.I),
    re.compile(r"\bI hear (?:that )?you\b", re.I),
    re.compile(r"\byou want\b", re.I),
    re.compile(r"\byou're (?:thinking|wondering|hoping)\b", re.I),
)

_OPEN_QUESTION_HEADS = (
    "what",
    "how",
    "why",
    "when",
    "where",
    "tell me",
    "can you say more",
    "could you say more",
)


def _classify_companion_turn(reply: str) -> str:
    """Return 'reflection' | 'open_question' | 'closed_question' | 'other'.

    Order: reflection check first, then open-question (sentence starting
    with an open head word), then any '?' as closed_question, else other.
    A reply can contain both — we return the first-matched category to
    keep the rate counts stable across sentence orderings. Reflection
    takes precedence because MITI counts a reflection-plus-question as a
    reflection in the primary rate.
    """
    if not reply:
        return "other"
    text = reply.strip()
    for pat in _REFLECTION_PATTERNS:
        if pat.search(text):
            return "reflection"
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for sent in sentences:
        s = sent.strip().lower()
        if not s or "?" not in sent:
            continue
        head = s.split()[0] if s.split() else ""
        if head in _OPEN_QUESTION_HEADS:
            return "open_question"
        if any(s.startswith(prefix) for prefix in _OPEN_QUESTION_HEADS):
            return "open_question"
    if "?" in text:
        return "closed_question"
    return "other"


# ---------------------------------------------------------------------------
# Per-dialog bundles
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExTESDialogResult:
    """One ExTES example scored end-to-end."""

    example_id: str
    bond: Optional[float]  # None if handoff before administration
    handoff_triggered: bool
    companion_turns: tuple[str, ...]
    reference_response: str
    companion_empathy_score: float  # average over the companion's turns
    reference_empathy_score: float  # single-pass on the reference
    companion_empathy_binary: int
    reference_empathy_binary: int


@dataclass(frozen=True)
class AnnoMIDialogResult:
    """One AnnoMI dialogue scored end-to-end."""

    transcript_id: str
    mi_quality: str
    bond: Optional[float]
    handoff_triggered: bool
    companion_turns: tuple[str, ...]
    companion_reflection_rate: float
    companion_open_question_rate: float
    baseline_reflection_rate: float
    baseline_open_question_rate: float


# ---------------------------------------------------------------------------
# Aggregate reports
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExTESReport:
    """Aggregate report over the ExTES scoring sweep."""

    per_dialog: tuple[ExTESDialogResult, ...]

    @property
    def n(self) -> int:
        return len(self.per_dialog)

    @property
    def bond_scores(self) -> tuple[float, ...]:
        return tuple(d.bond for d in self.per_dialog if d.bond is not None)

    @property
    def mean_bond(self) -> float:
        scores = self.bond_scores
        return mean(scores) if scores else 0.0

    @property
    def empath_kappa(self) -> float:
        if not self.per_dialog:
            return 0.0
        companion_labels = [d.companion_empathy_binary for d in self.per_dialog]
        reference_labels = [d.reference_empathy_binary for d in self.per_dialog]
        return _cohens_kappa_binary(companion_labels, reference_labels)

    @property
    def excluded_handoffs(self) -> int:
        return sum(1 for d in self.per_dialog if d.handoff_triggered)

    def meets_bond_gate(self, floor: float = _WOEBOT_BOND_FLOOR) -> bool:
        return self.n > 0 and self.mean_bond >= floor

    def meets_kappa_gate(self, floor: float = _EMPATH_KAPPA_FLOOR) -> bool:
        return self.n > 0 and self.empath_kappa >= floor


@dataclass(frozen=True)
class AnnoMIReport:
    """Aggregate report over the AnnoMI scoring sweep."""

    per_dialog: tuple[AnnoMIDialogResult, ...]

    @property
    def n(self) -> int:
        return len(self.per_dialog)

    @property
    def mean_bond(self) -> float:
        scores = [d.bond for d in self.per_dialog if d.bond is not None]
        return mean(scores) if scores else 0.0

    @property
    def mean_reflection_rate(self) -> float:
        if not self.per_dialog:
            return 0.0
        return mean(d.companion_reflection_rate for d in self.per_dialog)

    @property
    def mean_open_question_rate(self) -> float:
        if not self.per_dialog:
            return 0.0
        return mean(d.companion_open_question_rate for d in self.per_dialog)

    @property
    def mean_baseline_reflection_rate(self) -> float:
        if not self.per_dialog:
            return 0.0
        return mean(d.baseline_reflection_rate for d in self.per_dialog)

    @property
    def mean_baseline_open_question_rate(self) -> float:
        if not self.per_dialog:
            return 0.0
        return mean(d.baseline_open_question_rate for d in self.per_dialog)

    @property
    def excluded_handoffs(self) -> int:
        return sum(1 for d in self.per_dialog if d.handoff_triggered)

    def meets_mi_fidelity_gate(
        self,
        *,
        absolute_floor: float = _MI_FIDELITY_FLOOR,
        baseline_fraction: float = 0.50,
    ) -> bool:
        """Gate: reflection rate meets ``absolute_floor`` OR ≥ ``baseline_fraction``
        of the AnnoMI human-baseline reflection rate on this sample.
        """
        if self.n == 0:
            return False
        companion = self.mean_reflection_rate
        baseline = self.mean_baseline_reflection_rate
        if companion >= absolute_floor:
            return True
        if baseline > 0.0 and companion >= baseline_fraction * baseline:
            return True
        return False


@dataclass(frozen=True)
class CounselingDepthReport:
    """Pillar F unified report (one or both corpora)."""

    extes: Optional[ExTESReport] = None
    annomi: Optional[AnnoMIReport] = None
    pillar_f_bond_floor: float = _WOEBOT_BOND_FLOOR
    pillar_f_kappa_floor: float = _EMPATH_KAPPA_FLOOR
    pillar_f_mi_floor: float = _MI_FIDELITY_FLOOR

    @property
    def extes_bond_pass(self) -> bool:
        if self.extes is None:
            return False
        return self.extes.meets_bond_gate(self.pillar_f_bond_floor)

    @property
    def extes_kappa_pass(self) -> bool:
        if self.extes is None:
            return False
        return self.extes.meets_kappa_gate(self.pillar_f_kappa_floor)

    @property
    def annomi_mi_fidelity_pass(self) -> bool:
        if self.annomi is None:
            return False
        return self.annomi.meets_mi_fidelity_gate(absolute_floor=self.pillar_f_mi_floor)

    @property
    def pillar_f_pass(self) -> bool:
        """Pillar F partial PASS: all three available sub-gates pass.

        A corpus not run (``None``) does NOT disqualify — the partial-pass
        shape documented in the Phase 8 plan allows ExTES+AnnoMI to clear
        when PsyQA/D4 are externally gated. The gate counts only sub-gates
        with a result.
        """
        sub_results: list[bool] = []
        if self.extes is not None:
            sub_results.append(self.extes_bond_pass)
            sub_results.append(self.extes_kappa_pass)
        if self.annomi is not None:
            sub_results.append(self.annomi_mi_fidelity_pass)
        if not sub_results:
            return False
        return all(sub_results)


# ---------------------------------------------------------------------------
# Scoring entrypoints
# ---------------------------------------------------------------------------


# Type aliases for factories — same contract as Pillar B.
#
# ``system_factory``  : () -> CompanionSystem  (fresh substrate per dialog)
# ``agent_factory``   : (CompanionSystem, example_or_dialogue) -> CompanionAgent
# ``seeker_factory``  : (example_or_dialogue) -> SimulatedUser
SystemFactory = Callable[[], object]
AgentFactoryExTES = Callable[["CompanionSystemStub", ExTESExample], CompanionAgent]  # noqa: F821
AgentFactoryAnnoMI = Callable[["CompanionSystemStub", AnnoMIDialogue], CompanionAgent]  # noqa: F821
ExTESSeekerFactory = Callable[[ExTESExample], SimulatedUser]
AnnoMISeekerFactory = Callable[[AnnoMIDialogue], SimulatedUser]


def score_extes_sweep(
    examples: Iterable[ExTESExample],
    *,
    system_factory: Callable[[], object],
    agent_factory: Callable[[object, ExTESExample], CompanionAgent],
    seeker_factory: Callable[[ExTESExample], SimulatedUser],
    max_turns_per_dialog: int = 4,
    base_timestamp: Optional[datetime] = None,
    seconds_per_turn: float = 30.0,
    progress_sink: Optional[Callable[[int, ExTESDialogResult], None]] = None,
) -> ExTESReport:
    """Score a sweep of ExTES examples: Bond + lexical empathy for each.

    Fresh ``CompanionSystem`` + fresh ``CompanionAgent`` per example —
    same contract as Pillar B (no cross-dialog memory contamination).
    """
    base = base_timestamp or datetime.now(timezone.utc)
    per_dialog: list[ExTESDialogResult] = []
    for idx, example in enumerate(examples):
        system = system_factory()
        agent = agent_factory(system, example)
        seeker = seeker_factory(example)
        start = base + timedelta(days=idx)
        transcript = run_counseling_session(
            conversation_id=example.example_id,
            opening_utterance=example.client_utterance,
            companion_agent=agent,
            seeker=seeker,
            max_turns=max_turns_per_dialog,
            starting_timestamp=start,
            seconds_per_turn=seconds_per_turn,
        )
        companion_turns = tuple(
            t.utterance for t in transcript.turns if t.speaker == "companion"
        )
        companion_scores = [_lexical_empathy_score(t) for t in companion_turns]
        companion_mean_empathy = mean(companion_scores) if companion_scores else 0.0
        reference_empathy = _lexical_empathy_score(example.counselor_response)
        bond: Optional[float]
        if transcript.handoff_triggered or not companion_turns:
            bond = None
        else:
            admin_ts = start + timedelta(
                seconds=seconds_per_turn * max_turns_per_dialog + 60.0
            )
            try:
                rs = administer_wai_sr_simulated(
                    seeker,
                    user_id=f"extes-{example.example_id}",
                    administered_at_seconds=admin_ts.timestamp(),
                    subject_noun="Companion",
                )
                bond = score_wai_sr(rs).bond
            except Exception:
                # Seeker failed to produce a parsable Likert — exclude from
                # bond cohort, keep empathy result. Same discipline as MSC
                # harness: don't let one bad administration poison the sweep.
                bond = None
        result = ExTESDialogResult(
            example_id=example.example_id,
            bond=bond,
            handoff_triggered=transcript.handoff_triggered,
            companion_turns=companion_turns,
            reference_response=example.counselor_response,
            companion_empathy_score=companion_mean_empathy,
            reference_empathy_score=reference_empathy,
            companion_empathy_binary=_binarise_empathy(companion_mean_empathy),
            reference_empathy_binary=_binarise_empathy(reference_empathy),
        )
        per_dialog.append(result)
        if progress_sink is not None:
            progress_sink(idx, result)
    return ExTESReport(per_dialog=tuple(per_dialog))


def score_annomi_sweep(
    dialogues: Iterable[AnnoMIDialogue],
    *,
    system_factory: Callable[[], object],
    agent_factory: Callable[[object, AnnoMIDialogue], CompanionAgent],
    seeker_factory: Callable[[AnnoMIDialogue], SimulatedUser],
    max_turns_per_dialog: int = 6,
    base_timestamp: Optional[datetime] = None,
    seconds_per_turn: float = 30.0,
    progress_sink: Optional[Callable[[int, AnnoMIDialogResult], None]] = None,
) -> AnnoMIReport:
    """Score a sweep of AnnoMI dialogues: Bond + MI-fidelity for each.

    Fresh ``CompanionSystem`` + fresh ``CompanionAgent`` per dialogue.
    """
    base = base_timestamp or datetime.now(timezone.utc)
    per_dialog: list[AnnoMIDialogResult] = []
    for idx, dialogue in enumerate(dialogues):
        system = system_factory()
        agent = agent_factory(system, dialogue)
        seeker = seeker_factory(dialogue)
        start = base + timedelta(days=idx)
        first_client = dialogue.first_client_turn()
        if first_client is None:
            # Dialogue has no client utterance — skip (defensive).
            continue
        transcript = run_counseling_session(
            conversation_id=dialogue.transcript_id,
            opening_utterance=first_client.utterance_text,
            companion_agent=agent,
            seeker=seeker,
            max_turns=max_turns_per_dialog,
            starting_timestamp=start,
            seconds_per_turn=seconds_per_turn,
        )
        companion_turns = tuple(
            t.utterance for t in transcript.turns if t.speaker == "companion"
        )
        if companion_turns:
            labels = [_classify_companion_turn(t) for t in companion_turns]
            n = len(labels)
            companion_reflection_rate = sum(
                1 for l in labels if l == "reflection"
            ) / n
            companion_open_question_rate = sum(
                1 for l in labels if l == "open_question"
            ) / n
        else:
            companion_reflection_rate = 0.0
            companion_open_question_rate = 0.0
        bond: Optional[float]
        if transcript.handoff_triggered or not companion_turns:
            bond = None
        else:
            admin_ts = start + timedelta(
                seconds=seconds_per_turn * max_turns_per_dialog + 60.0
            )
            try:
                rs = administer_wai_sr_simulated(
                    seeker,
                    user_id=f"annomi-{dialogue.transcript_id}",
                    administered_at_seconds=admin_ts.timestamp(),
                    subject_noun="Companion",
                )
                bond = score_wai_sr(rs).bond
            except Exception:
                bond = None
        result = AnnoMIDialogResult(
            transcript_id=dialogue.transcript_id,
            mi_quality=dialogue.mi_quality,
            bond=bond,
            handoff_triggered=transcript.handoff_triggered,
            companion_turns=companion_turns,
            companion_reflection_rate=companion_reflection_rate,
            companion_open_question_rate=companion_open_question_rate,
            baseline_reflection_rate=therapist_reflection_rate(dialogue),
            baseline_open_question_rate=therapist_open_question_rate(dialogue),
        )
        per_dialog.append(result)
        if progress_sink is not None:
            progress_sink(idx, result)
    return AnnoMIReport(per_dialog=tuple(per_dialog))


__all__ = [
    "AnnoMIDialogResult",
    "AnnoMIReport",
    "CounselingDepthReport",
    "ExTESDialogResult",
    "ExTESReport",
    "make_annomi_seeker",
    "make_extes_seeker",
    "run_counseling_session",
    "score_annomi_sweep",
    "score_extes_sweep",
    # Primitives exported for tests + runners
    "_classify_companion_turn",
    "_cohens_kappa_binary",
    "_lexical_empathy_score",
]
