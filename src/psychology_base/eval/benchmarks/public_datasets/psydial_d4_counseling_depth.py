"""Phase 8 Pillar F — PsyDial-D4 Chinese-register counseling-depth sub-gate.

**Why this file (not direct reuse of psyqa_counseling_depth).** The
Chinese PsyQA scorer is dataset-polymorphic in machinery
(:func:`run_counseling_session`, :func:`_cohens_kappa_binary`,
:func:`administer_wai_sr_simulated`, :func:`parse_empathy_judgment`
all accept arbitrary CN counseling records), but its public surface is
typed on :class:`PsyQAExample`. Shipping a thin typed sibling here
keeps the PsyDial-D4 entry-points honest about which corpus they
target — important given the honest-labeling invariant on this
sub-corpus (counseling-TRAINING derivative, NOT the paper D4 at
X-LANCE SJTU EMNLP'22).

**Shared machinery (imported, not re-implemented).**
- Chinese judge prompt + :func:`make_llm_empathy_judge_zh` from
  :mod:`psyqa_counseling_depth` — the CN empathy judge is register-
  neutral across CN counseling corpora (same anchors, same ASCII
  response contract).
- Dialog loop + κ-across-conversants calc from
  :mod:`counseling_depth_scorer`.

**Gate contract (same as PsyQA — Pillar F plan).**

- PsyDial-D4 Bond ≥ 3.89 (Woebot aggregate floor)
- PsyDial-D4 EMPATH-κ ≥ 0.50 (κ-across-conversants under the CN LLM
  judge, Companion-side binary vs reference-side binary on the
  ``counselor_response``)

**Honest architectural prediction.** PsyQA refs rated 0.22 empathic
under the CN judge (real online counselors skew advice-heavy).
PsyDial-D4 refs are curated AI-counselor-training targets — the
training set is designed to model empathic reflection. Prediction:
PsyDial-D4 ref rate may be higher (0.4–0.8) than PsyQA (0.22), which
would narrow the Companion-vs-reference marginal gap and could
either (a) clear κ ≥ 0.50 (first corpus to clear) or (b) still fail
by independence (same architectural signature, just at a different
baseline). Either outcome is informative.

**NOT A κ-vs-lexical GATE.** Same contract as PsyQA LLM judge — the
gate is κ-across-conversants under one judge; it either clears 0.50
or reports an honest architectural miss.

**Nothing modified in:**
- :mod:`counseling_depth_scorer` (ExTES/AnnoMI lexical scorer — audit baseline)
- :mod:`counseling_depth_llm_judge` (ExTES LLM scorer — English register)
- :mod:`psyqa_counseling_depth` (PsyQA CN scorer — audit baseline for Chinese register)
- :mod:`psydial_d4` (adapter — flat loader, no LLM dependency)
"""

from __future__ import annotations

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
from psychology_base.eval.benchmarks.public_datasets.counseling_depth_scorer import (
    _cohens_kappa_binary,
    _lexical_empathy_score,
    run_counseling_session,
)
from psychology_base.eval.benchmarks.public_datasets.counseling_depth_llm_judge import (
    EmpathyJudge,
    ParsedEmpathyJudgment,
    parse_empathy_judgment,
)
from psychology_base.eval.benchmarks.public_datasets.esconv_wai_sr_harness import (
    administer_wai_sr_simulated,
)
from psychology_base.eval.benchmarks.public_datasets.psydial_d4 import (
    PsyDialD4Example,
)
from psychology_base.eval.benchmarks.public_datasets.psyqa_counseling_depth import (
    _PSYQA_SEEKER_PERSONA_ZH,
    build_empathy_prompt_zh,
    make_llm_empathy_judge_zh,
)
from psychology_base.llm.router import LLMRouter
from psychology_base.llm.types import ChatMessage, TaskTier


# ---------------------------------------------------------------------------
# Gate floors (same as PsyQA/ExTES)
# ---------------------------------------------------------------------------


_WOEBOT_BOND_FLOOR: float = 3.89
_EMPATH_KAPPA_FLOOR: float = 0.50


# ---------------------------------------------------------------------------
# PsyDial-D4 seeker persona (CN) — same register as PsyQA seeker. The
# dialog opening differs by data source (PsyQA opens with a question
# post; PsyDial-D4 opens with the first substantive conversational
# disclosure), but the help-seeker's position + voice are identical.
# ---------------------------------------------------------------------------


_PSYDIAL_D4_SEEKER_PERSONA_ZH: str = _PSYQA_SEEKER_PERSONA_ZH


def make_psydial_d4_seeker(
    router: LLMRouter,
    example: PsyDialD4Example,
    *,
    persona_prompt: str = _PSYDIAL_D4_SEEKER_PERSONA_ZH,
    tier: TaskTier = TaskTier.SIMPLE,
    max_tokens: Optional[int] = 200,
    temperature: Optional[float] = 0.7,
) -> SimulatedUser:
    """Build a :class:`SimulatedUser` whose first line is the PsyDial-D4
    first substantive user disclosure.

    Mirror of :func:`make_psyqa_seeker` with the PsyDial-D4 adapter
    type; the scenario prompt notes the multi-turn conversational
    context explicitly (PsyDial-D4 dialogs are multi-turn; PsyQA is
    one-shot question-answer).
    """
    opener = example.seeker_opening.strip()
    if not opener:
        raise ValueError(
            f"PsyDial-D4 example {example.example_id!r} has no seeker opening"
        )
    scenario = (
        "你正在用中文向陪伴AI进行一次口语化的对话咨询。保持第一人称、"
        "保持当事人视角，不要切换到咨询师角度。"
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
# Per-dialog + aggregate report shapes (parallel to LLMPsyQAReport)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LLMPsyDialD4DialogResult:
    """One PsyDial-D4 example scored end-to-end via the CN LLM semantic judge."""

    example_id: str
    bond: Optional[float]
    handoff_triggered: bool
    companion_turns: tuple[str, ...]
    reference_response: str
    companion_empathy_scores: tuple[float, ...]
    companion_empathy_binary_rate: float
    reference_empathy_score: float
    reference_empathy_binary: int
    companion_empathy_binary: int
    unresolved: bool


@dataclass(frozen=True)
class LLMPsyDialD4Report:
    """Aggregate report over the PsyDial-D4 LLM-judge scoring sweep."""

    per_dialog: tuple[LLMPsyDialD4DialogResult, ...]
    judge_routing: str = ""
    n_unresolved: int = 0
    unresolved_dialog_ids: tuple[str, ...] = ()

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
    def resolved_dialogs(self) -> tuple[LLMPsyDialD4DialogResult, ...]:
        return tuple(d for d in self.per_dialog if not d.unresolved)

    @property
    def empath_kappa(self) -> float:
        resolved = self.resolved_dialogs
        if not resolved:
            return 0.0
        companion_labels = [d.companion_empathy_binary for d in resolved]
        reference_labels = [d.reference_empathy_binary for d in resolved]
        return _cohens_kappa_binary(companion_labels, reference_labels)

    @property
    def mean_llm_judge_companion_empathy_binary_rate(self) -> float:
        if not self.per_dialog:
            return 0.0
        return mean(d.companion_empathy_binary_rate for d in self.per_dialog)

    @property
    def mean_llm_judge_reference_empathy_binary_rate(self) -> float:
        if not self.per_dialog:
            return 0.0
        return mean(d.reference_empathy_binary for d in self.per_dialog)

    @property
    def excluded_handoffs(self) -> int:
        return sum(1 for d in self.per_dialog if d.handoff_triggered)

    def meets_bond_gate(self, floor: float = _WOEBOT_BOND_FLOOR) -> bool:
        return self.n > 0 and self.mean_bond >= floor

    def meets_kappa_gate(self, floor: float = _EMPATH_KAPPA_FLOOR) -> bool:
        return self.n > 0 and self.empath_kappa >= floor


# ---------------------------------------------------------------------------
# Scorer entry-point (parallel surface to score_psyqa_sweep_llm /
# score_extes_sweep_llm — same dialog shape, same gate semantics)
# ---------------------------------------------------------------------------


def score_psydial_d4_sweep_llm(
    examples: Iterable[PsyDialD4Example],
    *,
    system_factory: Callable[[], object],
    agent_factory: Callable[[object, PsyDialD4Example], CompanionAgent],
    seeker_factory: Callable[[PsyDialD4Example], SimulatedUser],
    judge: EmpathyJudge,
    max_turns_per_dialog: int = 4,
    base_timestamp: Optional[datetime] = None,
    seconds_per_turn: float = 30.0,
    judge_routing: str = "",
    progress_sink: Optional[Callable[[int, LLMPsyDialD4DialogResult], None]] = None,
) -> LLMPsyDialD4Report:
    """Score a PsyDial-D4 sweep: Bond + CN LLM-judge empathy across
    Companion + Reference.

    Fresh ``CompanionSystem`` + fresh ``CompanionAgent`` per example —
    same contract as the PsyQA scorer. The judge is called with the
    seeker's immediately-preceding utterance as context so semantic
    rating happens *in situ*.
    """
    base = base_timestamp or datetime.now(timezone.utc)
    per_dialog: list[LLMPsyDialD4DialogResult] = []
    unresolved_ids: list[str] = []
    for idx, example in enumerate(examples):
        system = system_factory()
        agent = agent_factory(system, example)
        seeker = seeker_factory(example)
        start = base + timedelta(days=idx)
        transcript = run_counseling_session(
            conversation_id=example.example_id,
            opening_utterance=example.seeker_opening,
            companion_agent=agent,
            seeker=seeker,
            max_turns=max_turns_per_dialog,
            starting_timestamp=start,
            seconds_per_turn=seconds_per_turn,
        )

        turns = transcript.turns
        companion_turns: list[str] = []
        companion_scores: list[float] = []
        companion_binaries: list[int] = []
        all_unresolved_turns = True
        last_seeker: Optional[str] = None
        for t in turns:
            if t.speaker == "seeker":
                last_seeker = t.utterance
                continue
            companion_turns.append(t.utterance)
            context = [last_seeker] if last_seeker else []
            judgment = judge(t.utterance, context)
            if not judgment.unresolved:
                all_unresolved_turns = False
            if judgment.label == "yes":
                cont = judgment.confidence
            elif judgment.label == "no":
                cont = 1.0 - judgment.confidence
            else:
                cont = 0.5
            companion_scores.append(cont)
            companion_binaries.append(1 if cont >= 0.5 else 0)

        if companion_turns:
            companion_mean_rate = mean(companion_binaries)
        else:
            companion_mean_rate = 0.0
            all_unresolved_turns = True
        companion_dialog_binary = 1 if companion_mean_rate >= 0.5 else 0

        reference_judgment = judge(
            example.counselor_response, [example.seeker_opening]
        )
        if reference_judgment.label == "yes":
            reference_score = reference_judgment.confidence
        elif reference_judgment.label == "no":
            reference_score = 1.0 - reference_judgment.confidence
        else:
            reference_score = 0.5
        reference_binary = 1 if reference_score >= 0.5 else 0

        dialog_unresolved = all_unresolved_turns and reference_judgment.unresolved
        if dialog_unresolved:
            unresolved_ids.append(example.example_id)

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
                    user_id=f"psydial-d4-llm-{example.example_id}",
                    administered_at_seconds=admin_ts.timestamp(),
                    subject_noun="陪伴AI",
                )
                bond = score_wai_sr(rs).bond
            except Exception:
                bond = None

        result = LLMPsyDialD4DialogResult(
            example_id=example.example_id,
            bond=bond,
            handoff_triggered=transcript.handoff_triggered,
            companion_turns=tuple(companion_turns),
            reference_response=example.counselor_response,
            companion_empathy_scores=tuple(companion_scores),
            companion_empathy_binary_rate=companion_mean_rate,
            reference_empathy_score=reference_score,
            reference_empathy_binary=reference_binary,
            companion_empathy_binary=companion_dialog_binary,
            unresolved=dialog_unresolved,
        )
        per_dialog.append(result)
        if progress_sink is not None:
            progress_sink(idx, result)

    return LLMPsyDialD4Report(
        per_dialog=tuple(per_dialog),
        judge_routing=judge_routing,
        n_unresolved=len(unresolved_ids),
        unresolved_dialog_ids=tuple(unresolved_ids),
    )


# ---------------------------------------------------------------------------
# Judge-vs-lexical κ diagnostic (NOT a halt gate — reported only)
# ---------------------------------------------------------------------------


def judge_vs_lexical_kappa_psydial_d4(
    report: LLMPsyDialD4Report,
    examples: Sequence[PsyDialD4Example],
) -> float:
    """Diagnostic κ comparing CN LLM-judge binary labels vs lexical binary labels.

    The lexical scorer is English-tuned, so on Chinese PsyDial-D4
    companion turns it is expected to fire poorly — the diagnostic is
    kept for symmetry with the PsyQA / ExTES surface but is NOT a gate.
    """
    by_id = {d.example_id: d for d in report.per_dialog}
    judge_labels: list[int] = []
    lex_labels: list[int] = []
    for ex in examples:
        d = by_id.get(ex.example_id)
        if d is None or d.unresolved:
            continue
        if not d.companion_turns:
            continue
        lex_scores = [_lexical_empathy_score(t) for t in d.companion_turns]
        lex_mean = mean(lex_scores)
        lex_binary = 1 if lex_mean >= (1.0 / 3.0) else 0
        judge_labels.append(d.companion_empathy_binary)
        lex_labels.append(lex_binary)
    if not judge_labels:
        return 0.0
    return _cohens_kappa_binary(judge_labels, lex_labels)


__all__ = [
    "LLMPsyDialD4DialogResult",
    "LLMPsyDialD4Report",
    "judge_vs_lexical_kappa_psydial_d4",
    "make_psydial_d4_seeker",
    "score_psydial_d4_sweep_llm",
]
