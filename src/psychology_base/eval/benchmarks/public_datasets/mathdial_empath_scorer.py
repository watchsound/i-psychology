"""EMPATH judge × MathDial gold — the κ ≥ 0.60 floor from the pivot gate.

Scores a batch of MathDial teacher turns through an :class:`EMPATHJudge`
and compares the aggregated labels against the gold MathDial moves
(TELLING → DIRECT_ANSWER, non-TELLING → SOCRATIC). Produces the full
pivot-gate triple:

- auto-compliance rate (lexical classifier)
- MathDial-taxonomy overlap rate (lexical classifier vs gold)
- EMPATH κ against gold

**BORDERLINE handling.** EMPATH BORDERLINE verdicts drop out of the κ
denominator — a judge that cannot decide should not vote. The gold side
is forced to SOCRATIC/DIRECT_ANSWER by the MathDial 4-move taxonomy, so
κ is computed over the subset of turns where EMPATH gave a non-BORDERLINE
verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from psychology_base.applications.learning.tutor_agent import classify_tutor_move
from psychology_base.applications.measurement.empath_judge import (
    EMPATHJudge,
    EMPATHVerdict,
)
from psychology_base.applications.measurement.scaffolding_telemetry import TutorMove
from psychology_base.applications.measurement.socratic_compliance import (
    ComplianceLabel,
    InterAnnotatorAgreement,
    auto_label_from_tutor_move,
    cohen_kappa,
)
from psychology_base.eval.benchmarks.public_datasets.mathdial import (
    MATHDIAL_TO_TUTOR_MOVE,
    MathDialConversation,
    MathDialMove,
    MathDialTurn,
)


def gold_mathdial_to_compliance(move: MathDialMove) -> ComplianceLabel:
    """Map a gold MathDial 4-move label to the Socratic/direct-answer binary.

    TELLING → DIRECT_ANSWER (whether surrender- or scaffold-shaped is a
    psychology-base-side distinction; from MathDial's POV it is "telling").
    Everything else → SOCRATIC. BORDERLINE never appears on the gold side.
    """
    if move is MathDialMove.TELLING:
        return ComplianceLabel.DIRECT_ANSWER
    return ComplianceLabel.SOCRATIC


@dataclass(frozen=True)
class EMPATHMathDialReport:
    """Full three-floor pivot-gate report over a MathDial held-out sample."""

    sampled_teacher_turns: int

    # Lexical / taxonomy numbers (duplicate the MathDialAlignmentReport math
    # so the three pivot-gate floors sit on one object).
    taxonomy_matches: int

    # Auto-compliance (SOCRATIC rate per the lexical classifier → compliance
    # label chain).
    auto_socratic_count: int
    auto_direct_answer_count: int

    # EMPATH judge labels (same per-turn length, aligned to sampled_teacher_turns).
    empath_verdicts: tuple[EMPATHVerdict, ...]
    empath_gold_pairs: tuple[tuple[ComplianceLabel, ComplianceLabel], ...]
    """(empath_label, gold_label) pairs — BORDERLINE entries kept so the
    reader can count them; the κ computation filters them out."""

    all_proxy_verdicts: bool
    """True iff every EMPATH verdict came from the stub judge. The pivot
    gate requires a non-proxy run — report this up front for readers."""

    judge_parse_failures: int = 0
    """Count of LLM-judge turns whose JSON response could not be parsed
    and therefore defaulted to all-False sub-criteria (aggregating to
    BORDERLINE and dropping from the κ denominator). A judge whose JSON
    parser silently zeroes the denominator must not be read as a judge
    whose decisions are legitimately BORDERLINE — the gate checks this
    rate against ``max_judge_failure_rate`` so a parse-collapsed run
    does not masquerade as a passing κ=0.0 / n=0 result."""

    def taxonomy_overlap_rate(self) -> float:
        if self.sampled_teacher_turns == 0:
            return 0.0
        return self.taxonomy_matches / self.sampled_teacher_turns

    def auto_compliance_rate(self) -> float:
        denom = self.auto_socratic_count + self.auto_direct_answer_count
        if denom == 0:
            return 0.0
        return self.auto_socratic_count / denom

    def judge_failure_rate(self) -> float:
        """Fraction of sampled teacher turns whose judge response failed
        to parse into clean JSON. 0.0 when no turns were sampled."""
        if self.sampled_teacher_turns == 0:
            return 0.0
        return self.judge_parse_failures / self.sampled_teacher_turns

    def empath_kappa(self) -> InterAnnotatorAgreement:
        """Cohen's κ between EMPATH and gold, over non-BORDERLINE EMPATH turns.

        If every EMPATH verdict was BORDERLINE, returns κ=0.0 with n=0 —
        the caller should treat that as "judge produced no decisions"
        and surface the judge-failure count, not misread it as "κ=0".
        """
        filtered: list[tuple[ComplianceLabel, ComplianceLabel]] = [
            (e, g) for e, g in self.empath_gold_pairs
            if e is not ComplianceLabel.BORDERLINE
        ]
        if not filtered:
            return InterAnnotatorAgreement(kappa=0.0, n_items=0, po=0.0, pe=0.0)
        return cohen_kappa(
            [e for e, _ in filtered],
            [g for _, g in filtered],
        )

    def meets_pivot_gate(
        self,
        *,
        auto_compliance_floor: float = 0.95,
        taxonomy_overlap_floor: float = 0.90,
        kappa_floor: float = 0.60,
        min_turns: int = 100,
        max_judge_failure_rate: float = 0.20,
    ) -> bool:
        """Pivot §3 target: auto ≥95% + MathDial overlap ≥90% + EMPATH κ≥0.60.

        ``min_turns`` is the ≥100-turn held-out sample size the pivot
        requires. Non-proxy judge runs are a prerequisite — a report
        where every verdict is proxy returns False regardless of numbers.
        ``max_judge_failure_rate`` guards the silent-zero case: a run
        where the LLM-judge's JSON parser failed on more than 20% of
        turns is a judge failure, not a judge decision, and must not
        pass the gate even if the surviving κ clears the floor.
        """
        if self.all_proxy_verdicts:
            return False
        if self.sampled_teacher_turns < min_turns:
            return False
        if self.judge_failure_rate() > max_judge_failure_rate:
            return False
        if self.auto_compliance_rate() < auto_compliance_floor:
            return False
        if self.taxonomy_overlap_rate() < taxonomy_overlap_floor:
            return False
        return self.empath_kappa().kappa >= kappa_floor


def score_empath_against_mathdial(
    conversations: Sequence[MathDialConversation],
    judge: EMPATHJudge,
) -> EMPATHMathDialReport:
    """Pool EMPATH judge labels over the teacher turns of ``conversations``.

    Student turns and teacher turns without a gold ``move`` are skipped.
    The classifier is applied to each surviving teacher turn to produce
    the lexical label; the EMPATH judge is applied in parallel to produce
    the judge label. Both are compared against the gold MathDial move.
    """
    sampled = 0
    taxonomy_matches = 0
    auto_socratic = 0
    auto_direct = 0
    verdicts: list[EMPATHVerdict] = []
    empath_gold_pairs: list[tuple[ComplianceLabel, ComplianceLabel]] = []
    all_proxy = True
    parse_failures = 0

    for convo in conversations:
        # Student utterance paired with each teacher turn — the most-
        # recent student turn before the teacher turn. Walked inline so a
        # fresh teacher turn gets the right prior student context.
        last_student_utterance = ""
        for turn in convo.turns:
            if turn.role == "student":
                last_student_utterance = turn.utterance
                continue
            if turn.move is None:
                continue
            sampled += 1

            # Lexical taxonomy overlap.
            predicted_tutor_move = classify_tutor_move(turn.utterance)
            predicted_mathdial = _collapse_to_mathdial(predicted_tutor_move)
            if predicted_mathdial is turn.move:
                taxonomy_matches += 1

            # Auto-compliance (classifier → compliance label).
            auto_label = auto_label_from_tutor_move(predicted_tutor_move)
            if auto_label is ComplianceLabel.SOCRATIC:
                auto_socratic += 1
            elif auto_label is ComplianceLabel.DIRECT_ANSWER:
                auto_direct += 1
            # BORDERLINE (WORKED_STEP) drops out of auto denominator —
            # same convention `ComplianceReport.compliance_rate` uses.

            # EMPATH verdict.
            verdict = judge.judge(
                turn.utterance,
                problem_context=convo.problem,
                student_utterance=last_student_utterance,
            )
            verdicts.append(verdict)
            if not verdict.proxy:
                all_proxy = False
            if verdict.parse_failed:
                parse_failures += 1
            empath_gold_pairs.append(
                (verdict.label, gold_mathdial_to_compliance(turn.move))
            )

    return EMPATHMathDialReport(
        sampled_teacher_turns=sampled,
        taxonomy_matches=taxonomy_matches,
        auto_socratic_count=auto_socratic,
        auto_direct_answer_count=auto_direct,
        empath_verdicts=tuple(verdicts),
        empath_gold_pairs=tuple(empath_gold_pairs),
        all_proxy_verdicts=all_proxy if verdicts else True,
        judge_parse_failures=parse_failures,
    )


def _collapse_to_mathdial(move: TutorMove) -> MathDialMove:
    """Collapse our 5-enum tutor move to the gold 4-enum MathDial move.

    Deliberately duplicated from `TUTOR_MOVE_TO_MATHDIAL` to keep this
    scorer independent of mapping-table re-exports.
    """
    from psychology_base.eval.benchmarks.public_datasets.mathdial import (
        TUTOR_MOVE_TO_MATHDIAL,
    )
    return TUTOR_MOVE_TO_MATHDIAL[move]


__all__ = [
    "EMPATHMathDialReport",
    "gold_mathdial_to_compliance",
    "score_empath_against_mathdial",
]
