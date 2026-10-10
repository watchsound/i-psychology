"""EMPATH × MathDial gold scorer — κ, auto-compliance, and taxonomy overlap.

Exercises the three-floor pivot gate wiring with a scripted
:class:`EMPATHJudge`. No live LLM calls.
"""

from __future__ import annotations

import pytest

from psychology_base.applications.measurement.empath_judge import (
    EMPATHJudge,
    EMPATHSubCriterion,
    EMPATHVerdict,
)
from psychology_base.applications.measurement.socratic_compliance import ComplianceLabel
from psychology_base.eval.benchmarks.public_datasets.mathdial import (
    MathDialConversation,
    MathDialMove,
    MathDialTurn,
)
from psychology_base.eval.benchmarks.public_datasets.mathdial_empath_scorer import (
    EMPATHMathDialReport,
    gold_mathdial_to_compliance,
    score_empath_against_mathdial,
)


def _convo(turns: list[MathDialTurn], cid: str = "md_t") -> MathDialConversation:
    return MathDialConversation(
        conversation_id=cid,
        problem="Janet's ducks lay 16 eggs per day. She eats 3 and uses 4 for muffins.",
        ground_truth="16 - 3 - 4 = 9. The answer is 9.",
        student_incorrect_solution="16 * 2 = 32 dollars.",
        turns=tuple(turns),
    )


class ScriptedEMPATHJudge:
    """Returns pre-scripted verdicts per call. For wiring tests only."""

    def __init__(self, verdicts: list[EMPATHVerdict]) -> None:
        self._verdicts = list(verdicts)
        self.student_utterances: list[str] = []
        self.problem_contexts: list[str] = []

    def judge(
        self,
        tutor_utterance: str,
        *,
        problem_context: str = "",
        student_utterance: str = "",
    ) -> EMPATHVerdict:
        self.student_utterances.append(student_utterance)
        self.problem_contexts.append(problem_context)
        return self._verdicts.pop(0)


def _verdict(
    label: ComplianceLabel,
    *,
    proxy: bool = False,
    parse_failed: bool = False,
) -> EMPATHVerdict:
    """Build a verdict with sub-criteria that would deterministically fold to ``label``."""
    sc = {
        EMPATHSubCriterion.REQUESTS_REASONING: False,
        EMPATHSubCriterion.STATES_FINAL_ANSWER: False,
        EMPATHSubCriterion.PROVIDES_SCAFFOLD_HINT: False,
    }
    if label is ComplianceLabel.DIRECT_ANSWER:
        sc[EMPATHSubCriterion.STATES_FINAL_ANSWER] = True
    elif label is ComplianceLabel.SOCRATIC:
        sc[EMPATHSubCriterion.REQUESTS_REASONING] = True
    return EMPATHVerdict(
        label=label,
        sub_criteria=sc,
        rationale="t",
        proxy=proxy,
        parse_failed=parse_failed,
    )


# ---------------------------------------------------------------------------
# gold_mathdial_to_compliance
# ---------------------------------------------------------------------------


def test_gold_telling_maps_to_direct_answer() -> None:
    assert gold_mathdial_to_compliance(MathDialMove.TELLING) is ComplianceLabel.DIRECT_ANSWER


def test_gold_probing_focus_generic_all_map_to_socratic() -> None:
    for move in (MathDialMove.PROBING, MathDialMove.FOCUS, MathDialMove.GENERIC):
        assert gold_mathdial_to_compliance(move) is ComplianceLabel.SOCRATIC


# ---------------------------------------------------------------------------
# score_empath_against_mathdial — basic shape
# ---------------------------------------------------------------------------


def test_scorer_samples_only_labeled_teacher_turns() -> None:
    convo = _convo(
        [
            MathDialTurn(role="student", utterance="I got 32."),
            MathDialTurn(role="teacher", utterance="OK."),  # no move → skipped
            MathDialTurn(
                role="teacher",
                utterance="What do you notice about this?",
                move=MathDialMove.PROBING,
            ),
        ]
    )
    judge = ScriptedEMPATHJudge([_verdict(ComplianceLabel.SOCRATIC)])
    report = score_empath_against_mathdial([convo], judge)
    assert report.sampled_teacher_turns == 1
    assert report.auto_socratic_count == 1
    assert len(report.empath_verdicts) == 1


def test_scorer_passes_most_recent_student_utterance_to_judge() -> None:
    convo = _convo(
        [
            MathDialTurn(role="student", utterance="I got 32."),
            MathDialTurn(
                role="teacher",
                utterance="What do you notice?",
                move=MathDialMove.PROBING,
            ),
            MathDialTurn(role="student", utterance="Maybe 20?"),
            MathDialTurn(
                role="teacher",
                utterance="Why 20?",
                move=MathDialMove.PROBING,
            ),
        ]
    )
    judge = ScriptedEMPATHJudge(
        [_verdict(ComplianceLabel.SOCRATIC), _verdict(ComplianceLabel.SOCRATIC)]
    )
    score_empath_against_mathdial([convo], judge)
    assert judge.student_utterances == ["I got 32.", "Maybe 20?"]


def test_scorer_marks_all_proxy_true_only_when_every_verdict_is_proxy() -> None:
    convo = _convo(
        [
            MathDialTurn(
                role="teacher",
                utterance="What do you notice?",
                move=MathDialMove.PROBING,
            ),
            MathDialTurn(
                role="teacher",
                utterance="The answer is 9.",
                move=MathDialMove.TELLING,
            ),
        ]
    )
    all_proxy = ScriptedEMPATHJudge(
        [
            _verdict(ComplianceLabel.SOCRATIC, proxy=True),
            _verdict(ComplianceLabel.DIRECT_ANSWER, proxy=True),
        ]
    )
    report = score_empath_against_mathdial([convo], all_proxy)
    assert report.all_proxy_verdicts is True

    mixed = ScriptedEMPATHJudge(
        [
            _verdict(ComplianceLabel.SOCRATIC, proxy=True),
            _verdict(ComplianceLabel.DIRECT_ANSWER, proxy=False),
        ]
    )
    report = score_empath_against_mathdial([convo], mixed)
    assert report.all_proxy_verdicts is False


# ---------------------------------------------------------------------------
# κ computation — EMPATH vs gold, filtering BORDERLINE
# ---------------------------------------------------------------------------


def test_kappa_perfect_agreement_when_empath_matches_gold() -> None:
    convo = _convo(
        [
            MathDialTurn(
                role="teacher",
                utterance="What do you notice?",
                move=MathDialMove.PROBING,
            ),
            MathDialTurn(
                role="teacher",
                utterance="The answer is 9.",
                move=MathDialMove.TELLING,
            ),
            MathDialTurn(
                role="teacher",
                utterance="Focus on the eggs.",
                move=MathDialMove.FOCUS,
            ),
            MathDialTurn(
                role="teacher",
                utterance="The answer is 9.",
                move=MathDialMove.TELLING,
            ),
        ]
    )
    judge = ScriptedEMPATHJudge(
        [
            _verdict(ComplianceLabel.SOCRATIC),
            _verdict(ComplianceLabel.DIRECT_ANSWER),
            _verdict(ComplianceLabel.SOCRATIC),
            _verdict(ComplianceLabel.DIRECT_ANSWER),
        ]
    )
    report = score_empath_against_mathdial([convo], judge)
    kappa = report.empath_kappa()
    assert kappa.kappa == pytest.approx(1.0)
    assert kappa.n_items == 4


def test_kappa_filters_out_borderline_verdicts() -> None:
    convo = _convo(
        [
            MathDialTurn(
                role="teacher",
                utterance="x",
                move=MathDialMove.PROBING,
            ),
            MathDialTurn(
                role="teacher",
                utterance="x",
                move=MathDialMove.TELLING,
            ),
        ]
    )
    judge = ScriptedEMPATHJudge(
        [
            _verdict(ComplianceLabel.BORDERLINE),  # dropped
            _verdict(ComplianceLabel.DIRECT_ANSWER),  # kept, matches gold
        ]
    )
    report = score_empath_against_mathdial([convo], judge)
    kappa = report.empath_kappa()
    assert kappa.n_items == 1  # BORDERLINE dropped
    # When all remaining labels are identical, pe=1.0 and the fallback
    # branch returns κ=1.0 for po=1.0.
    assert kappa.kappa == pytest.approx(1.0)


def test_kappa_zero_denom_when_all_verdicts_borderline() -> None:
    convo = _convo(
        [
            MathDialTurn(role="teacher", utterance="x", move=MathDialMove.PROBING),
        ]
    )
    judge = ScriptedEMPATHJudge([_verdict(ComplianceLabel.BORDERLINE)])
    report = score_empath_against_mathdial([convo], judge)
    kappa = report.empath_kappa()
    assert kappa.n_items == 0
    assert kappa.kappa == 0.0


# ---------------------------------------------------------------------------
# auto_compliance_rate + taxonomy_overlap_rate
# ---------------------------------------------------------------------------


def test_auto_compliance_rate_excludes_borderline_from_denominator() -> None:
    """WORKED_STEP classifier output maps to BORDERLINE and drops out of the auto denom."""
    convo = _convo(
        [
            # Classifier → SOCRATIC_PROBE → SOCRATIC
            MathDialTurn(
                role="teacher",
                utterance="What do you notice?",
                move=MathDialMove.PROBING,
            ),
            # Classifier → FULL_ANSWER (declarative, >80 chars, no scaffold cue) → DIRECT_ANSWER
            MathDialTurn(
                role="teacher",
                utterance=(
                    "The derivative here is sinusoidal because cosine integrates to sine "
                    "and the chain rule yields exactly that composition result."
                ),
                move=MathDialMove.TELLING,
            ),
        ]
    )
    judge = ScriptedEMPATHJudge(
        [_verdict(ComplianceLabel.SOCRATIC), _verdict(ComplianceLabel.DIRECT_ANSWER)]
    )
    report = score_empath_against_mathdial([convo], judge)
    assert report.auto_socratic_count == 1
    assert report.auto_direct_answer_count == 1
    assert report.auto_compliance_rate() == pytest.approx(0.5)


def test_taxonomy_overlap_rate_matches_mathdial_alignment_math() -> None:
    convo = _convo(
        [
            MathDialTurn(
                role="teacher",
                utterance="What do you notice?",
                move=MathDialMove.PROBING,
            ),
            MathDialTurn(
                role="teacher",
                utterance="What do you notice?",
                move=MathDialMove.FOCUS,  # mismatch — classifier says PROBING
            ),
        ]
    )
    judge = ScriptedEMPATHJudge(
        [_verdict(ComplianceLabel.SOCRATIC), _verdict(ComplianceLabel.SOCRATIC)]
    )
    report = score_empath_against_mathdial([convo], judge)
    assert report.sampled_teacher_turns == 2
    assert report.taxonomy_matches == 1
    assert report.taxonomy_overlap_rate() == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# meets_pivot_gate — three-floor conjunction
# ---------------------------------------------------------------------------


def _well_shaped_corpus(n_pairs: int) -> list[MathDialConversation]:
    """Build ``2*n_pairs`` teacher turns alternating PROBING/TELLING, all
    classifier-shaped so lexical overlap + auto-compliance both come out high."""
    convos = []
    for i in range(n_pairs):
        convos.append(
            _convo(
                [
                    MathDialTurn(
                        role="teacher",
                        utterance=f"What do you notice about step {i}?",
                        move=MathDialMove.PROBING,
                    ),
                    MathDialTurn(
                        role="teacher",
                        utterance=(
                            "The derivative here is sinusoidal because cosine integrates "
                            "to sine and the chain rule yields exactly that result for "
                            "this composition."
                        ),
                        move=MathDialMove.TELLING,
                    ),
                ],
                cid=f"c{i}",
            )
        )
    return convos


def test_meets_pivot_gate_passes_when_all_three_floors_satisfied() -> None:
    convos = _well_shaped_corpus(50)  # 100 teacher turns total
    verdicts = []
    for _ in range(50):
        verdicts.append(_verdict(ComplianceLabel.SOCRATIC, proxy=False))
        verdicts.append(_verdict(ComplianceLabel.DIRECT_ANSWER, proxy=False))
    judge = ScriptedEMPATHJudge(verdicts)
    report = score_empath_against_mathdial(convos, judge)
    assert report.sampled_teacher_turns == 100
    assert report.taxonomy_overlap_rate() == pytest.approx(1.0)
    # auto side: with the FULL_ANSWER classifier output on the long declarative,
    # auto-compliance is 50/100 — tune the floor to pass this synthetic corpus.
    assert report.meets_pivot_gate(auto_compliance_floor=0.50) is True


def test_meets_pivot_gate_fails_when_all_verdicts_are_proxy() -> None:
    convos = _well_shaped_corpus(50)
    verdicts = []
    for _ in range(50):
        verdicts.append(_verdict(ComplianceLabel.SOCRATIC, proxy=True))
        verdicts.append(_verdict(ComplianceLabel.DIRECT_ANSWER, proxy=True))
    judge = ScriptedEMPATHJudge(verdicts)
    report = score_empath_against_mathdial(convos, judge)
    assert report.all_proxy_verdicts is True
    # Pivot gate refuses proxy regardless of numbers.
    assert report.meets_pivot_gate(auto_compliance_floor=0.0, kappa_floor=0.0) is False


def test_meets_pivot_gate_fails_below_min_turns() -> None:
    convos = _well_shaped_corpus(5)  # only 10 turns, below default 100
    judge = ScriptedEMPATHJudge(
        [_verdict(ComplianceLabel.SOCRATIC, proxy=False) for _ in range(10)]
    )
    report = score_empath_against_mathdial(convos, judge)
    assert report.meets_pivot_gate() is False
    # Lowering min_turns lets the taxonomy/κ floors drive the decision.
    assert report.meets_pivot_gate(min_turns=5, auto_compliance_floor=0.0, kappa_floor=-1.0) is True


# ---------------------------------------------------------------------------
# judge_parse_failures — guard against silent-zero κ denominator
# ---------------------------------------------------------------------------


def test_scorer_counts_parse_failed_verdicts() -> None:
    convos = _well_shaped_corpus(2)  # 4 teacher turns
    judge = ScriptedEMPATHJudge(
        [
            _verdict(ComplianceLabel.SOCRATIC, parse_failed=False),
            _verdict(ComplianceLabel.BORDERLINE, parse_failed=True),
            _verdict(ComplianceLabel.DIRECT_ANSWER, parse_failed=False),
            _verdict(ComplianceLabel.BORDERLINE, parse_failed=True),
        ]
    )
    report = score_empath_against_mathdial(convos, judge)
    assert report.judge_parse_failures == 2
    assert report.judge_failure_rate() == pytest.approx(0.5)


def test_judge_failure_rate_zero_when_no_turns_sampled() -> None:
    report = EMPATHMathDialReport(
        sampled_teacher_turns=0,
        taxonomy_matches=0,
        auto_socratic_count=0,
        auto_direct_answer_count=0,
        empath_verdicts=(),
        empath_gold_pairs=(),
        all_proxy_verdicts=True,
    )
    assert report.judge_failure_rate() == 0.0


def test_judge_parse_failures_defaults_to_zero_for_existing_callsites() -> None:
    """Dataclass default keeps every pre-flag construction site safe."""
    report = EMPATHMathDialReport(
        sampled_teacher_turns=1,
        taxonomy_matches=1,
        auto_socratic_count=1,
        auto_direct_answer_count=0,
        empath_verdicts=(_verdict(ComplianceLabel.SOCRATIC),),
        empath_gold_pairs=((ComplianceLabel.SOCRATIC, ComplianceLabel.SOCRATIC),),
        all_proxy_verdicts=False,
    )
    assert report.judge_parse_failures == 0
    assert report.judge_failure_rate() == 0.0


def test_meets_pivot_gate_fails_when_judge_failure_rate_exceeds_floor() -> None:
    """100 turns, 50 parse-failed (0.50 failure rate) → gate fails even if κ clears."""
    convos = _well_shaped_corpus(50)
    verdicts = []
    for i in range(50):
        # First turn in each convo: PROBING, judge produces SOCRATIC cleanly.
        verdicts.append(_verdict(ComplianceLabel.SOCRATIC, proxy=False))
        # Second turn: TELLING, but judge "parse-failed" → BORDERLINE.
        verdicts.append(
            _verdict(ComplianceLabel.BORDERLINE, proxy=False, parse_failed=True)
        )
    judge = ScriptedEMPATHJudge(verdicts)
    report = score_empath_against_mathdial(convos, judge)
    assert report.judge_parse_failures == 50
    assert report.judge_failure_rate() == pytest.approx(0.5)
    # Even with every surviving κ pair perfect, the parse-failure rate alone
    # must veto the gate.
    assert (
        report.meets_pivot_gate(
            auto_compliance_floor=0.0,
            taxonomy_overlap_floor=0.0,
            kappa_floor=0.0,
            min_turns=50,
        )
        is False
    )
    # Loosening the parse-failure floor past 0.5 flips it back to True.
    assert (
        report.meets_pivot_gate(
            auto_compliance_floor=0.0,
            taxonomy_overlap_floor=0.0,
            kappa_floor=0.0,
            min_turns=50,
            max_judge_failure_rate=0.6,
        )
        is True
    )


def test_meets_pivot_gate_default_floor_is_twenty_percent() -> None:
    """A judge failing exactly 25% of turns breaches the default 0.20 floor."""
    convos = _well_shaped_corpus(50)  # 100 turns
    verdicts = []
    for i in range(50):
        # Pair 1: both OK.
        verdicts.append(_verdict(ComplianceLabel.SOCRATIC, proxy=False))
        verdicts.append(_verdict(ComplianceLabel.DIRECT_ANSWER, proxy=False))
    # Mutate 25% into parse_failed BORDERLINE: replace indices 0..24 of the
    # DIRECT_ANSWER slot (every other verdict is DIRECT_ANSWER).
    for idx in range(1, 50, 2):  # 25 indices
        verdicts[idx] = _verdict(
            ComplianceLabel.BORDERLINE, proxy=False, parse_failed=True
        )
    judge = ScriptedEMPATHJudge(verdicts)
    report = score_empath_against_mathdial(convos, judge)
    assert report.judge_failure_rate() == pytest.approx(0.25)
    # Default max_judge_failure_rate=0.20 → 0.25 > 0.20 breaches.
    assert (
        report.meets_pivot_gate(
            auto_compliance_floor=0.0,
            taxonomy_overlap_floor=0.0,
            kappa_floor=0.0,
            min_turns=50,
        )
        is False
    )
