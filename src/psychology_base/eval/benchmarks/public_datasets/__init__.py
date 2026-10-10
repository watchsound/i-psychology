"""Public-dataset adapters for Phase 4b benchmark closure.

Added under the 2026-10-07 benchmark pivot (see
`docs/phases/phase-4b-benchmark-pivot.md`). Each submodule wraps a public
dataset — license-respecting, verify-before-cite — into a representation that
plugs into the existing measurement + agent harness without pulling the
dataset payloads into the repo.

Design notes:

- Datasets are **not vendored**. Each loader expects a user-provided JSONL
  shim path. A shim is produced from the dataset's native format by a small
  conversion script (TBD per dataset) so the loader stays size-agnostic and
  license-neutral. Tests use in-memory fixtures.
- Each adapter targets ONE benchmark-pivot gate from pivot §3. The gate is
  stated at the top of each submodule so later readers see what it is for
  without reading the pivot doc.
- Taxonomy alignment is explicit — where our internal taxonomy differs from
  the dataset's native taxonomy, a bidirectional mapping is published in
  the submodule and used by every scorer in that submodule.
"""

from psychology_base.eval.benchmarks.public_datasets.fantom import (
    FANToMContextTurn,
    FANToMConversation,
    FANToMItem,
    FANToMQuestion,
    FANToMQuestionType,
    iter_fantom_conversations,
    iter_fantom_items,
)
from psychology_base.eval.benchmarks.public_datasets.fantom_tom_scorer import (
    AnswerJudge,
    AnswerLabel,
    FANToMToMReport,
    FANToMVerdict,
    LLMFANToMJudge,
    Responder,
    lexical_answer_judge,
    score_fantom_tom,
)
from psychology_base.eval.benchmarks.public_datasets.esconv import (
    ESConvConversation,
    ESConvStrategy,
    ESConvTurn,
    StrategyDistributionReport,
    iter_esconv_conversations,
    seeker_opening_utterance,
    summarise_strategy_distribution,
)
from psychology_base.eval.benchmarks.public_datasets.esconv_wai_sr_harness import (
    CohortBondReport,
    SessionTranscript,
    SessionTurn,
    WAISRParseError,
    administer_wai_sr_simulated,
    aggregate_cohort_bond,
    make_esconv_seeker,
    run_esconv_session,
)
from psychology_base.eval.benchmarks.public_datasets.mathdial import (
    MATHDIAL_TO_TUTOR_MOVE,
    MathDialAlignmentReport,
    MathDialConversation,
    MathDialMove,
    MathDialTurn,
    TUTOR_MOVE_TO_MATHDIAL,
    iter_mathdial_conversations,
    score_tutor_against_mathdial,
)
from psychology_base.eval.benchmarks.public_datasets.mathdial_empath_scorer import (
    EMPATHMathDialReport,
    gold_mathdial_to_compliance,
    score_empath_against_mathdial,
)
from psychology_base.eval.benchmarks.public_datasets.mathdial_fade_harness import (
    FadeScheduleReport,
    MathDialSessionResult,
    aggregate_fade_schedule,
    extract_final_number,
    make_mathdial_student,
    run_mathdial_tutor_session,
    student_solved_correctly,
)
from psychology_base.eval.benchmarks.public_datasets.sycophancy_eval import (
    ARE_YOU_SURE_CHALLENGE,
    SYCO_SUBDATASET_TO_LOADING,
    SycoAnswerBias,
    SycoFeedbackBias,
    SycoSubDataset,
    SycophancyEvalItem,
    TaxonomyOverlapReport,
    compute_taxonomy_overlap,
    iter_sycophancy_eval_items,
    sycophancy_probe_from_item,
)
from psychology_base.eval.benchmarks.public_datasets.sycophancy_eval_scorer import (
    SycophancyScoreResult,
    WithoutRegressionReport,
    check_sycophancy_without_regression,
    score_sycophancy_eval,
)

__all__ = [
    "ARE_YOU_SURE_CHALLENGE",
    "AnswerJudge",
    "AnswerLabel",
    "CohortBondReport",
    "EMPATHMathDialReport",
    "ESConvConversation",
    "ESConvStrategy",
    "ESConvTurn",
    "FANToMContextTurn",
    "FANToMConversation",
    "FANToMItem",
    "FANToMQuestion",
    "FANToMQuestionType",
    "FANToMToMReport",
    "FANToMVerdict",
    "LLMFANToMJudge",
    "FadeScheduleReport",
    "MATHDIAL_TO_TUTOR_MOVE",
    "MathDialAlignmentReport",
    "MathDialConversation",
    "MathDialMove",
    "MathDialSessionResult",
    "MathDialTurn",
    "Responder",
    "SYCO_SUBDATASET_TO_LOADING",
    "SessionTranscript",
    "SessionTurn",
    "StrategyDistributionReport",
    "SycoAnswerBias",
    "SycoFeedbackBias",
    "SycoSubDataset",
    "SycophancyEvalItem",
    "SycophancyScoreResult",
    "TUTOR_MOVE_TO_MATHDIAL",
    "TaxonomyOverlapReport",
    "WAISRParseError",
    "WithoutRegressionReport",
    "administer_wai_sr_simulated",
    "aggregate_cohort_bond",
    "aggregate_fade_schedule",
    "check_sycophancy_without_regression",
    "compute_taxonomy_overlap",
    "extract_final_number",
    "gold_mathdial_to_compliance",
    "iter_esconv_conversations",
    "iter_fantom_conversations",
    "iter_fantom_items",
    "iter_mathdial_conversations",
    "iter_sycophancy_eval_items",
    "lexical_answer_judge",
    "make_esconv_seeker",
    "make_mathdial_student",
    "run_esconv_session",
    "run_mathdial_tutor_session",
    "score_empath_against_mathdial",
    "score_fantom_tom",
    "score_sycophancy_eval",
    "score_tutor_against_mathdial",
    "seeker_opening_utterance",
    "student_solved_correctly",
    "summarise_strategy_distribution",
    "sycophancy_probe_from_item",
]
