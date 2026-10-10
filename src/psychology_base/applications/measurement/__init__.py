"""Phase 4 measurement infrastructure.

Per docs/phases/phase-4-application-pilot-plan.md §6c:
    wai_sr               — Working Alliance Inventory Short Revised (5-item)
    scaffolding_telemetry — CoMeT 4-metric set for 学习陪伴
    socratic_compliance   — EMPATH-style turn-level labeling

Each module is pilot-shape-independent (works for P1/P2/P3/P4 unchanged).
"""

from psychology_base.applications.measurement.chain_reference_audit import (
    ChainReferenceLabel,
    ChainReferenceReport,
    ChainReferenceTurn,
    ReinterpretationAct,
    chain_reference_turn_to_argilla,
    kappa_on_chain_reference,
)
from psychology_base.applications.measurement.empath_judge import (
    EMPATHJudge,
    EMPATHSubCriterion,
    EMPATHVerdict,
    LLMEMPATHJudge,
    StubEMPATHJudge,
    aggregate_sub_criteria,
)
from psychology_base.applications.measurement.socratic_compliance import (
    ArgillaRecord,
    ComplianceLabel,
    ComplianceReport,
    InterAnnotatorAgreement,
    LabeledTurn,
    auto_label_from_tutor_move,
    cohen_kappa,
    labeled_turn_to_argilla,
    summarise_user,
    write_argilla_jsonl,
)
from psychology_base.applications.measurement.scaffolding_telemetry import (
    CohortMetrics,
    FadeEvent,
    ScaffoldingTelemetry,
    SessionSummary,
    TutorMove,
    TutorTurn,
    pool_cohort,
)
from psychology_base.applications.measurement.wai_sr import (
    WAI_SR_ITEMS,
    WAI_SR_SUBSCALES,
    WAISRResponse,
    WAISRResponseSet,
    administer_wai_sr_cli,
    score_wai_sr,
    WAISRScores,
)

__all__ = [
    "WAI_SR_ITEMS",
    "WAI_SR_SUBSCALES",
    "WAISRResponse",
    "WAISRResponseSet",
    "WAISRScores",
    "administer_wai_sr_cli",
    "score_wai_sr",
    "CohortMetrics",
    "FadeEvent",
    "ScaffoldingTelemetry",
    "SessionSummary",
    "TutorMove",
    "TutorTurn",
    "pool_cohort",
    "ArgillaRecord",
    "ComplianceLabel",
    "ComplianceReport",
    "InterAnnotatorAgreement",
    "LabeledTurn",
    "auto_label_from_tutor_move",
    "cohen_kappa",
    "labeled_turn_to_argilla",
    "summarise_user",
    "write_argilla_jsonl",
    "ChainReferenceLabel",
    "ChainReferenceReport",
    "ChainReferenceTurn",
    "ReinterpretationAct",
    "chain_reference_turn_to_argilla",
    "kappa_on_chain_reference",
    "EMPATHJudge",
    "EMPATHSubCriterion",
    "EMPATHVerdict",
    "LLMEMPATHJudge",
    "StubEMPATHJudge",
    "aggregate_sub_criteria",
]
