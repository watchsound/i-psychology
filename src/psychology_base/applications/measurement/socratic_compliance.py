"""EMPATH-style turn-level Socratic compliance labeling harness.

Per plan doc §2.2 (online research 2026-10-07) + §2.3 (OSS research 2026-10-07):
- Label schema aligned with **MathDial teacher-move taxonomy** (focus / probing /
  telling / generic) for cross-dataset defensibility.
- Export shape compatible with **Argilla** (OSS §2.3) for auditor labeling +
  inter-annotator-agreement computation (pingouin / irrCAC).
- Judge calibration follows **EMPATH** (arXiv 2606.30256) strict binary
  sub-criteria: raw-score LLM judging inflates on 10/19 metrics without
  calibration (one example: sensitive-context reintroduction 7.3 → 3.3).

Scope of this module:
- Deterministic *label-schema* + *roll-up math*, no LLM judge wired yet.
- The LLM judge wire-up is a separate module per §6d carried-debt
  (`src/psychology_base/eval/benchmarks/semantic_judge.py` → EMPATH-shape upgrade).
- Export to Argilla uses the Argilla-dataset JSONL shape; we do not import
  the `argilla` Python package here so Phase 4a stays dependency-light.

Gate target per docs/roadmap.md §Phase 4: Socratic no-direct-answer compliance
≥ 95% over 100 sampled turns per user, auditor-labeled.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional

from psychology_base.applications.measurement.scaffolding_telemetry import TutorMove


class ComplianceLabel(str, Enum):
    """EMPATH-style binary sub-criteria: direct-answer vs Socratic.

    `BORDERLINE` is the auditor-reserved option for cases where the label is
    genuinely ambiguous. The gate denominator excludes `BORDERLINE` turns.
    """

    SOCRATIC = "socratic"
    DIRECT_ANSWER = "direct_answer"
    BORDERLINE = "borderline"


# Map CoMeT/MathDial tutor moves to compliance labels. `WORKED_STEP` and
# `FULL_ANSWER` are both "telling" in MathDial; FULL_ANSWER is unambiguously
# a direct-answer violation; WORKED_STEP is a scaffolding move that is only
# a violation when it supplies more than the next step.
_MOVE_TO_LABEL: dict[TutorMove, ComplianceLabel] = {
    TutorMove.SOCRATIC_PROBE: ComplianceLabel.SOCRATIC,
    TutorMove.FOCUS_ATTENTION: ComplianceLabel.SOCRATIC,
    TutorMove.GENERIC_SUPPORT: ComplianceLabel.SOCRATIC,
    TutorMove.WORKED_STEP: ComplianceLabel.BORDERLINE,
    TutorMove.FULL_ANSWER: ComplianceLabel.DIRECT_ANSWER,
}


@dataclass(frozen=True)
class LabeledTurn:
    """One auditor-labeled tutor turn."""

    user_id: str
    session_id: str
    turn_index: int
    utterance: str
    tutor_move: Optional[TutorMove]  # None if the auditor rejects the model's label
    auditor_label: ComplianceLabel
    auditor_id: str


@dataclass(frozen=True)
class ComplianceReport:
    """Roll-up over a user's labeled sample."""

    user_id: str
    sampled_turns: int
    socratic_count: int
    direct_answer_count: int
    borderline_count: int

    def compliance_rate(self) -> float:
        """Fraction of non-borderline turns labeled SOCRATIC."""
        denom = self.socratic_count + self.direct_answer_count
        if denom == 0:
            return 0.0
        return self.socratic_count / denom

    def meets_roadmap_gate(
        self,
        compliance_floor: float = 0.95,
        min_turns: int = 100,
    ) -> bool:
        return (
            self.sampled_turns >= min_turns
            and self.compliance_rate() >= compliance_floor
        )


def auto_label_from_tutor_move(move: TutorMove) -> ComplianceLabel:
    """Model-side best-guess label; auditor overrides are authoritative."""
    return _MOVE_TO_LABEL[move]


def summarise_user(user_id: str, labeled_turns: Iterable[LabeledTurn]) -> ComplianceReport:
    turns = [t for t in labeled_turns if t.user_id == user_id]
    socratic = sum(1 for t in turns if t.auditor_label is ComplianceLabel.SOCRATIC)
    direct = sum(1 for t in turns if t.auditor_label is ComplianceLabel.DIRECT_ANSWER)
    borderline = sum(1 for t in turns if t.auditor_label is ComplianceLabel.BORDERLINE)
    return ComplianceReport(
        user_id=user_id,
        sampled_turns=len(turns),
        socratic_count=socratic,
        direct_answer_count=direct,
        borderline_count=borderline,
    )


# ---------------------------------------------------------------------------
# Argilla export — minimal JSONL shape, no `argilla` package import.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ArgillaRecord:
    """One row in the Argilla dataset JSONL export."""

    fields: dict[str, str]
    metadata: dict[str, str]
    external_id: str


def labeled_turn_to_argilla(labeled: LabeledTurn) -> ArgillaRecord:
    return ArgillaRecord(
        fields={
            "utterance": labeled.utterance,
            "tutor_move": labeled.tutor_move.value if labeled.tutor_move else "unknown",
        },
        metadata={
            "user_id": labeled.user_id,
            "session_id": labeled.session_id,
            "turn_index": str(labeled.turn_index),
            "auditor_id": labeled.auditor_id,
            "auditor_label": labeled.auditor_label.value,
        },
        external_id=f"{labeled.user_id}:{labeled.session_id}:{labeled.turn_index}",
    )


def write_argilla_jsonl(records: Iterable[ArgillaRecord], out_path: Path) -> int:
    """Write records to a JSONL file (Argilla's load-from-disk format).

    Returns the number of records written.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
            count += 1
    return count


# ---------------------------------------------------------------------------
# Inter-annotator agreement — Cohen's κ, two-rater case.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InterAnnotatorAgreement:
    kappa: float
    n_items: int
    po: float  # observed agreement
    pe: float  # expected agreement by chance


def cohen_kappa(
    labels_a: list[ComplianceLabel],
    labels_b: list[ComplianceLabel],
) -> InterAnnotatorAgreement:
    """Standard two-rater Cohen's κ over the three-label scheme.

    Returned κ follows the common convention: 1.0 = perfect agreement,
    0.0 = chance-level agreement, < 0 = worse than chance.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("rater label lists must have equal length")
    n = len(labels_a)
    if n == 0:
        raise ValueError("rater label lists must be non-empty")

    agreed = sum(1 for a, b in zip(labels_a, labels_b) if a is b)
    po = agreed / n

    categories = list(ComplianceLabel)
    pe = 0.0
    for cat in categories:
        pa_cat = sum(1 for a in labels_a if a is cat) / n
        pb_cat = sum(1 for b in labels_b if b is cat) / n
        pe += pa_cat * pb_cat

    if pe >= 1.0:
        kappa = 1.0 if po >= 1.0 else 0.0
    else:
        kappa = (po - pe) / (1.0 - pe)
    return InterAnnotatorAgreement(kappa=kappa, n_items=n, po=po, pe=pe)
