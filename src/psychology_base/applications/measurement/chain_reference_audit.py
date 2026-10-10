"""Chain-reference audit for 情感陪伴 (long-term emotional companion).

Measures how well the companion references prior memory-version-chain entries
when the user re-frames a past event — the narrative-therapy reconsolidation
move the memory substrate was built to enable (CLAUDE.md Pillar 1).

Gate per docs/roadmap.md §Phase 4: ≥ 80% of reinterpretation moments
appropriately cite prior version. Auditor-labeled.

Design: mirrors `socratic_compliance.py` shape — auditor-labeled turns +
inter-annotator-agreement + Argilla-compatible JSONL export.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional

from psychology_base.applications.measurement.socratic_compliance import (
    ArgillaRecord,
    InterAnnotatorAgreement,
)


class ReinterpretationAct(str, Enum):
    """Was this user turn a reinterpretation of a prior event?"""

    REINTERPRETATION = "reinterpretation"
    FIRST_MENTION = "first_mention"
    UNRELATED = "unrelated"


class ChainReferenceLabel(str, Enum):
    """Did the companion's response cite the prior version appropriately?"""

    APPROPRIATE_CITE = "appropriate_cite"       # cited prior version at the right moment
    MISSED_OPPORTUNITY = "missed_opportunity"   # reinterpretation moment, no cite
    INAPPROPRIATE_CITE = "inappropriate_cite"   # cited when the user did not reinterpret
    NOT_APPLICABLE = "not_applicable"           # user did not reinterpret, companion did not cite


@dataclass(frozen=True)
class ChainReferenceTurn:
    """One auditor-labeled turn pair (user utterance + companion response)."""

    user_id: str
    session_id: str
    turn_index: int
    user_utterance: str
    companion_utterance: str
    reinterpretation_act: ReinterpretationAct
    chain_reference_label: ChainReferenceLabel
    auditor_id: str
    referenced_event_id: Optional[str] = None
    referenced_version: Optional[int] = None


@dataclass(frozen=True)
class ChainReferenceReport:
    """Roll-up over a user's labeled sample."""

    user_id: str
    reinterpretation_count: int
    appropriate_cite_count: int
    missed_opportunity_count: int
    inappropriate_cite_count: int

    def chain_reference_rate(self) -> float:
        """Fraction of reinterpretation moments appropriately cited.

        Denominator = reinterpretation_count. Zero reinterpretations → 0.0
        (vacuously satisfies-the-floor cases are flagged as insufficient
        denominator by the gate check, not the rate).
        """
        if self.reinterpretation_count == 0:
            return 0.0
        return self.appropriate_cite_count / self.reinterpretation_count

    def meets_roadmap_gate(
        self,
        chain_reference_floor: float = 0.80,
        min_reinterpretations: int = 5,
    ) -> bool:
        return (
            self.reinterpretation_count >= min_reinterpretations
            and self.chain_reference_rate() >= chain_reference_floor
        )


def summarise_user(
    user_id: str,
    turns: Iterable[ChainReferenceTurn],
) -> ChainReferenceReport:
    user_turns = [t for t in turns if t.user_id == user_id]
    reinterp = [
        t for t in user_turns if t.reinterpretation_act is ReinterpretationAct.REINTERPRETATION
    ]
    appropriate = sum(
        1 for t in reinterp
        if t.chain_reference_label is ChainReferenceLabel.APPROPRIATE_CITE
    )
    missed = sum(
        1 for t in reinterp
        if t.chain_reference_label is ChainReferenceLabel.MISSED_OPPORTUNITY
    )
    inappropriate = sum(
        1 for t in user_turns
        if t.chain_reference_label is ChainReferenceLabel.INAPPROPRIATE_CITE
    )
    return ChainReferenceReport(
        user_id=user_id,
        reinterpretation_count=len(reinterp),
        appropriate_cite_count=appropriate,
        missed_opportunity_count=missed,
        inappropriate_cite_count=inappropriate,
    )


# ---------------------------------------------------------------------------
# Argilla export + κ — mirror the Socratic module's conventions.
# ---------------------------------------------------------------------------


def chain_reference_turn_to_argilla(turn: ChainReferenceTurn) -> ArgillaRecord:
    return ArgillaRecord(
        fields={
            "user_utterance": turn.user_utterance,
            "companion_utterance": turn.companion_utterance,
            "reinterpretation_act": turn.reinterpretation_act.value,
        },
        metadata={
            "user_id": turn.user_id,
            "session_id": turn.session_id,
            "turn_index": str(turn.turn_index),
            "auditor_id": turn.auditor_id,
            "chain_reference_label": turn.chain_reference_label.value,
            "referenced_event_id": turn.referenced_event_id or "",
            "referenced_version": (
                str(turn.referenced_version) if turn.referenced_version is not None else ""
            ),
        },
        external_id=f"{turn.user_id}:{turn.session_id}:{turn.turn_index}",
    )


def write_argilla_jsonl(records: Iterable[ArgillaRecord], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
            count += 1
    return count


def kappa_on_chain_reference(
    labels_a: list[ChainReferenceLabel],
    labels_b: list[ChainReferenceLabel],
) -> InterAnnotatorAgreement:
    """κ over the ChainReferenceLabel 4-way scheme.

    Implemented in-module instead of re-using the Socratic helper so each
    label scheme owns its own agreement computation surface.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("rater label lists must have equal length")
    n = len(labels_a)
    if n == 0:
        raise ValueError("rater label lists must be non-empty")

    agreed = sum(1 for a, b in zip(labels_a, labels_b) if a is b)
    po = agreed / n

    categories = list(ChainReferenceLabel)
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


__all__ = [
    "ChainReferenceLabel",
    "ChainReferenceReport",
    "ChainReferenceTurn",
    "ReinterpretationAct",
    "chain_reference_turn_to_argilla",
    "kappa_on_chain_reference",
    "summarise_user",
    "write_argilla_jsonl",
]
