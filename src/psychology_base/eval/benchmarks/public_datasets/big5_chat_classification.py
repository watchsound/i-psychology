"""Phase 8 Pillar C soft gate — BIG5-CHAT trait-condition-forward classification.

**Gate.** Plan §2 Pillar C soft bar: trait-condition-forward
classification ≥ 60% polarity accuracy on the labeled OCEAN dimension.
For each BIG5-CHAT dialog (one trait-conditioned utterance ``train_output``
+ ground-truth ``trait`` / ``level``), we feed the utterance through the
Phase 7 trait-inference pipeline and ask: *on the labeled trait, does
the inferred Likert score fall on the labeled side of the midpoint?*

**Why polarity, not MAE.** BIG5-CHAT is a classification corpus (high vs
low), not a Likert-scale corpus. Mapping to continuous MAE adds the
reader's choice of what "level=high" means numerically (4.5? 5.0?). The
labeled-dim polarity test is honest to the dataset: ``level=high`` ↔
inferred > ``LIKERT_MID``, ``level=low`` ↔ inferred < ``LIKERT_MID``.

**Baseline — 50%.** Binary polarity gives a 50% random floor, so the
60% soft gate asks the pipeline to be meaningfully better than coin
flip on the hardest version of the problem (one utterance per profile).
Phase 7 B4 ran on 50–500 utterances per profile; BIG5-CHAT single-turn
is a worst-case stress test for the lexical inference layer.

**Honesty posture.** The pipeline's lexical cue bank was tuned to
reproduce B4 synthetic self-labels ("i'm anxious", "i'm outgoing", ...).
BIG5-CHAT's generated dialogs embed trait signal in surface style, not
in explicit self-labels — a 60% result on this corpus would be strong
evidence the Phase 7 inference generalises beyond its own synthetic
corpus. A FAIL here is useful too: it quantifies the synthetic-vs-real
gap carried from Phase 7 §9.

The scorer emits per-dialog + per-cell diagnostics so a FAIL surfaces
*which* trait the pipeline is weakest on, not just the aggregate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Sequence

from psychology_base.applications.emotional.chat_import import (
    ImportFormat,
    ImportedTurn,
)
from psychology_base.applications.emotional.trait_inference import (
    confirm_candidates,
    infer_traits,
    ingest_confirmed_candidates,
)
from psychology_base.applications.emotional.trait_profile import (
    LIKERT_MID,
    OceanDim,
    TraitProfile,
)
from psychology_base.companion.system import CompanionSystem
from psychology_base.eval.benchmarks.public_datasets.big5_chat import (
    Big5ChatDialog,
    speaker_name,
)


_SOFT_GATE_FLOOR: float = 0.60
_DEFAULT_START = datetime(2026, 10, 8, 12, 0, 0, tzinfo=timezone.utc)


@dataclass(frozen=True)
class PerDialogClassResult:
    dialog_id: str
    trait: str
    level: str                                   # ground truth
    inferred_score: float                        # Likert on labeled dim
    inferred_confidence: float
    predicted_level: str                         # "high" / "low" / "neutral"
    correct: bool
    on_labeled_dim: bool                         # inference landed non-midpoint on labeled dim


@dataclass(frozen=True)
class Big5ChatClassReport:
    per_dialog: tuple[PerDialogClassResult, ...]
    polarity_accuracy: float                     # excludes neutral predictions
    polarity_accuracy_incl_neutral: float        # neutral counted as wrong
    per_trait_polarity_accuracy: dict[str, float]
    per_level_polarity_accuracy: dict[str, float]
    coverage: float                              # fraction with non-midpoint inferred score
    confusion: dict[str, dict[str, int]]         # labeled_trait → predicted_level → count
    n_dialogs: int
    n_neutral: int                               # count of neutral predictions
    soft_gate_floor: float = _SOFT_GATE_FLOOR

    def meets_soft_gate(self) -> bool:
        """Plan §2 Pillar C soft bar: polarity accuracy ≥ 0.60.

        Uses ``polarity_accuracy_incl_neutral`` — a neutral prediction
        is NOT a correct classification, so it counts against the gate.
        This is the honest reading: the pipeline must produce a signed
        prediction to claim classification credit.
        """
        return self.polarity_accuracy_incl_neutral >= self.soft_gate_floor


def _dim_from_trait(trait: str) -> OceanDim:
    return OceanDim(trait)


def _score_single_dialog(
    dialog: Big5ChatDialog,
    *,
    start_ts: datetime,
) -> PerDialogClassResult:
    turn = ImportedTurn(
        speaker="user",
        content=dialog.train_output,
        timestamp=start_ts,
        source_format=ImportFormat.GENERIC_JSONL,
        source_metadata={
            "dialog_id": dialog.dialog_id,
            "trait": dialog.trait,
            "level": dialog.level,
            "persony": speaker_name(dialog),
        },
    )
    system = CompanionSystem(top_k=4)
    cands = infer_traits([turn])
    # confidence_floor=0.0 — a single utterance never passes the Phase 7
    # D3 default of 0.5. The classification task explicitly measures what
    # the pipeline can infer from one utterance, so we admit every
    # candidate.
    confirmed = confirm_candidates(cands, confidence_floor=0.0)
    ingest_confirmed_candidates(system, confirmed)
    profile = TraitProfile(system=system, now=start_ts + timedelta(hours=1))
    labeled_dim = _dim_from_trait(dialog.trait)
    estimate = profile.get(labeled_dim)
    score = estimate.value
    confidence = estimate.confidence
    on_labeled = abs(score - LIKERT_MID) > 1e-9
    if score > LIKERT_MID:
        predicted = "high"
    elif score < LIKERT_MID:
        predicted = "low"
    else:
        predicted = "neutral"
    correct = predicted == dialog.level
    return PerDialogClassResult(
        dialog_id=dialog.dialog_id,
        trait=dialog.trait,
        level=dialog.level,
        inferred_score=score,
        inferred_confidence=confidence,
        predicted_level=predicted,
        correct=correct,
        on_labeled_dim=on_labeled,
    )


def score_dialogs(
    dialogs: Sequence[Big5ChatDialog],
    *,
    start_ts: datetime = _DEFAULT_START,
) -> Big5ChatClassReport:
    """Score a BIG5-CHAT dialog set for Pillar C soft gate."""
    per_dialog = tuple(
        _score_single_dialog(d, start_ts=start_ts) for d in dialogs
    )
    n = len(per_dialog)
    if n == 0:
        return Big5ChatClassReport(
            per_dialog=(),
            polarity_accuracy=0.0,
            polarity_accuracy_incl_neutral=0.0,
            per_trait_polarity_accuracy={},
            per_level_polarity_accuracy={},
            coverage=0.0,
            confusion={},
            n_dialogs=0,
            n_neutral=0,
        )
    n_neutral = sum(1 for r in per_dialog if r.predicted_level == "neutral")
    n_correct = sum(1 for r in per_dialog if r.correct)
    n_signed = n - n_neutral
    polarity_accuracy = (
        sum(1 for r in per_dialog if r.correct and r.predicted_level != "neutral")
        / n_signed
    ) if n_signed > 0 else 0.0
    polarity_accuracy_incl_neutral = n_correct / n

    per_trait: dict[str, list[bool]] = {}
    per_level: dict[str, list[bool]] = {}
    for r in per_dialog:
        per_trait.setdefault(r.trait, []).append(r.correct)
        per_level.setdefault(r.level, []).append(r.correct)
    per_trait_accuracy = {
        t: sum(xs) / len(xs) for t, xs in per_trait.items()
    }
    per_level_accuracy = {
        l: sum(xs) / len(xs) for l, xs in per_level.items()
    }

    coverage = sum(1 for r in per_dialog if r.on_labeled_dim) / n

    confusion: dict[str, dict[str, int]] = {}
    for r in per_dialog:
        row = confusion.setdefault(r.trait, {"high": 0, "low": 0, "neutral": 0})
        row[r.predicted_level] = row.get(r.predicted_level, 0) + 1

    return Big5ChatClassReport(
        per_dialog=per_dialog,
        polarity_accuracy=polarity_accuracy,
        polarity_accuracy_incl_neutral=polarity_accuracy_incl_neutral,
        per_trait_polarity_accuracy=per_trait_accuracy,
        per_level_polarity_accuracy=per_level_accuracy,
        coverage=coverage,
        confusion=confusion,
        n_dialogs=n,
        n_neutral=n_neutral,
    )


__all__ = [
    "Big5ChatClassReport",
    "PerDialogClassResult",
    "score_dialogs",
]
