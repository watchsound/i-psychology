"""MathDial adapter — Socratic-compliance alignment against gold teacher moves.

**Dataset provenance.** MathDial (Macina, Daheim, Shu, et al., 2023; arXiv
2305.14536) is a tutoring-dialogue dataset of ~2.9K multi-turn conversations
between a teacher and a (synthetic) student working through GSM8K problems.
Published under **CC BY-SA 4.0** (verify at the repo README before any
derivative distribution). The repo is at `github.com/eth-nlped/mathdial`.
Load-bearing attribution: any artefact derived from a MathDial payload must
cite the paper and preserve the CC BY-SA license on the derivative.

**Pivot gate this adapter targets.** `docs/phases/phase-4b-benchmark-pivot.md`
§3 row "学习陪伴 Socratic no-direct-answer compliance" — target ≥95%
auto-label + ≥90% MathDial-taxonomy overlap on a 100-turn held-out sample
+ EMPATH binary-sub-criteria κ ≥ 0.60.

**Why no vendor-download.** CC BY-SA 4.0 is permissive but share-alike
propagates to derivatives. We take no payload into the repo so that
downstream users of psychology-base aren't forced into CC BY-SA by
transitive inclusion of our test fixtures. The loader expects a
user-provided JSONL shim path.

**JSONL shim schema.** One conversation per line::

    {
      "conversation_id": "md_000123",
      "problem": "Janet's ducks lay 16 eggs per day...",
      "ground_truth": "Janet sells 16 - 3 - 4 = 9 eggs...",
      "student_incorrect_solution": "Janet makes 16 * 2 = $32...",
      "self_correction": true,
      "turns": [
        {"role": "teacher", "utterance": "What does Janet have left after breakfast?", "move": "probing"},
        {"role": "student", "utterance": "She has 16 - 3 = 13 eggs."},
        {"role": "teacher", "utterance": "Good. Now check the muffin step.", "move": "focus"},
        ...
      ],
      "metadata": {}
    }

Shim fields are the ones the alignment scorer load-bears on; raw MathDial
columns not consumed here (e.g. `confidence`, `ravens-progressive-score`)
are discarded by the conversion script. Student turns need no `move`.

**Taxonomy alignment.** MathDial's 4-move taxonomy
(`focus`/`probing`/`telling`/`generic`) maps to our 5-move
`TutorMove` with `telling` fanning out across WORKED_STEP vs FULL_ANSWER
depending on whether a scaffolding cue is present. The reverse map
collapses them. See `MATHDIAL_TO_TUTOR_MOVE` and `TUTOR_MOVE_TO_MATHDIAL`
below.

**No LLM dependency.** Scoring is deterministic — gold MathDial move is
compared against our classifier's output after a canonical collapse to the
MathDial taxonomy. The LLM-judge upgrade is a Phase 4b task (EMPATH binary
sub-criteria over the same held-out sample), tracked separately.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterator, Literal, Optional

from psychology_base.applications.learning.tutor_agent import classify_tutor_move
from psychology_base.applications.measurement.scaffolding_telemetry import TutorMove


class MathDialMove(str, Enum):
    """MathDial's 4-move teacher taxonomy (dataset-native)."""

    FOCUS = "focus"
    PROBING = "probing"
    TELLING = "telling"
    GENERIC = "generic"


MATHDIAL_TO_TUTOR_MOVE: dict[MathDialMove, TutorMove] = {
    MathDialMove.FOCUS: TutorMove.FOCUS_ATTENTION,
    MathDialMove.PROBING: TutorMove.SOCRATIC_PROBE,
    MathDialMove.GENERIC: TutorMove.GENERIC_SUPPORT,
    # MathDial `telling` fans out to WORKED_STEP (scaffold-shaped) or
    # FULL_ANSWER (surrender-shaped). Canonical collapse maps to FULL_ANSWER
    # because the auditor-level distinction between the two is a
    # psychology-base-side specialisation, not MathDial-native.
    MathDialMove.TELLING: TutorMove.FULL_ANSWER,
}


TUTOR_MOVE_TO_MATHDIAL: dict[TutorMove, MathDialMove] = {
    TutorMove.FOCUS_ATTENTION: MathDialMove.FOCUS,
    TutorMove.SOCRATIC_PROBE: MathDialMove.PROBING,
    TutorMove.GENERIC_SUPPORT: MathDialMove.GENERIC,
    TutorMove.WORKED_STEP: MathDialMove.TELLING,
    TutorMove.FULL_ANSWER: MathDialMove.TELLING,
}


@dataclass(frozen=True)
class MathDialTurn:
    """One turn in a MathDial conversation."""

    role: Literal["teacher", "student"]
    utterance: str
    move: Optional[MathDialMove] = None  # gold label; only present on teacher turns


@dataclass(frozen=True)
class MathDialConversation:
    """One MathDial conversation = one GSM8K problem + its tutoring dialogue."""

    conversation_id: str
    problem: str
    ground_truth: str
    student_incorrect_solution: str
    turns: tuple[MathDialTurn, ...]
    self_correction: Optional[bool] = None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class MathDialAlignmentReport:
    """Roll-up for the Socratic-compliance gate against MathDial gold.

    `taxonomy_overlap_rate` = fraction of teacher turns where our classifier's
    collapsed-to-MathDial label equals the gold MathDial move. This is the
    §3 pivot-gate "≥ 90% MathDial-taxonomy overlap" number.

    `socratic_rate_predicted` and `socratic_rate_gold` are the fractions of
    turns our side vs the gold labels as a non-TELLING move. The gap between
    them is a specific failure diagnostic: a positive gap (we say Socratic
    more than gold does) is the pattern we care about — our TutorAgent
    inflating self-reported compliance.

    Confusion matrix is a 4x4 dict keyed by (predicted, gold) MathDialMove
    pairs. For readers that want per-category breakdowns beyond the roll-up.
    """

    conversation_id: str
    sampled_teacher_turns: int
    taxonomy_matches: int
    socratic_rate_predicted: float
    socratic_rate_gold: float
    confusion: dict[tuple[MathDialMove, MathDialMove], int]

    def taxonomy_overlap_rate(self) -> float:
        if self.sampled_teacher_turns == 0:
            return 0.0
        return self.taxonomy_matches / self.sampled_teacher_turns

    def meets_pivot_gate(
        self,
        taxonomy_overlap_floor: float = 0.90,
        min_teacher_turns: int = 10,
    ) -> bool:
        """Pivot §3 target. Default floor 0.90 is held-out-sample rate; the
        gate denominator in the pivot is 100 sampled turns pooled across
        conversations, which the caller aggregates separately."""
        return (
            self.sampled_teacher_turns >= min_teacher_turns
            and self.taxonomy_overlap_rate() >= taxonomy_overlap_floor
        )


# ---------------------------------------------------------------------------
# JSONL shim loader
# ---------------------------------------------------------------------------

def iter_mathdial_conversations(
    shim_path: Path,
) -> Iterator[MathDialConversation]:
    """Stream MathDial conversations from a JSONL shim file.

    Streaming (not list-returning) matches the LoCoMo / LongMemEval loader
    convention — allows the alignment runner to pool across all 2.9K rows
    without loading the full payload into memory.

    Raises `ValueError` on malformed turns rather than silently dropping, so
    the shim-conversion script is forced to produce clean output.
    """
    with Path(shim_path).open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            turns = tuple(_parse_turn(t, line_no, idx) for idx, t in enumerate(obj.get("turns", [])))
            yield MathDialConversation(
                conversation_id=str(obj["conversation_id"]),
                problem=str(obj.get("problem", "")),
                ground_truth=str(obj.get("ground_truth", "")),
                student_incorrect_solution=str(obj.get("student_incorrect_solution", "")),
                turns=turns,
                self_correction=obj.get("self_correction"),
                metadata=dict(obj.get("metadata", {})),
            )


def _parse_turn(raw: dict, line_no: int, idx: int) -> MathDialTurn:
    role = raw.get("role")
    if role not in ("teacher", "student"):
        raise ValueError(
            f"MathDial shim line {line_no} turn {idx}: role must be 'teacher' or 'student', got {role!r}"
        )
    move_raw = raw.get("move")
    move: Optional[MathDialMove] = None
    if move_raw is not None:
        if role != "teacher":
            raise ValueError(
                f"MathDial shim line {line_no} turn {idx}: 'move' only valid on teacher turns"
            )
        move = MathDialMove(move_raw)
    utterance = raw.get("utterance")
    if not isinstance(utterance, str):
        raise ValueError(
            f"MathDial shim line {line_no} turn {idx}: 'utterance' must be a string"
        )
    return MathDialTurn(role=role, utterance=utterance, move=move)


# ---------------------------------------------------------------------------
# Alignment scorer — classifier output vs gold MathDial move
# ---------------------------------------------------------------------------

_NON_TELLING = {
    MathDialMove.FOCUS,
    MathDialMove.PROBING,
    MathDialMove.GENERIC,
}


def score_tutor_against_mathdial(
    conversation: MathDialConversation,
) -> MathDialAlignmentReport:
    """Run `classify_tutor_move` on each teacher utterance, compare to gold.

    Only teacher turns with a gold `move` are counted. Student turns and
    moveless teacher turns are skipped — if the shim is missing a move,
    the shim-conversion script dropped it and we should not penalise the
    classifier for a missing label.
    """
    teacher_turns_with_gold = [
        t for t in conversation.turns if t.role == "teacher" and t.move is not None
    ]
    confusion: dict[tuple[MathDialMove, MathDialMove], int] = {}
    matches = 0
    pred_socratic = 0
    gold_socratic = 0

    for turn in teacher_turns_with_gold:
        pred_tutor = classify_tutor_move(turn.utterance)
        pred_mathdial = TUTOR_MOVE_TO_MATHDIAL[pred_tutor]
        gold = turn.move  # non-None by filter
        assert gold is not None  # for the type-checker
        confusion[(pred_mathdial, gold)] = confusion.get((pred_mathdial, gold), 0) + 1
        if pred_mathdial is gold:
            matches += 1
        if pred_mathdial in _NON_TELLING:
            pred_socratic += 1
        if gold in _NON_TELLING:
            gold_socratic += 1

    n = len(teacher_turns_with_gold)
    return MathDialAlignmentReport(
        conversation_id=conversation.conversation_id,
        sampled_teacher_turns=n,
        taxonomy_matches=matches,
        socratic_rate_predicted=(pred_socratic / n) if n else 0.0,
        socratic_rate_gold=(gold_socratic / n) if n else 0.0,
        confusion=confusion,
    )
