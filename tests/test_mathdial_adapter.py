"""MathDial adapter — shim loader + taxonomy alignment scorer tests.

Phase 4b benchmark-pivot item D2 priority #1. Fixture is in-memory to keep
the repo free of CC BY-SA 4.0 derivative payload (see
`src/psychology_base/eval/benchmarks/public_datasets/mathdial.py` provenance
note). Real-data run awaits a `convert_mathdial` script that transforms the
upstream HuggingFace rows into our JSONL shim.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from psychology_base.applications.measurement.scaffolding_telemetry import TutorMove
from psychology_base.eval.benchmarks.public_datasets.mathdial import (
    MATHDIAL_TO_TUTOR_MOVE,
    MathDialConversation,
    MathDialMove,
    MathDialTurn,
    TUTOR_MOVE_TO_MATHDIAL,
    iter_mathdial_conversations,
    score_tutor_against_mathdial,
)


# ---------------------------------------------------------------------------
# Taxonomy map round-trip
# ---------------------------------------------------------------------------


def test_mathdial_to_tutor_covers_all_four_moves():
    for move in MathDialMove:
        assert move in MATHDIAL_TO_TUTOR_MOVE


def test_tutor_to_mathdial_covers_all_five_tutor_moves():
    for move in TutorMove:
        assert move in TUTOR_MOVE_TO_MATHDIAL


def test_telling_collapse_is_many_to_one():
    assert TUTOR_MOVE_TO_MATHDIAL[TutorMove.WORKED_STEP] is MathDialMove.TELLING
    assert TUTOR_MOVE_TO_MATHDIAL[TutorMove.FULL_ANSWER] is MathDialMove.TELLING


# ---------------------------------------------------------------------------
# Shim loader — happy path + malformed inputs
# ---------------------------------------------------------------------------


def _write_shim(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "mathdial_shim.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return path


def test_loader_round_trips_minimal_conversation(tmp_path):
    row = {
        "conversation_id": "md_001",
        "problem": "Janet's ducks lay 16 eggs...",
        "ground_truth": "9 eggs sold",
        "student_incorrect_solution": "32 dollars",
        "self_correction": True,
        "turns": [
            {"role": "teacher", "utterance": "What does Janet have left?", "move": "probing"},
            {"role": "student", "utterance": "She has 13 eggs."},
            {"role": "teacher", "utterance": "Good. Now check the muffins.", "move": "focus"},
        ],
    }
    path = _write_shim(tmp_path, [row])
    convos = list(iter_mathdial_conversations(path))
    assert len(convos) == 1
    convo = convos[0]
    assert convo.conversation_id == "md_001"
    assert convo.self_correction is True
    assert len(convo.turns) == 3
    assert convo.turns[0].move is MathDialMove.PROBING
    assert convo.turns[1].move is None  # student
    assert convo.turns[2].move is MathDialMove.FOCUS


def test_loader_streams_multiple_rows(tmp_path):
    rows = [
        {
            "conversation_id": f"md_{i:03d}",
            "problem": "p",
            "ground_truth": "g",
            "student_incorrect_solution": "s",
            "turns": [{"role": "teacher", "utterance": "x", "move": "generic"}],
        }
        for i in range(5)
    ]
    path = _write_shim(tmp_path, rows)
    convos = list(iter_mathdial_conversations(path))
    assert [c.conversation_id for c in convos] == [f"md_{i:03d}" for i in range(5)]


def test_loader_skips_blank_lines(tmp_path):
    path = tmp_path / "shim.jsonl"
    path.write_text(
        json.dumps(
            {
                "conversation_id": "md_x",
                "problem": "p",
                "ground_truth": "g",
                "student_incorrect_solution": "s",
                "turns": [],
            }
        )
        + "\n\n\n",
        encoding="utf-8",
    )
    convos = list(iter_mathdial_conversations(path))
    assert len(convos) == 1


def test_loader_rejects_move_on_student_turn(tmp_path):
    row = {
        "conversation_id": "md_bad",
        "problem": "p",
        "ground_truth": "g",
        "student_incorrect_solution": "s",
        "turns": [{"role": "student", "utterance": "x", "move": "probing"}],
    }
    path = _write_shim(tmp_path, [row])
    with pytest.raises(ValueError, match="'move' only valid on teacher"):
        list(iter_mathdial_conversations(path))


def test_loader_rejects_unknown_role(tmp_path):
    row = {
        "conversation_id": "md_bad",
        "problem": "p",
        "ground_truth": "g",
        "student_incorrect_solution": "s",
        "turns": [{"role": "narrator", "utterance": "x"}],
    }
    path = _write_shim(tmp_path, [row])
    with pytest.raises(ValueError, match="role must be"):
        list(iter_mathdial_conversations(path))


def test_loader_rejects_unknown_move_value(tmp_path):
    row = {
        "conversation_id": "md_bad",
        "problem": "p",
        "ground_truth": "g",
        "student_incorrect_solution": "s",
        "turns": [{"role": "teacher", "utterance": "x", "move": "coaching"}],
    }
    path = _write_shim(tmp_path, [row])
    with pytest.raises(ValueError):
        list(iter_mathdial_conversations(path))


def test_loader_rejects_non_string_utterance(tmp_path):
    row = {
        "conversation_id": "md_bad",
        "problem": "p",
        "ground_truth": "g",
        "student_incorrect_solution": "s",
        "turns": [{"role": "teacher", "utterance": 42, "move": "generic"}],
    }
    path = _write_shim(tmp_path, [row])
    with pytest.raises(ValueError, match="utterance.*must be a string"):
        list(iter_mathdial_conversations(path))


# ---------------------------------------------------------------------------
# Alignment scorer — classifier vs gold MathDial move
# ---------------------------------------------------------------------------


def _convo(turns: list[MathDialTurn]) -> MathDialConversation:
    return MathDialConversation(
        conversation_id="t",
        problem="p",
        ground_truth="g",
        student_incorrect_solution="s",
        turns=tuple(turns),
    )


def test_scorer_perfect_match_on_classifier_shaped_prompts():
    """Teacher turns authored to match the classifier's rules; expect perfect alignment."""
    convo = _convo(
        [
            # SOCRATIC_PROBE: question + probing cue
            MathDialTurn(
                role="teacher",
                utterance="What do you notice about the equation?",
                move=MathDialMove.PROBING,
            ),
            # FOCUS_ATTENTION: imperative "notice", no question
            MathDialTurn(
                role="teacher",
                utterance="Notice the sign on the right side.",
                move=MathDialMove.FOCUS,
            ),
            # FULL_ANSWER: declarative, >80 chars, no scaffold cue
            MathDialTurn(
                role="teacher",
                utterance=(
                    "The derivative here is sinusoidal because cosine integrates to sine "
                    "and the chain rule yields exactly that result for this composition."
                ),
                move=MathDialMove.TELLING,
            ),
            # GENERIC_SUPPORT: fallback — no question, no cue, short
            MathDialTurn(
                role="teacher",
                utterance="Great effort.",
                move=MathDialMove.GENERIC,
            ),
        ]
    )
    report = score_tutor_against_mathdial(convo)
    assert report.sampled_teacher_turns == 4
    assert report.taxonomy_matches == 4
    assert report.taxonomy_overlap_rate() == 1.0


def test_scorer_rejects_full_answer_as_probing():
    """A `telling` gold turn classified as FULL_ANSWER maps to TELLING — perfect match."""
    convo = _convo(
        [
            MathDialTurn(
                role="teacher",
                utterance=(
                    "The answer is nine because sixteen minus three minus four equals nine, "
                    "which is the number of eggs she sells at the market each day."
                ),
                move=MathDialMove.TELLING,
            ),
        ]
    )
    report = score_tutor_against_mathdial(convo)
    assert report.taxonomy_overlap_rate() == 1.0


def test_scorer_detects_sycophancy_disagreement():
    """Classifier mislabels a soft TELLING as GENERIC_SUPPORT (surface bug)."""
    convo = _convo(
        [
            # Shaped to miss both the TELLING cue list and the FULL_ANSWER
            # explicit-answer path: a short teacher turn that implies a
            # fact without quoting the student or using correction verbs.
            MathDialTurn(
                role="teacher",
                utterance="In the end that comes out even.",
                move=MathDialMove.TELLING,
            ),
        ]
    )
    report = score_tutor_against_mathdial(convo)
    # Classifier sees no cue → GENERIC_SUPPORT → MathDial GENERIC.
    # Gold is TELLING. Mismatch surfaces as 0% overlap on this one-turn convo.
    assert report.taxonomy_overlap_rate() == 0.0
    assert report.socratic_rate_predicted == 1.0
    assert report.socratic_rate_gold == 0.0


def test_scorer_skips_student_and_moveless_turns():
    convo = _convo(
        [
            MathDialTurn(role="student", utterance="I got 13."),
            MathDialTurn(role="teacher", utterance="OK."),  # no move -> skipped
            MathDialTurn(
                role="teacher",
                utterance="What is Janet left with?",
                move=MathDialMove.PROBING,
            ),
        ]
    )
    report = score_tutor_against_mathdial(convo)
    assert report.sampled_teacher_turns == 1
    assert report.taxonomy_overlap_rate() == 1.0


def test_scorer_empty_convo_returns_zero_rates():
    convo = _convo([])
    report = score_tutor_against_mathdial(convo)
    assert report.sampled_teacher_turns == 0
    assert report.taxonomy_overlap_rate() == 0.0
    assert report.socratic_rate_predicted == 0.0
    assert report.socratic_rate_gold == 0.0


def test_scorer_populates_confusion_matrix():
    convo = _convo(
        [
            MathDialTurn(
                role="teacher",
                utterance="What do you see?",
                move=MathDialMove.PROBING,
            ),
            MathDialTurn(
                role="teacher",
                utterance="What do you see?",
                move=MathDialMove.FOCUS,
            ),
        ]
    )
    report = score_tutor_against_mathdial(convo)
    # Both predicted PROBING; one gold PROBING, one gold FOCUS.
    assert report.confusion[(MathDialMove.PROBING, MathDialMove.PROBING)] == 1
    assert report.confusion[(MathDialMove.PROBING, MathDialMove.FOCUS)] == 1


# ---------------------------------------------------------------------------
# Pivot-gate helper
# ---------------------------------------------------------------------------


def test_meets_pivot_gate_requires_floor_and_min_turns():
    convo = _convo(
        [
            MathDialTurn(
                role="teacher",
                utterance=f"What is step {i}?",
                move=MathDialMove.PROBING,
            )
            for i in range(10)
        ]
    )
    report = score_tutor_against_mathdial(convo)
    assert report.sampled_teacher_turns == 10
    assert report.taxonomy_overlap_rate() == 1.0
    # 10 turns, 100% match: with defaults (floor=0.90, min_turns=10) passes.
    assert report.meets_pivot_gate() is True
    # Raise the floor to 1.01 — unreachable.
    assert report.meets_pivot_gate(taxonomy_overlap_floor=1.01) is False
    # Raise the min_turns to 11 — denominator fails.
    assert report.meets_pivot_gate(min_teacher_turns=11) is False
