"""Unit tests for the FANToM adapter.

Fixtures mimic the FANToM shim produced by ``provision_fantom.py`` so the
adapter can be tested without a live HuggingFace pull.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from psychology_base.eval.benchmarks.public_datasets.fantom import (
    FANToMContextTurn,
    FANToMConversation,
    FANToMItem,
    FANToMQuestion,
    FANToMQuestionType,
    iter_fantom_conversations,
    iter_fantom_items,
)


_SHIM_ROW_WITH_QUESTIONS = {
    "conversation_id": "fantom_001",
    "participants": ["Alice", "Bob", "Carol"],
    "context": [
        {"speaker": "Alice", "utterance": "I just adopted a puppy."},
        {"speaker": "Bob", "utterance": "That's great! What breed?"},
        {"speaker": "Alice", "utterance": "A corgi, from a rescue."},
        # Carol enters here per FANToM's absence-return structure
        {"speaker": "Carol", "utterance": "Hey, what did I miss?"},
    ],
    "questions": [
        {
            "question_type": "belief_inaccessible",
            "question_text": "What does Carol believe Alice adopted?",
            "target_person": "Carol",
            "correct_answer": "Carol does not know Alice adopted anything.",
            "wrong_answer": "Carol believes Alice adopted a corgi.",
            "tom_type": "false_belief",
            "tom_order": 1,
            "missed_info_accessibility": "inaccessible",
            "choices": None,
        },
        {
            "question_type": "belief_accessible",
            "question_text": "What does Bob believe Alice adopted?",
            "target_person": "Bob",
            "correct_answer": "A corgi.",
            "wrong_answer": "A labrador.",
            "tom_type": "true_belief",
            "tom_order": 1,
            "missed_info_accessibility": "accessible",
            "choices": None,
        },
        {
            "question_type": "fact",
            "question_text": "What breed did Alice adopt?",
            "target_person": None,
            "correct_answer": "A corgi.",
            "wrong_answer": None,
            "tom_type": None,
            "tom_order": None,
            "missed_info_accessibility": None,
            "choices": ["A corgi.", "A labrador.", "A poodle."],
        },
    ],
    "has_questions": True,
    "metadata": {"available_fields": ["context", "beliefQAs"]},
}

_SHIM_ROW_EMPTY = {
    "conversation_id": "fantom_002",
    "participants": [],
    "context": [],
    "questions": [],
    "has_questions": False,
    "metadata": {"available_fields": []},
}


@pytest.fixture
def shim_file(tmp_path: Path) -> Path:
    p = tmp_path / "test.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(_SHIM_ROW_WITH_QUESTIONS) + "\n")
        fh.write(json.dumps(_SHIM_ROW_EMPTY) + "\n")
    return p


def test_question_type_from_shim_matches_enum() -> None:
    assert FANToMQuestionType.from_shim("belief_inaccessible") is FANToMQuestionType.BELIEF_INACCESSIBLE
    assert FANToMQuestionType.from_shim("belief_accessible") is FANToMQuestionType.BELIEF_ACCESSIBLE
    assert FANToMQuestionType.from_shim("fact") is FANToMQuestionType.FACT
    assert FANToMQuestionType.from_shim("answerability") is FANToMQuestionType.ANSWERABILITY
    assert FANToMQuestionType.from_shim("info_accessibility") is FANToMQuestionType.INFO_ACCESSIBILITY
    assert FANToMQuestionType.from_shim("belief") is FANToMQuestionType.BELIEF


def test_question_type_from_shim_unknown_returns_none() -> None:
    assert FANToMQuestionType.from_shim("unknown_type") is None
    assert FANToMQuestionType.from_shim(None) is None


def test_iter_conversations_skips_empty_rows(shim_file: Path) -> None:
    convs = list(iter_fantom_conversations(shim_file))
    assert len(convs) == 1
    assert convs[0].conversation_id == "fantom_001"
    assert convs[0].participants == ("Alice", "Bob", "Carol")
    assert len(convs[0].context) == 4
    assert len(convs[0].questions) == 3


def test_iter_items_yields_one_per_question(shim_file: Path) -> None:
    items = list(iter_fantom_items(shim_file))
    assert len(items) == 3
    types = [item.question.question_type for item in items]
    assert types == [
        FANToMQuestionType.BELIEF_INACCESSIBLE,
        FANToMQuestionType.BELIEF_ACCESSIBLE,
        FANToMQuestionType.FACT,
    ]


def test_iter_items_filter_by_question_type(shim_file: Path) -> None:
    hard_only = list(
        iter_fantom_items(
            shim_file,
            question_types=(FANToMQuestionType.BELIEF_INACCESSIBLE,),
        )
    )
    assert len(hard_only) == 1
    assert hard_only[0].question.target_person == "Carol"
    assert hard_only[0].question.missed_info_accessibility == "inaccessible"


def test_iter_items_filter_multi_type(shim_file: Path) -> None:
    belief_only = list(
        iter_fantom_items(
            shim_file,
            question_types=(
                FANToMQuestionType.BELIEF_ACCESSIBLE,
                FANToMQuestionType.BELIEF_INACCESSIBLE,
            ),
        )
    )
    assert len(belief_only) == 2
    assert all(
        item.question.question_type
        in {FANToMQuestionType.BELIEF_ACCESSIBLE, FANToMQuestionType.BELIEF_INACCESSIBLE}
        for item in belief_only
    )


def test_render_context_round_trip(shim_file: Path) -> None:
    conv = next(iter_fantom_conversations(shim_file))
    rendered = conv.render_context()
    assert "Alice: I just adopted a puppy." in rendered
    assert "Carol: Hey, what did I miss?" in rendered
    # Order preserved
    assert rendered.index("Alice:") < rendered.index("Carol:")


def test_question_choices_tuple(shim_file: Path) -> None:
    items = list(iter_fantom_items(shim_file))
    fact_item = items[2]
    assert fact_item.question.choices == ("A corgi.", "A labrador.", "A poodle.")
    belief_item = items[0]
    assert belief_item.question.choices is None


def test_tom_order_parses_int_and_digit_string(tmp_path: Path) -> None:
    row = dict(_SHIM_ROW_WITH_QUESTIONS)
    row["conversation_id"] = "fantom_tom_order"
    row["questions"] = [
        {**_SHIM_ROW_WITH_QUESTIONS["questions"][0], "tom_order": "2"},
        {**_SHIM_ROW_WITH_QUESTIONS["questions"][0], "tom_order": 3},
        {**_SHIM_ROW_WITH_QUESTIONS["questions"][0], "tom_order": "not-a-number"},
    ]
    row["has_questions"] = True
    p = tmp_path / "to.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    items = list(iter_fantom_items(p))
    assert items[0].question.tom_order == 2
    assert items[1].question.tom_order == 3
    assert items[2].question.tom_order is None


def test_malformed_question_in_shim_row_is_skipped(tmp_path: Path) -> None:
    row = dict(_SHIM_ROW_WITH_QUESTIONS)
    row["conversation_id"] = "fantom_malformed"
    row["questions"] = [
        "not-a-dict",
        _SHIM_ROW_WITH_QUESTIONS["questions"][0],
    ]
    row["has_questions"] = True
    p = tmp_path / "mal.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    items = list(iter_fantom_items(p))
    assert len(items) == 1


def test_conversation_has_questions_property(shim_file: Path) -> None:
    convs = list(iter_fantom_conversations(shim_file))
    assert all(c.has_questions for c in convs)
    empty = FANToMConversation(
        conversation_id="x",
        participants=(),
        context=(),
        questions=(),
    )
    assert empty.has_questions is False


def test_item_dataclass_frozen() -> None:
    conv = FANToMConversation(
        conversation_id="x",
        participants=(),
        context=(),
        questions=(),
    )
    q = FANToMQuestion(
        question_type=FANToMQuestionType.FACT,
        question_text="?",
        target_person=None,
        correct_answer=None,
        wrong_answer=None,
        tom_type=None,
        tom_order=None,
        missed_info_accessibility=None,
        choices=None,
    )
    item = FANToMItem(conversation=conv, question=q)
    with pytest.raises(Exception):
        item.question = q  # type: ignore[misc]


def test_context_turn_unknown_speaker_renders_as_unknown(tmp_path: Path) -> None:
    row = dict(_SHIM_ROW_WITH_QUESTIONS)
    row["conversation_id"] = "fantom_unknown_speaker"
    row["context"] = [{"speaker": None, "utterance": "mystery line"}]
    row["questions"] = [_SHIM_ROW_WITH_QUESTIONS["questions"][0]]
    row["has_questions"] = True
    p = tmp_path / "u.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    conv = next(iter_fantom_conversations(p))
    rendered = conv.render_context()
    assert rendered == "Unknown: mystery line"


def test_empty_shim_file_yields_nothing(tmp_path: Path) -> None:
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    assert list(iter_fantom_conversations(p)) == []
    assert list(iter_fantom_items(p)) == []


def test_blank_lines_tolerated(tmp_path: Path) -> None:
    p = tmp_path / "blanks.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        fh.write("\n")
        fh.write(json.dumps(_SHIM_ROW_WITH_QUESTIONS) + "\n")
        fh.write("\n\n")
    convs = list(iter_fantom_conversations(p))
    assert len(convs) == 1


def test_context_turn_type() -> None:
    turn = FANToMContextTurn(speaker="Alice", utterance="hi")
    assert turn.speaker == "Alice"
    assert turn.utterance == "hi"
