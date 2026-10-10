"""Unit tests for the BigToM adapter and FANToM-shape projection.

Covers:

- ``BigToMScenarioCondition.from_shim`` round-trips the four forward
  conditions and raises on unknown input.
- ``iter_bigtom_items`` yields :class:`BigToMItem` objects with every
  required field; ``limit`` and ``scenario_condition`` filters work.
- ``bigtom_item_to_fantom_item`` projects to the shared FANToM shape
  with ``target_person=character``, ``speaker=character`` on the single
  context turn, and preserves scenario metadata.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from psychology_base.eval.benchmarks.public_datasets.bigtom import (
    BigToMItem,
    BigToMScenarioCondition,
    bigtom_item_to_fantom_item,
    iter_bigtom_items,
)
from psychology_base.eval.benchmarks.public_datasets.fantom import (
    FANToMQuestionType,
)


_SAMPLE_ROWS = [
    {
        "item_id": "bigtom-0-forward_belief_aware",
        "scenario_id": 0,
        "character": "Noor",
        "scenario_condition": "forward_belief_aware",
        "context": "Noor fills a pitcher with oat milk. A coworker swaps it; Noor sees.",
        "question": "Does Noor believe the pitcher contains oat or almond milk?",
        "correct_answer": "Noor believes the pitcher contains almond milk.",
        "incorrect_answer": "Noor believes the pitcher contains oat milk.",
        "order": 1,
    },
    {
        "item_id": "bigtom-0-forward_belief_unaware",
        "scenario_id": 0,
        "character": "Noor",
        "scenario_condition": "forward_belief_unaware",
        "context": "Noor fills a pitcher with oat milk. A coworker swaps it; Noor does NOT see.",
        "question": "Does Noor believe the pitcher contains oat or almond milk?",
        "correct_answer": "Noor believes the pitcher contains oat milk.",
        "incorrect_answer": "Noor believes the pitcher contains almond milk.",
        "order": 1,
    },
    {
        "item_id": "bigtom-0-forward_action_aware",
        "scenario_id": 0,
        "character": "Noor",
        "scenario_condition": "forward_action_aware",
        "context": "Noor fills a pitcher with oat milk. A coworker swaps it; Noor sees.",
        "question": "What will Noor do next?",
        "correct_answer": "Noor will use the almond milk or ask about the swap.",
        "incorrect_answer": "Noor will pour the oat milk as if nothing changed.",
        "order": 1,
    },
    {
        "item_id": "bigtom-0-forward_action_unaware",
        "scenario_id": 0,
        "character": "Noor",
        "scenario_condition": "forward_action_unaware",
        "context": "Noor fills a pitcher with oat milk. A coworker swaps it; Noor does NOT see.",
        "question": "What will Noor do next?",
        "correct_answer": "Noor will pour from the pitcher as if it still contains oat milk.",
        "incorrect_answer": "Noor will ask about the swap she witnessed.",
        "order": 1,
    },
]


@pytest.fixture
def sample_shim(tmp_path: Path) -> Path:
    shim = tmp_path / "items.jsonl"
    with shim.open("w", encoding="utf-8") as fh:
        for row in _SAMPLE_ROWS:
            fh.write(json.dumps(row))
            fh.write("\n")
    return shim


# ---------------------------------------------------------------------------
# BigToMScenarioCondition
# ---------------------------------------------------------------------------


def test_scenario_condition_from_shim_roundtrips_all_four() -> None:
    for value in (
        "forward_belief_aware",
        "forward_belief_unaware",
        "forward_action_aware",
        "forward_action_unaware",
    ):
        assert BigToMScenarioCondition.from_shim(value).value == value


def test_scenario_condition_from_shim_raises_on_unknown() -> None:
    with pytest.raises(ValueError, match="unknown BigToM scenario_condition"):
        BigToMScenarioCondition.from_shim("backward_belief_aware")


def test_scenario_condition_is_belief_flag() -> None:
    assert BigToMScenarioCondition.FORWARD_BELIEF_AWARE.is_belief is True
    assert BigToMScenarioCondition.FORWARD_BELIEF_UNAWARE.is_belief is True
    assert BigToMScenarioCondition.FORWARD_ACTION_AWARE.is_belief is False
    assert BigToMScenarioCondition.FORWARD_ACTION_UNAWARE.is_belief is False


def test_scenario_condition_is_unaware_flag() -> None:
    assert BigToMScenarioCondition.FORWARD_BELIEF_UNAWARE.is_unaware is True
    assert BigToMScenarioCondition.FORWARD_ACTION_UNAWARE.is_unaware is True
    assert BigToMScenarioCondition.FORWARD_BELIEF_AWARE.is_unaware is False
    assert BigToMScenarioCondition.FORWARD_ACTION_AWARE.is_unaware is False


# ---------------------------------------------------------------------------
# iter_bigtom_items
# ---------------------------------------------------------------------------


def test_iter_bigtom_items_reads_all_rows(sample_shim: Path) -> None:
    items = list(iter_bigtom_items(sample_shim))
    assert len(items) == 4
    assert all(isinstance(it, BigToMItem) for it in items)
    assert items[0].character == "Noor"
    assert items[0].scenario_condition is BigToMScenarioCondition.FORWARD_BELIEF_AWARE


def test_iter_bigtom_items_respects_limit(sample_shim: Path) -> None:
    items = list(iter_bigtom_items(sample_shim, limit=2))
    assert len(items) == 2


def test_iter_bigtom_items_filters_scenario_condition(sample_shim: Path) -> None:
    unaware_belief = list(
        iter_bigtom_items(
            sample_shim,
            scenario_condition=BigToMScenarioCondition.FORWARD_BELIEF_UNAWARE,
        )
    )
    assert len(unaware_belief) == 1
    assert unaware_belief[0].item_id == "bigtom-0-forward_belief_unaware"


# ---------------------------------------------------------------------------
# bigtom_item_to_fantom_item
# ---------------------------------------------------------------------------


def test_projection_sets_target_person_to_character(sample_shim: Path) -> None:
    item = next(iter_bigtom_items(sample_shim))
    fantom = bigtom_item_to_fantom_item(item)
    assert fantom.question.target_person == "Noor"
    assert fantom.conversation.participants == ("Noor",)


def test_projection_single_turn_speaker_is_character(sample_shim: Path) -> None:
    """REGRESSION PIN: the mentalistic path writes each turn with
    ``subject_id=speaker`` and recalls with ``scope=("subject",
    target_person)``. If the single context turn's speaker is NOT the
    character, recall returns empty and the mentalistic path collapses
    to the no-recall baseline."""

    item = next(iter_bigtom_items(sample_shim))
    fantom = bigtom_item_to_fantom_item(item)
    assert len(fantom.conversation.context) == 1
    assert fantom.conversation.context[0].speaker == "Noor"
    assert "Noor" in fantom.conversation.context[0].utterance


def test_projection_belief_question_type_mapped() -> None:
    item = BigToMItem(
        item_id="x",
        scenario_id=0,
        character="A",
        scenario_condition=BigToMScenarioCondition.FORWARD_BELIEF_UNAWARE,
        context="A does something.",
        question="What does A believe?",
        correct_answer="A believes X.",
        incorrect_answer="A believes Y.",
    )
    fantom = bigtom_item_to_fantom_item(item)
    assert fantom.question.question_type is FANToMQuestionType.BELIEF
    assert fantom.question.tom_type == "belief"
    assert fantom.question.missed_info_accessibility == "inaccessible"


def test_projection_action_question_type_is_none() -> None:
    item = BigToMItem(
        item_id="x",
        scenario_id=0,
        character="A",
        scenario_condition=BigToMScenarioCondition.FORWARD_ACTION_AWARE,
        context="A does something.",
        question="What does A do?",
        correct_answer="A does X.",
        incorrect_answer="A does Y.",
    )
    fantom = bigtom_item_to_fantom_item(item)
    assert fantom.question.question_type is None
    assert fantom.question.tom_type == "action"
    assert fantom.question.missed_info_accessibility == "accessible"


def test_projection_preserves_scenario_metadata(sample_shim: Path) -> None:
    item = next(iter_bigtom_items(sample_shim))
    fantom = bigtom_item_to_fantom_item(item)
    assert fantom.conversation.metadata["source"] == "bigtom"
    assert fantom.conversation.metadata["scenario_id"] == 0
    assert fantom.conversation.metadata["scenario_condition"] == "forward_belief_aware"


def test_projection_tom_order_is_one(sample_shim: Path) -> None:
    for item in iter_bigtom_items(sample_shim):
        fantom = bigtom_item_to_fantom_item(item)
        assert fantom.question.tom_order == 1
