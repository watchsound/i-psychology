"""Unit tests for the HiToM adapter and FANToM-shape projection.

Covers:

- ``_extract_agent_chain`` on each order (0–4), including the
  ``really think`` order-1 variant.
- ``HiToMOrder.from_int`` round-trip + raise on unknown.
- ``iter_hitom_items`` yields :class:`HiToMItem`; ``order`` filter and
  ``limit`` work.
- ``hitom_item_to_fantom_item`` sets ``target_person=primary_subject``
  (S-flat), speaker=primary_subject on the single turn, correct choice
  tuple ordering, metadata preserves full agent chain.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from psychology_base.eval.benchmarks.public_datasets.fantom import (
    FANToMQuestionType,
)
from psychology_base.eval.benchmarks.public_datasets.hitom import (
    HiToMItem,
    HiToMOrder,
    _extract_agent_chain,
    hitom_item_to_fantom_item,
    iter_hitom_items,
)


# ---------------------------------------------------------------------------
# _extract_agent_chain — regex pin
# ---------------------------------------------------------------------------


def test_extract_chain_order_0_empty() -> None:
    assert _extract_agent_chain("Where is the lettuce really?") == ()


def test_extract_chain_order_1_really_think() -> None:
    """REGRESSION PIN: order-1 HiToM questions use "really think"; the
    adverb must not block the first-agent capture."""

    assert _extract_agent_chain(
        "Where does Avery really think the lettuce is?"
    ) == ("Avery",)


def test_extract_chain_order_2_think_thinks() -> None:
    assert _extract_agent_chain(
        "Where does Charlotte think Avery thinks the lettuce is?"
    ) == ("Charlotte", "Avery")


def test_extract_chain_order_3() -> None:
    assert _extract_agent_chain(
        "Where does Owen think Charlotte thinks Avery thinks the lettuce is?"
    ) == ("Owen", "Charlotte", "Avery")


def test_extract_chain_order_4() -> None:
    assert _extract_agent_chain(
        "Where does Isabella think Owen thinks Charlotte thinks Avery thinks the lettuce is?"
    ) == ("Isabella", "Owen", "Charlotte", "Avery")


# ---------------------------------------------------------------------------
# HiToMOrder
# ---------------------------------------------------------------------------


def test_hitom_order_from_int_roundtrips() -> None:
    for n in range(5):
        assert int(HiToMOrder.from_int(n)) == n


def test_hitom_order_from_int_raises_on_unknown() -> None:
    with pytest.raises(ValueError, match="unknown HiToM order"):
        HiToMOrder.from_int(5)


# ---------------------------------------------------------------------------
# iter_hitom_items + shim fixture
# ---------------------------------------------------------------------------


_SAMPLE_ROWS = [
    {
        "sample_id": 0,
        "order": 0,
        "story": "1 Avery entered the living_room.\n2 The lettuce is in the green_drawer.",
        "question": "Where is the lettuce really?",
        "choices_raw": "A. blue_drawer, B. green_drawer",
        "choices": {"A": "blue_drawer", "B": "green_drawer"},
        "answer_text": "green_drawer",
        "answer_letter": "B",
        "story_length": 1,
        "prompting_type": "CoTP",
        "deception": False,
    },
    {
        "sample_id": 1,
        "order": 1,
        "story": "1 Avery entered the living_room.\n2 The lettuce is in the green_drawer.\n3 Avery moved the lettuce to the blue_pantry.",
        "question": "Where does Avery really think the lettuce is?",
        "choices_raw": "A. blue_pantry, B. green_drawer",
        "choices": {"A": "blue_pantry", "B": "green_drawer"},
        "answer_text": "blue_pantry",
        "answer_letter": "A",
        "story_length": 1,
        "prompting_type": "VP",
        "deception": False,
    },
    {
        "sample_id": 2,
        "order": 2,
        "story": "1 Avery, Charlotte entered.\n2 Avery exited.\n3 Charlotte moved the lettuce.",
        "question": "Where does Charlotte think Avery thinks the lettuce is?",
        "choices_raw": "A. blue_pantry, B. green_drawer",
        "choices": {"A": "blue_pantry", "B": "green_drawer"},
        "answer_text": "green_drawer",
        "answer_letter": "B",
        "story_length": 2,
        "prompting_type": "CoTP",
        "deception": True,
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


def test_iter_hitom_items_reads_all_rows(sample_shim: Path) -> None:
    items = list(iter_hitom_items(sample_shim))
    assert len(items) == 3
    assert all(isinstance(it, HiToMItem) for it in items)
    assert items[0].order is HiToMOrder.ORDER_0
    assert items[1].order is HiToMOrder.ORDER_1
    assert items[2].order is HiToMOrder.ORDER_2


def test_iter_hitom_items_filters_order(sample_shim: Path) -> None:
    only_2 = list(iter_hitom_items(sample_shim, order=HiToMOrder.ORDER_2))
    assert len(only_2) == 1
    assert only_2[0].sample_id == 2


def test_iter_hitom_items_respects_limit(sample_shim: Path) -> None:
    first_two = list(iter_hitom_items(sample_shim, limit=2))
    assert len(first_two) == 2


def test_hitom_item_primary_subject(sample_shim: Path) -> None:
    items = list(iter_hitom_items(sample_shim))
    assert items[0].primary_subject is None  # order 0, no agent
    assert items[1].primary_subject == "Avery"
    assert items[2].primary_subject == "Charlotte"


# ---------------------------------------------------------------------------
# hitom_item_to_fantom_item — S-flat projection
# ---------------------------------------------------------------------------


def test_projection_target_person_is_primary_subject_sflat(sample_shim: Path) -> None:
    """REGRESSION PIN: Pillar E ships S-flat (primary subject only).
    If this test turns red, S-chained was accidentally substituted —
    the plan §4.3 deferral would need re-opening."""

    items = list(iter_hitom_items(sample_shim))
    order_2 = next(it for it in items if it.order is HiToMOrder.ORDER_2)
    fantom = hitom_item_to_fantom_item(order_2)
    assert fantom.question.target_person == "Charlotte"
    # The chain still has Avery nested in it — but target_person is NOT Avery.
    assert order_2.agent_chain == ("Charlotte", "Avery")


def test_projection_order_0_target_person_is_none(sample_shim: Path) -> None:
    items = list(iter_hitom_items(sample_shim, order=HiToMOrder.ORDER_0))
    fantom = hitom_item_to_fantom_item(items[0])
    assert fantom.question.target_person is None
    assert fantom.question.question_type is FANToMQuestionType.FACT
    assert fantom.question.tom_type == "fact"


def test_projection_order_1_belief(sample_shim: Path) -> None:
    items = list(iter_hitom_items(sample_shim, order=HiToMOrder.ORDER_1))
    fantom = hitom_item_to_fantom_item(items[0])
    assert fantom.question.question_type is FANToMQuestionType.BELIEF
    assert fantom.question.tom_type == "belief"
    assert fantom.question.tom_order == 1


def test_projection_order_2_higher_order_belief(sample_shim: Path) -> None:
    items = list(iter_hitom_items(sample_shim, order=HiToMOrder.ORDER_2))
    fantom = hitom_item_to_fantom_item(items[0])
    assert fantom.question.question_type is FANToMQuestionType.BELIEF
    assert fantom.question.tom_type == "higher_order_belief"
    assert fantom.question.tom_order == 2


def test_projection_single_turn_speaker_is_primary_subject(sample_shim: Path) -> None:
    items = list(iter_hitom_items(sample_shim, order=HiToMOrder.ORDER_2))
    fantom = hitom_item_to_fantom_item(items[0])
    assert len(fantom.conversation.context) == 1
    assert fantom.conversation.context[0].speaker == "Charlotte"


def test_projection_choices_sorted_by_letter(sample_shim: Path) -> None:
    items = list(iter_hitom_items(sample_shim, order=HiToMOrder.ORDER_1))
    fantom = hitom_item_to_fantom_item(items[0])
    # Letters A, B → sorted → ("blue_pantry", "green_drawer")
    assert fantom.question.choices == ("blue_pantry", "green_drawer")


def test_projection_wrong_distractor_is_not_gold(sample_shim: Path) -> None:
    items = list(iter_hitom_items(sample_shim, order=HiToMOrder.ORDER_1))
    fantom = hitom_item_to_fantom_item(items[0])
    # Gold = "blue_pantry"; distractor must not equal gold
    assert fantom.question.wrong_answer == "green_drawer"
    assert fantom.question.wrong_answer != fantom.question.correct_answer


def test_projection_metadata_preserves_chain(sample_shim: Path) -> None:
    items = list(iter_hitom_items(sample_shim, order=HiToMOrder.ORDER_2))
    fantom = hitom_item_to_fantom_item(items[0])
    assert fantom.conversation.metadata["source"] == "hitom"
    assert fantom.conversation.metadata["agent_chain"] == ["Charlotte", "Avery"]
    assert fantom.conversation.metadata["order"] == 2
    assert fantom.conversation.metadata["prompting_type"] == "CoTP"
    assert fantom.conversation.metadata["deception"] is True


def test_projection_order_0_speaker_fallback() -> None:
    """Order 0 has no agent chain; the single-turn speaker must still
    be set so FANToMContextTurn's non-None invariant isn't tripped on
    the mentalistic path's observe() call."""

    item = HiToMItem(
        sample_id=99,
        order=HiToMOrder.ORDER_0,
        story="The lettuce is in the drawer.",
        question="Where is the lettuce really?",
        choices={"A": "drawer", "B": "shelf"},
        choices_raw="A. drawer, B. shelf",
        answer_text="drawer",
        answer_letter="A",
        story_length=1,
        prompting_type="CoTP",
        deception=False,
        agent_chain=(),
    )
    fantom = hitom_item_to_fantom_item(item)
    assert fantom.conversation.context[0].speaker == "Narrator"
    assert fantom.conversation.participants == ("Narrator",)
