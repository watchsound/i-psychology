"""Unit tests for the higher-order ToM scorer (Pillar E thin wrapper).

Covers:

- Mutex kwarg contract: ``responder`` XOR ``item_responder``; both at
  once raises; neither raises.
- Order-level slicing via ``cid_to_order``: ``accuracy(order=2)``
  returns only order-2 verdicts.
- ``architectural_delta(order=2)`` matches
  ``accuracy(mentalistic, 2) − accuracy(bare_llm, 2)``.
- ``meets_pillar_e_gate`` thresholds: delta < 0.30 fails; min_items
  shortfall fails; accuracy_mentalistic < 0.40 fails; clearing all
  three passes.
- ``item_responder`` branch runs a fresh responder per item.
"""

from __future__ import annotations

import pytest

from psychology_base.eval.benchmarks.public_datasets.bigtom import (
    BigToMItem,
    BigToMScenarioCondition,
    bigtom_item_to_fantom_item,
)
from psychology_base.eval.benchmarks.public_datasets.fantom import FANToMItem
from psychology_base.eval.benchmarks.public_datasets.higher_order_tom_scorer import (
    HigherOrderToMReport,
    score_higher_order_tom,
)
from psychology_base.eval.benchmarks.public_datasets.hitom import (
    HiToMItem,
    HiToMOrder,
    hitom_item_to_fantom_item,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_hitom_item(sample_id: int, order: HiToMOrder, *, correct: str, wrong: str) -> HiToMItem:
    primary = "Avery" if order >= HiToMOrder.ORDER_1 else None
    chain = (primary,) if primary else ()
    if order >= HiToMOrder.ORDER_2:
        chain = ("Charlotte", "Avery")
    if order >= HiToMOrder.ORDER_3:
        chain = ("Owen", "Charlotte", "Avery")

    q = {
        HiToMOrder.ORDER_0: "Where is the lettuce really?",
        HiToMOrder.ORDER_1: "Where does Avery really think the lettuce is?",
        HiToMOrder.ORDER_2: "Where does Charlotte think Avery thinks the lettuce is?",
        HiToMOrder.ORDER_3: "Where does Owen think Charlotte thinks Avery thinks the lettuce is?",
        HiToMOrder.ORDER_4: "Where does Isabella think Owen thinks Charlotte thinks Avery thinks the lettuce is?",
    }[order]
    return HiToMItem(
        sample_id=sample_id,
        order=order,
        story=f"1 {chain[0] if chain else 'A'} entered.\n2 The lettuce is in the {correct}.",
        question=q,
        choices={"A": correct, "B": wrong},
        choices_raw=f"A. {correct}, B. {wrong}",
        answer_text=correct,
        answer_letter="A",
        story_length=1,
        prompting_type="CoTP",
        deception=False,
        agent_chain=chain,
    )


def _make_bigtom_item(scenario_id: int, *, correct: str, wrong: str) -> BigToMItem:
    return BigToMItem(
        item_id=f"bigtom-{scenario_id}-forward_belief_unaware",
        scenario_id=scenario_id,
        character="Noor",
        scenario_condition=BigToMScenarioCondition.FORWARD_BELIEF_UNAWARE,
        context="Noor fills a pitcher with oat milk. A coworker swaps it; Noor does NOT see.",
        question="Does Noor believe the pitcher contains oat or almond milk?",
        correct_answer=correct,
        incorrect_answer=wrong,
    )


def _scripted_responder_for_items(
    items: list[FANToMItem], *, mentalistic_correct: bool, bare_correct: bool
) -> callable:
    """Return a responder that answers correctly / wrongly deterministically.

    The scorer calls the responder once per path per item (mentalistic
    then bare); we count calls and route by position so we can control
    accuracy per path.
    """

    # Count how many items on each path we expect; prompts are built in
    # the scorer so we distinguish by a substring that appears only in
    # one path's prompt builder.
    def responder(prompt: str) -> str:
        is_mentalistic = (
            "filtered observations below" in prompt
            or "co-presence intersection" in prompt
        )
        if is_mentalistic:
            # Find the matching item by question text appearing in the prompt
            for it in items:
                if it.question.question_text in prompt:
                    return it.question.correct_answer if mentalistic_correct else it.question.wrong_answer
            return "undecidable"
        else:
            for it in items:
                if it.question.question_text in prompt:
                    return it.question.correct_answer if bare_correct else it.question.wrong_answer
            return "undecidable"

    return responder


# ---------------------------------------------------------------------------
# Mutex kwarg contract
# ---------------------------------------------------------------------------


def test_mutex_requires_at_least_one_responder() -> None:
    with pytest.raises(ValueError, match="requires one of"):
        score_higher_order_tom([])


def test_mutex_rejects_both_responders() -> None:
    with pytest.raises(ValueError, match="EXACTLY ONE"):
        score_higher_order_tom(
            [],
            responder=lambda p: "x",
            item_responder=lambda it: (lambda p: "x"),
        )


# ---------------------------------------------------------------------------
# Order-level slicing + architectural delta
# ---------------------------------------------------------------------------


def test_report_slices_by_order() -> None:
    """Build a mixed-order batch; verify accuracy(order=X) only counts
    order-X verdicts."""

    hitom_1 = _make_hitom_item(1, HiToMOrder.ORDER_1, correct="green_drawer", wrong="blue_pantry")
    hitom_2 = _make_hitom_item(2, HiToMOrder.ORDER_2, correct="green_drawer", wrong="blue_pantry")
    items = [
        hitom_item_to_fantom_item(hitom_1),
        hitom_item_to_fantom_item(hitom_2),
    ]
    responder = _scripted_responder_for_items(items, mentalistic_correct=True, bare_correct=False)

    report = score_higher_order_tom(items, responder=responder, dataset_tag="hitom")
    assert isinstance(report, HigherOrderToMReport)
    assert report.cid_to_order == {items[0].conversation.conversation_id: 1, items[1].conversation.conversation_id: 2}

    # 2 items per order per path = 2 verdicts order-1 each path, 2 verdicts order-2 each path
    assert report.item_count(path="mentalistic", order=1) == 1
    assert report.item_count(path="mentalistic", order=2) == 1
    assert report.item_count(path="bare_llm", order=1) == 1
    assert report.item_count(path="bare_llm", order=2) == 1


def test_architectural_delta_order_slice() -> None:
    hitom_items = [
        _make_hitom_item(i, HiToMOrder.ORDER_2, correct="green_drawer", wrong="blue_pantry")
        for i in range(4)
    ]
    items = [hitom_item_to_fantom_item(it) for it in hitom_items]
    # Mentalistic 4/4 correct; bare 0/4 correct → delta=+1.0
    responder = _scripted_responder_for_items(items, mentalistic_correct=True, bare_correct=False)
    report = score_higher_order_tom(items, responder=responder)
    assert report.accuracy(path="mentalistic", order=2) == 1.0
    assert report.accuracy(path="bare_llm", order=2) == 0.0
    assert report.architectural_delta(order=2) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# meets_pillar_e_gate
# ---------------------------------------------------------------------------


def test_pillar_e_gate_passes_when_delta_and_floor_cleared() -> None:
    hitom_items = [
        _make_hitom_item(i, HiToMOrder.ORDER_2, correct="green_drawer", wrong="blue_pantry")
        for i in range(50)
    ]
    items = [hitom_item_to_fantom_item(it) for it in hitom_items]
    responder = _scripted_responder_for_items(items, mentalistic_correct=True, bare_correct=False)
    report = score_higher_order_tom(items, responder=responder)
    assert report.meets_pillar_e_gate(order=2, min_items=50) is True


def test_pillar_e_gate_fails_when_min_items_short() -> None:
    hitom_items = [
        _make_hitom_item(i, HiToMOrder.ORDER_2, correct="green_drawer", wrong="blue_pantry")
        for i in range(10)
    ]
    items = [hitom_item_to_fantom_item(it) for it in hitom_items]
    responder = _scripted_responder_for_items(items, mentalistic_correct=True, bare_correct=False)
    report = score_higher_order_tom(items, responder=responder)
    assert report.meets_pillar_e_gate(order=2, min_items=50) is False


def test_pillar_e_gate_fails_when_delta_below_floor() -> None:
    hitom_items = [
        _make_hitom_item(i, HiToMOrder.ORDER_2, correct="green_drawer", wrong="blue_pantry")
        for i in range(50)
    ]
    items = [hitom_item_to_fantom_item(it) for it in hitom_items]
    # BOTH paths all-correct → delta=0.0 < 0.30 → fail
    responder = _scripted_responder_for_items(items, mentalistic_correct=True, bare_correct=True)
    report = score_higher_order_tom(items, responder=responder)
    assert report.architectural_delta(order=2) == pytest.approx(0.0)
    assert report.meets_pillar_e_gate(order=2, min_items=50) is False


def test_pillar_e_gate_fails_when_mentalistic_accuracy_below_floor() -> None:
    hitom_items = [
        _make_hitom_item(i, HiToMOrder.ORDER_2, correct="green_drawer", wrong="blue_pantry")
        for i in range(50)
    ]
    items = [hitom_item_to_fantom_item(it) for it in hitom_items]
    # Both wrong → accuracy=0 both paths, delta=0 → fail on floor
    responder = _scripted_responder_for_items(items, mentalistic_correct=False, bare_correct=False)
    report = score_higher_order_tom(items, responder=responder)
    assert report.accuracy(path="mentalistic", order=2) == 0.0
    assert report.meets_pillar_e_gate(order=2, min_items=50, accuracy_floor=0.40) is False


# ---------------------------------------------------------------------------
# item_responder branch
# ---------------------------------------------------------------------------


def test_item_responder_called_per_item() -> None:
    hitom_items = [
        _make_hitom_item(i, HiToMOrder.ORDER_2, correct="green_drawer", wrong="blue_pantry")
        for i in range(3)
    ]
    items = [hitom_item_to_fantom_item(it) for it in hitom_items]
    build_count = {"n": 0}

    def item_responder_factory(it: FANToMItem):
        build_count["n"] += 1
        return _scripted_responder_for_items(items, mentalistic_correct=True, bare_correct=False)

    report = score_higher_order_tom(items, item_responder=item_responder_factory)
    assert build_count["n"] == 3
    # Each item built one responder, that responder served both paths
    assert report.accuracy(path="mentalistic", order=2) == 1.0
    assert report.accuracy(path="bare_llm", order=2) == 0.0


# ---------------------------------------------------------------------------
# BigToM order-1 scoring sanity
# ---------------------------------------------------------------------------


def test_bigtom_items_project_to_order_1_and_score() -> None:
    bigtom_items = [
        _make_bigtom_item(i, correct=f"gold-{i}", wrong=f"distractor-{i}")
        for i in range(3)
    ]
    items = [bigtom_item_to_fantom_item(it) for it in bigtom_items]
    responder = _scripted_responder_for_items(items, mentalistic_correct=True, bare_correct=False)
    report = score_higher_order_tom(items, responder=responder, dataset_tag="bigtom")
    assert report.cid_to_order == {items[0].conversation.conversation_id: 1,
                                    items[1].conversation.conversation_id: 1,
                                    items[2].conversation.conversation_id: 1}
    assert report.accuracy(path="mentalistic", order=1) == 1.0
    assert report.accuracy(path="bare_llm", order=1) == 0.0
    assert report.dataset_tag == "bigtom"


# ---------------------------------------------------------------------------
# HiToM-specific prompt builders — protocol conformance fix
# ---------------------------------------------------------------------------


from psychology_base.eval.benchmarks.public_datasets.higher_order_tom_scorer import (
    _build_hitom_bare_prompt,
    _build_hitom_mentalistic_prompt,
    _hitom_chain_scaffold,
    _hitom_choices_block,
)


def test_hitom_choices_block_renders_15_container_list() -> None:
    hitom = _make_hitom_item(0, HiToMOrder.ORDER_2, correct="green_bathtub", wrong="blue_pantry")
    item = hitom_item_to_fantom_item(hitom)
    block = _hitom_choices_block(item)
    assert "Possible answers" in block
    assert "- green_bathtub" in block
    assert "- blue_pantry" in block


def test_hitom_choices_block_empty_when_no_choices() -> None:
    """Guards the ``if not choices`` branch — empty-choices items (e.g.,
    BigToM-projected) must not emit a choices block."""
    import dataclasses

    hitom = _make_hitom_item(0, HiToMOrder.ORDER_2, correct="green_drawer", wrong="blue_pantry")
    item = hitom_item_to_fantom_item(hitom)
    stripped_q = dataclasses.replace(item.question, choices=())
    stripped = dataclasses.replace(item, question=stripped_q)
    assert _hitom_choices_block(stripped) == ""


def test_hitom_chain_scaffold_order_0_factual() -> None:
    hitom = _make_hitom_item(0, HiToMOrder.ORDER_0, correct="green_drawer", wrong="blue_pantry")
    item = hitom_item_to_fantom_item(hitom)
    scaffold = _hitom_chain_scaffold(item)
    assert "factual" in scaffold.lower()


def test_hitom_chain_scaffold_order_1_first_order() -> None:
    hitom = _make_hitom_item(0, HiToMOrder.ORDER_1, correct="green_drawer", wrong="blue_pantry")
    item = hitom_item_to_fantom_item(hitom)
    scaffold = _hitom_chain_scaffold(item)
    assert "first-order" in scaffold.lower()
    assert "Avery" in scaffold


def test_hitom_chain_scaffold_order_2_nested_belief() -> None:
    hitom = _make_hitom_item(0, HiToMOrder.ORDER_2, correct="green_drawer", wrong="blue_pantry")
    item = hitom_item_to_fantom_item(hitom)
    scaffold = _hitom_chain_scaffold(item)
    # Explicit nested-belief description + co-presence requirement.
    assert "Charlotte's model of" in scaffold
    assert "Avery's belief" in scaffold
    assert "same room" in scaffold
    assert "Charlotte, Avery" in scaffold


def test_hitom_chain_scaffold_order_3_three_level_nest() -> None:
    hitom = _make_hitom_item(0, HiToMOrder.ORDER_3, correct="green_drawer", wrong="blue_pantry")
    item = hitom_item_to_fantom_item(hitom)
    scaffold = _hitom_chain_scaffold(item)
    # Chain is (Owen, Charlotte, Avery). Expect two "model of" layers.
    assert "Owen's model of" in scaffold
    assert "Charlotte's model of" in scaffold
    assert "Avery's belief" in scaffold


def test_build_hitom_mentalistic_prompt_includes_choices_scaffold_and_answer_discipline() -> None:
    hitom = _make_hitom_item(0, HiToMOrder.ORDER_2, correct="green_drawer", wrong="blue_pantry")
    item = hitom_item_to_fantom_item(hitom)
    prompt = _build_hitom_mentalistic_prompt(item, "step 1: ...\nstep 2: ...")
    # Detection substring the scripted responder uses.
    assert "co-presence intersection" in prompt
    # Choices and chain scaffold both present.
    assert "- green_drawer" in prompt
    assert "- blue_pantry" in prompt
    assert "Charlotte's model of" in prompt
    # Answer discipline.
    assert "exact container name" in prompt
    assert item.question.question_text in prompt


def test_build_hitom_bare_prompt_shows_full_story_and_choices() -> None:
    hitom = _make_hitom_item(0, HiToMOrder.ORDER_2, correct="green_drawer", wrong="blue_pantry")
    item = hitom_item_to_fantom_item(hitom)
    prompt = _build_hitom_bare_prompt(item)
    # Bare path shows the full story (not the witness-filtered events).
    assert "Full story:" in prompt
    assert "lettuce" in prompt  # body of _make_hitom_item story
    # Choices and chain scaffold shown here too (standard HiToM protocol).
    assert "- green_drawer" in prompt
    assert "- blue_pantry" in prompt
    assert "Charlotte's model of" in prompt
    assert "exact container name" in prompt
    # Must NOT bleed the mentalistic detection substring.
    assert "co-presence intersection" not in prompt


def test_hitom_prompts_dispatch_at_scoring_time() -> None:
    """End-to-end: HiToM items get HiToM prompts, BigToM items get FANToM
    prompts. Scripted responder detects via substrings and asserts both
    shapes work in one mixed batch."""
    hitom_items = [
        _make_hitom_item(i, HiToMOrder.ORDER_2, correct="green_drawer", wrong="blue_pantry")
        for i in range(2)
    ]
    bigtom_items = [
        _make_bigtom_item(10 + i, correct=f"gold-{i}", wrong=f"distractor-{i}")
        for i in range(2)
    ]
    items = [hitom_item_to_fantom_item(it) for it in hitom_items]
    items.extend(bigtom_item_to_fantom_item(it) for it in bigtom_items)

    seen = {"hitom_mental": 0, "fantom_mental": 0, "hitom_bare": 0, "fantom_bare": 0}

    def responder(prompt: str) -> str:
        is_hitom_mental = "co-presence intersection" in prompt
        is_fantom_mental = "filtered observations below" in prompt
        is_hitom_bare = "Full story:" in prompt
        if is_hitom_mental:
            seen["hitom_mental"] += 1
        elif is_fantom_mental:
            seen["fantom_mental"] += 1
        elif is_hitom_bare:
            seen["hitom_bare"] += 1
        else:
            seen["fantom_bare"] += 1
        for it in items:
            if it.question.question_text in prompt:
                return it.question.correct_answer
        return "undecidable"

    score_higher_order_tom(items, responder=responder, dataset_tag="combined")
    assert seen["hitom_mental"] == 2
    assert seen["hitom_bare"] == 2
    assert seen["fantom_mental"] == 2
    assert seen["fantom_bare"] == 2
