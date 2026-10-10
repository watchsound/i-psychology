"""Regression pin — Pillar E ships Option S-flat scope selector.

Per `docs/phases/phase-8-e-plan.md` §4.3, Pillar E uses S-flat (first
agent in the chain only) rather than S-chained (recursive per-agent
scope filters). If this test turns red, S-chained was accidentally
substituted — the plan §4.3 deferral would need re-opening and the
Pillar E close-out doc's gate semantics would need to be re-reviewed.

The three invariants pinned here:

1. **HiToM order 2:** ``target_person`` is the OUTERMOST thinker (first
   element of ``agent_chain``), not the deepest.
2. **HiToM order 3:** same — outermost only.
3. **BigToM order 1:** ``target_person`` is the lone named character
   (trivially flat).

If the scope selector ever evolves to pass a tuple-of-agents instead of
a single string, this file must be updated in the same commit as the
design change.
"""

from __future__ import annotations

from psychology_base.eval.benchmarks.public_datasets.bigtom import (
    BigToMItem,
    BigToMScenarioCondition,
    bigtom_item_to_fantom_item,
)
from psychology_base.eval.benchmarks.public_datasets.hitom import (
    HiToMItem,
    HiToMOrder,
    hitom_item_to_fantom_item,
)


def test_sflat_hitom_order_2_target_is_outermost_thinker() -> None:
    item = HiToMItem(
        sample_id=0,
        order=HiToMOrder.ORDER_2,
        story="irrelevant",
        question="Where does Charlotte think Avery thinks the lettuce is?",
        choices={"A": "x", "B": "y"},
        choices_raw="A. x, B. y",
        answer_text="x",
        answer_letter="A",
        story_length=1,
        prompting_type="CoTP",
        deception=False,
        agent_chain=("Charlotte", "Avery"),
    )
    fantom = hitom_item_to_fantom_item(item)
    # S-flat invariant: outermost thinker, not deepest
    assert fantom.question.target_person == "Charlotte"
    assert fantom.question.target_person != "Avery"
    # target_person must be a plain string (not a tuple)
    assert isinstance(fantom.question.target_person, str)


def test_sflat_hitom_order_3_target_is_outermost_thinker() -> None:
    item = HiToMItem(
        sample_id=0,
        order=HiToMOrder.ORDER_3,
        story="irrelevant",
        question="Where does Owen think Charlotte thinks Avery thinks the lettuce is?",
        choices={"A": "x", "B": "y"},
        choices_raw="A. x, B. y",
        answer_text="x",
        answer_letter="A",
        story_length=1,
        prompting_type="CoTP",
        deception=False,
        agent_chain=("Owen", "Charlotte", "Avery"),
    )
    fantom = hitom_item_to_fantom_item(item)
    # S-flat: first element of the three-agent chain
    assert fantom.question.target_person == "Owen"


def test_sflat_bigtom_target_is_the_lone_character() -> None:
    item = BigToMItem(
        item_id="bigtom-0-forward_belief_unaware",
        scenario_id=0,
        character="Noor",
        scenario_condition=BigToMScenarioCondition.FORWARD_BELIEF_UNAWARE,
        context="x",
        question="What does Noor believe?",
        correct_answer="y",
        incorrect_answer="z",
    )
    fantom = bigtom_item_to_fantom_item(item)
    assert fantom.question.target_person == "Noor"
    assert isinstance(fantom.question.target_person, str)
