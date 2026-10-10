"""BigToM adapter — order-1 ToM via Gandhi et al. 2024 procedural scenarios.

**Target gate (Phase 8 Pillar E, sanity cross-check).** BigToM at order
1 is the lower-order sanity-check that complements the HiToM order-2
gate. The headline Pillar E metric is HiToM order 2
``accuracy_mentalistic − accuracy_bare_llm ≥ +0.30`` at ``min_items=50``;
BigToM is included to show the architectural delta holds at order 1
on an independent dataset family (procedurally generated causal
belief/action pairs vs HiToM's symbolic room/container worlds).

**Shape of items.** BigToM pairs each scenario with a *perception*
condition and a *forward* question type:

- **percept-aware** — the character WITNESSED the environmental change.
  Correct belief reflects the new state.
- **percept-unaware** — the character DID NOT WITNESS the change.
  Correct belief reflects the OLD state (classic false-belief).
- **forward-belief-{aware,unaware}** — "what does X believe about Y?"
- **forward-action-{aware,unaware}** — "what does X do next?"

Each CSV row materialises to **four forward items** (belief/action ×
aware/unaware) through the provisioner. All items carry
``order=1`` — the character has a belief ABOUT the world; no nested
"X believes Y believes Z" structure.

**Projection to FANToM shape.** The scorer reuses
``fantom_tom_scorer.score_fantom_tom`` to keep the mentalistic-vs-bare
delta measurement byte-identical across Pillar E's two sub-corpora.
``bigtom_item_to_fantom_item`` wraps each BigToM item as a
single-turn FANToM conversation where:

- ``participants=(character,)`` — the one named agent
- ``context=(FANToMContextTurn(speaker=character, utterance=story),)``
  — the story goes under the character's ``subject:`` tag so mentalistic
  recall with ``scope=("subject", character)`` surfaces it
- ``questions=(FANToMQuestion(target_person=character, tom_order=1, ...))``

**License.** BigToM is MIT (per Gandhi et al. 2024 upstream repo
``kanishkg/bigtom``). This shim is a derivative work and inherits MIT;
``.data/`` is gitignored.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterator

from psychology_base.eval.benchmarks.public_datasets.fantom import (
    FANToMContextTurn,
    FANToMConversation,
    FANToMItem,
    FANToMQuestion,
    FANToMQuestionType,
)


class BigToMScenarioCondition(Enum):
    """BigToM forward-condition taxonomy.

    Four conditions cross *question kind* (belief vs action) with
    *perceptual access* (aware vs unaware of the environmental
    change). The ``_UNAWARE`` variants are the ToM-load conditions —
    the character's belief has diverged from the current world state
    and the correct answer requires reasoning *inside* their epistemic
    frame.
    """

    FORWARD_BELIEF_AWARE = "forward_belief_aware"
    FORWARD_BELIEF_UNAWARE = "forward_belief_unaware"
    FORWARD_ACTION_AWARE = "forward_action_aware"
    FORWARD_ACTION_UNAWARE = "forward_action_unaware"

    @classmethod
    def from_shim(cls, raw: str) -> "BigToMScenarioCondition":
        for member in cls:
            if member.value == raw:
                return member
        raise ValueError(
            f"unknown BigToM scenario_condition: {raw!r}; "
            f"expected one of {[m.value for m in cls]}"
        )

    @property
    def is_belief(self) -> bool:
        return self in (
            BigToMScenarioCondition.FORWARD_BELIEF_AWARE,
            BigToMScenarioCondition.FORWARD_BELIEF_UNAWARE,
        )

    @property
    def is_unaware(self) -> bool:
        return self in (
            BigToMScenarioCondition.FORWARD_BELIEF_UNAWARE,
            BigToMScenarioCondition.FORWARD_ACTION_UNAWARE,
        )


@dataclass(frozen=True)
class BigToMItem:
    """One scorer-ready BigToM item.

    ``order`` is always 1 for BigToM — the character has a first-order
    belief about the world. ``scenario_id`` groups the four forward
    conditions that share a story stem; ``item_id`` is unique per
    forward condition.
    """

    item_id: str
    scenario_id: int
    character: str
    scenario_condition: BigToMScenarioCondition
    context: str
    question: str
    correct_answer: str
    incorrect_answer: str
    order: int = 1


def _parse_item(raw: dict[str, object]) -> BigToMItem:
    return BigToMItem(
        item_id=str(raw["item_id"]),
        scenario_id=int(raw["scenario_id"]),  # type: ignore[arg-type]
        character=str(raw["character"]),
        scenario_condition=BigToMScenarioCondition.from_shim(
            str(raw["scenario_condition"])
        ),
        context=str(raw["context"]),
        question=str(raw["question"]),
        correct_answer=str(raw["correct_answer"]),
        incorrect_answer=str(raw["incorrect_answer"]),
        order=int(raw.get("order", 1)),  # type: ignore[arg-type]
    )


def iter_bigtom_items(
    shim_path: str | Path,
    *,
    limit: int | None = None,
    scenario_condition: BigToMScenarioCondition | None = None,
) -> Iterator[BigToMItem]:
    """Yield BigToM items from the provisioner JSONL shim.

    ``scenario_condition`` filters to one of the four forward conditions;
    pass ``BigToMScenarioCondition.FORWARD_BELIEF_UNAWARE`` for the
    classic false-belief subset.
    """

    path = Path(shim_path)
    emitted = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            item = _parse_item(raw)
            if scenario_condition is not None and item.scenario_condition is not scenario_condition:
                continue
            yield item
            emitted += 1
            if limit is not None and emitted >= limit:
                return


def bigtom_item_to_fantom_item(item: BigToMItem) -> FANToMItem:
    """Project a BigToM item onto the shared FANToM scorer shape.

    The story text is written as a single context turn whose speaker is
    the character. On the mentalistic path, the scorer writes this turn
    with ``subject_id=character``; recall scoped to that subject
    surfaces the exact scenario text. The bare-LLM path sees the same
    text verbatim via ``render_context()`` so any accuracy delta is
    attributable to the retrieval-scope filter, not to prompt content.

    Belief questions map to ``FANToMQuestionType.BELIEF``; action
    questions carry ``question_type=None`` (BigToM's action variants
    don't match any FANToM belief-category label and we don't fabricate
    one). The ``scenario_condition`` string is stashed in conversation
    metadata so sub-filters (e.g. unaware-only) can slice the report.
    """

    question_type = (
        FANToMQuestionType.BELIEF if item.scenario_condition.is_belief else None
    )
    turn = FANToMContextTurn(speaker=item.character, utterance=item.context)
    conv = FANToMConversation(
        conversation_id=item.item_id,
        participants=(item.character,),
        context=(turn,),
        questions=(
            FANToMQuestion(
                question_type=question_type,
                question_text=item.question,
                target_person=item.character,
                correct_answer=item.correct_answer,
                wrong_answer=item.incorrect_answer,
                tom_type="belief" if item.scenario_condition.is_belief else "action",
                tom_order=item.order,
                missed_info_accessibility=(
                    "inaccessible" if item.scenario_condition.is_unaware else "accessible"
                ),
                choices=None,
            ),
        ),
        metadata={
            "source": "bigtom",
            "scenario_id": item.scenario_id,
            "scenario_condition": item.scenario_condition.value,
        },
    )
    return FANToMItem(conversation=conv, question=conv.questions[0])


__all__ = [
    "BigToMItem",
    "BigToMScenarioCondition",
    "bigtom_item_to_fantom_item",
    "iter_bigtom_items",
]
