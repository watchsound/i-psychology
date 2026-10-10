"""HiToM adapter — higher-order ToM (orders 0–4) via He/Wu et al. 2023.

**Target gate (Phase 8 Pillar E headline).** HiToM order 2 at
``min_items=50`` is the Pillar E hard gate:
``accuracy_mentalistic − accuracy_bare_llm ≥ +0.30`` with
``accuracy_mentalistic ≥ 0.40``. Order 3 at the same N is the soft
criterion; order 4 is open research (reported, not gated).

**Shape of items.** Each HiToM item carries a symbolic container-world
story (agents entering/leaving a room, moving an object) plus one
question. Question depth follows the agent chain:

- order 0 — "Where is the lettuce really?" (world truth; no agent)
- order 1 — "Where does Avery really think the lettuce is?"
- order 2 — "Where does Charlotte think Avery thinks the lettuce is?"
- order 3 — "Where does Owen think Charlotte thinks Avery thinks …"
- order 4 — "Where does Isabella think Owen thinks …"

The ``_CHAIN_RE`` regex extracts the ordered chain of agent names
from the question text. Order 0 yields ``()``; order 1 yields the
single named agent; higher orders yield the full nested list.

**Scope-selector: Option S-flat (plan §4.3 decision).** Only the
*first* agent in the chain is passed as ``target_person`` for recall
scoping. The remaining chain nests inside the LLM's answer-generation
step. S-chained (recursive per-agent scope filters) is deferred to
Phase 8+ per the plan — it would require projecting mental-state
chains onto the memory substrate's scope model, which is not shippable
in Pillar E's horizon.

**Projection to FANToM shape.** Each HiToM story becomes a single
``FANToMContextTurn`` whose speaker is the first agent in the chain
(so ``scope=("subject", first_agent)`` surfaces it on the mentalistic
path). ``choices`` carry the full MC list from the shim; the FANToM
scorer's lexical judge then compares the model's answer against
``choices`` and the gold ``answer_text``. ``tom_order`` reflects the
HiToM order 0–4; ``metadata`` preserves the full agent chain,
``story_length``, ``prompting_type``, and ``deception`` so sub-filters
can slice the report.

**License.** HiToM is Apache-2.0 upstream (``ying-hui-he/Hi-ToM_dataset``).
This shim is a derivative work and inherits Apache-2.0;
``.data/`` is gitignored.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Iterator

from psychology_base.eval.benchmarks.public_datasets.fantom import (
    FANToMContextTurn,
    FANToMConversation,
    FANToMItem,
    FANToMQuestion,
    FANToMQuestionType,
)


# Agent-chain regex. Handles:
#   order 1 "does Avery really think"  → Avery
#   order 2 "Charlotte think Avery thinks" → Charlotte, Avery
#   order 3 "Owen think Charlotte thinks Avery thinks" → Owen, Charlotte, Avery
# The optional "really" absorbs the order-1 adverb; "thinks?" matches
# both the first-position singular "think" (after "does") and the
# subsequent plural "thinks".
_CHAIN_RE = re.compile(r"\b([A-Z][a-z]+)\s+(?:really\s+)?thinks?\b")


class HiToMOrder(IntEnum):
    """HiToM question depth, 0 (world) through 4 (fifth-order ToM)."""

    ORDER_0 = 0
    ORDER_1 = 1
    ORDER_2 = 2
    ORDER_3 = 3
    ORDER_4 = 4

    @classmethod
    def from_int(cls, value: int) -> "HiToMOrder":
        for member in cls:
            if int(member) == int(value):
                return member
        raise ValueError(
            f"unknown HiToM order: {value!r}; expected 0..4"
        )


def _extract_agent_chain(question: str) -> tuple[str, ...]:
    """Return the ordered chain of agent names in a HiToM question.

    Order 0 ("Where is the X really?") yields ``()``. Order N≥1 yields
    a tuple of length N. The first element is the outermost thinker
    (used as S-flat scope target); subsequent elements nest inside.
    """

    return tuple(_CHAIN_RE.findall(question))


@dataclass(frozen=True)
class HiToMItem:
    """One scorer-ready HiToM item."""

    sample_id: int
    order: HiToMOrder
    story: str
    question: str
    choices: dict[str, str]  # letter → container
    choices_raw: str
    answer_text: str
    answer_letter: str | None
    story_length: int
    prompting_type: str
    deception: bool
    agent_chain: tuple[str, ...]

    @property
    def primary_subject(self) -> str | None:
        """The outermost thinker; ``None`` for order 0."""

        return self.agent_chain[0] if self.agent_chain else None


def _parse_item(raw: dict[str, object]) -> HiToMItem:
    choices_raw = str(raw.get("choices_raw") or "")
    choices_obj = raw.get("choices") or {}
    if not isinstance(choices_obj, dict):
        choices_obj = {}
    choices: dict[str, str] = {
        str(k): str(v) for k, v in choices_obj.items()
    }
    answer_letter_raw = raw.get("answer_letter")
    answer_letter: str | None = (
        str(answer_letter_raw) if answer_letter_raw not in (None, "") else None
    )
    question_text = str(raw["question"])
    return HiToMItem(
        sample_id=int(raw["sample_id"]),  # type: ignore[arg-type]
        order=HiToMOrder.from_int(int(raw["order"])),  # type: ignore[arg-type]
        story=str(raw["story"]),
        question=question_text,
        choices=choices,
        choices_raw=choices_raw,
        answer_text=str(raw["answer_text"]),
        answer_letter=answer_letter,
        story_length=int(raw["story_length"]),  # type: ignore[arg-type]
        prompting_type=str(raw["prompting_type"]),
        deception=bool(raw["deception"]),
        agent_chain=_extract_agent_chain(question_text),
    )


def iter_hitom_items(
    shim_path: str | Path,
    *,
    limit: int | None = None,
    order: HiToMOrder | None = None,
) -> Iterator[HiToMItem]:
    """Yield HiToM items from the provisioner JSONL shim.

    ``order`` filters to one HiToM depth; the gate run uses
    ``order=HiToMOrder.ORDER_2``.
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
            if order is not None and item.order is not order:
                continue
            yield item
            emitted += 1
            if limit is not None and emitted >= limit:
                return


def _wrong_distractor(item: HiToMItem) -> str | None:
    """Pick a stable distractor from ``choices`` ≠ ``answer_text``.

    Used as the FANToM ``wrong_answer`` field so the lexical judge has
    a two-sided match. Deterministic: picks the first choice (by
    sorted letter) that isn't the gold.
    """

    for letter in sorted(item.choices.keys()):
        container = item.choices[letter]
        if container and container != item.answer_text:
            return container
    return None


def hitom_item_to_fantom_item(item: HiToMItem) -> FANToMItem:
    """Project a HiToM item onto the shared FANToM scorer shape.

    S-flat scope-selector: ``target_person`` is set to the first agent
    in the chain (``primary_subject``), or ``None`` for order 0. The
    story text is attributed to the same speaker on the single context
    turn so mentalistic recall surfaces it. All agents in the chain and
    the ``prompting_type`` / ``deception`` / ``story_length`` axes are
    stashed in metadata for sub-filtering.
    """

    speaker = item.primary_subject or "Narrator"
    turn = FANToMContextTurn(speaker=speaker, utterance=item.story)
    choices_tuple = (
        tuple(item.choices[k] for k in sorted(item.choices.keys()))
        if item.choices
        else None
    )
    conv = FANToMConversation(
        conversation_id=f"hitom-{item.sample_id}",
        participants=tuple(dict.fromkeys(item.agent_chain)) or (speaker,),
        context=(turn,),
        questions=(
            FANToMQuestion(
                question_type=FANToMQuestionType.BELIEF if item.order >= HiToMOrder.ORDER_1 else FANToMQuestionType.FACT,
                question_text=item.question,
                target_person=item.primary_subject,
                correct_answer=item.answer_text,
                wrong_answer=_wrong_distractor(item),
                tom_type="higher_order_belief" if item.order >= HiToMOrder.ORDER_2 else ("belief" if item.order == HiToMOrder.ORDER_1 else "fact"),
                tom_order=int(item.order),
                missed_info_accessibility=None,
                choices=choices_tuple,
            ),
        ),
        metadata={
            "source": "hitom",
            "sample_id": item.sample_id,
            "order": int(item.order),
            "agent_chain": list(item.agent_chain),
            "story_length": item.story_length,
            "prompting_type": item.prompting_type,
            "deception": item.deception,
        },
    )
    return FANToMItem(conversation=conv, question=conv.questions[0])


__all__ = [
    "HiToMItem",
    "HiToMOrder",
    "_extract_agent_chain",
    "hitom_item_to_fantom_item",
    "iter_hitom_items",
]
