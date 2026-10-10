"""FANToM adapter — ToM in conversational asymmetric-info setups.

**Target gate:** Phase 5 B2 FANToM κ ≥ 0.40 against our mentalistic-routed
answer path, with a `κ_our − κ_bare_LLM ≥ 0.03` architectural-contribution
check (see `docs/phases/phase-5-research-plan.md` §2).

**Why FANToM.** FANToM (Kim et al., EMNLP 2023) is the only off-the-shelf
ToM benchmark that tests reasoning *from inside a partial-info mental
state* — a participant leaves the room, information is shared, the
participant returns, and the model is asked what the (now-absent) person
believes. This maps directly to our Pillar 2 Store #2 (`other-as-user-
represents-them`): the user's internal model of someone else is built
from what the user has *disclosed* about them, which is itself a partial-
info window on the real person. Scoring our substrate against FANToM
measures whether the mentalistic-routing architecture produces an
advantage over a bare LLM reading the same conversation.

**License.** FANToM is MIT per the official repo (`skywalker023/fantom`).
This adapter reads JSONL shims produced by `benchmarks/scripts/
provision_fantom.py`; the shim is a derivative work and inherits MIT.
``.data/fantom/`` is gitignored to keep transitive distribution clean.

**Shape of items.** FANToM yields two levels of structure: one
conversation produces multiple questions (fact / belief:accessible /
belief:inaccessible / answerability / info-accessibility). The scorer
consumes one `FANToMItem` at a time (a `(conversation, question)` pair);
`iter_fantom_items` is the primary iteration surface. Conversation-level
iteration is also exposed via `iter_fantom_conversations` for scorers
that need to see all questions in a group (e.g. per-conversation κ).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, Iterator


class FANToMQuestionType(Enum):
    """FANToM question taxonomy.

    - ``BELIEF_ACCESSIBLE`` — target person was present when the info was
      shared; a correct ToM answer attributes the info to them.
    - ``BELIEF_INACCESSIBLE`` — target person was absent; a correct ToM
      answer must NOT attribute the info to them. This is the hard case
      and the headline FANToM metric.
    - ``BELIEF`` — belief question whose accessibility was not labelled
      by the shim (older FANToM dumps that merge the two variants).
    - ``FACT`` — a ground-truth-fact question about what was said in the
      conversation. Diagnostic — tells us whether the model read the
      conversation at all.
    - ``ANSWERABILITY`` — "can this question be answered given what you
      see?" — tests whether the model recognises info-asymmetry.
    - ``INFO_ACCESSIBILITY`` — "which participants have access to this
      info?" — the structural prior for belief:inaccessible.
    """

    BELIEF_ACCESSIBLE = "belief_accessible"
    BELIEF_INACCESSIBLE = "belief_inaccessible"
    BELIEF = "belief"
    FACT = "fact"
    ANSWERABILITY = "answerability"
    INFO_ACCESSIBILITY = "info_accessibility"

    @classmethod
    def from_shim(cls, raw: str | None) -> "FANToMQuestionType | None":
        if raw is None:
            return None
        for member in cls:
            if member.value == raw:
                return member
        return None


@dataclass(frozen=True)
class FANToMContextTurn:
    """One utterance in a FANToM conversation."""

    speaker: str | None
    utterance: str


@dataclass(frozen=True)
class FANToMQuestion:
    """One scorer-ready FANToM question item.

    ``correct_answer`` + ``wrong_answer`` + ``choices`` capture the three
    scoring regimes FANToM supports:

    - *Free-form generation.* Compare the model's natural-language answer
      against ``correct_answer`` (string match or LLM-judged).
    - *Binary distractor.* Pick between ``correct_answer`` and
      ``wrong_answer``.
    - *Multi-choice.* Pick from ``choices`` — correct one is
      ``correct_answer``.

    Scorers decide which regime to use; the adapter carries all three.
    """

    question_type: FANToMQuestionType | None
    question_text: str
    target_person: str | None
    correct_answer: str | None
    wrong_answer: str | None
    tom_type: str | None
    tom_order: int | None
    missed_info_accessibility: str | None
    choices: tuple[str, ...] | None


@dataclass(frozen=True)
class FANToMConversation:
    """One FANToM conversation + the question group attached to it."""

    conversation_id: str
    participants: tuple[str, ...]
    context: tuple[FANToMContextTurn, ...]
    questions: tuple[FANToMQuestion, ...]
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def has_questions(self) -> bool:
        return bool(self.questions)

    def render_context(self) -> str:
        """Human-readable rendering of the conversation.

        Used both by the mentalistic-routing scorer (as the disclosure
        pool fed into the companion's Store #2) and by the bare-LLM
        baseline (as the context block preceding the question). Keeping
        a single rendering shared between both paths removes a confound
        — any delta between the two paths is architectural, not
        formatting.
        """

        lines: list[str] = []
        for turn in self.context:
            speaker = turn.speaker or "Unknown"
            lines.append(f"{speaker}: {turn.utterance}")
        return "\n".join(lines)


@dataclass(frozen=True)
class FANToMItem:
    """One `(conversation, question)` scorer unit."""

    conversation: FANToMConversation
    question: FANToMQuestion


def _parse_question(raw: dict[str, object]) -> FANToMQuestion:
    choices_raw = raw.get("choices")
    choices: tuple[str, ...] | None
    if isinstance(choices_raw, list) and choices_raw:
        choices = tuple(str(c) for c in choices_raw)
    else:
        choices = None
    tom_order_raw = raw.get("tom_order")
    tom_order: int | None
    if isinstance(tom_order_raw, int):
        tom_order = tom_order_raw
    elif isinstance(tom_order_raw, str) and tom_order_raw.strip().isdigit():
        tom_order = int(tom_order_raw.strip())
    else:
        tom_order = None
    return FANToMQuestion(
        question_type=FANToMQuestionType.from_shim(
            raw.get("question_type") if isinstance(raw.get("question_type"), str) else None
        ),
        question_text=str(raw.get("question_text") or ""),
        target_person=(
            str(raw["target_person"]) if raw.get("target_person") not in (None, "") else None
        ),
        correct_answer=(
            str(raw["correct_answer"]) if raw.get("correct_answer") not in (None, "") else None
        ),
        wrong_answer=(
            str(raw["wrong_answer"]) if raw.get("wrong_answer") not in (None, "") else None
        ),
        tom_type=(
            str(raw["tom_type"]) if raw.get("tom_type") not in (None, "") else None
        ),
        tom_order=tom_order,
        missed_info_accessibility=(
            str(raw["missed_info_accessibility"])
            if raw.get("missed_info_accessibility") not in (None, "")
            else None
        ),
        choices=choices,
    )


def _parse_conversation(raw: dict[str, object]) -> FANToMConversation:
    ctx_raw = raw.get("context") or []
    context: list[FANToMContextTurn] = []
    if isinstance(ctx_raw, list):
        for entry in ctx_raw:
            if isinstance(entry, dict):
                context.append(
                    FANToMContextTurn(
                        speaker=(str(entry["speaker"]) if entry.get("speaker") else None),
                        utterance=str(entry.get("utterance") or ""),
                    )
                )
    participants_raw = raw.get("participants") or []
    if isinstance(participants_raw, list):
        participants = tuple(str(p) for p in participants_raw)
    else:
        participants = ()
    questions_raw = raw.get("questions") or []
    questions: list[FANToMQuestion] = []
    if isinstance(questions_raw, list):
        for q in questions_raw:
            if isinstance(q, dict):
                questions.append(_parse_question(q))
    metadata_raw = raw.get("metadata") or {}
    metadata = metadata_raw if isinstance(metadata_raw, dict) else {}
    return FANToMConversation(
        conversation_id=str(raw.get("conversation_id") or "unknown"),
        participants=participants,
        context=tuple(context),
        questions=tuple(questions),
        metadata=dict(metadata),
    )


def iter_fantom_conversations(shim_path: str | Path) -> Iterator[FANToMConversation]:
    """Yield FANToM conversations from a provision-script JSONL shim.

    Rows with ``has_questions=False`` are retained in the shim (so
    structural audits can see them) but skipped here, since scoring on
    an empty question group is meaningless.
    """

    path = Path(shim_path)
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            conv = _parse_conversation(raw)
            if conv.has_questions:
                yield conv


def iter_fantom_items(
    shim_path: str | Path,
    *,
    question_types: Iterable[FANToMQuestionType] | None = None,
) -> Iterator[FANToMItem]:
    """Yield one `(conversation, question)` pair per question.

    ``question_types`` filters to a specific subset — pass
    ``(FANToMQuestionType.BELIEF_INACCESSIBLE,)`` for the hard-case-only
    run, which is the primary FANToM metric.
    """

    allowed: set[FANToMQuestionType] | None = (
        set(question_types) if question_types is not None else None
    )
    for conv in iter_fantom_conversations(shim_path):
        for question in conv.questions:
            if allowed is not None and question.question_type not in allowed:
                continue
            yield FANToMItem(conversation=conv, question=question)


__all__ = [
    "FANToMContextTurn",
    "FANToMConversation",
    "FANToMItem",
    "FANToMQuestion",
    "FANToMQuestionType",
    "iter_fantom_conversations",
    "iter_fantom_items",
]
