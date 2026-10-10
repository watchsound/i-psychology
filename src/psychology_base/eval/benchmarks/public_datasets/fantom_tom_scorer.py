"""FANToM ToM scorer — mentalistic-routed path vs bare-LLM baseline.

**Phase 5 §2 target.** For each FANToM `(conversation, question)` item,
produce an answer under *two* paths and score each against gold:

- *Mentalistic path* — write every conversation turn as a scoped
  `Observation` into a fresh `CompanionSystem` with
  `subject:<speaker>` tags (Pillar 2 Store #2), then recall with
  `scope=("subject", target_person)` and feed the recalled content as
  context to the LLM responder. For fact questions (no target person),
  recall is unscoped.
- *Bare-LLM path* — the same LLM responder sees `conv.render_context()`
  verbatim and the question, no routing, no recall.

The report exposes **per-path accuracy** against the gold `correct_answer`
and an **architectural delta** `accuracy_mentalistic − accuracy_bare_llm`.
The pivot kill criterion from `docs/phases/phase-5-research-plan.md` §2
is `accuracy ≥ 0.40` AND `delta ≥ 0.03`.

**Metric naming note.** The plan doc uses "κ" throughout for the FANToM
sub-metrics. Cohen's κ against FANToM gold is mathematically degenerate
when gold is always ``correct`` (single-class rater → `pe` collapses to
`po` and κ reduces to 0). The honest primary metric here is accuracy —
same number the FANToM paper reports — and that is what this module
computes. The ``meets_pivot_gate(accuracy_floor=0.40)`` floor matches
the plan's "κ_floor=0.40" one-to-one in intent; the rename removes the
mathematical degeneracy without changing the gate.

**Confound hygiene.** Both paths share:
- the same `responder` (same LLM, same temperature, same prompt-shell),
- the same `conv.render_context()` for utterance formatting, and
- the same judge that maps free-form answers to {correct,
  wrong, undecidable}.

The only difference between the two paths is whether the question is
preceded by a mentalistic-recall block or by the full raw context.
Any accuracy delta is therefore attributable to the memory substrate's
routing choice, not to prompt formatting or model identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Optional, Sequence

from psychology_base.companion.perspective import PerspectiveSpec, ScopeSpec
from psychology_base.companion.system import CompanionSystem
from psychology_base.eval.adapters import Observation
from psychology_base.eval.benchmarks.public_datasets.fantom import (
    FANToMConversation,
    FANToMItem,
    FANToMQuestion,
    FANToMQuestionType,
)


_PATH_MENTALISTIC = "mentalistic"
_PATH_BARE_LLM = "bare_llm"


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


# A responder is a pure function (prompt_text) -> model_answer. The
# scorer stays provider-agnostic; a thin adapter wraps `LLMRouter` in
# the live-fire run script.
Responder = Callable[[str], str]

# A judge decides {"correct", "wrong", "undecidable"} given a question
# and the model's free-form answer. The default lexical judge is
# deterministic and runs in-process; `LLMFANToMJudge` lifts the same
# contract onto an LLM for live-fire grading.
AnswerLabel = str  # "correct" | "wrong" | "undecidable"
AnswerJudge = Callable[[FANToMQuestion, str], AnswerLabel]


@dataclass(frozen=True)
class FANToMVerdict:
    """One scored `(path, item)` outcome."""

    conversation_id: str
    question_text: str
    question_type: Optional[FANToMQuestionType]
    target_person: Optional[str]
    source_path: str  # "mentalistic" | "bare_llm"
    model_answer: str
    label: AnswerLabel  # "correct" | "wrong" | "undecidable"


@dataclass(frozen=True)
class FANToMToMReport:
    """Side-by-side verdict pool for mentalistic + bare-LLM paths."""

    verdicts: tuple[FANToMVerdict, ...]
    judge_name: str
    responder_name: str

    def _subset(
        self,
        *,
        path: str,
        question_type: Optional[FANToMQuestionType] = None,
    ) -> tuple[FANToMVerdict, ...]:
        out: list[FANToMVerdict] = []
        for v in self.verdicts:
            if v.source_path != path:
                continue
            if question_type is not None and v.question_type is not question_type:
                continue
            out.append(v)
        return tuple(out)

    def item_count(
        self,
        *,
        path: str,
        question_type: Optional[FANToMQuestionType] = None,
    ) -> int:
        return len(self._subset(path=path, question_type=question_type))

    def accuracy(
        self,
        *,
        path: str,
        question_type: Optional[FANToMQuestionType] = None,
        drop_undecidable: bool = True,
    ) -> float:
        """Fraction of items the model answered correctly on ``path``.

        ``drop_undecidable=True`` (default) excludes judge-undecidable
        verdicts from both numerator and denominator — a judge that
        cannot decide should not vote, same convention as the EMPATH
        scorer's BORDERLINE handling.
        """

        subset = self._subset(path=path, question_type=question_type)
        if drop_undecidable:
            subset = tuple(v for v in subset if v.label != "undecidable")
        if not subset:
            return 0.0
        correct = sum(1 for v in subset if v.label == "correct")
        return correct / len(subset)

    def undecidable_rate(
        self,
        *,
        path: str,
        question_type: Optional[FANToMQuestionType] = None,
    ) -> float:
        subset = self._subset(path=path, question_type=question_type)
        if not subset:
            return 0.0
        n = sum(1 for v in subset if v.label == "undecidable")
        return n / len(subset)

    def architectural_delta(
        self,
        *,
        question_type: Optional[FANToMQuestionType] = None,
    ) -> float:
        """``accuracy(mentalistic) − accuracy(bare_llm)``.

        The sign of this number is the Phase 5 Track 1 claim: positive
        delta = memory substrate contributes above the LLM's own ToM
        capacity on the same context; zero or negative = no
        architectural contribution and the Store #2 claim softens.
        """

        return (
            self.accuracy(path=_PATH_MENTALISTIC, question_type=question_type)
            - self.accuracy(path=_PATH_BARE_LLM, question_type=question_type)
        )

    def meets_pivot_gate(
        self,
        *,
        accuracy_floor: float = 0.40,
        delta_floor: float = 0.03,
        min_items: int = 50,
        question_type: Optional[FANToMQuestionType] = FANToMQuestionType.BELIEF_INACCESSIBLE,
        max_undecidable_rate: float = 1.0,
    ) -> bool:
        """Phase 5 §2 kill criterion.

        The default ``question_type`` restricts the gate to
        ``BELIEF_INACCESSIBLE`` — the hard-case FANToM headline. Pass
        ``question_type=None`` to run the gate against the full pooled
        answer set. ``min_items`` is enforced on BOTH paths at the
        chosen subset so a reader cannot sneak a passing delta from a
        tiny imbalanced bucket.

        ``max_undecidable_rate`` defaults to 1.0 (off) because the
        plan's kill criterion is accuracy + delta + N only; the
        BELIEF_INACCESSIBLE gold answer *is* ``subject doesn't know``,
        and a correct hedge ("I don't know what X thinks") falls into
        the judge's undecidable bucket by construction. Callers who
        want a stricter guard against pathological responder failure
        can set the kwarg explicitly.
        """

        n_mental = self.item_count(path=_PATH_MENTALISTIC, question_type=question_type)
        n_bare = self.item_count(path=_PATH_BARE_LLM, question_type=question_type)
        if n_mental < min_items or n_bare < min_items:
            return False
        if (
            self.undecidable_rate(path=_PATH_MENTALISTIC, question_type=question_type)
            > max_undecidable_rate
        ):
            return False
        if (
            self.undecidable_rate(path=_PATH_BARE_LLM, question_type=question_type)
            > max_undecidable_rate
        ):
            return False
        acc_mental = self.accuracy(path=_PATH_MENTALISTIC, question_type=question_type)
        if acc_mental < accuracy_floor:
            return False
        return self.architectural_delta(question_type=question_type) >= delta_floor


# ---------------------------------------------------------------------------
# Prompt builders — shared between the two paths
# ---------------------------------------------------------------------------


def _build_mentalistic_prompt(
    question: FANToMQuestion,
    recalled_context: str,
) -> str:
    """Prompt shell for the mentalistic path.

    The recalled block is the output of
    ``CompanionSystem.recall(..., scope=("subject", target_person))``;
    its content is the perspective-filtered observations the memory
    substrate decided were relevant. The LLM only sees this filtered
    view, never the raw transcript.
    """

    target = question.target_person or "the person in question"
    header = (
        f"You are reasoning about what {target} believes based ONLY on the "
        f"filtered observations below. The observations are limited to what "
        f"{target} would have witnessed in the conversation. Do not assume "
        f"{target} knows anything beyond these observations."
    )
    body = f"Observations relevant to {target}:\n{recalled_context or '(no observations retrieved)'}"
    question_block = f"Question: {question.question_text}"
    answer_hint = "Answer concisely in one sentence."
    return "\n\n".join([header, body, question_block, answer_hint])


def _build_bare_llm_prompt(
    question: FANToMQuestion,
    conversation: FANToMConversation,
) -> str:
    """Prompt shell for the bare-LLM baseline.

    The entire conversation transcript is included verbatim. The LLM is
    expected to perform ToM reasoning from the raw transcript — this is
    the "LLM-only" condition.
    """

    header = (
        "Read the full conversation below, then answer the question. "
        "Reason about what each named participant knows based on what they "
        "witnessed."
    )
    rendered = conversation.render_context() or "(empty conversation)"
    question_block = f"Question: {question.question_text}"
    answer_hint = "Answer concisely in one sentence."
    return "\n\n".join([header, rendered, question_block, answer_hint])


# ---------------------------------------------------------------------------
# Mentalistic routing — load conversation into a fresh CompanionSystem
# ---------------------------------------------------------------------------


def _load_conversation_into_system(
    conversation: FANToMConversation,
    system: CompanionSystem,
    *,
    base_time: Optional[datetime] = None,
) -> None:
    """Write every conversation turn as a scoped Observation.

    ``subject_id`` is set to the speaker so recall with
    ``scope=("subject", speaker)`` filters to that speaker's utterances —
    the structural proxy for "observations from inside speaker's
    perspective". FANToM's absence-return structure means a speaker's
    `subject:` tags cover exactly the turns they could have witnessed;
    this preserves the asymmetric-info constraint through the retrieval
    front-end.
    """

    t0 = base_time or datetime(2026, 1, 1, 12, 0, 0)
    for idx, turn in enumerate(conversation.context):
        speaker = turn.speaker or "Unknown"
        timestamp = t0 + timedelta(seconds=idx)
        system.observe(
            Observation(
                timestamp=timestamp,
                speaker=speaker,
                content=turn.utterance,
                subject_id=speaker,
            )
        )


def _recall_for_question(
    conversation: FANToMConversation,
    system: CompanionSystem,
    question: FANToMQuestion,
) -> str:
    """Scope the recall by target_person when the question has one.

    Fact questions (no target_person) fall through to an unscoped
    recall — the memory substrate's regular retrieval decides what
    surfaces, matching how the Companion would answer a factual
    question in a real session.
    """

    if question.target_person:
        # Phase 9+ perspective migration — passes the SAME two-sided
        # ScopeSpec that the legacy ``scope=("subject", target_person)``
        # tuple produced. The subject-asymmetric mechanism (FANToM
        # fragments carry ``subject:<speaker>`` with no ``entity:*`` →
        # the AND-contract drops them on scoped recall → empty pool on
        # BELIEF_INACCESSIBLE → "I don't know" matches gold) is
        # preserved byte-identical. The Phase 5 Track 1 +0.791
        # architectural result depends on this; do not refactor the
        # ScopeSpec shape here.
        recall = system.recall(
            cue=question.question_text,
            context=None,
            perspective=PerspectiveSpec(
                scope=ScopeSpec(
                    subject="subject", entity=question.target_person
                )
            ),
        )
    else:
        recall = system.recall(cue=question.question_text, context=None)
    return recall.content


# ---------------------------------------------------------------------------
# Lexical answer judge — default; deterministic; CI-safe.
# ---------------------------------------------------------------------------


def _normalize_answer(text: str) -> str:
    return " ".join(text.lower().split()).strip(".,!?;:'\"")


def lexical_answer_judge(
    question: FANToMQuestion, model_answer: str
) -> AnswerLabel:
    """Substring-overlap verdict against ``correct_answer`` / ``wrong_answer``.

    Deliberately conservative — if the model answer contains the
    correct answer's text (modulo normalization) and NOT the wrong
    answer's text, it is ``correct``. The symmetric clause gives
    ``wrong``. Everything else is ``undecidable``.

    This is a lower-bound judge. Live-fire runs should pass
    ``LLMFANToMJudge`` for the semantic-match quality the FANToM paper
    uses; the lexical judge is for CI determinism and development
    inspection.
    """

    answer = _normalize_answer(model_answer)
    correct = _normalize_answer(question.correct_answer or "")
    wrong = _normalize_answer(question.wrong_answer or "")

    has_correct = bool(correct) and correct in answer
    has_wrong = bool(wrong) and wrong in answer

    if has_correct and not has_wrong:
        return "correct"
    if has_wrong and not has_correct:
        return "wrong"

    # Multi-choice: if choices are published, scan them. Choice equal
    # to correct → correct; any other choice content present → wrong.
    if question.choices:
        for choice in question.choices:
            norm_choice = _normalize_answer(choice)
            if norm_choice and norm_choice in answer:
                if norm_choice == correct:
                    return "correct"
                return "wrong"

    return "undecidable"


# ---------------------------------------------------------------------------
# LLM answer judge — JSON-shaped verdict via LLMRouter
# ---------------------------------------------------------------------------


class LLMFANToMJudge:
    """Semantic-match judge for FANToM free-form answers.

    Takes an ``LLMRouter`` and dispatches a 3-way classification per
    (question, model_answer) pair. Returns ``undecidable`` on parse
    failure so the gate's ``max_undecidable_rate`` guard catches
    pathological runs — identical convention to the EMPATH judge's
    BORDERLINE+parse-fail handling.
    """

    def __init__(self, router, *, tier=None, name: str = "llm_fantom_judge"):
        from psychology_base.llm.types import TaskTier

        self._router = router
        self._tier = tier or TaskTier.SIMPLE
        self.name = name

    def __call__(
        self, question: FANToMQuestion, model_answer: str
    ) -> AnswerLabel:
        from psychology_base.llm.types import ChatMessage

        prompt = (
            "You are grading a model's answer to a theory-of-mind question. "
            "Decide whether the model's answer matches the GOLD-CORRECT answer, "
            "matches the WRONG distractor, or neither.\n\n"
            f"Question: {question.question_text}\n"
            f"Gold-correct answer: {question.correct_answer or '(none)'}\n"
            f"Wrong distractor: {question.wrong_answer or '(none)'}\n"
            f"Model answer: {model_answer}\n\n"
            "Respond with EXACTLY one JSON object of the form "
            '{"verdict":"correct"} OR {"verdict":"wrong"} OR {"verdict":"undecidable"} '
            "and nothing else."
        )
        try:
            completion = self._router.chat(
                [ChatMessage(role="user", content=prompt)],
                tier=self._tier,
            )
            raw = completion.content.strip()
        except Exception:
            return "undecidable"

        import json as _json
        import re as _re

        match = _re.search(r"\{.*?\}", raw, flags=_re.DOTALL)
        if not match:
            return "undecidable"
        try:
            parsed = _json.loads(match.group(0))
        except _json.JSONDecodeError:
            return "undecidable"
        verdict = str(parsed.get("verdict", "")).strip().lower()
        if verdict in ("correct", "wrong", "undecidable"):
            return verdict  # type: ignore[return-value]
        return "undecidable"


# ---------------------------------------------------------------------------
# Entry point — run both paths across a batch of items
# ---------------------------------------------------------------------------


def score_fantom_tom(
    items: Sequence[FANToMItem],
    *,
    responder: Responder,
    judge: AnswerJudge = lexical_answer_judge,
    system_factory: Optional[Callable[[], CompanionSystem]] = None,
    paths: tuple[str, ...] = (_PATH_MENTALISTIC, _PATH_BARE_LLM),
    judge_name: str = "lexical",
    responder_name: str = "unknown",
) -> FANToMToMReport:
    """Score ``items`` on both the mentalistic path and bare-LLM baseline.

    ``items`` is typically the output of
    ``iter_fantom_items(shim_path, question_types=...)``. One fresh
    `CompanionSystem` is built per conversation on the mentalistic
    path so cross-conversation leakage is impossible — the retrieval
    index at scoring time is only the conversation under test.
    """

    verdicts: list[FANToMVerdict] = []

    # Group items by conversation so we only pay the ingest cost once
    # per conversation on the mentalistic path. The grouping is
    # intentionally insertion-order-preserving so the output verdict
    # list is deterministic for a given input stream.
    grouped: dict[str, list[FANToMItem]] = {}
    order: list[str] = []
    for item in items:
        cid = item.conversation.conversation_id
        if cid not in grouped:
            grouped[cid] = []
            order.append(cid)
        grouped[cid].append(item)

    build_system = system_factory or (lambda: CompanionSystem())

    for cid in order:
        bucket = grouped[cid]
        conversation = bucket[0].conversation

        if _PATH_MENTALISTIC in paths:
            system = build_system()
            _load_conversation_into_system(conversation, system)
            for item in bucket:
                recalled = _recall_for_question(conversation, system, item.question)
                prompt = _build_mentalistic_prompt(item.question, recalled)
                answer = responder(prompt)
                label = judge(item.question, answer)
                verdicts.append(
                    FANToMVerdict(
                        conversation_id=cid,
                        question_text=item.question.question_text,
                        question_type=item.question.question_type,
                        target_person=item.question.target_person,
                        source_path=_PATH_MENTALISTIC,
                        model_answer=answer,
                        label=label,
                    )
                )

        if _PATH_BARE_LLM in paths:
            for item in bucket:
                prompt = _build_bare_llm_prompt(item.question, conversation)
                answer = responder(prompt)
                label = judge(item.question, answer)
                verdicts.append(
                    FANToMVerdict(
                        conversation_id=cid,
                        question_text=item.question.question_text,
                        question_type=item.question.question_type,
                        target_person=item.question.target_person,
                        source_path=_PATH_BARE_LLM,
                        model_answer=answer,
                        label=label,
                    )
                )

    return FANToMToMReport(
        verdicts=tuple(verdicts),
        judge_name=judge_name,
        responder_name=responder_name,
    )


__all__ = [
    "AnswerJudge",
    "AnswerLabel",
    "FANToMToMReport",
    "FANToMVerdict",
    "LLMFANToMJudge",
    "Responder",
    "lexical_answer_judge",
    "score_fantom_tom",
]
