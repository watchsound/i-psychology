"""Unit tests for the FANToM ToM scorer.

Covers lexical judge semantics, both scoring paths with a scripted
responder, report math (per-path accuracy, architectural delta,
pivot-gate boolean), and conversation-isolation invariants.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from psychology_base.eval.benchmarks.public_datasets.fantom import (
    FANToMContextTurn,
    FANToMConversation,
    FANToMItem,
    FANToMQuestion,
    FANToMQuestionType,
)
from psychology_base.eval.benchmarks.public_datasets.fantom_tom_scorer import (
    FANToMToMReport,
    FANToMVerdict,
    lexical_answer_judge,
    score_fantom_tom,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _mk_conv(cid: str) -> FANToMConversation:
    return FANToMConversation(
        conversation_id=cid,
        participants=("Alice", "Bob", "Carol"),
        context=(
            FANToMContextTurn(speaker="Alice", utterance="I adopted a corgi."),
            FANToMContextTurn(speaker="Bob", utterance="Great! Rescue?"),
            FANToMContextTurn(speaker="Alice", utterance="Yes, from a shelter."),
            FANToMContextTurn(speaker="Carol", utterance="Hey, what did I miss?"),
        ),
        questions=(),
    )


def _mk_question(
    question_type: FANToMQuestionType,
    *,
    target: str | None,
    correct: str,
    wrong: str | None,
    text: str = "What breed?",
) -> FANToMQuestion:
    return FANToMQuestion(
        question_type=question_type,
        question_text=text,
        target_person=target,
        correct_answer=correct,
        wrong_answer=wrong,
        tom_type=None,
        tom_order=None,
        missed_info_accessibility=None,
        choices=None,
    )


@pytest.fixture
def items() -> tuple[FANToMItem, ...]:
    conv = _mk_conv("fantom_x")
    q_hard = _mk_question(
        FANToMQuestionType.BELIEF_INACCESSIBLE,
        target="Carol",
        correct="Carol does not know Alice adopted anything.",
        wrong="Carol believes Alice adopted a corgi.",
        text="What does Carol believe Alice adopted?",
    )
    q_easy = _mk_question(
        FANToMQuestionType.BELIEF_ACCESSIBLE,
        target="Bob",
        correct="A corgi.",
        wrong="A labrador.",
        text="What does Bob believe Alice adopted?",
    )
    conv_with_q = replace(conv, questions=(q_hard, q_easy))
    return (
        FANToMItem(conversation=conv_with_q, question=q_hard),
        FANToMItem(conversation=conv_with_q, question=q_easy),
    )


# ---------------------------------------------------------------------------
# Lexical judge semantics
# ---------------------------------------------------------------------------


def test_lexical_judge_correct_substring_match() -> None:
    q = _mk_question(
        FANToMQuestionType.FACT,
        target=None,
        correct="A corgi.",
        wrong="A labrador.",
    )
    assert lexical_answer_judge(q, "I think it was a corgi") == "correct"


def test_lexical_judge_wrong_substring_match() -> None:
    q = _mk_question(
        FANToMQuestionType.FACT,
        target=None,
        correct="A corgi.",
        wrong="A labrador.",
    )
    assert lexical_answer_judge(q, "A labrador I believe") == "wrong"


def test_lexical_judge_undecidable_when_neither_present() -> None:
    q = _mk_question(
        FANToMQuestionType.FACT,
        target=None,
        correct="A corgi.",
        wrong="A labrador.",
    )
    assert lexical_answer_judge(q, "I don't know.") == "undecidable"


def test_lexical_judge_undecidable_when_both_present() -> None:
    q = _mk_question(
        FANToMQuestionType.FACT,
        target=None,
        correct="A corgi.",
        wrong="A labrador.",
    )
    # Model mentions both — ambiguous.
    assert (
        lexical_answer_judge(q, "Could be a corgi or a labrador")
        == "undecidable"
    )


def test_lexical_judge_uses_choices_fallback() -> None:
    q = FANToMQuestion(
        question_type=FANToMQuestionType.FACT,
        question_text="?",
        target_person=None,
        correct_answer="A corgi.",
        wrong_answer=None,
        tom_type=None,
        tom_order=None,
        missed_info_accessibility=None,
        choices=("A corgi.", "A labrador.", "A poodle."),
    )
    assert lexical_answer_judge(q, "A poodle") == "wrong"
    assert lexical_answer_judge(q, "A corgi") == "correct"


# ---------------------------------------------------------------------------
# Scorer behaviour
# ---------------------------------------------------------------------------


def test_scorer_yields_one_verdict_per_path_per_item(items) -> None:
    report = score_fantom_tom(
        items,
        responder=lambda prompt: "A corgi.",
        judge=lexical_answer_judge,
    )
    # 2 items × 2 paths = 4 verdicts
    assert len(report.verdicts) == 4
    paths = {v.source_path for v in report.verdicts}
    assert paths == {"mentalistic", "bare_llm"}


def test_scorer_single_path_opt_in(items) -> None:
    report = score_fantom_tom(
        items,
        responder=lambda prompt: "A corgi.",
        judge=lexical_answer_judge,
        paths=("bare_llm",),
    )
    assert len(report.verdicts) == 2
    assert all(v.source_path == "bare_llm" for v in report.verdicts)


def test_scorer_responder_sees_different_prompts_per_path(items) -> None:
    seen: list[str] = []

    def _recorder(prompt: str) -> str:
        seen.append(prompt)
        return "A corgi."

    score_fantom_tom(items, responder=_recorder, judge=lexical_answer_judge)
    assert len(seen) == 4
    mentalistic_prompts = [p for p in seen if "Observations relevant to" in p]
    bare_prompts = [p for p in seen if "Read the full conversation" in p]
    assert len(mentalistic_prompts) == 2
    assert len(bare_prompts) == 2


def test_mentalistic_prompt_includes_target_person(items) -> None:
    captured: dict[str, str] = {}

    def _splitter(prompt: str) -> str:
        if "Observations relevant to Carol" in prompt:
            captured["carol"] = prompt
        if "Observations relevant to Bob" in prompt:
            captured["bob"] = prompt
        return "A corgi."

    score_fantom_tom(items, responder=_splitter, judge=lexical_answer_judge)
    assert "carol" in captured
    assert "bob" in captured


def test_bare_llm_prompt_includes_full_transcript(items) -> None:
    captured: list[str] = []

    def _recorder(prompt: str) -> str:
        if "Read the full conversation" in prompt:
            captured.append(prompt)
        return "A corgi."

    score_fantom_tom(items, responder=_recorder, judge=lexical_answer_judge)
    assert captured
    # Every speaker's turn should be in the raw transcript.
    for p in captured:
        assert "Alice:" in p
        assert "Bob:" in p
        assert "Carol:" in p


# ---------------------------------------------------------------------------
# Report math
# ---------------------------------------------------------------------------


def test_report_accuracy_counts_correct_only(items) -> None:
    # Mentalistic path gets both right; bare-LLM gets both wrong.
    def _responder(prompt: str) -> str:
        if "Observations relevant to" in prompt:
            # Mentalistic: give the correct hard+easy answers.
            if "Carol" in prompt:
                return "Carol does not know Alice adopted anything."
            return "A corgi."
        # Bare-LLM: wrong.
        return "A labrador."

    report = score_fantom_tom(items, responder=_responder, judge=lexical_answer_judge)
    assert report.accuracy(path="mentalistic") == 1.0
    assert report.accuracy(path="bare_llm") == 0.0


def test_report_architectural_delta(items) -> None:
    def _responder(prompt: str) -> str:
        if "Observations relevant to" in prompt:
            if "Carol" in prompt:
                return "Carol does not know Alice adopted anything."
            return "A corgi."
        return "A labrador."

    report = score_fantom_tom(items, responder=_responder, judge=lexical_answer_judge)
    assert report.architectural_delta() == pytest.approx(1.0)


def test_report_accuracy_filter_by_question_type(items) -> None:
    def _responder(prompt: str) -> str:
        return "A corgi."  # correct for Bob-easy, wrong for Carol-hard

    report = score_fantom_tom(items, responder=_responder, judge=lexical_answer_judge)
    # BELIEF_ACCESSIBLE: "A corgi." == correct_answer → correct.
    acc_easy = report.accuracy(
        path="mentalistic", question_type=FANToMQuestionType.BELIEF_ACCESSIBLE
    )
    # BELIEF_INACCESSIBLE: "A corgi." contains "corgi" which appears in
    # the WRONG answer ("Carol believes Alice adopted a corgi.") → wrong.
    acc_hard = report.accuracy(
        path="mentalistic", question_type=FANToMQuestionType.BELIEF_INACCESSIBLE
    )
    assert acc_easy == 1.0
    assert acc_hard == 0.0


def test_report_undecidable_dropped_from_accuracy_denominator() -> None:
    conv = _mk_conv("fantom_u")
    q = _mk_question(
        FANToMQuestionType.FACT,
        target=None,
        correct="A corgi.",
        wrong="A labrador.",
    )
    conv_with_q = replace(conv, questions=(q,))
    items = (FANToMItem(conversation=conv_with_q, question=q),)
    # Responder gives neither correct nor wrong.
    report = score_fantom_tom(
        items,
        responder=lambda prompt: "I have no idea.",
        judge=lexical_answer_judge,
    )
    assert report.accuracy(path="mentalistic") == 0.0
    assert report.undecidable_rate(path="mentalistic") == 1.0


# ---------------------------------------------------------------------------
# Pivot gate
# ---------------------------------------------------------------------------


def _bulk_items(n: int, *, correct_mentalistic: bool, correct_bare: bool) -> tuple[FANToMItem, ...]:
    """Build ``n`` BELIEF_INACCESSIBLE items with disjoint IDs for gate tests."""
    out: list[FANToMItem] = []
    for i in range(n):
        conv = FANToMConversation(
            conversation_id=f"c{i}",
            participants=("A", "B"),
            context=(FANToMContextTurn(speaker="A", utterance=f"turn {i}"),),
            questions=(),
        )
        q = _mk_question(
            FANToMQuestionType.BELIEF_INACCESSIBLE,
            target="B",
            correct="correct_phrase",
            wrong="wrong_phrase",
            text=f"q{i}",
        )
        out.append(FANToMItem(conversation=replace(conv, questions=(q,)), question=q))
    return tuple(out)


def test_pivot_gate_requires_min_items() -> None:
    items = _bulk_items(10, correct_mentalistic=True, correct_bare=False)

    def _responder(prompt: str) -> str:
        # Mentalistic correct, bare-LLM wrong → delta=1.0 clears delta floor;
        # the only free knob is min_items.
        if "Observations relevant to" in prompt:
            return "correct_phrase"
        return "wrong_phrase"

    report = score_fantom_tom(items, responder=_responder, judge=lexical_answer_judge)
    assert report.meets_pivot_gate(min_items=50) is False
    assert report.meets_pivot_gate(min_items=5) is True


def test_pivot_gate_requires_delta_floor() -> None:
    items = _bulk_items(60, correct_mentalistic=True, correct_bare=True)

    def _responder(prompt: str) -> str:
        # Both paths return correct → delta == 0 → gate fails on delta.
        return "correct_phrase"

    report = score_fantom_tom(items, responder=_responder, judge=lexical_answer_judge)
    assert report.meets_pivot_gate(min_items=50) is False
    # Zero-delta is explicitly below the 0.03 floor.
    assert report.architectural_delta() == pytest.approx(0.0)


def test_pivot_gate_passes_when_delta_and_floor_cleared() -> None:
    items = _bulk_items(60, correct_mentalistic=True, correct_bare=False)

    def _responder(prompt: str) -> str:
        if "Observations relevant to" in prompt:
            return "correct_phrase"
        return "wrong_phrase"

    report = score_fantom_tom(items, responder=_responder, judge=lexical_answer_judge)
    assert report.meets_pivot_gate(min_items=50) is True
    assert report.architectural_delta() == pytest.approx(1.0)


def test_pivot_gate_fails_on_high_undecidable_rate_when_opted_in() -> None:
    items = _bulk_items(60, correct_mentalistic=True, correct_bare=False)

    def _responder(prompt: str) -> str:
        return "I don't know"

    report = score_fantom_tom(items, responder=_responder, judge=lexical_answer_judge)
    assert report.undecidable_rate(path="mentalistic") == 1.0
    assert (
        report.meets_pivot_gate(min_items=50, max_undecidable_rate=0.20) is False
    )


# ---------------------------------------------------------------------------
# Conversation isolation — no cross-leakage via the retrieval index
# ---------------------------------------------------------------------------


def test_each_conversation_gets_its_own_system() -> None:
    """A fresh CompanionSystem per conversation = no cross-leakage.

    The scorer builds a system per conversation; a question about
    conv-A's participant cannot pull observations written for conv-B.
    """

    # Conv 1 — only Alice talks about adopting a corgi
    conv1 = FANToMConversation(
        conversation_id="c1",
        participants=("Alice",),
        context=(FANToMContextTurn(speaker="Alice", utterance="I adopted a corgi."),),
        questions=(),
    )
    q1 = _mk_question(
        FANToMQuestionType.FACT,
        target="Alice",
        correct="A corgi.",
        wrong=None,
    )
    item1 = FANToMItem(conversation=replace(conv1, questions=(q1,)), question=q1)

    # Conv 2 — Alice talks about a cat; no "corgi" anywhere.
    conv2 = FANToMConversation(
        conversation_id="c2",
        participants=("Alice",),
        context=(FANToMContextTurn(speaker="Alice", utterance="I adopted a cat."),),
        questions=(),
    )
    q2 = _mk_question(
        FANToMQuestionType.FACT,
        target="Alice",
        correct="A cat.",
        wrong=None,
    )
    item2 = FANToMItem(conversation=replace(conv2, questions=(q2,)), question=q2)

    seen_prompts: list[str] = []

    def _recorder(prompt: str) -> str:
        if "Observations relevant to Alice" in prompt:
            seen_prompts.append(prompt)
        return ""

    score_fantom_tom(
        (item1, item2),
        responder=_recorder,
        judge=lexical_answer_judge,
        paths=("mentalistic",),
    )
    assert len(seen_prompts) == 2
    # Prompt for conv2 must not contain "corgi" — isolation holds.
    corgi_leaked_into_cat_prompt = any(
        "corgi" in p.lower() and "cat" in p.lower() for p in seen_prompts
    )
    assert not corgi_leaked_into_cat_prompt


# ---------------------------------------------------------------------------
# Report immutability
# ---------------------------------------------------------------------------


def test_report_is_frozen() -> None:
    report = FANToMToMReport(verdicts=(), judge_name="x", responder_name="y")
    with pytest.raises(Exception):
        report.verdicts = ()  # type: ignore[misc]


def test_verdict_is_frozen() -> None:
    v = FANToMVerdict(
        conversation_id="c",
        question_text="?",
        question_type=None,
        target_person=None,
        source_path="mentalistic",
        model_answer="",
        label="undecidable",
    )
    with pytest.raises(Exception):
        v.label = "correct"  # type: ignore[misc]
