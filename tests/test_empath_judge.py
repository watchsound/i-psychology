"""EMPATH-style binary-sub-criteria Socratic judge — shape tests.

Covers the stub judge + the LLM-judge JSON parser + the sub-criteria
aggregation rule. The MathDial-side scorer (κ against gold) is tested
separately in ``test_mathdial_empath_scorer.py``.
"""

from __future__ import annotations

from psychology_base.applications.measurement.empath_judge import (
    EMPATHSubCriterion,
    EMPATHVerdict,
    LLMEMPATHJudge,
    StubEMPATHJudge,
    aggregate_sub_criteria,
    _parse_empath_json,
)
from psychology_base.applications.measurement.socratic_compliance import ComplianceLabel


# ---------------------------------------------------------------------------
# aggregate_sub_criteria — the deterministic fold
# ---------------------------------------------------------------------------


def test_aggregate_final_answer_wins_over_scaffold() -> None:
    label = aggregate_sub_criteria(
        {
            EMPATHSubCriterion.REQUESTS_REASONING: True,
            EMPATHSubCriterion.STATES_FINAL_ANSWER: True,
            EMPATHSubCriterion.PROVIDES_SCAFFOLD_HINT: True,
        }
    )
    assert label is ComplianceLabel.DIRECT_ANSWER


def test_aggregate_reasoning_or_scaffold_without_final_answer_is_socratic() -> None:
    sc_only = aggregate_sub_criteria(
        {
            EMPATHSubCriterion.REQUESTS_REASONING: False,
            EMPATHSubCriterion.STATES_FINAL_ANSWER: False,
            EMPATHSubCriterion.PROVIDES_SCAFFOLD_HINT: True,
        }
    )
    reasoning_only = aggregate_sub_criteria(
        {
            EMPATHSubCriterion.REQUESTS_REASONING: True,
            EMPATHSubCriterion.STATES_FINAL_ANSWER: False,
            EMPATHSubCriterion.PROVIDES_SCAFFOLD_HINT: False,
        }
    )
    assert sc_only is ComplianceLabel.SOCRATIC
    assert reasoning_only is ComplianceLabel.SOCRATIC


def test_aggregate_all_false_is_borderline() -> None:
    label = aggregate_sub_criteria(
        {
            EMPATHSubCriterion.REQUESTS_REASONING: False,
            EMPATHSubCriterion.STATES_FINAL_ANSWER: False,
            EMPATHSubCriterion.PROVIDES_SCAFFOLD_HINT: False,
        }
    )
    assert label is ComplianceLabel.BORDERLINE


# ---------------------------------------------------------------------------
# StubEMPATHJudge
# ---------------------------------------------------------------------------


def test_stub_labels_socratic_question_as_socratic() -> None:
    judge = StubEMPATHJudge()
    verdict = judge.judge("What do you notice first in this equation?")
    assert verdict.label is ComplianceLabel.SOCRATIC
    assert verdict.proxy is True
    assert verdict.sub_criteria[EMPATHSubCriterion.REQUESTS_REASONING] is True


def test_stub_labels_the_answer_is_X_as_direct_answer() -> None:
    judge = StubEMPATHJudge()
    verdict = judge.judge("The answer is 9 — Janet sells 9 eggs.")
    assert verdict.label is ComplianceLabel.DIRECT_ANSWER
    assert verdict.sub_criteria[EMPATHSubCriterion.STATES_FINAL_ANSWER] is True


def test_stub_labels_notice_cue_as_scaffold_hint() -> None:
    judge = StubEMPATHJudge()
    verdict = judge.judge("Notice the sign on the right side.")
    assert verdict.label is ComplianceLabel.SOCRATIC
    assert verdict.sub_criteria[EMPATHSubCriterion.PROVIDES_SCAFFOLD_HINT] is True


def test_stub_labels_bare_praise_as_borderline() -> None:
    judge = StubEMPATHJudge()
    verdict = judge.judge("Great effort.")
    assert verdict.label is ComplianceLabel.BORDERLINE


def test_stub_scaffold_hint_suppressed_when_final_answer_fires() -> None:
    """Scaffold-cue words shouldn't inflate scaffold_hint when the answer is given."""
    judge = StubEMPATHJudge()
    verdict = judge.judge("First, the answer is 9.")
    assert verdict.sub_criteria[EMPATHSubCriterion.STATES_FINAL_ANSWER] is True
    assert verdict.sub_criteria[EMPATHSubCriterion.PROVIDES_SCAFFOLD_HINT] is False
    assert verdict.label is ComplianceLabel.DIRECT_ANSWER


# ---------------------------------------------------------------------------
# LLM judge — JSON parser
# ---------------------------------------------------------------------------


def test_parse_empath_json_round_trips_clean_object() -> None:
    parsed, parse_failed = _parse_empath_json(
        '{"requests_reasoning": true, "states_final_answer": false, '
        '"provides_scaffold_hint": true}'
    )
    assert parsed[EMPATHSubCriterion.REQUESTS_REASONING] is True
    assert parsed[EMPATHSubCriterion.STATES_FINAL_ANSWER] is False
    assert parsed[EMPATHSubCriterion.PROVIDES_SCAFFOLD_HINT] is True
    assert parse_failed is False


def test_parse_empath_json_tolerates_prose_prefix() -> None:
    parsed, parse_failed = _parse_empath_json(
        "Here is my judgement.\n"
        '{"requests_reasoning": false, "states_final_answer": true, '
        '"provides_scaffold_hint": false}\n'
        "Rationale: the tutor stated 9."
    )
    assert parsed[EMPATHSubCriterion.STATES_FINAL_ANSWER] is True
    assert parse_failed is False


def test_parse_empath_json_returns_all_false_on_unparseable_text() -> None:
    parsed, parse_failed = _parse_empath_json("I'm not sure.")
    assert all(v is False for v in parsed.values())
    assert parse_failed is True


def test_parse_empath_json_returns_all_false_on_non_dict_json() -> None:
    parsed, parse_failed = _parse_empath_json("[1, 2, 3]")
    assert all(v is False for v in parsed.values())
    assert parse_failed is True


def test_parse_empath_json_coerces_truthy_values() -> None:
    parsed, parse_failed = _parse_empath_json(
        '{"requests_reasoning": 1, "states_final_answer": 0, '
        '"provides_scaffold_hint": "yes"}'
    )
    assert parsed[EMPATHSubCriterion.REQUESTS_REASONING] is True
    assert parsed[EMPATHSubCriterion.STATES_FINAL_ANSWER] is False
    assert parsed[EMPATHSubCriterion.PROVIDES_SCAFFOLD_HINT] is True
    assert parse_failed is False


def test_parse_empath_json_flags_malformed_json_body() -> None:
    """A JSON-shaped block that fails to decode must flag parse_failed."""
    parsed, parse_failed = _parse_empath_json(
        'Here is my answer: {this is not: "valid"'
    )
    assert all(v is False for v in parsed.values())
    assert parse_failed is True


def test_parse_empath_json_handles_nested_object_fields() -> None:
    """The balanced-brace scanner must capture nested objects whole.

    Guard against regressing to a non-recursive regex: a future prompt
    revision may add a nested ``rationale: {...}`` field, and the parser
    must still extract the five booleans without the inner braces
    truncating the outer object.
    """
    parsed, parse_failed = _parse_empath_json(
        '{"requests_reasoning": true, "states_final_answer": false, '
        '"provides_scaffold_hint": true, "corrects_with_fact": false, '
        '"affirms_or_greets": false, '
        '"rationale": {"detail": "tutor asked", "confidence": 0.9}}'
    )
    assert parse_failed is False
    assert parsed[EMPATHSubCriterion.REQUESTS_REASONING] is True
    assert parsed[EMPATHSubCriterion.PROVIDES_SCAFFOLD_HINT] is True


def test_parse_empath_json_tolerates_brace_inside_quoted_rationale() -> None:
    """A ``{`` inside a string literal must not break depth counting."""
    parsed, parse_failed = _parse_empath_json(
        '{"requests_reasoning": true, "states_final_answer": false, '
        '"provides_scaffold_hint": false, "corrects_with_fact": false, '
        '"affirms_or_greets": false, '
        '"note": "the tutor said \\"compute f(x) = {x+1}\\" to the student"}'
    )
    assert parse_failed is False
    assert parsed[EMPATHSubCriterion.REQUESTS_REASONING] is True


def test_parse_empath_json_flags_unbalanced_braces_as_parse_failed() -> None:
    """An opening ``{`` with no matching close must flag parse_failed, not
    silently return a truncated prefix as valid."""
    parsed, parse_failed = _parse_empath_json(
        'Here is my answer: {"requests_reasoning": true, '
        '"states_final_answer": false'  # no closing }
    )
    assert parse_failed is True
    assert all(v is False for v in parsed.values())


def test_empath_verdict_parse_failed_defaults_false() -> None:
    """Dataclass default keeps all existing kwargs-only call-sites safe."""
    verdict = EMPATHVerdict(
        label=ComplianceLabel.SOCRATIC,
        sub_criteria={EMPATHSubCriterion.REQUESTS_REASONING: True},
        rationale="stub",
        proxy=True,
    )
    assert verdict.parse_failed is False


# ---------------------------------------------------------------------------
# LLM judge — end-to-end with a scripted router
# ---------------------------------------------------------------------------


class _FakeCompletion:
    def __init__(self, text: str) -> None:
        self.content = text


class _FakeRouter:
    """Minimal router stub returning pre-scripted JSON per call."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list = []

    def chat(self, messages, *, tier=None, max_tokens=None, temperature=None):
        self.calls.append(messages)
        return _FakeCompletion(self._replies.pop(0))


def test_llm_judge_emits_non_proxy_verdict_with_parsed_criteria() -> None:
    router = _FakeRouter(
        [
            '{"requests_reasoning": true, "states_final_answer": false, '
            '"provides_scaffold_hint": false}'
        ]
    )
    judge = LLMEMPATHJudge(router)
    verdict = judge.judge(
        "What do you notice?",
        problem_context="word problem",
        student_utterance="I got 32.",
    )
    assert verdict.proxy is False
    assert verdict.label is ComplianceLabel.SOCRATIC
    assert verdict.sub_criteria[EMPATHSubCriterion.REQUESTS_REASONING] is True
    # Verify the prompt mentions the student utterance so the judge has context.
    (msgs,) = (router.calls,)
    prompt_text = msgs[0][0].content
    assert "I got 32." in prompt_text


def test_llm_judge_falls_back_to_borderline_on_bad_json() -> None:
    router = _FakeRouter(["I cannot answer that."])
    judge = LLMEMPATHJudge(router)
    verdict = judge.judge("some utterance")
    assert verdict.label is ComplianceLabel.BORDERLINE
    # Still marked non-proxy — the judge TRIED, it just failed to parse.
    # Downstream filters BORDERLINE out of the κ denominator.
    assert verdict.proxy is False
    # And the parse-failure is visible so a 100%-fail run cannot silently
    # zero the κ denominator.
    assert verdict.parse_failed is True


def test_llm_judge_clean_json_sets_parse_failed_false() -> None:
    router = _FakeRouter(
        [
            '{"requests_reasoning": true, "states_final_answer": false, '
            '"provides_scaffold_hint": false}'
        ]
    )
    judge = LLMEMPATHJudge(router)
    verdict = judge.judge("some utterance")
    assert verdict.parse_failed is False
