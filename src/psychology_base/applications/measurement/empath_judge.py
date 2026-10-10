"""EMPATH-style binary-sub-criteria Socratic-compliance judge.

Per the Phase 4b benchmark pivot gate (`docs/phases/phase-4b-benchmark-pivot.md`
§3 row 学习陪伴 Socratic compliance): the third floor is
"EMPATH binary-sub-criteria κ ≥ 0.60" against gold MathDial / auditor labels.

**What EMPATH gets us.** Raw-score LLM judges inflate on 10/19 metrics
without calibration (EMPATH 2024, arXiv 2606.30256; one documented case:
sensitive-context reintroduction raw-score 7.3 → calibrated 3.3). The fix
is to decompose the final judgement into a short list of *binary*
yes/no sub-criteria and aggregate deterministically. A model that is bad
at "is this Socratic?" (continuous) can still be good at "does this
response contain the final numeric answer?" (binary).

**Three sub-criteria** cover the Socratic / direct-answer / borderline
decision over a tutor turn:

- :attr:`REQUESTS_REASONING` — the response asks the student to explain,
  attempt, or notice something.
- :attr:`STATES_FINAL_ANSWER` — the response contains the complete answer
  to the problem (the surrender tell).
- :attr:`PROVIDES_SCAFFOLD_HINT` — the response names a next step or
  focuses attention on a specific part of the problem without stating
  the answer.

Aggregation (deterministic, in the judge not the LLM):

- ``STATES_FINAL_ANSWER=True`` → ``DIRECT_ANSWER`` (surrender beats scaffold)
- ``REQUESTS_REASONING=True OR PROVIDES_SCAFFOLD_HINT=True`` → ``SOCRATIC``
- otherwise → ``BORDERLINE`` (ambiguous; excluded from κ denominator)

**Two judges, same protocol.**

- :class:`StubEMPATHJudge` — deterministic, LLM-free. Derived from the
  shipped :func:`classify_tutor_move` lexical classifier so CI can run.
  Marked ``proxy=True`` on every verdict.
- :class:`LLMEMPATHJudge` — wires an :class:`LLMRouter`; one chat per
  tutor turn, structured JSON output with the three booleans. Marked
  ``proxy=False``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, Sequence

from psychology_base.applications.measurement.scaffolding_telemetry import TutorMove
from psychology_base.applications.measurement.socratic_compliance import (
    ComplianceLabel,
)


class EMPATHSubCriterion(str, Enum):
    """Binary sub-criteria over a tutor turn."""

    REQUESTS_REASONING = "requests_reasoning"
    STATES_FINAL_ANSWER = "states_final_answer"
    PROVIDES_SCAFFOLD_HINT = "provides_scaffold_hint"
    CORRECTS_WITH_FACT = "corrects_with_fact"
    """TRUE iff the tutor directly corrects the student's error by stating
    a math fact, pointing out the student's wrong number, or using a
    rhetorical question that re-states the student's own words to show
    the error. MathDial labels these turns as TELLING (not PROBING or
    FOCUS), so they must fold to DIRECT_ANSWER even without an explicit
    'the answer is X'. Added 2026-10-08 to close the EMPATH κ gap on
    TELLING turns the three-criterion aggregation was mis-labelling as
    BORDERLINE / SOCRATIC."""
    AFFIRMS_OR_GREETS = "affirms_or_greets"
    """TRUE iff the tutor turn is purely affirmational, congratulatory,
    or a social greeting with no math content — ``Super!``, ``Well done``,
    ``Exactly correct``, ``Hi!``, ``hey``, ``youre right``, etc. MathDial
    labels these as GENERIC (support), which collapses to the SOCRATIC
    side of the compliance binary. Added 2026-10-08 to pull short-praise
    turns out of BORDERLINE (where the other four criteria legitimately
    all vote FALSE) and into SOCRATIC, lifting κ against gold without
    inflating DIRECT_ANSWER."""


@dataclass(frozen=True)
class EMPATHVerdict:
    """One judge verdict on one tutor turn."""

    label: ComplianceLabel
    sub_criteria: dict[EMPATHSubCriterion, bool]
    rationale: str
    proxy: bool
    """True iff the verdict came from the deterministic stub judge rather
    than an LLM. Downstream reports MUST disclose this — stub-judge κ is
    not substitutable for LLM-judge κ against the pivot gate."""
    parse_failed: bool = False
    """True iff the LLM response could not be parsed into clean JSON, so
    the sub-criteria defaulted to all-False (aggregating to BORDERLINE and
    dropping out of the κ denominator). Downstream reports count these so
    a judge whose JSON parser silently zeroes the κ denominator can't
    masquerade as a judge whose decisions are legitimately BORDERLINE.
    Stub judges never set this (they produce booleans from lexical cues,
    not JSON)."""


def aggregate_sub_criteria(
    sub_criteria: dict[EMPATHSubCriterion, bool],
) -> ComplianceLabel:
    """Fold the booleans into a compliance label.

    Priority: TELLING signals (``STATES_FINAL_ANSWER`` or
    ``CORRECTS_WITH_FACT``) override scaffolding signals. Fully-telling
    turns beat partially-scaffolded telling. If neither TELLING signal
    is set but a scaffolding signal is, the turn is SOCRATIC. Otherwise
    BORDERLINE (drops out of κ denominator)."""
    if sub_criteria.get(EMPATHSubCriterion.STATES_FINAL_ANSWER, False):
        return ComplianceLabel.DIRECT_ANSWER
    if sub_criteria.get(EMPATHSubCriterion.CORRECTS_WITH_FACT, False):
        return ComplianceLabel.DIRECT_ANSWER
    if (
        sub_criteria.get(EMPATHSubCriterion.REQUESTS_REASONING, False)
        or sub_criteria.get(EMPATHSubCriterion.PROVIDES_SCAFFOLD_HINT, False)
        or sub_criteria.get(EMPATHSubCriterion.AFFIRMS_OR_GREETS, False)
    ):
        return ComplianceLabel.SOCRATIC
    return ComplianceLabel.BORDERLINE


class EMPATHJudge(Protocol):
    """Protocol for a Socratic-compliance judge over tutor turns."""

    def judge(
        self,
        tutor_utterance: str,
        *,
        problem_context: str = "",
        student_utterance: str = "",
    ) -> EMPATHVerdict: ...


# ---------------------------------------------------------------------------
# Stub judge — deterministic
# ---------------------------------------------------------------------------


_FINAL_ANSWER_CUES: tuple[str, ...] = (
    "the answer is",
    "the correct answer",
    "the result is",
    "the solution is",
    "equals",
    "comes to",
)
_SCAFFOLD_HINT_CUES: tuple[str, ...] = (
    "notice",
    "look at",
    "focus on",
    "step 1",
    "first,",
    "step by step",
    "next step",
)
_AFFIRMS_OR_GREETS_CUES: tuple[str, ...] = (
    "exactly correct",
    "exactly right",
    "that's right",
    "that is right",
    "well done",
    "super!",
    "perfect!",
    "fantastic!",
    "awesome!",
    "excellent!",
    "excellent calculation",
    "excellent,",
    "great job",
    "great work",
    "nicely done",
    "youre right",
    "you're right",
    "correct!",
    "correct.",
    "good job",
    "yep",
    "hey",
    "hi!",
    "hello!",
    "well done.",
    "well done,",
    "that's the correct",
    "that is the correct",
    "the correct solution",
    "the correct reasoning",
    "the correct answer.  well",  # praise around "the correct answer"
)
_CORRECTS_WITH_FACT_CUES: tuple[str, ...] = (
    "you said",
    "you just said",
    "you jusy said",
    "that's not right",
    "that's not quite right",
    "that is not right",
    "still not quite right",
    "still not right",
    "not quite right",
    "not quite correct",
    "isn't quite correct",
    "isn't correct",
    "is not correct",
    "not quite.",
    "that's not quite",
    "not correct, so",
    "that's incorrect",
    "that is incorrect",
    "that's wrong",
    "that is wrong",
    " doesn't equal",
    "does not equal",
    "will never equal",
    "re-calculate",
    "recalculate",
    "redo the problem",
    "redo this",
    "remember that",
    "remember, ",
    "look back",
    "look again",
    "going back",
    "is incorrect",
    "not correct",
)


class StubEMPATHJudge:
    """LLM-free judge derived from lexical heuristics.

    Deliberately conservative on :attr:`STATES_FINAL_ANSWER` — only fires
    on the explicit "the answer is X" family of cues, so the stub does
    not inflate the DIRECT_ANSWER rate against auditor labels. The LLM
    judge replaces this with semantic detection of "contains the complete
    answer to the problem".
    """

    def judge(
        self,
        tutor_utterance: str,
        *,
        problem_context: str = "",
        student_utterance: str = "",
    ) -> EMPATHVerdict:
        text = tutor_utterance.lower()
        states_final_answer = any(cue in text for cue in _FINAL_ANSWER_CUES)
        # Praise with no math-content wins over corrects_with_fact: a
        # praise-dominant turn like "That's right, well done." must not
        # fall into DIRECT_ANSWER even if a trailing phrase happens to
        # resemble a correction cue.
        is_praise_dominant = (
            not states_final_answer
            and len(text.strip()) <= 160
            and any(cue in text for cue in _AFFIRMS_OR_GREETS_CUES)
        )
        corrects_with_fact = (
            not states_final_answer
            and not is_praise_dominant
            and any(cue in text for cue in _CORRECTS_WITH_FACT_CUES)
        )
        requests_reasoning = (
            not corrects_with_fact
            and "?" in text
            and any(
                cue in text
                for cue in ("what", "why", "how", "can you", "could you")
            )
        )
        provides_scaffold_hint = (
            not states_final_answer
            and not corrects_with_fact
            and any(cue in text for cue in _SCAFFOLD_HINT_CUES)
        )
        affirms_or_greets = is_praise_dominant and not corrects_with_fact
        sub_criteria = {
            EMPATHSubCriterion.REQUESTS_REASONING: requests_reasoning,
            EMPATHSubCriterion.STATES_FINAL_ANSWER: states_final_answer,
            EMPATHSubCriterion.PROVIDES_SCAFFOLD_HINT: provides_scaffold_hint,
            EMPATHSubCriterion.CORRECTS_WITH_FACT: corrects_with_fact,
            EMPATHSubCriterion.AFFIRMS_OR_GREETS: affirms_or_greets,
        }
        return EMPATHVerdict(
            label=aggregate_sub_criteria(sub_criteria),
            sub_criteria=sub_criteria,
            rationale=(
                f"stub: reasoning={requests_reasoning} "
                f"final_answer={states_final_answer} "
                f"scaffold_hint={provides_scaffold_hint} "
                f"corrects_with_fact={corrects_with_fact} "
                f"affirms_or_greets={affirms_or_greets}"
            ),
            proxy=True,
        )


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------


_EMPATH_PROMPT: str = (
    "You are an auditor labeling a math tutor's turn against five binary "
    "criteria. Reason briefly, then output a single JSON object with "
    "five booleans — no other text after the JSON.\n\n"
    "Criteria:\n"
    "- requests_reasoning: TRUE iff the tutor asks the student a GENUINE "
    "question that invites the student to explain, attempt, or notice "
    "something ON THEIR OWN. Mark FALSE if the question is rhetorical, "
    "if it re-states the student's words to show a contradiction, or if "
    "the tutor has already supplied the computation and is only asking "
    "the student to confirm (e.g. '…he had 990 cards. how many times a "
    "week does he tear?' — tutor supplied the number, so this is a "
    "walkthrough-step, not a reasoning request).\n"
    "- states_final_answer: TRUE iff the tutor's turn contains the complete "
    "final answer to the problem (not an intermediate step). Praise words "
    "with no computation ('correct', 'correct.', 'exactly right', 'well "
    "done') are FALSE here — they do not state the answer, they only "
    "affirm. Intermediate computations ('36 plates per day, times this by "
    "4') are also FALSE here but are TRUE under corrects_with_fact.\n"
    "- provides_scaffold_hint: TRUE iff the tutor names a specific next "
    "step or focuses the student's attention on a specific part of the "
    "problem WITHOUT stating the complete answer AND WITHOUT doing the "
    "computation for the student. Pointing to the student's PRIOR WORK "
    "to review it (e.g. 'review your calculation', 'in your third step "
    "you divided…') is a scaffold hint; so is 'notice the sign', 'look "
    "at the next step'. If the tutor is doing the arithmetic for the "
    "student rather than asking the student to do it, that is NOT a "
    "scaffold hint — mark FALSE here and TRUE under corrects_with_fact.\n"
    "- corrects_with_fact: TRUE iff the tutor directly corrects the "
    "student's error or does the arithmetic for the student. This "
    "includes: stating a math fact that contradicts the student's answer, "
    "re-quoting the student's own words to show the contradiction, "
    "saying 'that's not right' / 'that's incorrect' / 'not quite' / "
    "'this isn't quite correct', asking the student to 're-calculate' "
    "or 'redo' the problem, using a rhetorical question that re-states "
    "the student's words, OR performing the intermediate arithmetic "
    "step for the student (e.g. 'if they used 12 plates per meal and "
    "there were 3 meals a day, it would be 36 plates per day', 'he had "
    "the 90$, plus the 1.8 and the 1.2, whats the answer?', '36 plates "
    "per day, times this by 4'). This is MathDial's 'telling' move: the "
    "tutor is giving the student math, not asking the student to produce "
    "it themselves. Do NOT mark TRUE if the turn is pure praise with no "
    "arithmetic ('correct', 'well done', 'exactly right').\n"
    "- affirms_or_greets: TRUE iff the tutor turn is pure affirmation, "
    "congratulation, or social greeting with no math content and no "
    "scaffolding step — 'Super!', 'Well done', 'Exactly correct', 'That's "
    "right', 'Hi!', 'hey', 'correct.', 'you're right'. Praise sentences "
    "that also include a scaffolding question ('Great, so what's your "
    "next step?') are FALSE here (set provides_scaffold_hint TRUE "
    "instead). This maps MathDial's GENERIC/support turns.\n\n"
    "Mutual-exclusion priority (apply in order, and when a higher-priority "
    "criterion is TRUE the lower ones should be FALSE unless the turn "
    "genuinely does both independent things):\n"
    "  1. If states_final_answer is TRUE, nothing else can also be TRUE.\n"
    "  2. If corrects_with_fact is TRUE, requests_reasoning / "
    "provides_scaffold_hint / affirms_or_greets must be FALSE.\n"
    "  3. If affirms_or_greets is TRUE and the turn has no scaffolding "
    "step and no genuine reasoning request, the others are FALSE.\n"
    "  4. requests_reasoning and provides_scaffold_hint can coexist.\n\n"
    "PROBLEM CONTEXT: {problem_context}\n"
    "STUDENT'S LAST TURN: {student_utterance}\n"
    "TUTOR'S TURN TO LABEL: {tutor_utterance}\n\n"
    'Respond with: {{"requests_reasoning": true/false, '
    '"states_final_answer": true/false, '
    '"provides_scaffold_hint": true/false, '
    '"corrects_with_fact": true/false, '
    '"affirms_or_greets": true/false}}'
)


def _find_balanced_json_object(text: str) -> str | None:
    """Return the first brace-balanced JSON object substring in ``text``.

    Walks the input tracking brace depth with string-literal awareness, so
    a ``{`` or ``}`` inside a quoted string doesn't throw off the count
    and nested objects are returned whole. Returns None if no outer-most
    balanced object exists. Deliberately not a regex: Python's ``re`` is
    non-recursive, so ``\\{[^{}]*\\}`` would silently miss any nested
    rationale field a future prompt revision adds.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if escaped:
            escaped = False
            continue
        if in_string:
            if ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _parse_empath_json(
    response_text: str,
) -> tuple[dict[EMPATHSubCriterion, bool], bool]:
    """Extract the five-boolean JSON block from an LLM response.

    Robust to prose wrappers around the JSON object AND to nested fields
    the prompt may grow in the future (e.g. a ``rationale: {...}`` object).
    Returns the parsed booleans and a ``parse_failed`` flag — True iff the
    response contained no balanced JSON object or the block could not be
    decoded. In the failure case all booleans default to False, the
    aggregated label is BORDERLINE, and that turn drops out of the κ
    denominator. The caller is expected to count ``parse_failed``
    separately: a run with 100% parse failures silently returns κ=0 over
    n=0 items, which is a judge failure, not a judge decision.
    """
    default = {
        EMPATHSubCriterion.REQUESTS_REASONING: False,
        EMPATHSubCriterion.STATES_FINAL_ANSWER: False,
        EMPATHSubCriterion.PROVIDES_SCAFFOLD_HINT: False,
        EMPATHSubCriterion.CORRECTS_WITH_FACT: False,
        EMPATHSubCriterion.AFFIRMS_OR_GREETS: False,
    }
    block = _find_balanced_json_object(response_text)
    if block is None:
        return default, True
    try:
        obj = json.loads(block)
    except (json.JSONDecodeError, ValueError):
        return default, True
    if not isinstance(obj, dict):
        return default, True
    parsed = {
        EMPATHSubCriterion.REQUESTS_REASONING: bool(obj.get("requests_reasoning", False)),
        EMPATHSubCriterion.STATES_FINAL_ANSWER: bool(obj.get("states_final_answer", False)),
        EMPATHSubCriterion.PROVIDES_SCAFFOLD_HINT: bool(obj.get("provides_scaffold_hint", False)),
        EMPATHSubCriterion.CORRECTS_WITH_FACT: bool(obj.get("corrects_with_fact", False)),
        EMPATHSubCriterion.AFFIRMS_OR_GREETS: bool(obj.get("affirms_or_greets", False)),
    }
    return parsed, False


class LLMEMPATHJudge:
    """LLM-backed EMPATH judge.

    One chat per tutor turn; the ``LLMRouter`` cache keeps re-runs over the
    same held-out sample free after the first pass. ``tier`` defaults to
    ``TaskTier.SIMPLE`` per the project's local-first posture — override to
    ``TaskTier.ADVANCED`` for the final held-out audit if the stub-vs-LLM
    calibration MAD is high.
    """

    def __init__(self, router: Any, *, tier: Any = None, max_tokens: int = 200) -> None:
        self.router = router
        self.tier = tier
        self.max_tokens = max_tokens

    def judge(
        self,
        tutor_utterance: str,
        *,
        problem_context: str = "",
        student_utterance: str = "",
    ) -> EMPATHVerdict:
        from psychology_base.llm.types import ChatMessage, TaskTier

        tier = self.tier if self.tier is not None else TaskTier.SIMPLE
        prompt = _EMPATH_PROMPT.format(
            problem_context=problem_context or "(not supplied)",
            student_utterance=student_utterance or "(not supplied)",
            tutor_utterance=tutor_utterance,
        )
        completion = self.router.chat(
            [ChatMessage(role="user", content=prompt)],
            tier=tier,
            max_tokens=self.max_tokens,
        )
        text = getattr(completion, "content", "") or getattr(completion, "text", "")
        sub_criteria, parse_failed = _parse_empath_json(text)
        return EMPATHVerdict(
            label=aggregate_sub_criteria(sub_criteria),
            sub_criteria=sub_criteria,
            rationale=text.strip(),
            proxy=False,
            parse_failed=parse_failed,
        )


__all__ = [
    "EMPATHJudge",
    "EMPATHSubCriterion",
    "EMPATHVerdict",
    "LLMEMPATHJudge",
    "StubEMPATHJudge",
    "aggregate_sub_criteria",
]
