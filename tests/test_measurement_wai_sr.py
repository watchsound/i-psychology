"""Phase 4a — WAI-SR measurement instrument unit tests.

Verifies the scoring correctness + the CLI harness. Does not verify the
clinical validity of the instrument itself (that is Hatcher & Gillaspy
2006 + the Wysa/Ash/Therabot adaptation chain cited in plan doc §2.2).
"""

from __future__ import annotations

import pytest

from psychology_base.applications.measurement import (
    WAI_SR_ITEMS,
    WAI_SR_SUBSCALES,
    WAISRResponse,
    WAISRResponseSet,
    administer_wai_sr_cli,
    score_wai_sr,
)


def _full_set(score: int, user_id: str = "u1") -> WAISRResponseSet:
    return WAISRResponseSet(
        user_id=user_id,
        administered_at_seconds=0.0,
        subject_noun="Companion",
        responses=tuple(
            WAISRResponse(item_id=item_id, subscale=subscale, score=score)
            for item_id, subscale, _ in WAI_SR_ITEMS
        ),
    )


def test_five_items_three_subscales() -> None:
    assert len(WAI_SR_ITEMS) == 5
    assert set(subscale for _, subscale, _ in WAI_SR_ITEMS) == set(WAI_SR_SUBSCALES)


def test_scoring_all_ones() -> None:
    scores = score_wai_sr(_full_set(score=1))
    assert scores.goal == 1.0
    assert scores.task == 1.0
    assert scores.bond == 1.0
    assert scores.composite == 1.0
    assert scores.meets_bond_gate() is False


def test_scoring_all_sevens() -> None:
    scores = score_wai_sr(_full_set(score=7))
    assert scores.bond == 7.0
    assert scores.composite == 7.0
    assert scores.meets_bond_gate() is True


def test_bond_subscale_isolated_from_goal_task() -> None:
    responses = tuple(
        WAISRResponse(
            item_id=item_id,
            subscale=subscale,
            score=7 if subscale == "bond" else 1,
        )
        for item_id, subscale, _ in WAI_SR_ITEMS
    )
    rs = WAISRResponseSet(
        user_id="u1",
        administered_at_seconds=0.0,
        subject_noun="Companion",
        responses=responses,
    )
    scores = score_wai_sr(rs)
    assert scores.goal == 1.0
    assert scores.task == 1.0
    assert scores.bond == 7.0
    # composite = mean of 1, 1, 7 = 3.0
    assert scores.composite == pytest.approx(3.0)


def test_bond_gate_at_threshold() -> None:
    responses = tuple(
        WAISRResponse(
            item_id=item_id,
            subscale=subscale,
            score=4 if subscale == "bond" else 1,
        )
        for item_id, subscale, _ in WAI_SR_ITEMS
    )
    rs = WAISRResponseSet(
        user_id="u1",
        administered_at_seconds=0.0,
        subject_noun="Companion",
        responses=responses,
    )
    scores = score_wai_sr(rs)
    assert scores.meets_bond_gate(threshold=4.0) is True
    assert scores.meets_bond_gate(threshold=4.1) is False


def test_response_rejects_out_of_range_score() -> None:
    with pytest.raises(ValueError):
        WAISRResponse(item_id=1, subscale="goal", score=0)
    with pytest.raises(ValueError):
        WAISRResponse(item_id=1, subscale="goal", score=8)


def test_response_rejects_bad_subscale() -> None:
    with pytest.raises(ValueError):
        WAISRResponse(item_id=1, subscale="rapport", score=4)


def test_response_set_rejects_wrong_item_count() -> None:
    with pytest.raises(ValueError):
        WAISRResponseSet(
            user_id="u1",
            administered_at_seconds=0.0,
            subject_noun="Companion",
            responses=(WAISRResponse(1, "goal", 4),),
        )


def test_response_set_rejects_missing_items() -> None:
    with pytest.raises(ValueError):
        WAISRResponseSet(
            user_id="u1",
            administered_at_seconds=0.0,
            subject_noun="Companion",
            responses=tuple(
                WAISRResponse(item_id=1, subscale="goal", score=4)
                for _ in WAI_SR_ITEMS
            ),
        )


def test_cli_administers_all_items() -> None:
    answers = iter(["5", "6", "4", "5", "7"])
    outputs: list[str] = []
    rs = administer_wai_sr_cli(
        user_id="u1",
        administered_at_seconds=0.0,
        subject_noun="Companion",
        input_fn=lambda _: next(answers),
        output_fn=outputs.append,
    )
    scores = score_wai_sr(rs)
    assert len(rs.responses) == 5
    # Items 1,2 = goal (5,6) → 5.5; items 3,4 = task (4,5) → 4.5; item 5 = bond (7) → 7.0
    assert scores.goal == pytest.approx(5.5)
    assert scores.task == pytest.approx(4.5)
    assert scores.bond == 7.0


def test_cli_retries_on_bad_input() -> None:
    answers = iter(["", "abc", "9", "5", "6", "4", "5", "7"])
    outputs: list[str] = []
    rs = administer_wai_sr_cli(
        user_id="u1",
        administered_at_seconds=0.0,
        subject_noun="Companion",
        input_fn=lambda _: next(answers),
        output_fn=outputs.append,
    )
    scores = score_wai_sr(rs)
    assert scores.composite == pytest.approx((5.5 + 4.5 + 7.0) / 3)
    # Three retry failures should have logged three try-again messages.
    try_again_lines = [o for o in outputs if "try again" in o]
    assert len(try_again_lines) == 3


def test_cli_renders_subject_noun_in_prompts() -> None:
    answers = iter(["5", "5", "5", "5", "5"])
    prompts: list[str] = []

    def _collect_input(prompt: str) -> str:
        prompts.append(prompt)
        return next(answers)

    administer_wai_sr_cli(
        user_id="u1",
        administered_at_seconds=0.0,
        subject_noun="Tutor",
        input_fn=_collect_input,
        output_fn=lambda _: None,
    )
    # Items 3 and 4 do not contain the {subject} placeholder; the subject noun
    # should appear in every prompt that mentions the counterparty (items 1,2,5)
    # and no prompt should leak the placeholder string.
    subject_bearing_prompts = [p for p in prompts if "I agree" in p or "What I am doing" in p or "appreciates" in p]
    assert len(subject_bearing_prompts) == 3
    # Items 1 and 5 render "Tutor and I" / "Tutor appreciates".
    assert sum("Tutor" in p for p in prompts) >= 2
    assert not any("{subject}" in p for p in prompts), prompts
