"""Phase 8 Pillar C v4 — LLM semantic judge scorer tests.

In-memory fixtures only. No network, no real LLM router. Covers:

- prompt builder: trait validation, trait description + few-shot block
  structure, speaker + text injection
- response parser: ``POLARITY=`` + ``LIKERT=`` parsing, Likert / polarity
  cross-check clamping, unresolved on missing fields, case-insensitive,
  whitespace-tolerant
- scorer end-to-end with a scripted judge that returns deterministic
  verdicts per dialog
- unresolved items excluded from denominators, logged on
  ``unresolved_dialog_ids``, not counted against the gate
- per-trait / per-level aggregation + confusion matrix wiring
- κ helper on sequences shorter than and equal to the lexical scorer's
  output shape; length mismatch raises
"""

from __future__ import annotations

import pytest

from psychology_base.eval.benchmarks.public_datasets.big5_chat import (
    Big5ChatDialog,
)
from psychology_base.eval.benchmarks.public_datasets.big5_chat_classification_llm import (
    Big5ChatClassReportLLM,
    ParsedJudgment,
    build_classification_prompt,
    cohens_kappa,
    parse_judgment,
    score_dialogs_llm,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _d(dialog_id: str, trait: str, level: str, output: str) -> Big5ChatDialog:
    return Big5ChatDialog(
        dialog_id=dialog_id,
        env_idx=0,
        trait=trait,
        level=level,
        train_input="context",
        train_output=output,
        persony="Alex",
    )


def _scripted_judge(verdicts: dict[str, tuple[str, float]]):
    """Return a judge callable that looks up verdicts by dialog_id."""

    def _judge(dialog: Big5ChatDialog) -> ParsedJudgment:
        if dialog.dialog_id not in verdicts:
            return ParsedJudgment(
                polarity="unresolved",
                likert=3.0,
                raw_response="(no scripted verdict)",
            )
        polarity, likert = verdicts[dialog.dialog_id]
        return ParsedJudgment(
            polarity=polarity,
            likert=likert,
            raw_response=f"POLARITY={polarity}\nLIKERT={int(likert)}",
        )

    return _judge


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def test_prompt_builder_rejects_unknown_trait() -> None:
    with pytest.raises(ValueError, match="Unknown trait"):
        build_classification_prompt(trait="charisma", text="hi", speaker="A")


def test_prompt_builder_emits_system_plus_user_message_pair() -> None:
    msgs = build_classification_prompt(
        trait="openness", text="let's try something new", speaker="Alex"
    )
    assert len(msgs) == 2
    assert msgs[0].role == "system"
    assert msgs[1].role == "user"
    assert "classifier" in msgs[0].content.lower()


def test_prompt_builder_injects_trait_and_text_into_user_message() -> None:
    msgs = build_classification_prompt(
        trait="agreeableness",
        text="thanks so much for your help",
        speaker="Bea",
    )
    user = msgs[1].content
    assert "agreeableness" in user.lower()
    assert "thanks so much for your help" in user
    assert "Bea" in user
    # Few-shot block renders at least the three polarities as anchors.
    assert user.lower().count("polarity=high") >= 1
    assert user.lower().count("polarity=low") >= 1
    assert user.lower().count("polarity=neutral") >= 1


def test_prompt_builder_formats_response_contract() -> None:
    msgs = build_classification_prompt(
        trait="neuroticism", text="i'm fine", speaker="C"
    )
    user = msgs[1].content
    # The response contract literal must survive into the prompt body.
    assert "POLARITY=<high|low|neutral>" in user
    assert "LIKERT=<1|2|3|4|5>" in user


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------


def test_parse_judgment_canonical_format() -> None:
    out = parse_judgment("POLARITY=high\nLIKERT=5")
    assert out.polarity == "high"
    assert out.likert == 5.0


def test_parse_judgment_case_insensitive_polarity() -> None:
    out = parse_judgment("POLARITY=HIGH\nLIKERT=4")
    assert out.polarity == "high"


def test_parse_judgment_whitespace_around_equals() -> None:
    out = parse_judgment("POLARITY = low \nLIKERT = 2")
    assert out.polarity == "low"
    assert out.likert == 2.0


def test_parse_judgment_markdown_wrapper_tolerated() -> None:
    out = parse_judgment(
        "Here is my call:\n```\nPOLARITY=neutral\nLIKERT=3\n```\n"
    )
    assert out.polarity == "neutral"
    assert out.likert == 3.0


def test_parse_judgment_missing_polarity_is_unresolved() -> None:
    out = parse_judgment("LIKERT=4\n(the model forgot the first line)")
    assert out.polarity == "unresolved"


def test_parse_judgment_missing_likert_is_unresolved() -> None:
    out = parse_judgment("POLARITY=high (lost the likert line)")
    assert out.polarity == "unresolved"


def test_parse_judgment_clamps_mismatched_polarity_likert() -> None:
    # Model says HIGH but Likert=2 — polarity is the primary gate label,
    # so we clamp the Likert into the half-space above the midpoint.
    out = parse_judgment("POLARITY=high\nLIKERT=2")
    assert out.polarity == "high"
    assert out.likert > 3.0

    # Opposite: LOW polarity but Likert=5 → clamp below midpoint.
    out = parse_judgment("POLARITY=low\nLIKERT=5")
    assert out.polarity == "low"
    assert out.likert < 3.0

    # Neutral polarity always forces Likert to the midpoint.
    out = parse_judgment("POLARITY=neutral\nLIKERT=5")
    assert out.polarity == "neutral"
    assert out.likert == 3.0


def test_parse_judgment_empty_text_is_unresolved() -> None:
    out = parse_judgment("")
    assert out.polarity == "unresolved"
    assert out.likert == 3.0


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


def test_score_dialogs_llm_empty_returns_zero_report() -> None:
    r = score_dialogs_llm([], judge=_scripted_judge({}))
    assert r.n_dialogs == 0
    assert r.n_unresolved == 0
    assert r.polarity_accuracy == 0.0
    assert r.polarity_accuracy_incl_neutral == 0.0
    assert not r.meets_soft_gate()


def test_score_dialogs_llm_perfect_judge_passes_gate() -> None:
    dialogs = [
        _d("d1", "openness", "high", "what if we tried something wild"),
        _d("d2", "openness", "low", "let's stick to the usual"),
        _d("d3", "extraversion", "high", "come hang out, it'll be a blast!"),
    ]
    verdicts = {
        "d1": ("high", 5.0),
        "d2": ("low", 1.0),
        "d3": ("high", 5.0),
    }
    r = score_dialogs_llm(dialogs, judge=_scripted_judge(verdicts))
    assert r.n_dialogs == 3
    assert r.n_unresolved == 0
    assert r.polarity_accuracy_incl_neutral == 1.0
    assert r.polarity_accuracy == 1.0
    assert r.coverage == 1.0
    assert r.meets_soft_gate()


def test_score_dialogs_llm_neutral_counts_against_incl_gate() -> None:
    dialogs = [
        _d("d1", "openness", "high", "what if we tried something new"),
        _d("d2", "agreeableness", "high", "hi"),
    ]
    verdicts = {
        "d1": ("high", 5.0),
        "d2": ("neutral", 3.0),
    }
    r = score_dialogs_llm(dialogs, judge=_scripted_judge(verdicts))
    assert r.polarity_accuracy == 1.0           # signed-only
    assert r.polarity_accuracy_incl_neutral == 0.5  # incl-neutral
    assert r.n_neutral == 1
    assert r.coverage == 0.5


def test_score_dialogs_llm_unresolved_excluded_from_denominators() -> None:
    dialogs = [
        _d("d1", "openness", "high", "what if we tried something new"),
        _d("d2", "openness", "low", "let's stick to the usual"),
        _d("mystery", "neuroticism", "high", "???"),
    ]
    verdicts = {
        "d1": ("high", 5.0),
        "d2": ("low", 1.0),
        # d_mystery is NOT in the scripted dict → judge returns unresolved.
    }
    r = score_dialogs_llm(dialogs, judge=_scripted_judge(verdicts))
    assert r.n_dialogs == 2
    assert r.n_unresolved == 1
    assert "mystery" in r.unresolved_dialog_ids
    # Accuracy denominator is the resolved subset only.
    assert r.polarity_accuracy_incl_neutral == 1.0
    assert r.meets_soft_gate()


def test_score_dialogs_llm_wrong_polarity_lowers_accuracy() -> None:
    dialogs = [
        _d("d1", "openness", "high", "what if we tried"),
        _d("d2", "openness", "high", "what if we tried"),
    ]
    verdicts = {
        "d1": ("low", 1.0),
        "d2": ("high", 5.0),
    }
    r = score_dialogs_llm(dialogs, judge=_scripted_judge(verdicts))
    # One correct, one wrong, both signed → 0.5 both metrics.
    assert r.polarity_accuracy == 0.5
    assert r.polarity_accuracy_incl_neutral == 0.5
    assert not r.meets_soft_gate()


def test_score_dialogs_llm_per_trait_aggregation() -> None:
    dialogs = [
        _d("d1", "openness", "high", "..."),
        _d("d2", "openness", "low", "..."),
        _d("d3", "extraversion", "high", "..."),
    ]
    verdicts = {
        "d1": ("high", 5.0),       # correct
        "d2": ("high", 5.0),       # wrong (said high, label low)
        "d3": ("high", 5.0),       # correct
    }
    r = score_dialogs_llm(dialogs, judge=_scripted_judge(verdicts))
    assert r.per_trait_polarity_accuracy["openness"] == 0.5
    assert r.per_trait_polarity_accuracy["extraversion"] == 1.0


def test_score_dialogs_llm_confusion_matrix_wires_predicted_counts() -> None:
    dialogs = [
        _d("d1", "openness", "high", "..."),
        _d("d2", "openness", "low", "..."),
        _d("d3", "openness", "high", "..."),
    ]
    verdicts = {
        "d1": ("high", 5.0),
        "d2": ("low", 1.0),
        "d3": ("neutral", 3.0),
    }
    r = score_dialogs_llm(dialogs, judge=_scripted_judge(verdicts))
    conf = r.confusion["openness"]
    assert conf["high"] == 1
    assert conf["low"] == 1
    assert conf["neutral"] == 1


def test_score_dialogs_llm_routing_tag_preserved() -> None:
    r = score_dialogs_llm(
        [_d("d1", "openness", "high", "...")],
        judge=_scripted_judge({"d1": ("high", 5.0)}),
        routing="ollama / qwen2.5:7b (reason=ollama_available)",
    )
    assert "ollama" in r.routing


def test_score_dialogs_llm_per_dialog_rows_are_frozen_dataclasses() -> None:
    r = score_dialogs_llm(
        [_d("d1", "openness", "high", "...")],
        judge=_scripted_judge({"d1": ("high", 5.0)}),
    )
    row = r.per_dialog[0]
    assert row.dialog_id == "d1"
    assert row.predicted_level == "high"
    assert row.correct is True
    # Dataclass is frozen — mutation attempt should raise.
    with pytest.raises(Exception):
        row.correct = False  # type: ignore[misc]


def test_score_dialogs_llm_all_unresolved_short_circuits_to_empty_report() -> None:
    dialogs = [
        _d("d1", "openness", "high", "..."),
        _d("d2", "openness", "low", "..."),
    ]
    r = score_dialogs_llm(dialogs, judge=_scripted_judge({}))
    assert r.n_dialogs == 0
    assert r.n_unresolved == 2
    assert set(r.unresolved_dialog_ids) == {"d1", "d2"}
    assert r.polarity_accuracy_incl_neutral == 0.0


# ---------------------------------------------------------------------------
# κ calibration helper
# ---------------------------------------------------------------------------


def test_cohens_kappa_perfect_agreement_returns_one() -> None:
    a = ["high", "low", "neutral", "high", "low"]
    b = ["high", "low", "neutral", "high", "low"]
    assert cohens_kappa(a, b) == pytest.approx(1.0)


def test_cohens_kappa_zero_agreement_is_negative() -> None:
    a = ["high", "high", "high", "high"]
    b = ["low", "low", "low", "low"]
    # Both labelers collapsed to one category each — po=0, pe=0 ⇒ falls
    # back to po (0.0) per helper policy (undefined denominator).
    kappa = cohens_kappa(a, b)
    assert kappa <= 0.0


def test_cohens_kappa_mismatched_lengths_raises() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        cohens_kappa(["high", "low"], ["high"])


def test_cohens_kappa_empty_returns_zero() -> None:
    assert cohens_kappa([], []) == 0.0


def test_cohens_kappa_partial_agreement_is_between_zero_and_one() -> None:
    a = ["high", "high", "low", "low", "neutral", "neutral"]
    b = ["high", "low", "low", "high", "neutral", "neutral"]
    kappa = cohens_kappa(a, b)
    assert 0.0 < kappa < 1.0


# ---------------------------------------------------------------------------
# Calibration round-trip (judge-vs-lexical on a scripted shim)
# ---------------------------------------------------------------------------


def test_calibration_round_trip_llm_vs_lexical_on_shared_fixture() -> None:
    """One-call round-trip: run both the LLM scorer (scripted judge) and
    the lexical scorer over the same tiny shim, extract per-dialog
    polarity, feed through :func:`cohens_kappa`.

    The point of the test is not to pin κ to a specific value (the
    scripted judge is intentionally perfect-mirror on cue hits, so κ=1
    here) but to prove the plumbing path — the calibration script can
    run the same signatures end-to-end without network.
    """

    from psychology_base.eval.benchmarks.public_datasets.big5_chat_classification import (
        score_dialogs as score_dialogs_lexical,
    )

    dialogs = [
        _d("d1", "neuroticism", "high", "i'm anxious about tomorrow"),
        _d("d2", "extraversion", "low", "i'm introverted and prefer alone"),
    ]
    # Scripted judge mirrors ground truth so κ=1 against a lexical
    # pipeline that also hits these cues (both will say high / low).
    verdicts = {
        "d1": ("high", 5.0),
        "d2": ("low", 1.0),
    }
    llm_rep = score_dialogs_llm(dialogs, judge=_scripted_judge(verdicts))
    lex_rep = score_dialogs_lexical(dialogs)

    def _as_labels(rep) -> list[str]:
        return [row.predicted_level for row in rep.per_dialog]

    labels_llm = _as_labels(llm_rep)
    labels_lex = _as_labels(lex_rep)
    assert len(labels_llm) == len(labels_lex)
    kappa = cohens_kappa(labels_llm, labels_lex)
    assert kappa == pytest.approx(1.0)
