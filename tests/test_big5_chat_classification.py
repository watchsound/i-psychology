"""Phase 8 Pillar C — BIG5-CHAT classification scorer tests.

In-memory fixtures only. No network, no LLM. Exercises:
- lexical-cue hits that should produce the correct polarity call
- empty-signal utterances that fall to neutral and count against the gate
- per-trait / per-level aggregation + confusion matrix wiring
- soft-gate boundary at 0.60 accuracy
"""

from __future__ import annotations

from psychology_base.eval.benchmarks.public_datasets.big5_chat import (
    Big5ChatDialog,
)
from psychology_base.eval.benchmarks.public_datasets.big5_chat_classification import (
    score_dialogs,
)


def _d(dialog_id: str, trait: str, level: str, output: str) -> Big5ChatDialog:
    return Big5ChatDialog(
        dialog_id=dialog_id,
        env_idx=0,
        trait=trait,
        level=level,
        train_input="context",
        train_output=output,
    )


def test_empty_input_returns_zero_report() -> None:
    r = score_dialogs([])
    assert r.n_dialogs == 0
    assert r.polarity_accuracy == 0.0
    assert r.polarity_accuracy_incl_neutral == 0.0
    assert not r.meets_soft_gate()


def test_scorer_handles_single_dialog() -> None:
    # Explicit high-neuroticism self-label hits the lexical cue bank.
    dialog = _d("d1", "neuroticism", "high", "i'm anxious about everything")
    r = score_dialogs([dialog])
    assert r.n_dialogs == 1
    first = r.per_dialog[0]
    assert first.dialog_id == "d1"
    assert first.trait == "neuroticism"
    assert first.level == "high"


def test_lexical_hit_classifies_high_neuroticism_correctly() -> None:
    # "i'm anxious" is in the explicit high-neuroticism cue bank, so
    # inference should score neuroticism above the midpoint.
    dialog = _d("d1", "neuroticism", "high", "i'm anxious about tomorrow")
    r = score_dialogs([dialog])
    first = r.per_dialog[0]
    assert first.on_labeled_dim
    assert first.predicted_level == "high"
    assert first.correct


def test_lexical_hit_classifies_low_extraversion_correctly() -> None:
    dialog = _d("d1", "extraversion", "low", "i'm introverted and prefer alone")
    r = score_dialogs([dialog])
    first = r.per_dialog[0]
    assert first.predicted_level == "low"
    assert first.correct


def test_neutral_utterance_lands_neutral_or_implicit() -> None:
    # A neutral content string with no cues should land near the midpoint
    # on the labeled dim; the scorer records this as neutral or very
    # weakly signed (implicit baseline). Either way, coverage drops.
    dialog = _d("d1", "agreeableness", "high", "went to the grocery store today")
    r = score_dialogs([dialog])
    first = r.per_dialog[0]
    # Agreeableness has no implicit signal in Phase 7 — expect neutral.
    assert first.predicted_level == "neutral"
    assert not first.on_labeled_dim


def test_neutral_counts_against_incl_gate_but_not_signed_accuracy() -> None:
    dialogs = [
        _d("d1", "neuroticism", "high", "i'm anxious about tomorrow"),   # correct
        _d("d2", "agreeableness", "high", "went to the store"),          # neutral
    ]
    r = score_dialogs(dialogs)
    # Signed accuracy ignores neutral; one signed call, correct → 1.0.
    assert r.polarity_accuracy == 1.0
    # Incl-neutral counts the neutral as wrong → 0.5.
    assert r.polarity_accuracy_incl_neutral == 0.5
    assert r.n_neutral == 1


def test_per_trait_aggregation() -> None:
    dialogs = [
        _d("d1", "neuroticism", "high", "i'm anxious today"),            # correct
        _d("d2", "neuroticism", "low", "i'm anxious today"),             # wrong (predicts high)
        _d("d3", "extraversion", "high", "i love people, so social!"),   # correct
    ]
    r = score_dialogs(dialogs)
    assert r.per_trait_polarity_accuracy["neuroticism"] == 0.5
    assert r.per_trait_polarity_accuracy["extraversion"] == 1.0


def test_per_level_aggregation() -> None:
    dialogs = [
        _d("d1", "neuroticism", "high", "i'm anxious"),             # correct
        _d("d2", "extraversion", "high", "i love people, social!"), # correct
        _d("d3", "neuroticism", "low", "i'm calm"),                 # correct (lexical low cue)
    ]
    r = score_dialogs(dialogs)
    assert r.per_level_polarity_accuracy["high"] == 1.0
    assert r.per_level_polarity_accuracy["low"] == 1.0


def test_confusion_matrix_wires_predicted_counts() -> None:
    dialogs = [
        _d("d1", "neuroticism", "high", "i'm anxious today"),
        _d("d2", "neuroticism", "low", "i'm calm about everything"),
    ]
    r = score_dialogs(dialogs)
    conf_n = r.confusion["neuroticism"]
    # Two dialogs, both lexically clean — one high prediction, one low.
    assert conf_n["high"] == 1
    assert conf_n["low"] == 1
    assert conf_n["neutral"] == 0


def test_soft_gate_passes_at_60pct() -> None:
    # 6 correct, 4 wrong (incl-neutral) → 60%: ceiling-boundary pass.
    correct = [
        _d(f"c{i}", "neuroticism", "high", "i'm anxious today") for i in range(6)
    ]
    wrong = [
        _d(f"w{i}", "agreeableness", "high", "went to the store")
        for i in range(4)
    ]
    r = score_dialogs(correct + wrong)
    assert r.polarity_accuracy_incl_neutral == 0.6
    assert r.meets_soft_gate()


def test_soft_gate_fails_below_60pct() -> None:
    correct = [
        _d(f"c{i}", "neuroticism", "high", "i'm anxious") for i in range(5)
    ]
    wrong = [
        _d(f"w{i}", "agreeableness", "high", "went to the store")
        for i in range(5)
    ]
    r = score_dialogs(correct + wrong)
    assert r.polarity_accuracy_incl_neutral == 0.5
    assert not r.meets_soft_gate()


def test_coverage_reports_signal_fraction() -> None:
    dialogs = [
        _d("d1", "neuroticism", "high", "i'm anxious"),              # on-labeled
        _d("d2", "agreeableness", "high", "went to the store"),      # neutral
    ]
    r = score_dialogs(dialogs)
    assert r.coverage == 0.5
