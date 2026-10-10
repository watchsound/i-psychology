"""Unit tests for the PATIENT-Ψ adapter.

Covers plan §4.2:

- Shim round-trip: :func:`iter_patient_psi_profiles` yields the normalised
  profile fields in file order with ``limit`` applied.
- SI lexical inference: word-boundary match prevents "burden" from
  false-positiving on "overburdened" (regression pin — bug caught at
  N=3 shakeout on all three public-sample profiles' history fields).
- ``suicidal_ideation_only`` filter applies BEFORE ``limit``.
- :meth:`CognitiveModel.dominant_core_belief_family` collapses the
  three families to a single label with "mixed"/"none" sentinels.
- Explicit ``suicidal_ideation_flag`` field on a shim row overrides
  the lexical inference (makes the shim-swap path to the gated
  dataset trivial).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from psychology_base.eval.benchmarks.public_datasets.patient_psi import (
    CognitiveModel,
    PatientPsiProfile,
    _lexical_si_flag,
    _text_contains_any,
    iter_patient_psi_profiles,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _row(
    *,
    profile_id: str = "1-1",
    name: str = "Abe",
    style_types: tuple[str, ...] = ("plain", "verbose"),
    history: str = "Mom works two jobs.",
    helpless_belief: tuple[str, ...] = ("I am incompetent.",),
    unlovable_belief: tuple[str, ...] = (),
    worthless_belief: tuple[str, ...] = (),
    intermediate_belief: str = "If I avoid challenges, I'll be okay.",
    coping_strategies: str = "Avoids asking for help.",
    situation: str = "Thinking about bills.",
    auto_thought: str = "What if I run out of money?",
    emotion: tuple[str, ...] = ("anxious",),
    behavior: str = "Continues to sit on couch.",
    extra: dict | None = None,
) -> dict:
    row = {
        "profile_id": profile_id,
        "name": name,
        "type": list(style_types),
        "history": history,
        "helpless_belief": list(helpless_belief),
        "unlovable_belief": list(unlovable_belief),
        "worthless_belief": list(worthless_belief),
        "intermediate_belief": intermediate_belief,
        "coping_strategies": coping_strategies,
        "situation": situation,
        "auto_thought": auto_thought,
        "emotion": list(emotion),
        "behavior": behavior,
    }
    if extra:
        row.update(extra)
    return row


def _write_shim(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# Shim round-trip
# ---------------------------------------------------------------------------


def test_iter_patient_psi_profiles_round_trip(tmp_path: Path) -> None:
    shim = _write_shim(
        tmp_path / "profiles.jsonl",
        [
            _row(profile_id="1-1", name="Abe"),
            _row(profile_id="1-2", name="Abe", situation="At work."),
        ],
    )
    profiles = list(iter_patient_psi_profiles(shim))
    assert len(profiles) == 2
    assert profiles[0].profile_id == "1-1"
    assert profiles[1].profile_id == "1-2"
    assert profiles[0].cognitive_model.auto_thought == "What if I run out of money?"
    assert profiles[0].style_types == ("plain", "verbose")
    assert profiles[0].cognitive_model.core_beliefs_helpless == (
        "I am incompetent.",
    )


def test_iter_patient_psi_profiles_respects_limit(tmp_path: Path) -> None:
    shim = _write_shim(
        tmp_path / "profiles.jsonl",
        [_row(profile_id=f"1-{i}") for i in range(5)],
    )
    profiles = list(iter_patient_psi_profiles(shim, limit=2))
    assert len(profiles) == 2
    assert [p.profile_id for p in profiles] == ["1-0", "1-1"]


def test_iter_patient_psi_profiles_skips_blank_lines(tmp_path: Path) -> None:
    shim = tmp_path / "profiles.jsonl"
    shim.write_text(
        json.dumps(_row(profile_id="1-1"))
        + "\n\n"
        + json.dumps(_row(profile_id="1-2"))
        + "\n",
        encoding="utf-8",
    )
    profiles = list(iter_patient_psi_profiles(shim))
    assert [p.profile_id for p in profiles] == ["1-1", "1-2"]


def test_iter_patient_psi_profiles_raises_on_missing_field(tmp_path: Path) -> None:
    bad = _row()
    bad.pop("situation")
    shim = _write_shim(tmp_path / "profiles.jsonl", [bad])
    with pytest.raises(ValueError, match="missing fields"):
        list(iter_patient_psi_profiles(shim))


def test_iter_patient_psi_profiles_accepts_str_type_and_emotion(
    tmp_path: Path,
) -> None:
    row = _row()
    row["type"] = "plain"
    row["emotion"] = "anxious"
    shim = _write_shim(tmp_path / "profiles.jsonl", [row])
    profiles = list(iter_patient_psi_profiles(shim))
    assert profiles[0].style_types == ("plain",)
    assert profiles[0].cognitive_model.emotion == ("anxious",)


# ---------------------------------------------------------------------------
# SI lexical inference — regression pin on word-boundary match
# ---------------------------------------------------------------------------


def test_text_contains_any_word_boundary_prevents_overburdened_match() -> None:
    """REGRESSION: initial lexical SI inference substring-matched "burden"
    inside "overburdened", flagging all three public-sample profiles as
    SI despite them being depression-presentation. The fix is word-boundary
    matching; this test pins the fix."""

    assert _text_contains_any("Mom is overburdened by work.", ("burden",)) is False
    assert _text_contains_any("I'm a burden to my family.", ("burden",)) is True


def test_text_contains_any_phrase_hits() -> None:
    assert _text_contains_any("there is no way out of this", ("no way out",)) is True
    assert _text_contains_any("I'd rather give up on this project", ("give up",)) is True


def test_text_contains_any_case_insensitive() -> None:
    assert _text_contains_any("HOPELESS right now", ("hopeless",)) is True


def test_lexical_si_flag_fires_on_means_cue() -> None:
    row = _row(auto_thought="I keep thinking about pills.")
    assert _lexical_si_flag(row) is True


def test_lexical_si_flag_fires_on_self_harm_cue() -> None:
    row = _row(behavior="I want to hurt myself.")
    assert _lexical_si_flag(row) is True


def test_lexical_si_flag_fires_on_hopelessness_cue() -> None:
    row = _row(auto_thought="There's no point anymore.")
    assert _lexical_si_flag(row) is True


def test_lexical_si_flag_does_not_fire_on_depression_only() -> None:
    """Public PATIENT-Ψ sample: all 3 profiles are depression-presentation
    (Abe at 3 situations), none should trip the SI inference. Pin this
    behaviour so SI-subset-empty short-circuits honestly."""

    row = _row(
        auto_thought="I'll never be good enough at this job.",
        emotion=("sad", "frustrated"),
        behavior="Sits on couch, procrastinates.",
        history="Dad left when I was young. Mom is overburdened.",
    )
    assert _lexical_si_flag(row) is False


# ---------------------------------------------------------------------------
# suicidal_ideation_only filter semantics — filter BEFORE limit
# ---------------------------------------------------------------------------


def test_suicidal_ideation_only_filter_applies_before_limit(tmp_path: Path) -> None:
    rows = [
        _row(profile_id="1-1", auto_thought="normal thought"),  # not SI
        _row(profile_id="1-2", auto_thought="I keep thinking about pills."),  # SI
        _row(profile_id="1-3", auto_thought="also normal"),  # not SI
        _row(profile_id="1-4", auto_thought="I want to end my life."),  # SI
        _row(profile_id="1-5", auto_thought="also fine"),  # not SI
    ]
    shim = _write_shim(tmp_path / "profiles.jsonl", rows)

    si_profiles = list(
        iter_patient_psi_profiles(shim, suicidal_ideation_only=True, limit=10)
    )
    assert [p.profile_id for p in si_profiles] == ["1-2", "1-4"]

    si_capped = list(
        iter_patient_psi_profiles(shim, suicidal_ideation_only=True, limit=1)
    )
    assert [p.profile_id for p in si_capped] == ["1-2"]


def test_suicidal_ideation_flag_explicit_override_wins(tmp_path: Path) -> None:
    """If a shim row carries an explicit ``suicidal_ideation_flag``
    boolean, the lexical inference is skipped and the explicit value is
    used. This makes the gated-dataset drop-in replacement a shim-swap
    with no re-engineering."""

    row = _row(auto_thought="I keep thinking about pills.")  # would lex-flag True
    row["suicidal_ideation_flag"] = False
    shim = _write_shim(tmp_path / "profiles.jsonl", [row])
    profiles = list(iter_patient_psi_profiles(shim))
    assert profiles[0].suicidal_ideation_flag is False


# ---------------------------------------------------------------------------
# CognitiveModel helpers
# ---------------------------------------------------------------------------


def test_dominant_core_belief_family_helpless() -> None:
    cm = CognitiveModel(
        core_beliefs_helpless=("I am incompetent.",),
        core_beliefs_unlovable=(),
        core_beliefs_worthless=(),
        intermediate_belief="",
        coping_strategies="",
        situation="",
        auto_thought="",
        emotion=(),
        behavior="",
    )
    assert cm.dominant_core_belief_family() == "helpless"


def test_dominant_core_belief_family_mixed_and_none() -> None:
    mixed = CognitiveModel(
        core_beliefs_helpless=("a",),
        core_beliefs_unlovable=("b",),
        core_beliefs_worthless=(),
        intermediate_belief="",
        coping_strategies="",
        situation="",
        auto_thought="",
        emotion=(),
        behavior="",
    )
    empty = CognitiveModel(
        core_beliefs_helpless=(),
        core_beliefs_unlovable=(),
        core_beliefs_worthless=(),
        intermediate_belief="",
        coping_strategies="",
        situation="",
        auto_thought="",
        emotion=(),
        behavior="",
    )
    assert mixed.dominant_core_belief_family() == "mixed"
    assert empty.dominant_core_belief_family() == "none"


def test_presenting_concern_composes_from_quadruple() -> None:
    profile = PatientPsiProfile(
        profile_id="1-1",
        name="Abe",
        style_types=("plain",),
        history="",
        cognitive_model=CognitiveModel(
            core_beliefs_helpless=("I am incompetent.",),
            core_beliefs_unlovable=(),
            core_beliefs_worthless=(),
            intermediate_belief="",
            coping_strategies="",
            situation="Thinking about bills.",
            auto_thought="What if I run out of money?",
            emotion=("anxious",),
            behavior="",
        ),
        suicidal_ideation_flag=False,
    )
    concern = profile.presenting_concern()
    assert "Abe" in concern
    assert "helpless" in concern
    assert "Thinking about bills." in concern
    assert "What if I run out of money?" in concern
