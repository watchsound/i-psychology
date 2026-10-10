"""Phase 4a — Socratic-compliance labeling + Argilla-shape export + κ."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from psychology_base.applications.measurement import (
    ComplianceLabel,
    LabeledTurn,
    TutorMove,
    auto_label_from_tutor_move,
    cohen_kappa,
    labeled_turn_to_argilla,
    summarise_user,
    write_argilla_jsonl,
)


def _labeled(
    user_id: str,
    turn_index: int,
    tutor_move: TutorMove,
    auditor_label: ComplianceLabel,
    session_id: str = "s1",
) -> LabeledTurn:
    return LabeledTurn(
        user_id=user_id,
        session_id=session_id,
        turn_index=turn_index,
        utterance=f"turn {turn_index} content",
        tutor_move=tutor_move,
        auditor_label=auditor_label,
        auditor_id="auditor-1",
    )


def test_auto_label_from_move_maps_canonically() -> None:
    assert auto_label_from_tutor_move(TutorMove.SOCRATIC_PROBE) is ComplianceLabel.SOCRATIC
    assert auto_label_from_tutor_move(TutorMove.FOCUS_ATTENTION) is ComplianceLabel.SOCRATIC
    assert auto_label_from_tutor_move(TutorMove.GENERIC_SUPPORT) is ComplianceLabel.SOCRATIC
    assert auto_label_from_tutor_move(TutorMove.WORKED_STEP) is ComplianceLabel.BORDERLINE
    assert auto_label_from_tutor_move(TutorMove.FULL_ANSWER) is ComplianceLabel.DIRECT_ANSWER


def test_summarise_user_counts_labels() -> None:
    labeled = [
        _labeled("u1", 0, TutorMove.SOCRATIC_PROBE, ComplianceLabel.SOCRATIC),
        _labeled("u1", 1, TutorMove.SOCRATIC_PROBE, ComplianceLabel.SOCRATIC),
        _labeled("u1", 2, TutorMove.FULL_ANSWER, ComplianceLabel.DIRECT_ANSWER),
        _labeled("u1", 3, TutorMove.WORKED_STEP, ComplianceLabel.BORDERLINE),
        _labeled("u2", 0, TutorMove.SOCRATIC_PROBE, ComplianceLabel.SOCRATIC),
    ]
    r = summarise_user("u1", labeled)
    assert r.sampled_turns == 4
    assert r.socratic_count == 2
    assert r.direct_answer_count == 1
    assert r.borderline_count == 1
    # compliance_rate excludes borderline from the denominator.
    assert r.compliance_rate() == pytest.approx(2 / 3)


def test_meets_roadmap_gate_at_default_thresholds() -> None:
    # Build a 100-turn sample: 96 socratic, 2 direct, 2 borderline → 96/98 ≈ 0.9796
    labeled = []
    for i in range(96):
        labeled.append(
            _labeled("u1", i, TutorMove.SOCRATIC_PROBE, ComplianceLabel.SOCRATIC)
        )
    for i in range(96, 98):
        labeled.append(_labeled("u1", i, TutorMove.FULL_ANSWER, ComplianceLabel.DIRECT_ANSWER))
    for i in range(98, 100):
        labeled.append(_labeled("u1", i, TutorMove.WORKED_STEP, ComplianceLabel.BORDERLINE))
    r = summarise_user("u1", labeled)
    assert r.meets_roadmap_gate() is True

    # Flip two borderline into direct_answer → 96/100 = 0.96, still passes.
    labeled[-2] = _labeled("u1", 98, TutorMove.FULL_ANSWER, ComplianceLabel.DIRECT_ANSWER)
    labeled[-1] = _labeled("u1", 99, TutorMove.FULL_ANSWER, ComplianceLabel.DIRECT_ANSWER)
    r = summarise_user("u1", labeled)
    assert r.meets_roadmap_gate() is True  # 96/100 = 0.96 ≥ 0.95


def test_fails_gate_with_too_few_turns() -> None:
    labeled = [
        _labeled("u1", i, TutorMove.SOCRATIC_PROBE, ComplianceLabel.SOCRATIC)
        for i in range(50)
    ]
    r = summarise_user("u1", labeled)
    assert r.compliance_rate() == 1.0
    assert r.meets_roadmap_gate() is False


def test_fails_gate_below_compliance_floor() -> None:
    labeled = []
    for i in range(90):
        labeled.append(_labeled("u1", i, TutorMove.SOCRATIC_PROBE, ComplianceLabel.SOCRATIC))
    for i in range(90, 100):
        labeled.append(_labeled("u1", i, TutorMove.FULL_ANSWER, ComplianceLabel.DIRECT_ANSWER))
    r = summarise_user("u1", labeled)
    assert r.compliance_rate() == pytest.approx(0.90)
    assert r.meets_roadmap_gate() is False


def test_argilla_export_jsonl_round_trip(tmp_path: Path) -> None:
    labeled = [
        _labeled("u1", 0, TutorMove.SOCRATIC_PROBE, ComplianceLabel.SOCRATIC),
        _labeled("u1", 1, TutorMove.FULL_ANSWER, ComplianceLabel.DIRECT_ANSWER),
    ]
    records = [labeled_turn_to_argilla(lt) for lt in labeled]
    out = tmp_path / "argilla.jsonl"
    count = write_argilla_jsonl(records, out)
    assert count == 2
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["fields"]["utterance"] == "turn 0 content"
    assert first["fields"]["tutor_move"] == "socratic_probe"
    assert first["metadata"]["auditor_label"] == "socratic"
    assert first["external_id"] == "u1:s1:0"


def test_cohen_kappa_perfect_agreement() -> None:
    labels = [ComplianceLabel.SOCRATIC] * 10 + [ComplianceLabel.DIRECT_ANSWER] * 5
    r = cohen_kappa(labels, labels)
    assert r.kappa == pytest.approx(1.0)
    assert r.po == pytest.approx(1.0)


def test_cohen_kappa_zero_at_chance() -> None:
    # 10 socratic + 10 direct from rater A; rater B samples same marginals
    # but assignment is independent → κ ≈ 0.
    labels_a = [ComplianceLabel.SOCRATIC, ComplianceLabel.DIRECT_ANSWER] * 10
    labels_b = [ComplianceLabel.SOCRATIC] * 10 + [ComplianceLabel.DIRECT_ANSWER] * 10
    r = cohen_kappa(labels_a, labels_b)
    # Observed agreement = 10/20 = 0.5. Expected agreement = 0.5*0.5 + 0.5*0.5 = 0.5. κ = 0.
    assert r.kappa == pytest.approx(0.0)


def test_cohen_kappa_moderate() -> None:
    # Classic kappa case: 20 items, mostly agree.
    labels_a = [ComplianceLabel.SOCRATIC] * 15 + [ComplianceLabel.DIRECT_ANSWER] * 5
    labels_b = [ComplianceLabel.SOCRATIC] * 12 + [ComplianceLabel.DIRECT_ANSWER] * 3 + \
               [ComplianceLabel.SOCRATIC] * 2 + [ComplianceLabel.DIRECT_ANSWER] * 3
    r = cohen_kappa(labels_a, labels_b)
    # Landis-Koch κ fair-to-moderate band (0.21-0.60).
    assert 0.21 < r.kappa < 0.60
    assert r.n_items == 20


def test_cohen_kappa_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        cohen_kappa([ComplianceLabel.SOCRATIC], [])
