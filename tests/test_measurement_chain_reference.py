"""Phase 4a — chain-reference audit unit tests.

Mirrors the Socratic-compliance module's test shape. Verifies the roll-up
math over the 4-way ChainReferenceLabel scheme + Argilla export round-trip
+ κ over the 4-way scheme.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from psychology_base.applications.measurement import (
    ChainReferenceLabel,
    ChainReferenceTurn,
    ReinterpretationAct,
    chain_reference_turn_to_argilla,
    kappa_on_chain_reference,
)
from psychology_base.applications.measurement.chain_reference_audit import (
    summarise_user,
    write_argilla_jsonl,
)


def _turn(
    user_id: str,
    turn_index: int,
    reinterp: ReinterpretationAct,
    cref: ChainReferenceLabel,
    session_id: str = "s1",
) -> ChainReferenceTurn:
    return ChainReferenceTurn(
        user_id=user_id,
        session_id=session_id,
        turn_index=turn_index,
        user_utterance=f"user turn {turn_index}",
        companion_utterance=f"companion turn {turn_index}",
        reinterpretation_act=reinterp,
        chain_reference_label=cref,
        auditor_id="auditor-1",
    )


def test_report_counts_reinterpretation_label_breakdown() -> None:
    turns = [
        _turn("u1", 0, ReinterpretationAct.REINTERPRETATION, ChainReferenceLabel.APPROPRIATE_CITE),
        _turn("u1", 1, ReinterpretationAct.REINTERPRETATION, ChainReferenceLabel.APPROPRIATE_CITE),
        _turn("u1", 2, ReinterpretationAct.REINTERPRETATION, ChainReferenceLabel.MISSED_OPPORTUNITY),
        _turn("u1", 3, ReinterpretationAct.FIRST_MENTION, ChainReferenceLabel.NOT_APPLICABLE),
        _turn("u1", 4, ReinterpretationAct.UNRELATED, ChainReferenceLabel.INAPPROPRIATE_CITE),
    ]
    r = summarise_user("u1", turns)
    assert r.reinterpretation_count == 3
    assert r.appropriate_cite_count == 2
    assert r.missed_opportunity_count == 1
    assert r.inappropriate_cite_count == 1
    assert r.chain_reference_rate() == pytest.approx(2 / 3)


def test_meets_roadmap_gate_at_default_thresholds() -> None:
    # 5 reinterpretations, 4 appropriate → 0.80 exactly.
    turns = [
        _turn("u1", i, ReinterpretationAct.REINTERPRETATION, ChainReferenceLabel.APPROPRIATE_CITE)
        for i in range(4)
    ]
    turns.append(
        _turn("u1", 4, ReinterpretationAct.REINTERPRETATION, ChainReferenceLabel.MISSED_OPPORTUNITY)
    )
    r = summarise_user("u1", turns)
    assert r.chain_reference_rate() == pytest.approx(0.80)
    assert r.meets_roadmap_gate() is True


def test_fails_gate_below_floor() -> None:
    turns = [
        _turn("u1", i, ReinterpretationAct.REINTERPRETATION, ChainReferenceLabel.APPROPRIATE_CITE)
        for i in range(3)
    ]
    turns.extend(
        _turn("u1", i + 3, ReinterpretationAct.REINTERPRETATION, ChainReferenceLabel.MISSED_OPPORTUNITY)
        for i in range(2)
    )
    r = summarise_user("u1", turns)
    assert r.chain_reference_rate() == pytest.approx(0.60)
    assert r.meets_roadmap_gate() is False


def test_fails_gate_too_few_reinterpretations() -> None:
    turns = [
        _turn("u1", i, ReinterpretationAct.REINTERPRETATION, ChainReferenceLabel.APPROPRIATE_CITE)
        for i in range(3)
    ]
    r = summarise_user("u1", turns)
    assert r.chain_reference_rate() == 1.0
    # Only 3 reinterpretations; default min is 5.
    assert r.meets_roadmap_gate() is False


def test_argilla_export_round_trip(tmp_path: Path) -> None:
    turns = [
        _turn("u1", 0, ReinterpretationAct.REINTERPRETATION, ChainReferenceLabel.APPROPRIATE_CITE),
        _turn("u1", 1, ReinterpretationAct.UNRELATED, ChainReferenceLabel.NOT_APPLICABLE),
    ]
    records = [chain_reference_turn_to_argilla(t) for t in turns]
    out = tmp_path / "argilla-cref.jsonl"
    count = write_argilla_jsonl(records, out)
    assert count == 2
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["fields"]["reinterpretation_act"] == "reinterpretation"
    assert first["metadata"]["chain_reference_label"] == "appropriate_cite"
    assert first["external_id"] == "u1:s1:0"


def test_kappa_perfect_agreement() -> None:
    labels = [
        ChainReferenceLabel.APPROPRIATE_CITE,
        ChainReferenceLabel.MISSED_OPPORTUNITY,
        ChainReferenceLabel.NOT_APPLICABLE,
    ] * 3
    r = kappa_on_chain_reference(labels, labels)
    assert r.kappa == pytest.approx(1.0)


def test_kappa_rejects_mismatched_length() -> None:
    with pytest.raises(ValueError):
        kappa_on_chain_reference([ChainReferenceLabel.APPROPRIATE_CITE], [])


def test_zero_reinterpretations_returns_zero_rate() -> None:
    r = summarise_user("u1", [])
    assert r.reinterpretation_count == 0
    assert r.chain_reference_rate() == 0.0
    assert r.meets_roadmap_gate() is False
