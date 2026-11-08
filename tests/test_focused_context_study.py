"""Frozen public-source comparison and negative-path report validation."""
from __future__ import annotations

import json

import pytest

from benchmarks.scripts import phase9_carryover_run as cli
from psychology_base.eval.benchmarks import chinese_retrieval as shared
from psychology_base.eval.benchmarks import focused_context as study

DIRECTORY = cli.ROOT / "docs/experiments/phase-9-focused-context"
PROTOCOL = cli.ROOT / "benchmarks/fixtures/focused_context/protocol.json"


def validate_archived(report, cases, controls):
    policy = json.loads((DIRECTORY / 'validation.json').read_text(encoding='utf-8'))['selection_policy']
    return study.validate_report(report, cases, controls, expected_production_policy=policy)


def frozen():
    from tests.test_cue_context_study import frozen as earlier
    _, cases, controls, _ = earlier()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    report = json.loads((DIRECTORY / "observed-regression-reviewed.json").read_text(encoding="utf-8"))
    return protocol, cases, controls, report


def test_complete_four_arm_report_recomputes_frozen_decision():
    protocol, cases, controls, report = frozen()
    study.validate_protocol(protocol)
    validate_archived(report, cases, controls)
    assert report["matrix"] == {"expected": 1296, "actual": 1296, "unique": 1296}
    validation = json.loads((DIRECTORY / "validation.json").read_text(encoding="utf-8"))
    assert report["code_sha256"] == validation["code_sha256"]
    report_path = "docs/experiments/phase-9-focused-context/observed-regression-reviewed.json"
    assert shared.digest(cli.ROOT / report_path) == validation["artifacts_sha256"][report_path]
    assert report["protocol_sha256"] == shared.digest(PROTOCOL)
    assert report["old_heldout_unscored"] == 60 and report["fresh_evaluation_scored"] == 0
    assert report["production_context_selection_enabled"] is False


def test_all_corrected_baseline_records_match_with_explicit_focus_input():
    _, _, _, report = frozen()
    corrected = json.loads((cli.ROOT / "docs/experiments/phase-9-script-dispatch/corrected-observed.json").read_text(encoding="utf-8"))
    previous = {(r["case"], r["candidate"], r["state"], r["absent"]): r
                for r in corrected["trials"] if r["candidate"] in study.ARMS[:2]}
    actual = [r for r in report["trials"] if r["candidate"] in study.ARMS[:2]]
    ignored = {"latency_ms", "peak_python_bytes", "focus_ref", "witness_ids", "witness_activation"}
    assert len(actual) == len(previous) == 648
    for row in actual:
        old = previous[row["case"], row["candidate"], row["state"], row["absent"]]
        assert {k: v for k, v in row.items() if k not in ignored} == {k: v for k, v in old.items() if k not in ignored}


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "arm", "focus", "witness_ids", "witness_state",
    "witness_nan", "foreign_ids", "gate", "advance", "heldout", "inventory", "trace", "control"])
def test_forged_or_partial_focused_reports_are_rejected(corruption):
    _, cases, controls, report = frozen()
    row = next(r for r in report["trials"] if r["candidate"] == "focus_companion")
    if corruption == "missing":
        report["trials"].pop()
    elif corruption == "duplicate":
        report["trials"][-1] = report["trials"][0]
    elif corruption == "arm":
        row["candidate"] = "unfrozen"
    elif corruption == "focus":
        row["focus_ref"] = [-1, -1]
    elif corruption == "witness_ids":
        row["witness_ids"] = ["foreign"]
    elif corruption == "witness_state":
        row["witness_activation"] = {}
    elif corruption == "witness_nan":
        row["witness_activation"][next(iter(row["witness_activation"]))] = float("nan")
    elif corruption == "foreign_ids":
        row["selected_ids"][0] = "foreign"
    elif corruption == "gate":
        report["gates"]["witness_baseline_view"] = not report["gates"]["witness_baseline_view"]
    elif corruption == "advance":
        report["advance_to_fresh_freeze"] = not report["advance_to_fresh_freeze"]
    elif corruption == "heldout":
        report["fresh_evaluation_scored"] = 60
    elif corruption == "inventory":
        report["source_inventories"].pop()
    elif corruption == "trace":
        report["traces"].pop()
    else:
        report["software_controls"]["focus_companion"].pop()
    with pytest.raises(ValueError):
        validate_archived(report, cases, controls)


def test_historical_focused_policy_does_not_qualify_current_runtime():
    _, cases, controls, report = frozen()
    with pytest.raises(ValueError):
        study.validate_report(report, cases, controls)


@pytest.mark.parametrize("field,value", [("primary", "focus_shared"), ("support", "tuned"),
    ("top_k", 7), ("witness_primary_policy", "focused")])
def test_protocol_rejects_unregistered_candidate_changes(field, value):
    protocol, _, _, _ = frozen()
    protocol["fixed"][field] = value
    with pytest.raises(ValueError):
        study.validate_protocol(protocol)


def test_last_supplied_question_contract_rejects_a_changed_pointer_before_recall():
    from psychology_base.eval.benchmarks import recall_carryover as carry

    invalid = {"query": [0, 3], "cue_refs": [[0, 0], [0, 1]]}
    with pytest.raises(ValueError, match="last supplied cue"):
        carry.prepare_recall([], invalid, None, "production", "cold", focus_current=True)


def test_cli_preserves_existing_evidence_before_running(tmp_path):
    output = tmp_path / "existing.json"
    output.write_text("original evidence", encoding="utf-8")
    assert cli.main(["focused-diagnose", "--protocol", str(PROTOCOL), "--output", str(output)]) == 1
    assert output.read_text(encoding="utf-8") == "original evidence"
