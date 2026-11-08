"""Frozen fresh phases using the existing context scorer and resource owner."""
from __future__ import annotations

from functools import partial
import json
from pathlib import Path
import platform
import sys

from psychology_base.eval.benchmarks import chinese_retrieval as shared
from psychology_base.eval.benchmarks import cue_context as observed
from psychology_base.eval.benchmarks import focused_context as focused
from psychology_base.eval.benchmarks import recall_carryover as carry
from psychology_base.eval.benchmarks import scaled_context as scaled
from psychology_base.eval.benchmarks import turn_association as association
from psychology_base.i18n import tokenizer_identity

ARMS = ("ordinary_v4", "metadata_only", "scaled_focus")
SCHEMA = "scaled-context-fresh-report-v1"
FIXTURES = observed.root() / "benchmarks/fixtures/scaled_context"
EXECUTION_INPUTS = {"docs/phases/phase-9-fresh-context-execution.md", "tests/test_fresh_context_execution.py",
    "benchmarks/fixtures/scaled_context/fresh-protocol.json", "benchmarks/fixtures/scaled_context/fresh-catalog.json",
    "benchmarks/fixtures/scaled_context/fresh-audit.json", "docs/experiments/phase-9-scaled-context/validation.json"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_protocol(protocol):
    previous = read(observed.root() / "docs/experiments/phase-9-scaled-context/validation.json")
    for name in ("fresh-protocol.json", "fresh-catalog.json", "fresh-audit.json"):
        path = FIXTURES / name
        key = path.relative_to(observed.root()).as_posix()
        if shared.digest(path) != previous["artifacts_sha256"][key]:
            raise ValueError("Fresh inputs differ from the prior freeze")
    if protocol != read(FIXTURES / "fresh-protocol.json"):
        raise ValueError("Fresh protocol configuration differs from freeze")
    if not previous["observed_gates_passed"] or protocol["arms"] != list(ARMS):
        raise ValueError("Observed candidate has not qualified")
    if any(shared.digest(observed.root() / path) != sha for path, sha in protocol["inputs_sha256"].items()):
        raise ValueError("Fresh protocol prerequisites changed")


def validate_execution():
    path = FIXTURES / "fresh-execution.json"
    execution = read(path)
    if (execution["schema"] != "fresh-context-execution-v1" or execution["status"] != "frozen-before-fresh-scoring"
            or set(execution["inputs"]) != EXECUTION_INPUTS
            or any(shared.digest(observed.root() / name) != sha for name, sha in execution["inputs"].items())):
        raise ValueError("Fresh execution contracts differ from freeze")
    return shared.digest(path)


def fingerprints():
    return observed.fingerprints(extra_paths=(Path(focused.__file__), Path(scaled.__file__), Path(__file__)))


def resource_config(protocol):
    return {name: protocol["resources"][name] for name in ("sizes", "warmup", "repeats")}


def source_fingerprints(protocol):
    pins = {pin["path"]: pin["sha256"] for pin in protocol["sources"].values()}
    resource = protocol["resources"]["source"]
    return pins | {resource["path"]: resource["sha256"]}


def resource_gates(stress, protocol):
    checks = association.stress_gates(stress["measurements"], arms=ARMS, config=resource_config(protocol),
        max_latency_ratio=protocol["gates"]["max_p95_latency_ratio"],
        max_memory_ratio=protocol["gates"]["max_peak_python_memory_ratio"])
    if stress["gates"] != checks or type(stress["passed"]) is not bool or stress["passed"] != all(checks.values()):
        raise ValueError("Fresh resource decision differs from actual measurements")
    return checks


def quality_decisions(rows, *, arms=ARMS, gates):
    summary, comparisons, checks = observed.decisions(rows, arms=arms)
    for baseline in arms[:2]:
        checks[f"minimum_cold_gain_over_{baseline}"] = (
            summary[arms[-1]]["cold"]["recall_at_6"] - summary[baseline]["cold"]["recall_at_6"]
            >= gates["minimum_cold_recall_gain"] - 1e-12)
        for domain in (None, *sorted({r["source"].split("/")[0] for r in rows})):
            subset = rows if domain is None else [r for r in rows if r["source"].split("/")[0] == domain]
            partial_summary = shared.summarize_trials(subset, arms, carry.STATES)
            for state in carry.STATES:
                checks[f'{domain or "pooled"}_{state}_recall_over_{baseline}'] = (
                    partial_summary[arms[-1]][state]["recall_at_6"] >= partial_summary[baseline][state]["recall_at_6"] - 1e-12)
    return summary, comparisons, checks


def validate_report(report, catalog, protocol, *, development=None, expected_development_sha256=None,
                    expected_production_policy=None):
    """An explicit trusted producer pin permits historical integrity checks only."""
    policy = observed.memory.RECALL_SELECTION_POLICY if expected_production_policy is None else expected_production_policy
    if not isinstance(policy, str) or not policy:
        raise ValueError("Expected production policy must be a nonempty string")
    phase = report.get("phase")
    if report.get("schema") != SCHEMA or phase not in ("development", "evaluation"):
        raise ValueError("Not a frozen fresh development/evaluation report")
    if phase == "evaluation" and (
            not isinstance(expected_development_sha256, str) or len(expected_development_sha256) != 64
            or any(char not in "0123456789abcdef" for char in expected_development_sha256)
            or report.get("development_report_sha256") != expected_development_sha256):
        raise ValueError("Fresh evaluation development report identity differs from the actual input")
    if report["source_sha256"] != source_fingerprints(protocol):
        raise ValueError("Fresh report source fingerprints differ from freeze")
    if (report["protocol_sha256"] != shared.digest(FIXTURES / "fresh-protocol.json")
            or report["catalog_sha256"] != protocol["catalog_sha256"]
            or report["controls_sha256"] != protocol["controls_sha256"]
            or report["execution_sha256"] != shared.digest(FIXTURES / "fresh-execution.json")):
        raise ValueError("Fresh report configuration fingerprints differ from freeze")
    cases = [case for case in catalog["cases"] if case["split"] == phase]
    controls = read(observed.root() / "benchmarks/fixtures/recall_carryover/controls.json")
    matrix, summary, comparisons, checks = observed.validate_evidence(report, cases, controls, arms=ARMS,
        extra_checks=partial(focused.extra_checks, primary=ARMS[-1]),
        decision_fn=partial(quality_decisions, gates=protocol["gates"]))
    focused.validate_inputs(report, cases)
    stress_checks = resource_gates(report["stress"], protocol)
    quality_passed, resources_passed = all(checks.values()), all(stress_checks.values())
    eligible = quality_passed and resources_passed
    if (report["matrix"] != matrix or report["summary"] != summary or report["comparisons"] != comparisons
            or report["gates"] != checks or report["quality_passed"] is not quality_passed
            or report["advance"] is not eligible or report["resources_passed"] is not resources_passed
            or report["qualified_for_policy_review"] is not (phase == "evaluation" and eligible)
            or report["fresh_development_scored"] != 30
            or report["fresh_evaluation_scored"] != (60 if phase == "evaluation" else 0)
            or report["old_heldout_unscored"] != 60 or report["integration_eligible"] is not False
            or report["production_context_selection_enabled"] is not False
            or report["production_policy"] != policy):
        raise ValueError("Fresh report decision differs from its evidence")
    if phase == "development":
        if report["resource_origin"] != "measured-development" or report["development_report_sha256"] is not None:
            raise ValueError("Fresh development resources were not measured")
    else:
        if development is None:
            raise ValueError("Evaluation validation requires actual development evidence")
        validate_report(development, catalog, protocol, expected_production_policy=expected_production_policy)
        if (not development["advance"] or report["resource_origin"] != "validated-development"
                or report["stress"] != development["stress"]
                or report["code_sha256"] != development["code_sha256"]
                or report["environment"] != development["environment"]
                or any(report[key] != development[key] for key in (
                    "protocol_sha256", "catalog_sha256", "controls_sha256", "execution_sha256", "source_sha256"))):
            raise ValueError("Evaluation resources or runtime differ from qualified development")


def run(args):
    from benchmarks.scripts import phase9_carryover_run as cli

    protocol = read(args.protocol)
    validate_protocol(protocol)
    if args.command not in ("scaled-development", "scaled-evaluation") or args.reference_report is not None:
        raise ValueError("Unsupported fresh phase or reference-report option")
    phase = "development" if args.command == "scaled-development" else "evaluation"
    if phase == "evaluation" and args.development_report is None:
        raise ValueError("Fresh evaluation requires a qualified development report")
    if phase == "development" and args.development_report is not None:
        raise ValueError("Development cannot consume a development report")
    inputs = cli.input_fingerprints(args)
    if (inputs["catalog_sha256"] != protocol["catalog_sha256"]
            or inputs["controls_sha256"] != protocol["controls_sha256"]):
        raise ValueError("Fresh catalog or controls differ from freeze")
    catalog = read(args.catalog)
    before = fingerprints()
    development = None
    if phase == "evaluation":
        development = read(args.development_report)
        validate_report(development, catalog, protocol)
        if not development["advance"] or development["phase"] != "development":
            raise ValueError("Development did not qualify; evaluation remains unscored")
    execution_sha = validate_execution()
    identity = {"python": sys.version, "platform": platform.platform(), "tokenizer": tokenizer_identity("jieba")}
    if development is not None and (development["code_sha256"] != before or development["environment"] != identity
            or development["execution_sha256"] != execution_sha
            or any(development[key] != inputs[key] for key in ("protocol_sha256", "catalog_sha256", "controls_sha256"))):
        raise ValueError("Development did not qualify with identical code, inputs and tokenizer")
    # No source load or held-out recall occurs above the qualification lock.
    sources = association.load_sources(args.data_root, protocol)
    allowed = {key: set(groups) - set(protocol["reserved_background_groups"][key])
               for key, groups in protocol["allowed_groups"].items()}
    backgrounds = {key: set(groups) for key, groups in protocol["reserved_background_groups"].items()}
    if carry.build_catalog(sources, protocol, allowed_dialogues=allowed, background_dialogues=backgrounds) != catalog:
        raise ValueError("Fresh catalog differs from pinned source windows")
    source_pins = source_fingerprints(protocol)
    pin = protocol["resources"]["source"]
    if any(shared.digest(args.data_root / path) != sha for path, sha in source_pins.items()):
        raise ValueError("Fresh source files differ from freeze")
    cases = [case for case in catalog["cases"] if case["split"] == phase]
    inventories = []
    rows = association.score_cases(sources, cases, protocol, arms=ARMS, system_factory=scaled.system_for_arm,
                                   inventories=inventories, focus_current=True)
    controls = read(args.controls)
    control_reports = {arm: carry.control_results(controls, "production", system_factory=lambda arm=arm:
                       scaled.system_for_arm(arm)) for arm in ARMS}
    summary, comparisons, checks = quality_decisions(rows, gates=protocol["gates"])
    checks.update(focused.extra_checks(rows, primary=ARMS[-1]))
    checks["software_controls"] = all(c["passed"] for group in control_reports.values() for c in group)
    traces = observed.changed_case_traces([(sources, cases, protocol)], comparisons, arms=ARMS,
                                          system_factory=scaled.system_for_arm, focus_current=True)
    print(json.dumps({"phase": phase, "quality_passed": all(checks.values()),
        "failed_quality_guards": [key for key, passed in checks.items() if not passed],
        "recall_at_6": {arm: {state: summary[arm][state]["recall_at_6"] for state in carry.STATES}
                        for arm in ARMS}}, indent=2), flush=True)
    if development is None:
        resource_source = shared.load_source(args.data_root / pin["path"], {"source": pin})
        stress = association.stress(resource_source, pin, arms=ARMS, system_factory=scaled.system_for_arm,
            config=resource_config(protocol), focus_current=True,
            max_latency_ratio=protocol["gates"]["max_p95_latency_ratio"],
            max_memory_ratio=protocol["gates"]["max_peak_python_memory_ratio"])
    else:
        stress = development["stress"]
    report = {"schema": SCHEMA, "phase": phase, **inputs, "code_sha256": before,
        "execution_sha256": execution_sha, "environment": identity, "source_sha256": source_pins,
        "matrix": carry.trial_matrix(rows, cases, ARMS), "trials": rows, "source_inventories": inventories,
        "summary": summary, "comparisons": comparisons, "gates": checks, "software_controls": control_reports,
        "traces": traces, "stress": stress, "quality_passed": all(checks.values()),
        "resources_passed": stress["passed"], "advance": all(checks.values()) and stress["passed"],
        "qualified_for_policy_review": phase == "evaluation" and all(checks.values()) and stress["passed"],
        "resource_origin": "validated-development" if development is not None else "measured-development",
        "development_report_sha256": inputs.get("development_report_sha256"),
        "fresh_development_scored": 30, "fresh_evaluation_scored": 60 if phase == "evaluation" else 0,
        "old_heldout_unscored": 60, "integration_eligible": False,
        "production_context_selection_enabled": False, "production_policy": observed.memory.RECALL_SELECTION_POLICY,
        "limitations": ["Repeated stored source questions and mechanical neighboring-answer labels.",
            "Absent nonempty-rate nonincrease does not establish abstention or generated-answer quality.",
            "Resource test is bounded at 512 fragments; no month-horizon, clinical or SOTA qualification."]}
    validate_report(report, catalog, protocol, development=development,
                    expected_development_sha256=inputs.get("development_report_sha256"))
    validate_protocol(protocol)
    if (before != fingerprints() or execution_sha != validate_execution() or inputs != cli.input_fingerprints(args)
            or any(shared.digest(args.data_root / path) != sha for path, sha in source_pins.items())):
        raise ValueError("Fresh runtime or source/configuration changed during execution")
    return report
