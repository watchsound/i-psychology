"""Freeze coherent source cases, select on development, then evaluate once.

Run as python -m benchmarks.scripts.phase9_carryover_run --help.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import sys

from psychology_base.eval.benchmarks import recall_carryover as study
from psychology_base.eval.benchmarks import chinese_retrieval as shared
from psychology_base.i18n import tokenizer_identity


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "benchmarks/fixtures/recall_carryover"


def fingerprints(*, extra_paths=()):
    paths = [Path(study.__file__), Path(shared.__file__), Path(study.memory.__file__),
             Path(study.witness.__file__), ROOT / "src/psychology_base/companion/system.py",
             ROOT / "src/psychology_base/i18n/__init__.py", Path(__file__), *extra_paths]
    return {str(path.relative_to(ROOT)): shared.digest(path) for path in paths}


def load_inputs(protocol_path, catalog_path, controls_path, data_root, split):
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["candidates"] != list(study.POLICIES) or protocol["fixed"] != {
            "top_k": 6, "tokenizer": "jieba", "witness_active": True, "byte_cap": None,
            "seed": 0, "states": list(study.STATES), "paired_absent": True}:
        raise ValueError("Unsupported carryover protocol configuration")
    selection = {"target_window_turns": 8, "distractor_windows": 2, "distractor_window_turns": 8}
    mandatory = ("cold_recall_must_not_decrease", "followup_recall_must_not_decrease",
                 "switch_mrr_must_not_decrease", "absent_nonempty_rate_must_not_increase",
                 "software_controls_must_pass")
    if (any(protocol["selection"][key] != value for key, value in selection.items())
            or any(protocol["gates"][key] is not True for key in mandatory)):
        raise ValueError("Unsupported carryover selection or gate configuration")
    if split not in ("development", "evaluation"):
        raise ValueError("Unsupported study split")
    if shared.digest(catalog_path) != protocol["catalog_sha256"] or shared.digest(controls_path) != protocol["controls_sha256"]:
        raise ValueError("Catalog or software controls differ from freeze")
    sources = study.load_sources(data_root, protocol, split)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    cases = [case for case in catalog["cases"] if case["split"] == split]
    if study.build_catalog(sources, protocol)["cases"] != cases:
        raise ValueError("Catalog/source mismatch")
    controls = json.loads(controls_path.read_text(encoding="utf-8"))
    return protocol, sources, cases, controls


def input_fingerprints(args):
    paths = {"protocol": args.protocol, "catalog": args.catalog, "controls": args.controls}
    paths.update({name: getattr(args, name) for name in ("development_report", "reference_report")
                  if getattr(args, name, None) is not None})
    return {f"{name}_sha256": shared.digest(path) for name, path in paths.items()}


def validate_evidence(report, cases, controls, protocol):
    """Recompute decisions from a complete report; hashes alone are insufficient.

    This checks internal consistency, not authenticity of an arbitrary report.
    Production replay separately checks its source IDs against the real engine.
    """
    phase, selected = report["phase"], report["selected"]
    if phase not in ("development", "evaluation") or selected not in study.POLICIES[2:]:
        raise ValueError("Report has no adoptable development selection")
    policies = study.POLICIES if phase == "development" else ("baseline", selected)
    if report["schema"] != protocol["schema"] or report["matrix"] != study.trial_matrix(report["trials"], cases, policies):
        raise ValueError("Report schema or trial matrix differs")
    for row in report["trials"]:
        ids, gold = row["selected_ids"], row["gold_ids"]
        if (len(ids) > 6 or len(ids) != len(set(ids)) or bool(gold) == row["absent"]
                or any(row[key] != value for key, value in shared.score_context(ids, set(gold)).items())
                or any(not math.isfinite(row[key]) or row[key] < 0 for key in ("latency_ms", "peak_python_bytes"))):
            raise ValueError("Report contains inconsistent trial scores or resources")
    expected_controls = {c["id"] for c in controls["cases"]}
    if set(report["software_controls"]) != set(policies):
        raise ValueError("Report software-control policies differ")
    for results in report["software_controls"].values():
        if (len(results) != len(expected_controls) or {c["case"] for c in results} != expected_controls
                or any(type(c["passed"]) is not bool for c in results)):
            raise ValueError("Incomplete or duplicated software controls")
    summary = shared.summarize_trials(report["trials"], policies, study.STATES)
    guards, winner = study.selection(summary, report["software_controls"], protocol["gates"])
    if (report["summary"] != summary or report["guards"] != guards
            or phase == "development" and selected != winner):
        raise ValueError("Report summary, guards or development selection differs from its evidence")


def run(args):
    before = fingerprints()
    inputs = input_fingerprints(args)
    phase = args.command
    protocol, sources, cases, controls = load_inputs(args.protocol, args.catalog, args.controls, args.data_root, phase)
    identity = tokenizer_identity("jieba")
    development = None
    selected = None
    if phase == "evaluation":
        if args.development_report is None:
            raise ValueError("Evaluation requires the frozen development report")
        development = json.loads(args.development_report.read_text(encoding="utf-8"))
        if (development["phase"] != "development" or development["code_sha256"] != before
                or any(development[key] != inputs[key] for key in ("protocol_sha256", "catalog_sha256", "controls_sha256"))
                or development["environment"]["tokenizer"] != identity):
            raise ValueError("Development source/configuration differs; do not retune on evaluation")
        development_cases = [c for c in json.loads(args.catalog.read_text(encoding="utf-8"))["cases"]
                             if c["split"] == "development"]
        validate_evidence(development, development_cases, controls, protocol)
        selected = development["selected"]
        if selected is None:
            raise ValueError("No development candidate passed; evaluation remains unscored")
    policies = study.POLICIES if phase == "development" else ("baseline", selected)
    control_reports = {policy: study.control_results(controls, policy) for policy in policies}
    rows = []
    for case in cases:
        source = sources[case["source"]]
        for absent in (False, True):
            base, ids, _ = shared.prepare_trial(source, case, "precise", absent=absent, source_sessions=True)
            for policy in policies:
                for state in study.STATES:
                    rows.append(study.evaluate(source, case, base, ids, policy, state, absent))
        print(f"Completed {case['id']}", flush=True)
    matrix = study.trial_matrix(rows, cases, policies)
    summary = shared.summarize_trials(rows, policies, study.STATES)
    guards, winner = study.selection(summary, control_reports, protocol["gates"])
    if phase == "development":
        selected = winner
    domain_guards = {}
    if phase == "evaluation":
        for source in sources:
            domain = shared.summarize_trials([r for r in rows if r["source"] == source], policies, study.STATES)
            domain_guards[source] = {s: domain[selected][s]["recall_at_6"] >= domain["baseline"][s]["recall_at_6"]
                                     for s in ("cold", "followup")}
    if fingerprints() != before:
        raise ValueError("Executed code changed during scoring")
    if input_fingerprints(args) != inputs:
        raise ValueError("Study inputs changed during scoring")
    return {"schema": protocol["schema"], "phase": phase, "selected": selected,
            **inputs, "code_sha256": before,
            "development_report_sha256": inputs.get("development_report_sha256"),
            "environment": {"python": sys.version, "platform": platform.platform(), "tokenizer": identity},
            "matrix": matrix,
            "software_controls": control_reports, "summary": summary, "guards": guards,
            "domain_guards": domain_guards,
            "integration_eligible": phase == "evaluation" and all(guards[selected].values())
                                    and all(all(g.values()) for g in domain_guards.values()),
            "trials": rows, "limitations": protocol["limitations"]}


def verify_production(args, *, audit_change=None):
    """Replay the ordinary engine; strict unless an explicit audit handles changes.

    The public CLI remains strict. Correctness-repair research may supply an
    input-exposure validator, which must reject unexplained changed records.
    """
    if args.reference_report is None:
        raise ValueError("Production verification requires --reference-report")
    inputs = input_fingerprints(args)
    reference = json.loads(args.reference_report.read_text(encoding="utf-8"))
    if reference.get("selected") != "novel_cue" or reference.get("phase") not in ("development", "evaluation"):
        raise ValueError("Reference must be a frozen novel-cue development/evaluation report")
    protocol, sources, cases, controls = load_inputs(args.protocol, args.catalog, args.controls, args.data_root, reference["phase"])
    if any(reference[key] != inputs[key] for key in ("protocol_sha256", "catalog_sha256", "controls_sha256")):
        raise ValueError("Reference configuration differs")
    validate_evidence(reference, cases, controls, protocol)
    tokenizer_identity("jieba")
    before = fingerprints()
    control_report = study.control_results(controls, "production")
    if not all(row["passed"] for row in control_report):
        raise ValueError("Production failed a frozen software control")
    expected = {(r["case"], r["state"], r["absent"]): r for r in reference["trials"]
                if r["candidate"] == "novel_cue"}
    rows = []
    changes = []
    omitted = {"candidate", "latency_ms", "peak_python_bytes"}
    for case in cases:
        source = sources[case["source"]]
        for absent in (False, True):
            base, ids, _ = shared.prepare_trial(source, case, "precise", absent=absent, source_sessions=True)
            for state in study.STATES:
                actual = study.evaluate(source, case, base, ids, "production", state, absent)
                previous = expected[case["id"], state, absent]
                if ({k: v for k, v in actual.items() if k not in omitted}
                        != {k: v for k, v in previous.items() if k not in omitted}):
                    if audit_change is None:
                        raise ValueError(f"Production differs from selected candidate: {case['id']}/{state}/{absent}")
                    changes.append(audit_change(actual, previous))
                rows.append(actual)
    if len(rows) != len(expected) or fingerprints() != before:
        raise ValueError("Incomplete or changed-code production verification")
    if input_fingerprints(args) != inputs:
        raise ValueError("Study inputs changed during scoring")
    return {"schema": "carryover-production-audit-v1" if audit_change else "carryover-production-verification-v1",
            "phase": reference["phase"],
            **inputs,
            "code_sha256": before, "policy": study.memory.RECALL_SELECTION_POLICY,
            "matched_trials": len(rows) - len(changes), "software_controls": control_report, "trials": rows,
            **({"compared_trials": len(rows), "changes": changes} if audit_change else {})}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "development", "evaluation", "verify-production", "diagnose-graph",
                                           "association-development", "association-evaluation", "context-diagnose", "focused-diagnose",
                                           "scaled-diagnose", "scaled-development", "scaled-evaluation",
                                           "application-prepare", "application-resources", "application-preflight", "application-smoke", "application-calibration", "application-development",
                                           "application-evaluation", "application-validate", "navigation-diagnostic"))
    parser.add_argument("--protocol", type=Path, default=FIXTURES / "protocol.json")
    parser.add_argument("--catalog", type=Path, default=FIXTURES / "catalog.json")
    parser.add_argument("--controls", type=Path, default=FIXTURES / "controls.json")
    parser.add_argument("--data-root", type=Path, default=ROOT / ".data/kdconv/data")
    parser.add_argument("--development-report", type=Path)
    parser.add_argument("--reference-report", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--application-annotations", type=Path, default=ROOT / ".data/application_context/annotations.json")
    parser.add_argument("--application-calibration", type=Path)
    parser.add_argument("--application-review-manifest-sha256",
                        help="Require the frozen criterion-review manifest when validating qualification")
    parser.add_argument("--application-preflight", type=Path)
    parser.add_argument("--application-smoke", type=Path)
    parser.add_argument("--application-resume", type=Path)
    parser.add_argument("--application-profile", type=Path,
                        help="Explicit approved local or $1 hybrid development screening overlay")
    parser.add_argument("--application-resources", type=Path)
    parser.add_argument("--llm-config", type=Path)
    parser.add_argument("--navigation-plan", type=Path,
                        help="Explicit frozen local-only navigation diagnostic profile")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
            shared.write_new(args.catalog, study.build_catalog(study.load_sources(args.data_root, protocol), protocol))
            print(f"catalog_sha256={shared.digest(args.catalog)} controls_sha256={shared.digest(args.controls)}")
        else:
            if args.output is None or args.output.exists():
                raise ValueError("Use a new --output path; previous evidence is never overwritten")
            if args.command == "navigation-diagnostic":
                from psychology_base.eval.benchmarks.application_context import navigation_diagnostic
                report = navigation_diagnostic(args)
            elif args.command.startswith("application-"):
                from psychology_base.eval.benchmarks.application_context import run as application_run
                report = application_run(args)
            elif args.command in ("scaled-development", "scaled-evaluation"):
                from psychology_base.eval.benchmarks.fresh_context import run as fresh_run
                report = fresh_run(args)
            elif args.command == "scaled-diagnose":
                from psychology_base.eval.benchmarks.scaled_context import run as scaled_run
                report = scaled_run(args)
            elif args.command == "focused-diagnose":
                from psychology_base.eval.benchmarks.focused_context import run as focused_run
                report = focused_run(args)
            elif args.command == "context-diagnose":
                from psychology_base.eval.benchmarks.cue_context import run as context_run
                report = context_run(args)
            elif args.command.startswith("association-"):
                from psychology_base.eval.benchmarks.turn_association import run as association_run
                report = association_run(args)
            elif args.command == "diagnose-graph":
                from psychology_base.eval.benchmarks.graph_priors import diagnose
                report = diagnose(args)
            else:
                report = verify_production(args) if args.command == "verify-production" else run(args)
            shared.write_new(args.output, report)
            if args.command == "navigation-diagnostic":
                verdict = {key: report[key] for key in ("phase", "local_embedding_requests", "hosted_provider_requests",
                           "observed_development_screens", "integration_eligible")}
            elif args.command.startswith("application-"):
                verdict = {key: report[key] for key in ("phase", "families", "cases_scored", "passed", "validated", "decision", "budget") if key in report}
            elif args.command in ("scaled-development", "scaled-evaluation"):
                verdict = {"quality_passed": report["quality_passed"], "resources_passed": report["resources_passed"],
                           "advance": report["advance"], "qualified_for_policy_review": report["qualified_for_policy_review"]}
            elif args.command in ("context-diagnose", "focused-diagnose", "scaled-diagnose"):
                verdict = {"advance_to_fresh_freeze": report["advance_to_fresh_freeze"],
                           "gates": report["gates"], "integration_eligible": report["integration_eligible"]}
            elif args.command.startswith("association-"):
                verdict = {"advance": report["advance"], "gates": report["gates"],
                           "integration_eligible": report["integration_eligible"]}
            elif args.command == "diagnose-graph":
                verdict = {"candidates_passing_observed_gate": report["candidates_passing_observed_gate"],
                           "integration_eligible": report["integration_eligible"]}
            elif "matched_trials" in report:
                verdict = {"matched_trials": report["matched_trials"]}
            else:
                verdict = {"selected": report["selected"], "guards": report["guards"],
                           "integration_eligible": report["integration_eligible"]}
            print(json.dumps(verdict, indent=2))
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"Carryover evaluation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
