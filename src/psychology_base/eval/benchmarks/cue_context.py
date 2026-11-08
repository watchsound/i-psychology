"""Regression-first experiment in the existing six-slot reconstruction path.

No new source cases are scored here. A pass permits a separately frozen fresh
study, never production activation. A failure terminates this fixed candidate.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import platform
import sys

from psychology_base.companion import reconstruction as memory
from psychology_base.eval.adapters import make_fragment
from psychology_base.eval.benchmarks import chinese_retrieval as shared
from psychology_base.eval.benchmarks import graph_priors as graph
from psychology_base.eval.benchmarks import recall_carryover as carry
from psychology_base.eval.benchmarks import turn_association as association
from psychology_base.i18n import tokenizer_identity


SCHEMA = "cue-context-regression-v1"
ARMS = ("ordinary_v4", "metadata_only", "cue_context")
FIXED = {"top_k": 6, "tokenizer": "jieba", "byte_cap": None, "seed": 0,
         "witness_active": True, "states": list(carry.STATES), "paired_absent": True,
         "support": "half-maximum-normalized-direct-cue-seed", "normalize_selection_mass": True,
         "known_cases": 24, "observed_train_cases": 30, "fresh_cases_scored": 0}


def reconstruction():
    return shared._bind(memory.reconstruct, _selection_scores=memory.cue_context_scores)


def system_for_arm(arm, **overrides):
    if arm not in ARMS:
        raise ValueError("Unknown cue-context arm")
    if overrides.get("fragment_context_selection", "identity") != "identity":
        raise ValueError("Historical cue arms require identity runtime selection")
    system = graph.system_for_policy("production", **overrides)
    return carry.bind_reconstruction(system, reconstruction()) if arm == "cue_context" else system


def fingerprints(*, extra_paths=()):
    return association.fingerprints() | {
        str(path.relative_to(root())): shared.digest(path) for path in (Path(__file__), *extra_paths)}


def root():
    from benchmarks.scripts import phase9_carryover_run as cli
    return cli.ROOT


def validate_protocol(protocol):
    if (protocol["schema"] != SCHEMA or protocol["status"] != "frozen-before-source-scoring"
            or protocol["arms"] != list(ARMS) or protocol["fixed"] != FIXED
            or protocol["gates"] != {"no_lost_answer_hits": True, "metadata_selection_parity": True,
                "pooled_and_domain_mrr_nonloss": True, "absent_rate_nonincrease": True,
                "software_controls": True}):
        raise ValueError("Unsupported cue-context protocol")
    required = {"benchmarks/fixtures/recall_carryover/protocol.json",
                "benchmarks/fixtures/recall_carryover/catalog.json",
                "benchmarks/fixtures/recall_carryover/controls.json",
                "benchmarks/fixtures/turn_association/protocol-v2.json",
                "benchmarks/fixtures/turn_association/catalog.json",
                "docs/phases/phase-9-cue-context-plan.md"}
    if set(protocol["inputs"]) != required or any(
            shared.digest(root() / path) != digest for path, digest in protocol["inputs"].items()):
        raise ValueError("Cue-context inputs differ from freeze")


def regression_inputs(data_root):
    """Load exactly the two previously observed sets with existing validation."""
    from benchmarks.scripts import phase9_carryover_run as cli
    old = cli.FIXTURES
    protocol, sources, cases, controls = cli.load_inputs(
        old / "protocol.json", old / "catalog.json", old / "controls.json", data_root, "evaluation")
    previous = root() / "benchmarks/fixtures/turn_association"
    train_protocol = json.loads((previous / "protocol-v2.json").read_text(encoding="utf-8"))
    association.validate_protocol(train_protocol)
    train_sources = association.load_sources(data_root, train_protocol)
    catalog = json.loads((previous / "catalog.json").read_text(encoding="utf-8"))
    if association.catalog_for(train_sources, train_protocol) != catalog:
        raise ValueError("Observed train catalog differs from pinned sources")
    train_cases = [c for c in catalog["cases"] if c["split"] == "development"]
    adapted = protocol | {"upstream": protocol["selection"]["upstream"],
        "revision": train_protocol["revision"],
        "sources": {k: v | {"path": f"{k}.json"} for k, v in protocol["sources"].items()}}
    pins = {f"{key}.json": pin["sha256"] for key, pin in protocol["sources"].items()
            if pin["split"] == "evaluation"}
    pins.update({pin["path"]: pin["sha256"] for pin in train_protocol["sources"].values()})
    if len(cases) != 24 or len(train_cases) != 30:
        raise ValueError("Observed regression sample differs")
    return [(sources, cases, adapted), (train_sources, train_cases, train_protocol)], controls, pins


def decisions(rows, *, arms=ARMS):
    comparisons = association.comparison(rows, arms=arms)
    checks = {f"no_lost_hits_over_{arm}": not comparisons[arm]["losses"] for arm in arms[:2]}
    checks["metadata_selection_parity"] = not comparisons["ordinary_v4"]["metadata_only_source_changes"]
    summary = shared.summarize_trials(rows, arms, carry.STATES)
    for domain in (None, *sorted({r["source"].split("/")[0] for r in rows})):
        subset = rows if domain is None else [r for r in rows if r["source"].split("/")[0] == domain]
        partial = shared.summarize_trials(subset, arms, carry.STATES)
        for arm in arms[:2]:
            for state in carry.STATES:
                checks[f'{domain or "pooled"}_{state}_mrr_over_{arm}'] = (
                    partial[arms[-1]][state]["mrr_at_6"] >= partial[arm][state]["mrr_at_6"] - 1e-12)
                checks[f'{domain or "pooled"}_{state}_absent_over_{arm}'] = (
                    partial[arms[-1]][state]["absent_nonempty_rate"] <= partial[arm][state]["absent_nonempty_rate"])
    return summary, comparisons, checks


def validate_evidence(report, cases, controls, *, arms=ARMS, extra_checks=None, decision_fn=decisions):
    """One source/ID/control/trace validator shared across study phases."""
    matrix = carry.trial_matrix(report["trials"], cases, arms)
    by_case = {c["id"]: c for c in cases}
    inventories = report["source_inventories"]
    expected = {(c["id"], arm, absent) for c in cases for arm in arms for absent in (False, True)}
    inventory_keys = [(item["case"], item["candidate"], item["absent"]) for item in inventories]
    if len(inventory_keys) != len(expected) or set(inventory_keys) != expected:
        raise ValueError("Incomplete cue-context source inventories")
    by_inventory = dict(zip(inventory_keys, inventories))
    for (case_id, arm, absent), item in by_inventory.items():
        history = by_case[case_id]["absent_history" if absent else "history"]
        if (type(absent) is not bool or len(item["ids"]) != len(history)
                or sorted(item["ids"].values()) != sorted(history)):
            raise ValueError("Cue-context inventory differs from available source history")
        if arm not in arms[:2] and item["ids"] != by_inventory[case_id, "metadata_only", absent]["ids"]:
            raise ValueError("Cue-context intervention changes observation IDs")
    for row in report["trials"]:
        case = by_case[row["case"]]
        ids, gold, refs = row["selected_ids"], row["gold_ids"], row["selected_sources"]
        available_ids = by_inventory[row["case"], row["candidate"], row["absent"]]["ids"]
        available = case["absent_history" if row["absent"] else "history"]
        if (row["source"] != case["source"] or len(ids) > 6 or len(ids) != len(set(ids))
                or len(ids) != len(refs) or any(ref not in available for ref in refs)
                or any(fid not in available_ids for fid in ids)
                or refs != [available_ids.get(fid) for fid in ids]
                or gold != sorted(fid for fid, ref in available_ids.items() if ref in case["gold"])
                or len({tuple(ref) for ref in refs}) != len(refs) or bool(gold) == row["absent"]
                or any(row[k] != v for k, v in shared.score_context(ids, set(gold)).items())
                or (bool(set(ids) & set(gold)) != any(ref in case["gold"] for ref in refs))
                or any(type(row[k]) not in (int, float) or not math.isfinite(row[k]) or row[k] < 0
                       for k in ("latency_ms", "peak_python_bytes"))):
            raise ValueError("Cue-context trial differs from source IDs or scores")
    summary, comparisons, checks = decision_fn(report["trials"], arms=arms)
    if extra_checks is not None:
        checks.update(extra_checks(report["trials"]))
    if set(report["software_controls"]) != set(arms):
        raise ValueError("Cue-context control arms differ")
    for results in report["software_controls"].values():
        if len(results) != len(controls["cases"]) or {r["case"] for r in results} != {c["id"] for c in controls["cases"]}:
            raise ValueError("Incomplete cue-context software controls")
        indexed = {r["case"]: r for r in results}
        for case in controls["cases"]:
            row = indexed[case["id"]]
            available = [make_fragment(text, shared.BASE + shared.timedelta(minutes=i))
                         for i, text in enumerate(case["facts"])]
            required = sorted(available[i].id for i in case["required"])
            selected = row["selected_ids"]
            available_ids = {f.id for f in available}
            passed = (set(required) <= set(selected) <= available_ids and len(selected) <= 6
                and len(selected) == len(set(selected)) and not any(
                    case.get("forbidden_text", "\0") in f.content for f in available if f.id in selected))
            if row["required_ids"] != required or type(row["passed"]) is not bool or row["passed"] != passed:
                raise ValueError("Cue-context software-control decision differs from IDs")
    checks["software_controls"] = all(c["passed"] for results in report["software_controls"].values() for c in results)
    changed = {(case, state) for result in comparisons.values()
               for field in ("wins", "losses") for case, state, _ in result[field]}
    expected_traces = {(case, state, arm) for case, state in changed for arm in arms}
    actual_traces = [(trace["case"], trace["state"], trace["arm"]) for trace in report["traces"]]
    if set(actual_traces) != expected_traces or len(actual_traces) != len(expected_traces):
        raise ValueError("Incomplete cue-context changed-case traces")
    indexed = {(r["case"], r["state"], r["candidate"]): r for r in report["trials"] if not r["absent"]}
    for trace in report["traces"]:
        row = indexed[trace["case"], trace["state"], trace["arm"]]
        fragments = trace["fragments"]
        inventory = by_inventory[trace["case"], trace["arm"], False]["ids"]
        ids = [f["id"] for f in fragments]
        score_keys = ("seed_and_residual", "hop_1", "hop_2", "final_score", "selection_score")
        if (len(ids) != len(inventory) or set(ids) != set(inventory)
                or any(f["source"] != inventory.get(f["id"]) for f in fragments)
                or any(type(f[key]) not in (int, float) or not math.isfinite(f[key]) or f[key] < 0
                       for f in fragments for key in score_keys)
                or any(f["rank"] is not None and (type(f["rank"]) is not int or f["rank"] < 1
                           or f["selection_score"] <= 0) for f in fragments)):
            raise ValueError("Cue-context trace differs from its full source inventory or scores")
        ranked = sorted((f for f in fragments if f["rank"] is not None), key=lambda f: f["rank"])
        # Reconstruction ranks the positive distinct pool in descending score,
        # preserving input order on ties. Check lower ranks used by diagnoses too.
        expected_ranked = sorted((f for f in fragments if f["rank"] is not None),
                                 key=lambda f: f["selection_score"], reverse=True)
        if ([f["rank"] for f in ranked] != list(range(1, len(ranked) + 1))
                or [f["id"] for f in ranked] != [f["id"] for f in expected_ranked]):
            raise ValueError("Cue-context trace ranks differ from selection scores")
        if (trace["selected_ids"] != row["selected_ids"] or trace["gold_ids"] != row["gold_ids"]
                or [f["id"] for f in ranked[:6]] != row["selected_ids"]
                or not math.isclose(sum(f["selection_score"] for f in trace["fragments"]),
                                    sum(f["final_score"] for f in trace["fragments"]), rel_tol=1e-12, abs_tol=1e-12)
                or any(not math.isclose(f["final_score"], f["seed_and_residual"] + f["hop_1"] + f["hop_2"],
                                       rel_tol=1e-12, abs_tol=1e-12) for f in trace["fragments"])):
            raise ValueError("Cue-context trace differs from actual selected IDs or activation")
    return matrix, summary, comparisons, checks


def validate_report(report, cases, controls, *, arms=ARMS, schema=SCHEMA, extra_checks=None,
                    expected_production_policy=None):
    """Validate current evidence, or an archive under its trusted producer pin."""
    policy = memory.RECALL_SELECTION_POLICY if expected_production_policy is None else expected_production_policy
    if not isinstance(policy, str) or not policy:
        raise ValueError("Expected production policy must be a nonempty string")
    matrix, summary, comparisons, checks = validate_evidence(
        report, cases, controls, arms=arms, extra_checks=extra_checks)
    if (report["schema"] != schema or report["phase"] != "observed-regression" or report["matrix"] != matrix
            or report["summary"] != summary or report["comparisons"] != comparisons or report["gates"] != checks
            or type(report["advance_to_fresh_freeze"]) is not bool
            or report["advance_to_fresh_freeze"] != all(checks.values())
            or report["fresh_development_scored"] != 0 or report["fresh_evaluation_scored"] != 0
            or report["old_heldout_unscored"] != 60 or report["integration_eligible"] is not False
            or report["production_context_selection_enabled"] is not False
            or report["production_policy"] != policy):
        raise ValueError("Cue-context report decision differs from evidence")


def changed_case_traces(datasets, comparisons, *, arms=ARMS, system_factory=system_for_arm,
                        focus_current=False):
    # Trace all changed positive selections, not a hand-picked set of wins.
    changed = {(case, state) for result in comparisons.values()
               for field in ("wins", "losses") for case, state, _ in result[field]}
    traces = []
    for sources, observed, source_protocol in datasets:
        for case in observed:
            for state in carry.STATES:
                if (case["id"], state) not in changed:
                    continue
                for arm in arms:
                    base, ids, _ = association.setup(sources[case["source"]], case, arm, source_protocol,
                                                     arms=arms, system_factory=system_factory)
                    traces.append(graph.trace_recall(sources[case["source"]], case, base, ids, state,
                                                     focus_current=focus_current) | {"arm": arm})
    return traces


def run(args, *, arms=ARMS, schema=SCHEMA, system_factory=system_for_arm,
        protocol_validator=validate_protocol, extra_paths=(), focus_current=False, extra_checks=None):
    from benchmarks.scripts import phase9_carryover_run as cli

    if (args.catalog.resolve() != (cli.FIXTURES / "catalog.json").resolve()
            or args.controls.resolve() != (cli.FIXTURES / "controls.json").resolve()
            or args.development_report is not None or args.reference_report is not None):
        raise ValueError("Context diagnosis uses the protocol-pinned observed catalogs and controls; incompatible CLI inputs")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    protocol_validator(protocol)
    before, protocol_sha = fingerprints(extra_paths=extra_paths), shared.digest(args.protocol)
    datasets, controls, pins = regression_inputs(args.data_root)
    identity = tokenizer_identity("jieba")
    rows, cases, inventories = [], [], []
    for sources, observed, source_protocol in datasets:
        cases.extend(observed)
        rows.extend(association.score_cases(sources, observed, source_protocol,
                                            arms=arms, system_factory=system_factory, inventories=inventories,
                                            focus_current=focus_current))
    control_reports = {arm: carry.control_results(controls, "production", system_factory=lambda: system_factory(arm))
                       for arm in arms}
    summary, comparisons, checks = decisions(rows, arms=arms)
    if extra_checks is not None:
        checks.update(extra_checks(rows))
    checks["software_controls"] = all(c["passed"] for results in control_reports.values() for c in results)
    traces = changed_case_traces(datasets, comparisons, arms=arms, system_factory=system_factory,
                                 focus_current=focus_current)
    report = {"schema": schema, "phase": "observed-regression", "protocol_sha256": protocol_sha,
        "code_sha256": before, "source_sha256": pins,
        "environment": {"python": sys.version, "platform": platform.platform(), "tokenizer": identity},
        "matrix": carry.trial_matrix(rows, cases, arms), "trials": rows, "summary": summary,
        "source_inventories": inventories,
        "comparisons": comparisons, "gates": checks, "software_controls": control_reports, "traces": traces,
        "advance_to_fresh_freeze": all(checks.values()), "integration_eligible": False,
        "production_policy": memory.RECALL_SELECTION_POLICY, "production_context_selection_enabled": False,
        "fresh_development_scored": 0, "fresh_evaluation_scored": 0, "old_heldout_unscored": 60,
        "limitations": ["Previously observed public factual cases; no fresh generalization estimate.",
                        "Synthetic software contracts are not clinical or model-quality evidence."]}
    validate_report(report, cases, controls, arms=arms, schema=schema, extra_checks=extra_checks)
    protocol_validator(protocol)
    if (before != fingerprints(extra_paths=extra_paths) or protocol_sha != shared.digest(args.protocol)
            or any(shared.digest(args.data_root / path) != digest for path, digest in pins.items())):
        raise ValueError("Code or source inputs changed during cue-context scoring")
    return report
