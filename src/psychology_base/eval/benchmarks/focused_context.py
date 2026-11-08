"""One frozen focused-selection candidate plus shared-policy ablation."""
from __future__ import annotations

from pathlib import Path
import math

from psychology_base.companion import reconstruction as memory
from psychology_base.eval.benchmarks import chinese_retrieval as shared
from psychology_base.eval.benchmarks import cue_context as observed
from psychology_base.eval.benchmarks import graph_priors as graph
from psychology_base.eval.benchmarks import recall_carryover as carry


ARMS = ("ordinary_v4", "metadata_only", "focus_shared", "focus_companion")
SCHEMA = "focused-context-regression-v1"
FIXED = observed.FIXED | {"support": "half-maximum-normalized-active-question-seed",
    "focus": "last-provided-cue-turn", "primary": "focus_companion",
    "witness_primary_policy": "production-identity", "ablation": "focus_shared"}
GATES = {"no_lost_answer_hits": True, "metadata_selection_parity": True,
    "pooled_and_domain_mrr_nonloss": True, "absent_rate_nonincrease": True,
    "software_controls": True, "witness_baseline_view": True}
INPUTS = {"benchmarks/fixtures/cue_context/protocol.json",
    "docs/phases/phase-9-focused-context-plan.md", "tests/test_focused_context.py",
    "docs/experiments/phase-9-script-dispatch/corrected-observed.json"}


def reconstruction():
    return shared._bind(memory.reconstruct, _selection_scores=memory.focused_context_scores)


def system_for_arm(arm, **overrides):
    if arm not in ARMS:
        raise ValueError("Unknown focused-context arm")
    if overrides.get("fragment_context_selection", "identity") != "identity":
        raise ValueError("Historical focused arms require identity runtime selection")
    system = graph.system_for_policy("production", **overrides)
    if arm in ARMS[:2]:
        return system
    return carry.bind_reconstruction(system, reconstruction(),
        witness_reconstruct=memory.reconstruct if arm == "focus_companion" else None)


def validate_protocol(protocol, *, schema=SCHEMA, arms=ARMS, fixed=FIXED, inputs=INPUTS):
    if (protocol["schema"] != schema or protocol["status"] != "frozen-before-source-scoring"
            or protocol["fixed"] != fixed or protocol["arms"] != list(arms) or protocol["gates"] != GATES):
        raise ValueError("Unsupported focused-context protocol")
    if set(protocol["inputs"]) != inputs or any(
            shared.digest(observed.root() / path) != sha for path, sha in protocol["inputs"].items()):
        raise ValueError("Focused-context inputs differ from freeze")
    import json
    original = observed.root() / "benchmarks/fixtures/cue_context/protocol.json"
    observed.validate_protocol(json.loads(original.read_text(encoding="utf-8")))


def extra_checks(rows, *, primary=ARMS[-1]):
    indexed = {(r["case"], r["candidate"], r["state"], r["absent"]): r for r in rows}
    views, focus = True, True
    for row in rows:
        if row["candidate"] != primary:
            continue
        baseline = indexed[row["case"], "metadata_only", row["state"], row["absent"]]
        views = views and (row["witness_ids"] == baseline["witness_ids"]
                           and row["witness_activation"] == baseline["witness_activation"])
        focus = focus and row["focus_ref"] == baseline["focus_ref"]
    return {"witness_baseline_view": views, "focus_input_parity": focus}


def validate_report(report, cases, controls, *, arms=ARMS, schema=SCHEMA,
                    expected_production_policy=None):
    observed.validate_report(report, cases, controls, arms=arms, schema=schema,
                             extra_checks=lambda rows: extra_checks(rows, primary=arms[-1]),
                             expected_production_policy=expected_production_policy)
    validate_inputs(report, cases)


def validate_inputs(report, cases):
    """Validate focus and independent Witness ownership in any study phase."""
    for row in report["trials"]:
        case = next(c for c in cases if c["id"] == row["case"])
        inventory = next(i["ids"] for i in report["source_inventories"] if
            (i["case"], i["candidate"], i["absent"]) == (row["case"], row["candidate"], row["absent"]))
        if (row["focus_ref"] != case["query"] or case["query"] != case["cue_refs"][-1]
                or len(row["witness_ids"]) > 6 or len(set(row["witness_ids"])) != len(row["witness_ids"])
                or not set(row["witness_ids"]) <= set(inventory)
                or not set(row["witness_activation"]) <= set(inventory)
                or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
                       for v in row["witness_activation"].values())):
            raise ValueError("Focused report input or Witness view differs")


def run(args):
    report = observed.run(args, arms=ARMS, schema=SCHEMA, system_factory=system_for_arm,
        protocol_validator=validate_protocol, extra_paths=(Path(__file__),), focus_current=True, extra_checks=extra_checks)
    datasets, controls, _ = observed.regression_inputs(args.data_root)
    validate_report(report, [case for _, cases, _ in datasets for case in cases], controls)
    return report
