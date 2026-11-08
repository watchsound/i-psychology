"""Fixed scale-consistency study through the existing focused-context workflow."""
from __future__ import annotations

from functools import partial
import json
from pathlib import Path

from psychology_base.eval.benchmarks import cue_context as observed
from psychology_base.eval.benchmarks import focused_context as focused
from psychology_base.eval.benchmarks import graph_priors as graph


ARMS = ("ordinary_v4", "metadata_only", "focus_companion", "scaled_focus")
SCHEMA = "scaled-context-regression-v1"
FIXED = focused.FIXED | {"support": "half-maximum-normalized-active-question-seed-times-selection-mass",
    "primary": ARMS[-1], "ablation": "focus_companion"}
INPUTS = {"benchmarks/fixtures/focused_context/protocol.json",
    "docs/phases/phase-9-scaled-context-plan.md", "tests/test_scaled_context.py",
    "docs/experiments/phase-9-focused-context/review-validation.json",
    "docs/experiments/phase-9-focused-context/review-observed.json"}


def system_for_arm(arm, **overrides):
    if arm in ARMS[:-1]:
        return focused.system_for_arm(arm, **overrides)
    if arm != ARMS[-1]:
        raise ValueError("Unknown scaled-context arm")
    if overrides.get("fragment_context_selection", "scaled_focus") != "scaled_focus":
        raise ValueError("Scaled primary requires the scaled_focus context selection")
    return graph.system_for_policy("production", **(overrides | {"fragment_context_selection": "scaled_focus"}))


def validate_protocol(protocol):
    focused.validate_protocol(protocol, schema=SCHEMA, arms=ARMS, fixed=FIXED, inputs=INPUTS)
    original = observed.root() / "benchmarks/fixtures/focused_context/protocol.json"
    focused.validate_protocol(json.loads(original.read_text(encoding="utf-8")))


def validate_report(report, cases, controls, *, expected_production_policy=None):
    focused.validate_report(report, cases, controls, arms=ARMS, schema=SCHEMA,
                            expected_production_policy=expected_production_policy)


def run(args):
    report = observed.run(args, arms=ARMS, schema=SCHEMA, system_factory=system_for_arm,
        protocol_validator=validate_protocol, extra_paths=(Path(__file__), Path(focused.__file__)),
        focus_current=True, extra_checks=partial(focused.extra_checks, primary=ARMS[-1]))
    datasets, controls, _ = observed.regression_inputs(args.data_root)
    validate_report(report, [case for _, cases, _ in datasets for case in cases], controls)
    return report
