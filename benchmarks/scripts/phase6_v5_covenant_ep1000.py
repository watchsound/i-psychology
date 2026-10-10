"""Phase 6 §3 hard gate — V5 covenant 8/8 held at ep1000.

Reuses the Phase 3 ``v5_covenant_probes.yaml`` set but extends
``check_at_ep`` from 200 to 1000 and sweeps across
``{baseline, S1 EbbinghausPolicy, S2 CapacityOnlyPolicy}``.

**Hard gate:** 100% of 8 covenant rules held at ep1000 under every
(policy × probe) cell. Zero drift tolerated. V5 drift at ep1000 halts
the Phase 6 sequence immediately as a safety failure, not a research
iteration (plan §3).

**Output.** JSON to
``docs/experiments/phase-6-month-horizon/v5-covenant-ep1000.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from psychology_base.companion.aging import CapacityOnlyPolicy, EbbinghausPolicy
from psychology_base.companion.system import CompanionSystem
from psychology_base.eval.benchmarks.v5_covenant import (
    V5Probe,
    V5ProbeSet,
    load_v5_probes,
    run_v5_probes,
)


REPO = Path(__file__).resolve().parent.parent.parent
PROBES = REPO / "benchmarks" / "scripts" / "v5_covenant_probes.yaml"
OUT_PATH = (
    REPO
    / "docs"
    / "experiments"
    / "phase-6-month-horizon"
    / "v5-covenant-ep1000.json"
)


def _retarget_horizon(probe_set: V5ProbeSet, ep: int) -> V5ProbeSet:
    """Return a probe set with every probe's ``check_at_ep`` set to ``ep``.

    V5Probe is a Pydantic model — ``model_copy(update=...)`` gives us an
    immutable rebuild without mutating the shared loaded probe set.
    """
    retargeted = [p.model_copy(update={"check_at_ep": ep}) for p in probe_set.probes]
    return probe_set.model_copy(update={"probes": retargeted})


def _factory(policy_name: str, byte_cap: int):
    def make():
        if policy_name == "S1":
            policy = EbbinghausPolicy(half_life=5.0, base_importance=1.0)
        elif policy_name == "S2":
            policy = CapacityOnlyPolicy(consolidate_every=24)
        elif policy_name == "baseline":
            policy = None
        else:
            raise ValueError(f"unknown policy_name {policy_name!r}")
        return CompanionSystem(
            byte_cap=byte_cap,
            witness_active=False,
            seed=0,
            aging_policy=policy,
        )

    return make


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ep", type=int, default=1000)
    parser.add_argument("--byte-cap", type=int, default=1024)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    probe_set = _retarget_horizon(load_v5_probes(PROBES), args.ep)
    print(
        f"phase6 v5 covenant ep{args.ep}: byte_cap={args.byte_cap} "
        f"probes={len(probe_set.probes)}",
        flush=True,
    )

    out: dict = {
        "experiment": "phase-6-v5-covenant-ep1000",
        "run_id": f"v5-ep{args.ep}-run-01",
        "ep": args.ep,
        "byte_cap": args.byte_cap,
        "probe_count": len(probe_set.probes),
        "cells": {},
    }

    all_pass = True
    for policy in ("baseline", "S1", "S2"):
        print(f"  running {policy} ...", flush=True)
        result = run_v5_probes(probe_set, _factory(policy, args.byte_cap))
        cell = {
            "policy": policy,
            "overall_pass": result.overall_pass,
            "per_covenant_pass": {str(k): v for k, v in result.per_covenant_pass.items()},
            "failures": [
                {
                    "probe_id": o.probe_id,
                    "covenant_id": o.covenant_id,
                    "covenant_name": o.covenant_name,
                }
                for o in result.outcomes
                if not o.survived
            ],
            "min_fragment_count_at_check": min(
                o.fragment_count_at_check for o in result.outcomes
            ),
            "max_fragment_count_at_check": max(
                o.fragment_count_at_check for o in result.outcomes
            ),
        }
        out["cells"][policy] = cell
        print(
            f"    overall_pass={cell['overall_pass']} "
            f"failures={len(cell['failures'])} "
            f"noncov_frag_range=[{cell['min_fragment_count_at_check']}, "
            f"{cell['max_fragment_count_at_check']}]",
            flush=True,
        )
        all_pass = all_pass and cell["overall_pass"]

    out["all_cells_pass"] = all_pass
    if all_pass:
        out["verdict"] = (
            f"V5 covenant 8/8 held at ep{args.ep} under every policy ✓ "
            "(hard gate)"
        )
    else:
        failing_cells = [
            p for p, c in out["cells"].items() if not c["overall_pass"]
        ]
        out["verdict"] = (
            f"V5 covenant FAIL at ep{args.ep} in cells {failing_cells} — "
            "Phase 6 sequence halted (safety failure, not research iteration)"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  - {out['verdict']}")
    print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()
