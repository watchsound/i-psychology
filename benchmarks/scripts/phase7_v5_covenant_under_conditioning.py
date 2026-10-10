"""Phase 7 §6 item 7 — V5 covenant regression under trait conditioning.

**The gate.** V5 covenant 8/8 holds under every (policy × OCEAN-extreme
pre-population × ep1000) cell. Zero drift tolerated. Phase 6's §3 hard
gate must continue to clear when the companion store carries the kind
of trait-inferred fragments Phase 7 will produce in production.

**What is tested.** Trait conditioning is a prompt-level mechanism —
it never writes to the store, and `run_v5_probes` runs at storage
level with no agent and no LLM. So the regression surface is NOT the
prompt. The regression surface is byte_cap pressure: Phase 7 will add
20–30 trait fragments per onboarded user (self_explicit + self_implicit
across 5 OCEAN dims). Those fragments compete with covenant fragments
and noise observations for byte_cap. If they displace covenants, V5
fails. If they do not, Phase 7 is safe to ship.

**Pre-population shape.** For each OCEAN extreme (10 cells: high/low
on each dim), we pre-populate the fresh store with 6 explicit + 6
implicit trait fragments on that dim at `score = {5.0, 1.0}`. This
mirrors what `infer_traits` + `confirm_candidates` +
`ingest_confirmed_candidates` would produce for a user whose chat
import strongly displays that extreme. Fragments are timestamped
BEFORE the covenant seed so they are maximally stale under exponential
decay — the hardest regression case.

**Output.** JSON to
``docs/experiments/phase-7-personalization/v5-covenant-under-conditioning.json``.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from psychology_base.applications.emotional.trait_inference import (
    SOURCE_INFERRED_CONFIRMED,
)
from psychology_base.applications.emotional.trait_profile import OceanDim
from psychology_base.companion.aging import CapacityOnlyPolicy, EbbinghausPolicy
from psychology_base.companion.system import CompanionSystem
from psychology_base.eval.adapters import Observation
from psychology_base.eval.benchmarks.v5_covenant import (
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
    / "phase-7-personalization"
    / "v5-covenant-under-conditioning.json"
)


# Timestamps BEFORE the Phase 6 default `_default_timestamp` floor
# (2026-07-01), so trait fragments are maximally decayed by the time
# the covenant seed is observed at step 0.
_PREPOP_FLOOR = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)


def _retarget_horizon(probe_set: V5ProbeSet, ep: int) -> V5ProbeSet:
    retargeted = [p.model_copy(update={"check_at_ep": ep}) for p in probe_set.probes]
    return probe_set.model_copy(update={"probes": retargeted})


def _pre_populate_trait_fragments(
    system: CompanionSystem,
    *,
    dim: OceanDim,
    score: float,
    n_per_layer: int = 6,
) -> None:
    """Seed trait fragments representing one OCEAN extreme.

    Non-covenant tags only — these ARE expected to be evictable; the
    test is whether they displace covenants, not whether they survive.
    """
    layer_label = "high" if score >= 4.0 else "low"
    for layer in ("explicit", "implicit"):
        for i in range(n_per_layer):
            system.observe(
                Observation(
                    timestamp=_PREPOP_FLOOR + timedelta(minutes=i),
                    speaker="user",
                    content=f"{layer_label}-{dim.value}-{layer}-{i}",
                    tags=(
                        f"subject:user",
                        f"trait:{dim.value}",
                        f"trait:{dim.value}:{layer}",
                        f"trait:{dim.value}:score:{score:.1f}",
                        SOURCE_INFERRED_CONFIRMED,
                    ),
                )
            )


def _factory_with_prepop(
    policy_name: str,
    byte_cap: int,
    dim: OceanDim,
    score: float,
) -> Callable[[], CompanionSystem]:
    def make() -> CompanionSystem:
        if policy_name == "S1":
            policy = EbbinghausPolicy(half_life=5.0, base_importance=1.0)
        elif policy_name == "S2":
            policy = CapacityOnlyPolicy(consolidate_every=24)
        elif policy_name == "baseline":
            policy = None
        else:
            raise ValueError(f"unknown policy_name {policy_name!r}")
        system = CompanionSystem(
            byte_cap=byte_cap,
            witness_active=False,
            seed=0,
            aging_policy=policy,
        )
        _pre_populate_trait_fragments(system, dim=dim, score=score)
        return system

    return make


def _extreme_cells() -> list[tuple[OceanDim, float]]:
    """10 cells = 5 dims × {high=5.0, low=1.0}."""
    out: list[tuple[OceanDim, float]] = []
    for dim in OceanDim:
        out.append((dim, 5.0))
        out.append((dim, 1.0))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ep", type=int, default=1000)
    parser.add_argument("--byte-cap", type=int, default=1024)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    probe_set = _retarget_horizon(load_v5_probes(PROBES), args.ep)
    print(
        f"phase7 v5 under conditioning ep{args.ep}: "
        f"byte_cap={args.byte_cap} probes={len(probe_set.probes)} "
        f"cells={len(_extreme_cells()) * 3}",
        flush=True,
    )

    out: dict = {
        "experiment": "phase-7-v5-covenant-under-conditioning",
        "run_id": f"phase7-v5-ep{args.ep}-run-01",
        "ep": args.ep,
        "byte_cap": args.byte_cap,
        "probe_count": len(probe_set.probes),
        "cells": [],
    }

    all_pass = True
    for policy in ("baseline", "S1", "S2"):
        for dim, score in _extreme_cells():
            label = f"{'high' if score >= 4.0 else 'low'}-{dim.value}"
            print(f"  running {policy} / {label} ...", flush=True)
            result = run_v5_probes(
                probe_set,
                _factory_with_prepop(policy, args.byte_cap, dim, score),
            )
            cell = {
                "policy": policy,
                "extreme": label,
                "ocean_dim": dim.value,
                "score": score,
                "overall_pass": result.overall_pass,
                "per_covenant_pass": {
                    str(k): v for k, v in result.per_covenant_pass.items()
                },
                "failures": [
                    {
                        "probe_id": o.probe_id,
                        "covenant_id": o.covenant_id,
                        "covenant_name": o.covenant_name,
                    }
                    for o in result.outcomes
                    if not o.survived
                ],
                "min_noncov_frag_at_check": min(
                    o.fragment_count_at_check for o in result.outcomes
                ),
                "max_noncov_frag_at_check": max(
                    o.fragment_count_at_check for o in result.outcomes
                ),
            }
            out["cells"].append(cell)
            all_pass = all_pass and cell["overall_pass"]
            if not cell["overall_pass"]:
                print(
                    f"    FAIL overall_pass={cell['overall_pass']} "
                    f"failures={len(cell['failures'])}",
                    flush=True,
                )

    out["all_cells_pass"] = all_pass
    if all_pass:
        out["verdict"] = (
            f"V5 covenant 8/8 held at ep{args.ep} under every "
            "(policy × OCEAN-extreme) cell with trait pre-population ✓ "
            "(Phase 7 §6 hard gate)"
        )
    else:
        failing = [
            (c["policy"], c["extreme"]) for c in out["cells"]
            if not c["overall_pass"]
        ]
        out["verdict"] = (
            f"V5 covenant FAIL at ep{args.ep} under conditioning in cells "
            f"{failing} — Phase 7 sequence halted (safety failure)"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  - {out['verdict']}")
    print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()
