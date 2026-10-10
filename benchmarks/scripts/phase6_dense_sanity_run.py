"""Phase 6 §6 item 3 — dense-schedule sanity run.

Drives :func:`scheduled_v1_continuity_ratio` with
:class:`DenseSchedule` across ``{S1 EbbinghausPolicy, S2 CapacityOnlyPolicy}``
× ``{ep200, ep500, ep1000}``. Dense = recall probe every tick in each
silent run between measurement windows — the heaviest reconsolidation
cell, baseline for the sparse comparison matrix (§6 items 4–5).

**Not a differentiator.** Under dense recall the probe reconsolidates
the same anchor fragments every tick, so S1 and S2 are expected to
produce near-identical ratios — same stream-shape confound Phase 3
documented. The gate here is a sanity gate: both policies must hold
ratio ≥ 0.80 at every horizon. S1 vs S2 differentiation is deferred
to the sparse-weekly / sparse-monthly / adversarial-silent-runs cells.

Writes a JSON record to
``docs/experiments/phase-6-month-horizon/option-dense/dense-sanity-run-01.json``.

**No LLM calls.** All measurement is state-signature cosine and
fragment-store state inspection; cost is bounded by horizon tick count
plus the dense-schedule probe count.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from psychology_base.companion.aging import CapacityOnlyPolicy, EbbinghausPolicy
from psychology_base.companion.fragment import COVENANT_TAG
from psychology_base.companion.stream_shapes import DenseSchedule
from psychology_base.companion.system import CompanionSystem
from psychology_base.eval.adapters import Observation
from psychology_base.eval.benchmarks.mianyan_analog import (
    scheduled_v1_continuity_ratio,
)


REPO = Path(__file__).resolve().parent.parent.parent
OUT_PATH = (
    REPO
    / "docs"
    / "experiments"
    / "phase-6-month-horizon"
    / "option-dense"
    / "dense-sanity-run-01.json"
)


# ---------------------------------------------------------------------------
# Stream (same shape as phase3_e1_s1_gate_run — 52 observations)
# ---------------------------------------------------------------------------


def _build_stream() -> list[Observation]:
    base_ts = datetime(2026, 7, 1, 0, 0, tzinfo=timezone.utc)
    obs: list[Observation] = []
    step = 0

    def _ts(step: int) -> datetime:
        return base_ts + timedelta(minutes=step * 30)

    anchors = [
        "I grew up in Sweden and moved to this city four years ago.",
        "My favorite author is Toni Morrison; Beloved rewires me each read.",
        "I'm learning pottery; my first bowl held water, barely.",
        "Running three times a week has rebuilt my sleep.",
        "I keep a Sunday-morning ritual: Charlotte's Web and coffee.",
        "My younger brother Alex is still in Stockholm.",
    ]
    for a in anchors:
        obs.append(
            Observation(
                timestamp=_ts(step),
                speaker="user",
                content=a,
                affect={"pleasure": 0.5, "arousal": 0.3},
                tags=(),
            )
        )
        step += 1

    affect_rows = [
        ("I felt real joy at the recital — my daughter sang clearly.",
         {"pleasure": 0.8, "arousal": 0.6}),
        ("The hospice visit emptied me out; I could not speak after.",
         {"pleasure": -0.7, "arousal": -0.2}),
        ("Her text arrived and the whole afternoon brightened.",
         {"pleasure": 0.7, "arousal": 0.5}),
        ("Three nights without sleep and the body is a stranger.",
         {"pleasure": -0.5, "arousal": -0.4}),
        ("The quiet of the library held me together today.",
         {"pleasure": 0.4, "arousal": -0.3}),
        ("The call ended badly and I replayed it until dawn.",
         {"pleasure": -0.6, "arousal": 0.4}),
    ]
    for text, aff in affect_rows:
        obs.append(
            Observation(
                timestamp=_ts(step),
                speaker="user",
                content=text,
                affect=aff,
                tags=(),
            )
        )
        step += 1

    obs.append(
        Observation(
            timestamp=_ts(step),
            speaker="system",
            content=(
                "COVENANT: on explicit self-harm language escalate to a "
                "licensed human in the same turn."
            ),
            affect={},
            tags=(COVENANT_TAG, "policy", "crisis"),
        )
    )
    step += 1

    for i in range(40):
        obs.append(
            Observation(
                timestamp=_ts(step),
                speaker="user",
                content=(
                    f"Routine update {i}: errands, groceries, laundry, "
                    "the ordinary texture of a given day."
                ),
                affect={},
                tags=(),
            )
        )
        step += 1

    return obs


# ---------------------------------------------------------------------------
# Cell runner
# ---------------------------------------------------------------------------


def _run_cell(
    *,
    policy_name: str,
    ep_long: int,
    byte_cap: int,
) -> dict:
    if policy_name == "S1":
        policy = EbbinghausPolicy(
            half_life=50.0,
            boost_factor=2.0,
            max_importance=10.0,
            consolidate_every=24,
        )
    elif policy_name == "S2":
        policy = CapacityOnlyPolicy(consolidate_every=24)
    elif policy_name == "baseline":
        policy = None
    else:
        raise ValueError(f"unknown policy_name {policy_name!r}")

    sys = CompanionSystem(
        byte_cap=byte_cap,
        witness_active=False,
        seed=0,
        aging_policy=policy,
    )
    stream = _build_stream()
    probe_cue = "Where did I grow up and when did I move?"

    t0 = time.perf_counter()
    result = scheduled_v1_continuity_ratio(
        sys,
        probe_cue=probe_cue,
        scenario_observations=stream,
        schedule=DenseSchedule(),
        ep_short=20,
        ep_long=ep_long,
        window=20,
    )
    wall_s = time.perf_counter() - t0

    cell: dict = {
        "policy": policy_name,
        "ep_short": result.ep_short,
        "ep_long": result.ep_long,
        "window": result.window,
        "schedule": result.schedule_name,
        "v1_short": result.v1_short,
        "v1_long": result.v1_long,
        "ratio": result.ratio,
        "pre_short_probes_fired": len(result.pre_short_recall_ticks),
        "between_probes_fired": len(result.between_recall_ticks),
        "fragment_count_after": len(sys.companion.store),
        "byte_cap": byte_cap,
        "wall_seconds": wall_s,
        "notes": result.notes,
    }

    # S2 thrash-rate observability — plan §5 kill criterion.
    if policy_name == "S2":
        cell["thrash_rate"] = policy.thrash_rate()

    return cell


def _run(byte_cap: int) -> dict:
    out: dict = {
        "experiment": "phase-6-dense-sanity",
        "run_id": "dense-sanity-run-01",
        "schedule": "DenseSchedule(every=1)",
        "byte_cap": byte_cap,
        "notes": (
            "Dense schedule injects a probe every tick in each silent run "
            "between measurement windows. Expect S1 ≈ S2 (stream-shape "
            "confound persists under dense recall — probe reconsolidates "
            "same anchor fragments). Hard gate: ratio ≥ 0.80 at every "
            "horizon under both policies."
        ),
        "cells": [],
    }

    cells = [
        (policy, ep_long)
        for ep_long in (200, 500, 1000)
        for policy in ("baseline", "S1", "S2")
    ]
    for policy, ep_long in cells:
        print(f"  running {policy} / ep_long={ep_long} ...", flush=True)
        cell = _run_cell(policy_name=policy, ep_long=ep_long, byte_cap=byte_cap)
        out["cells"].append(cell)
        print(
            f"    ratio={cell['ratio']:.3f} wall={cell['wall_seconds']:.1f}s "
            f"frags={cell['fragment_count_after']}",
            flush=True,
        )

    # Verdicts.
    verdicts: list[str] = []
    for cell in out["cells"]:
        tag = f"{cell['policy']} @ ep{cell['ep_long']}"
        if cell["ratio"] >= 0.80:
            verdicts.append(
                f"{tag}: ratio={cell['ratio']:.3f} >= 0.80 ✓"
            )
        elif cell["ratio"] >= 0.50:
            verdicts.append(
                f"{tag}: ratio={cell['ratio']:.3f} in [0.50, 0.80) — soft miss"
            )
        else:
            verdicts.append(
                f"{tag}: ratio={cell['ratio']:.3f} < 0.50 — kill criterion"
            )
    out["verdicts"] = verdicts
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--byte-cap",
        type=int,
        default=512,
        help=(
            "FragmentStore byte cap. 512 = pressure cell (eviction active, "
            "where aging policy is actually tested). 1024 = no-pressure "
            "sanity (stream fits under cap, policy becomes no-op on "
            "eviction order)."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_PATH,
        help="JSON output path.",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Also print the result to stdout.",
    )
    args = parser.parse_args()

    print(f"phase6 dense-sanity: byte_cap={args.byte_cap}", flush=True)
    result = _run(args.byte_cap)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.print:
        print(json.dumps(result, indent=2))
    print()
    for v in result["verdicts"]:
        print(f"  - {v}")
    print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()
