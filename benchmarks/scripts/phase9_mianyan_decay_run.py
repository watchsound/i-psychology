"""Phase 9+ 绵延 decay study — extend Phase 6 sparse-recall harness past ep1000.

Phase 6 close-out §5 flagged the carried-debt item: *"绵延 metric beyond ep1000
— the plan declared ep1000 as the horizon; nothing says the ratio continues to
hold at ep5000+. A longer-horizon stress is Phase 7-or-later research (not v1
scope)."*

This runner is the stress sweep that closes that debt. It reuses the Phase 6
``_build_stream`` + ``_run_cell`` cell logic verbatim (import-only, no copy),
and sweeps a longer horizon ladder — ep1000 / ep2000 / ep3000 / ep5000 /
ep10000 — across the same ``{baseline, S1, S2} × {sparse-weekly,
sparse-monthly, adversarial-silent-runs}`` matrix Phase 6 used.

**Decision gate.** Per CLAUDE.md and the Phase 6 decision stance: if the
ratio falls below 0.80 at any horizon, **declare the limitation** — record
the measured ratio, the horizon at which it crossed, and the cadence under
which it crossed. Do not retrofit the metric or the policy. Honest
documentation of where the ratio breaks is the deliverable.

**Scope.** This extends the measurement; it does not design a new mechanism.
If ep10000 holds, that is a mechanism-viability result (not a SOTA claim;
Ouroboros's mianyan decay past ep20 remains an open 绵延 track per the
CLAUDE.md thesis). If ep10000 breaks, that is an honest open-problem flag
for a future Phase 9+ mechanism-design study.

**Output.** JSON records to
``docs/experiments/phase-9-mianyan-decay/`` plus a comparison memo at close-out
time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from psychology_base.companion.stream_shapes import (
    AdversarialSilentRuns,
    MonthlySchedule,
    WeeklySchedule,
)

# Reuse Phase 6 cell helpers verbatim — the Phase 9+ study is a horizon
# extension, not a mechanism change, so the cell contract is byte-identical.
from benchmarks.scripts.phase6_sparse_recall_run import _run_cell, _run_sweep  # noqa: E402


REPO = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO / "docs" / "experiments" / "phase-9-mianyan-decay"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--byte-cap", type=int, default=512)
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[1000, 2000, 3000, 5000, 10000],
        help=(
            "Longer-horizon ladder beyond Phase 6's ep1000 ceiling. Default "
            "sweep stress-tests the 绵延 ratio out to ep10000 under the "
            "same sparse schedules Phase 6 used."
        ),
    )
    parser.add_argument(
        "--adversarial-ep",
        type=int,
        default=5000,
        help=(
            "Horizon for adversarial-silent-runs cell. Default ep5000 "
            "exercises ~33 silent-run segments under ticks=(5,50,150)."
        ),
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bundle: dict = {"byte_cap": args.byte_cap, "sweeps": {}}

    # --- Weekly ---
    print(f"\nphase9 mianyan-decay sparse-weekly: byte_cap={args.byte_cap}", flush=True)
    weekly = _run_sweep(
        schedule=WeeklySchedule(),  # 168-tick period
        schedule_label="sparse-weekly",
        horizons=tuple(args.horizons),
        byte_cap=args.byte_cap,
    )
    (OUT_DIR / "sparse-weekly-run-01.json").write_text(
        json.dumps(weekly, indent=2), encoding="utf-8"
    )
    bundle["sweeps"]["sparse-weekly"] = weekly

    # --- Monthly ---
    print(f"\nphase9 mianyan-decay sparse-monthly: byte_cap={args.byte_cap}", flush=True)
    monthly = _run_sweep(
        schedule=MonthlySchedule(),  # 720-tick period
        schedule_label="sparse-monthly",
        horizons=tuple(args.horizons),
        byte_cap=args.byte_cap,
    )
    (OUT_DIR / "sparse-monthly-run-01.json").write_text(
        json.dumps(monthly, indent=2), encoding="utf-8"
    )
    bundle["sweeps"]["sparse-monthly"] = monthly

    # --- Adversarial ---
    # Phase 6 plan §2 cadence: recalls at ep5 / ep50 / ep150 within each
    # silent-run segment. Extending the horizon exposes the ratio to more
    # silent-run resets than any other schedule — this is the stress cell.
    adv_schedule = AdversarialSilentRuns(ticks=(5, 50, 150))
    print(
        f"\nphase9 mianyan-decay adversarial-silent-runs: byte_cap={args.byte_cap}",
        flush=True,
    )
    adv = _run_sweep(
        schedule=adv_schedule,
        schedule_label="adversarial-silent-runs",
        horizons=(args.adversarial_ep,),
        byte_cap=args.byte_cap,
    )
    (OUT_DIR / "adversarial-silent-runs-run-01.json").write_text(
        json.dumps(adv, indent=2), encoding="utf-8"
    )
    bundle["sweeps"]["adversarial-silent-runs"] = adv

    print("\n=== VERDICTS ===")
    for label, sweep in bundle["sweeps"].items():
        print(f"\n[{label}]")
        for v in sweep["verdicts"]:
            print(f"  - {v}")

    # Roll-up: find the lowest ratio across every cell. This is the honest
    # "where does the ratio break" signal. If min >= 0.80, Phase 9+ decay
    # study PASSES the Phase 6 gate out to the longest sweep horizon. If
    # min < 0.80, record the horizon + schedule + policy where it crossed.
    all_cells = []
    for sweep in bundle["sweeps"].values():
        all_cells.extend(sweep["cells"])

    min_cell = min(all_cells, key=lambda c: c["ratio"])
    max_ep_passing = max(
        (c["ep_long"] for c in all_cells if c["ratio"] >= 0.80),
        default=0,
    )
    broken_cells = [
        {
            "policy": c["policy"],
            "schedule": c["schedule"],
            "ep_long": c["ep_long"],
            "ratio": c["ratio"],
        }
        for c in all_cells
        if c["ratio"] < 0.80
    ]

    rollup = {
        "min_ratio_cell": {
            "policy": min_cell["policy"],
            "schedule": min_cell["schedule"],
            "ep_long": min_cell["ep_long"],
            "ratio": min_cell["ratio"],
        },
        "max_ep_long_at_or_above_0_80": max_ep_passing,
        "broken_cells_below_0_80": broken_cells,
        "verdict": (
            "PASS: 绵延 ratio ≥ 0.80 holds across the full sweep horizon."
            if not broken_cells
            else f"LIMITATION: {len(broken_cells)} cell(s) crossed below 0.80."
        ),
    }
    bundle["rollup"] = rollup

    (OUT_DIR / "rollup.json").write_text(
        json.dumps(rollup, indent=2), encoding="utf-8"
    )

    print("\n=== ROLLUP ===")
    print(f"  min-ratio cell: {rollup['min_ratio_cell']}")
    print(f"  max ep_long ≥ 0.80: {rollup['max_ep_long_at_or_above_0_80']}")
    print(f"  broken cells: {len(broken_cells)}")
    print(f"  verdict: {rollup['verdict']}")
    print(f"\nWrote 4 reports to: {OUT_DIR}")


if __name__ == "__main__":
    main()
