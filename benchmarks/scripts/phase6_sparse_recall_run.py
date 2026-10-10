"""Phase 6 §6 items 4+5 — sparse-weekly / sparse-monthly / adversarial-silent-runs.

Drives :func:`scheduled_v1_continuity_ratio` under three sparse-recall
schedules × ``{baseline, S1 EbbinghausPolicy, S2 CapacityOnlyPolicy}``
× ``{ep200, ep500, ep1000}``. The dense-sanity run shipped alongside
established the baseline — this script's job is to see whether sparse
recall actually differentiates S1 from S2 (Phase 3's documented
stream-shape confound resolver, Phase 6 §3 soft gate).

**Hard gates (per Phase 6 §3):**
- V1 continuity ratio ≥ 0.80 at **ep500 sparse-monthly** (the hardest
  combined cell — a silent run longer than the horizon under
  byte-cap pressure).

**Soft gates:**
- ≥ one cell shows ≥ 0.05 absolute delta between S1 and S2 on V1 ratio
  under sparse recall → stream-shape-confound resolved.

**Decision stance.** If the ratio falls below 0.80 at ep500
sparse-monthly, **declare the limitation** per plan §3 — do not
retrofit the metric. Record the measured ratio, the horizon at which
it crossed, and continue.

**Output.** Three JSON records to
``docs/experiments/phase-6-month-horizon/option-sparse/``:
- ``sparse-weekly-run-01.json``
- ``sparse-monthly-run-01.json``
- ``adversarial-silent-runs-run-01.json``

Plus a comparison memo synthesized at close-out time.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from psychology_base.companion.aging import CapacityOnlyPolicy, EbbinghausPolicy
from psychology_base.companion.fragment import COVENANT_TAG
from psychology_base.companion.stream_shapes import (
    AdversarialSilentRuns,
    MonthlySchedule,
    StreamSchedule,
    WeeklySchedule,
)
from psychology_base.companion.system import CompanionSystem
from psychology_base.eval.adapters import Observation
from psychology_base.eval.benchmarks.mianyan_analog import (
    scheduled_v1_continuity_ratio,
)


REPO = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO / "docs" / "experiments" / "phase-6-month-horizon" / "option-sparse"


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


def _run_cell(
    *,
    policy_name: str,
    schedule: StreamSchedule,
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
        schedule=schedule,
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
    if policy_name == "S2":
        cell["thrash_rate"] = policy.thrash_rate()
    return cell


def _run_sweep(
    *,
    schedule: StreamSchedule,
    schedule_label: str,
    horizons: tuple[int, ...],
    byte_cap: int,
) -> dict:
    out: dict = {
        "experiment": f"phase-6-{schedule_label}",
        "run_id": f"{schedule_label}-run-01",
        "schedule": schedule_label,
        "byte_cap": byte_cap,
        "cells": [],
    }
    for ep_long in horizons:
        for policy in ("baseline", "S1", "S2"):
            print(
                f"  [{schedule_label}] {policy} / ep{ep_long} ...",
                flush=True,
            )
            cell = _run_cell(
                policy_name=policy,
                schedule=schedule,
                ep_long=ep_long,
                byte_cap=byte_cap,
            )
            out["cells"].append(cell)
            print(
                f"    ratio={cell['ratio']:.3f} probes_pre="
                f"{cell['pre_short_probes_fired']} between="
                f"{cell['between_probes_fired']} wall="
                f"{cell['wall_seconds']:.1f}s",
                flush=True,
            )

    # S1 vs S2 differentiation (Phase 6 §3 soft gate).
    deltas: dict[int, float] = {}
    for ep_long in horizons:
        s1 = next(
            c for c in out["cells"] if c["policy"] == "S1" and c["ep_long"] == ep_long
        )
        s2 = next(
            c for c in out["cells"] if c["policy"] == "S2" and c["ep_long"] == ep_long
        )
        deltas[ep_long] = abs(s1["ratio"] - s2["ratio"])
    out["s1_vs_s2_abs_delta_per_horizon"] = {str(k): v for k, v in deltas.items()}
    out["max_s1_s2_delta"] = max(deltas.values()) if deltas else 0.0

    verdicts: list[str] = []
    for cell in out["cells"]:
        tag = f"{cell['policy']} @ ep{cell['ep_long']}"
        if cell["ratio"] >= 0.80:
            verdicts.append(f"{tag}: ratio={cell['ratio']:.3f} ✓")
        elif cell["ratio"] >= 0.50:
            verdicts.append(f"{tag}: ratio={cell['ratio']:.3f} soft-miss")
        else:
            verdicts.append(f"{tag}: ratio={cell['ratio']:.3f} KILL")
    if out["max_s1_s2_delta"] >= 0.05:
        verdicts.append(
            f"S1 vs S2 differentiation Δ={out['max_s1_s2_delta']:.3f} "
            f">= 0.05 ✓ (stream-shape confound resolved)"
        )
    else:
        verdicts.append(
            f"S1 vs S2 Δ={out['max_s1_s2_delta']:.3f} < 0.05 — "
            f"stream-shape confound persists under {schedule_label}"
        )
    out["verdicts"] = verdicts
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--byte-cap", type=int, default=512)
    parser.add_argument(
        "--horizons-fixed",
        type=int,
        nargs="+",
        default=[200, 500],
        help=(
            "Horizons for weekly + monthly. Monthly on ep200/500 exercises "
            "the hardest cell (silent run shorter or equal to cadence)."
        ),
    )
    parser.add_argument(
        "--horizons-weekly-extended",
        type=int,
        nargs="+",
        default=[1000],
        help=(
            "Extra weekly horizons where multiple probes fit — ep1000 "
            "fires ~5 weekly probes under default cadence."
        ),
    )
    parser.add_argument(
        "--adversarial-ep",
        type=int,
        default=500,
        help="Horizon for adversarial-silent-runs cell. Plan §3 names ep500.",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bundle: dict = {"byte_cap": args.byte_cap, "sweeps": {}}

    # --- Weekly ---
    print(f"\nphase6 sparse-weekly: byte_cap={args.byte_cap}", flush=True)
    weekly = _run_sweep(
        schedule=WeeklySchedule(),  # 168-tick period, offset = period - 1
        schedule_label="sparse-weekly",
        horizons=tuple(args.horizons_fixed + args.horizons_weekly_extended),
        byte_cap=args.byte_cap,
    )
    (OUT_DIR / "sparse-weekly-run-01.json").write_text(
        json.dumps(weekly, indent=2), encoding="utf-8"
    )
    bundle["sweeps"]["sparse-weekly"] = weekly

    # --- Monthly ---
    print(f"\nphase6 sparse-monthly: byte_cap={args.byte_cap}", flush=True)
    monthly = _run_sweep(
        schedule=MonthlySchedule(),  # 720-tick period
        schedule_label="sparse-monthly",
        horizons=tuple(args.horizons_fixed + args.horizons_weekly_extended),
        byte_cap=args.byte_cap,
    )
    (OUT_DIR / "sparse-monthly-run-01.json").write_text(
        json.dumps(monthly, indent=2), encoding="utf-8"
    )
    bundle["sweeps"]["sparse-monthly"] = monthly

    # --- Adversarial ---
    # Phase 6 plan §2: "recalls at ep5 / ep50 / ep150, silent in between".
    # These are tick indices within each silent-run segment.
    adv_schedule = AdversarialSilentRuns(ticks=(5, 50, 150))
    print(f"\nphase6 adversarial-silent-runs: byte_cap={args.byte_cap}", flush=True)
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

    print(f"\nWrote 3 reports to: {OUT_DIR}")


if __name__ == "__main__":
    main()
