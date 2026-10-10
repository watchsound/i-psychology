"""Phase 7 §6 item 5 — B4 import-to-profile accuracy run.

Measures import → trait-inference → TraitProfile accuracy against the
pre-assigned OCEAN vectors of the ``default_personas()`` fixture (10
single-dim-extreme personas) at three message-volume cells
(N={50, 200, 500}).

Pass bar per plan §9: ≥ 3 of 5 dimensions within ±0.5 Likert at N=200.
Below the bar ships with the Phase 7 §3 soft-gate documentation
("inference remains noisy below N imported turns") — this is a
mechanism-viability floor, not a SOTA claim.

Writes JSON to
``docs/experiments/phase-7-personalization/b4-import-to-profile.json``.

No LLM calls — the inference pass is lexical v1; synthetic persona
generation is deterministic under seed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from psychology_base.eval.benchmarks.b4_import_to_profile import (
    B4Report,
    OceanDim,
    default_personas,
    score_corpus,
)


DEFAULT_OUT = (
    Path(__file__).resolve().parent.parent.parent
    / "docs"
    / "experiments"
    / "phase-7-personalization"
    / "b4-import-to-profile.json"
)


def _serialise_report(report: B4Report) -> dict:
    return {
        "n_personas": report.n_personas,
        "n_messages_per_persona": report.n_messages_per_persona,
        "mae_per_dim_mean": {d.value: v for d, v in report.mae_per_dim_mean.items()},
        "mean_dims_within_half_likert": report.mean_dims_within_half_likert,
        "fraction_dims_within_half_likert_mean": report.fraction_dims_within_half_likert_mean,
        "pass_bar_dims_within_half_likert": report.pass_bar_dims_within_half_likert,
        "meets_pass_bar": report.meets_pass_bar(),
        "per_persona": [
            {
                "persona_id": r.persona_id,
                "n_messages": r.n_messages,
                "ground_truth": {d.value: v for d, v in r.ground_truth.items()},
                "inferred": {d.value: v for d, v in r.inferred.items()},
                "confidence": {d.value: v for d, v in r.confidence.items()},
                "mae_per_dim": {d.value: v for d, v in r.mae_per_dim.items()},
                "dims_within_half_likert": r.dims_within_half_likert,
            }
            for r in report.per_persona
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-messages",
        type=int,
        nargs="+",
        default=[50, 200, 500],
        help="Message-volume cells to measure.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output JSON path.",
    )
    args = parser.parse_args()

    personas = default_personas()
    cells: list[dict] = []
    overall_pass = True
    t0 = time.time()
    for n in args.n_messages:
        report = score_corpus(personas, n_messages=n, seed=args.seed)
        cell = _serialise_report(report)
        cells.append(cell)
        if n == 200 and not report.meets_pass_bar():
            overall_pass = False
        print(
            f"N={n:>4}  mean_dims_within_half_likert="
            f"{report.mean_dims_within_half_likert:.2f}  "
            f"MAE_mean={ {d.value: round(v, 2) for d, v in report.mae_per_dim_mean.items()} }"
        )
    wall = time.time() - t0

    out = {
        "phase": "phase-7",
        "benchmark": "b4-import-to-profile",
        "cells": cells,
        "overall_pass_at_n200": overall_pass,
        "wall_seconds": wall,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}  wall={wall:.1f}s")


if __name__ == "__main__":
    main()
