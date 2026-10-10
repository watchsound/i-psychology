"""Phase 7 §6 item 9 — B5 stereotype-collapse run.

Measures whether trait conditioning collapses response space when
10 varied TraitProfile seeds are fed 20 benign prompts.

Pass bar per plan §6: conditioned diversity ≥ 0.70 (theoretical max 1.0).

Writes JSON to
``docs/experiments/phase-7-personalization/b5-stereotype-collapse.json``.

No LLM — the responder is a deterministic SHA-of-config surrogate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from psychology_base.eval.benchmarks.b5_stereotype_collapse import (
    B5Report,
    BENIGN_PROMPTS,
    default_seeds,
    score_corpus,
)


DEFAULT_OUT = (
    Path(__file__).resolve().parent.parent.parent
    / "docs"
    / "experiments"
    / "phase-7-personalization"
    / "b5-stereotype-collapse.json"
)


def _serialise(report: B5Report) -> dict:
    return {
        "n_seeds": report.n_seeds,
        "n_prompts": report.n_prompts,
        "unique_responses_baseline": report.unique_responses_baseline,
        "unique_responses_conditioned": report.unique_responses_conditioned,
        "unique_addenda": report.unique_addenda,
        "diversity_baseline": report.diversity_baseline,
        "diversity_conditioned": report.diversity_conditioned,
        "pass_bar_floor": 0.70,
        "meets_pass_bar": report.meets_pass_bar(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    seeds = default_seeds()
    print(f"phase7 B5 stereotype-collapse: seeds={len(seeds)} prompts={len(BENIGN_PROMPTS)}")
    report = score_corpus(seeds)
    payload = _serialise(report)
    print(
        f"  diversity_baseline={report.diversity_baseline:.3f} "
        f"(expected ≈ {1 / report.n_seeds:.2f})",
        flush=True,
    )
    print(
        f"  diversity_conditioned={report.diversity_conditioned:.3f} "
        f"(floor 0.70)",
        flush=True,
    )
    print(
        f"  unique_addenda={report.unique_addenda} / {report.n_seeds} seeds",
        flush=True,
    )
    verdict = (
        "B5 PASS — conditioning does not collapse response space (≥ 0.70)"
        if report.meets_pass_bar()
        else "B5 FAIL — conditioning collapses response space below 0.70 floor"
    )

    out = {
        "experiment": "phase-7-b5-stereotype-collapse",
        "report": payload,
        "verdict": verdict,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  - {verdict}")
    print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()
