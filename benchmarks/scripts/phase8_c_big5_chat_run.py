"""Phase 8 Pillar C soft gate — BIG5-CHAT trait classification live-fire.

Reads the BIG5-CHAT shim produced by ``provision_big5_chat.py``, runs
every dialog through the Phase 7 trait-inference pipeline, and reports
polarity-classification accuracy on the labeled OCEAN dimension.

Pillar C soft bar (per
`docs/phases/phase-8-dataset-replay-expansion-plan.md` §2):
- BIG5-CHAT trait-condition-forward classification ≥ 60%

No LLM dependency — the scorer is pure Python over the lexical cue bank
+ TraitProfile reconstruction. A full 100-dialog scoring finishes in a
couple seconds on a laptop CPU.

Usage::

    python benchmarks/scripts/phase8_c_big5_chat_run.py --limit 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


# Line-buffered stdout keeps progress visible when a parent harness pipes
# to a file or tails the log.
sys.stdout.reconfigure(line_buffering=True)

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))

from psychology_base.eval.benchmarks.public_datasets.big5_chat import (  # noqa: E402
    iter_big5_chat_dialogs,
)
from psychology_base.eval.benchmarks.public_datasets.big5_chat_classification import (  # noqa: E402
    score_dialogs,
)


DEFAULT_SHIM = REPO / ".data" / "big5_chat" / "train.jsonl"
DEFAULT_OUT = (
    REPO
    / "docs"
    / "experiments"
    / "phase-8-dataset-replay-expansion"
    / "pillar-c"
    / "big5-chat-classification.json"
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="phase8_c_big5_chat_run")
    p.add_argument("--shim", type=Path, default=DEFAULT_SHIM)
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max dialogs to score. Default: all rows in the shim.",
    )
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.shim.exists():
        raise SystemExit(
            f"Shim not found at {args.shim}; run provision_big5_chat.py first."
        )
    dialogs = []
    for dialog in iter_big5_chat_dialogs(args.shim):
        dialogs.append(dialog)
        if args.limit is not None and len(dialogs) >= args.limit:
            break
    print(f"Loaded {len(dialogs)} BIG5-CHAT dialogs from {args.shim}")

    t0 = time.time()
    report = score_dialogs(dialogs)
    wall = time.time() - t0

    pass_soft = report.meets_soft_gate()
    print(
        f"\nn_dialogs={report.n_dialogs} n_neutral={report.n_neutral} "
        f"coverage={report.coverage:.3f}"
    )
    print(
        f"  polarity_accuracy={report.polarity_accuracy:.3f} "
        f"(signed only; neutral excluded)"
    )
    print(
        f"  polarity_accuracy_incl_neutral={report.polarity_accuracy_incl_neutral:.3f} "
        f"(gate metric)"
    )
    print("  per-trait polarity accuracy:")
    for trait, acc in sorted(report.per_trait_polarity_accuracy.items()):
        print(f"    {trait:16s} {acc:.3f}")
    print("  per-level polarity accuracy:")
    for level, acc in sorted(report.per_level_polarity_accuracy.items()):
        print(f"    {level:6s} {acc:.3f}")
    print("  confusion (labeled → predicted):")
    for trait, row in sorted(report.confusion.items()):
        print(f"    {trait:16s} {row}")
    print(f"  pillar_c_soft_PASS = {pass_soft} (floor accuracy ≥ 0.60)")
    print(f"  wall = {wall:.1f}s")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "benchmark": "phase8_c_big5_chat_classification",
        "shim": str(args.shim),
        "limit": args.limit,
        "n_dialogs": report.n_dialogs,
        "n_neutral": report.n_neutral,
        "coverage": report.coverage,
        "polarity_accuracy": report.polarity_accuracy,
        "polarity_accuracy_incl_neutral": report.polarity_accuracy_incl_neutral,
        "per_trait_polarity_accuracy": report.per_trait_polarity_accuracy,
        "per_level_polarity_accuracy": report.per_level_polarity_accuracy,
        "confusion": report.confusion,
        "soft_gate_floor": report.soft_gate_floor,
        "pillar_c_soft_pass": pass_soft,
        "wall_s": round(wall, 1),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
