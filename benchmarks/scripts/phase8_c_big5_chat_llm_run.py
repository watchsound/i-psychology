"""Phase 8 Pillar C v4 — BIG5-CHAT classification via LLM semantic judge.

Reads the BIG5-CHAT shim produced by ``provision_big5_chat.py``, routes
each dialog's ``train_output`` through the LLM semantic judge
(``big5_chat_classification_llm.py``), and reports polarity accuracy on
the labeled OCEAN dimension. Mirrors ``phase8_c_big5_chat_run.py``
schema with two added fields: ``n_unresolved`` + ``routing``.

Pillar C soft gate (per
`docs/phases/phase-8-dataset-replay-expansion-plan.md` §2):
- BIG5-CHAT trait-condition-forward classification ≥ 60%

The lexical scorer (``big5_chat_classification.py``) is left untouched
as the audit baseline — this runner is the v4 experiment that
hypothesises the F4 LLM-judge intervention from the root-cause
analysis will cross the 0.60 floor by lifting coverage.

Usage::

    python benchmarks/scripts/phase8_c_big5_chat_llm_run.py \\
        --mode cloud --limit 200
"""

from __future__ import annotations

import argparse
import dataclasses as _dc
import datetime as _dt
import json
import os
import sys
import time
from pathlib import Path


sys.stdout.reconfigure(line_buffering=True)

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))

from psychology_base.eval.benchmarks.public_datasets.big5_chat import (  # noqa: E402
    iter_big5_chat_dialogs,
)
from psychology_base.eval.benchmarks.public_datasets.big5_chat_classification_llm import (  # noqa: E402
    make_llm_judge,
    score_dialogs_llm,
)
from psychology_base.llm.config import load_config  # noqa: E402
from psychology_base.llm.router import LLMRouter  # noqa: E402
from psychology_base.llm.types import TaskTier  # noqa: E402


DEFAULT_SHIM = REPO / ".data" / "big5_chat" / "train.jsonl"
DEFAULT_OUT = (
    REPO
    / "docs"
    / "experiments"
    / "phase-8-dataset-replay-expansion"
    / "pillar-c"
    / "big5-chat-classification-v4-llm.json"
)


def _config_with_overrides(mode: str):
    cfg = load_config()
    if mode == "local":
        cfg = _dc.replace(
            cfg,
            local_ollama=_dc.replace(
                cfg.local_ollama, enabled=True, force_cloud_chat=False
            ),
        )
    elif mode == "cloud":
        cfg = _dc.replace(
            cfg,
            local_ollama=_dc.replace(cfg.local_ollama, force_cloud_chat=True),
        )
    return cfg


def _build_router(mode: str, tier: TaskTier) -> tuple[LLMRouter, str]:
    cfg = _config_with_overrides(mode)
    if mode == "cloud":
        env = cfg.get_provider(cfg.active_provider).api_key_env
        if env and not os.environ.get(env):
            raise SystemExit(f"--mode cloud requires {env} set.")
    router = LLMRouter(cfg)
    dec = router.route(tier)[1]
    if mode == "local" and dec.provider != "ollama":
        raise SystemExit(
            f"--mode local but router routed to {dec.provider} ({dec.reason})."
        )
    return router, f"{dec.provider} / {dec.model_id} (reason={dec.reason})"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="phase8_c_big5_chat_llm_run")
    p.add_argument("--shim", type=Path, default=DEFAULT_SHIM)
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max dialogs to score. Default: all rows in the shim.",
    )
    p.add_argument("--mode", choices=("local", "cloud"), default="cloud")
    p.add_argument("--tier", choices=("simple", "advanced"), default="simple")
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable LLM response cache (default: cache enabled).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tier = TaskTier.ADVANCED if args.tier == "advanced" else TaskTier.SIMPLE

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

    router, routing_note = _build_router(args.mode, tier)
    print(f"routing: {routing_note}")

    judge = make_llm_judge(router, tier=tier, use_cache=not args.no_cache)

    t0 = time.time()
    report = score_dialogs_llm(dialogs, judge=judge, routing=routing_note)
    wall = time.time() - t0

    pass_soft = report.meets_soft_gate()
    print(
        f"\nn_input={len(dialogs)} n_dialogs={report.n_dialogs} "
        f"n_neutral={report.n_neutral} n_unresolved={report.n_unresolved} "
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
    if report.n_unresolved > 0:
        print(f"  unresolved_dialog_ids = {list(report.unresolved_dialog_ids)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "benchmark": "phase8_c_big5_chat_classification_v4_llm",
        "shim": str(args.shim),
        "limit": args.limit,
        "mode": args.mode,
        "tier": args.tier,
        "routing": routing_note,
        "n_input": len(dialogs),
        "n_dialogs": report.n_dialogs,
        "n_neutral": report.n_neutral,
        "n_unresolved": report.n_unresolved,
        "unresolved_dialog_ids": list(report.unresolved_dialog_ids),
        "coverage": report.coverage,
        "polarity_accuracy": report.polarity_accuracy,
        "polarity_accuracy_incl_neutral": report.polarity_accuracy_incl_neutral,
        "per_trait_polarity_accuracy": report.per_trait_polarity_accuracy,
        "per_level_polarity_accuracy": report.per_level_polarity_accuracy,
        "confusion": report.confusion,
        "soft_gate_floor": report.soft_gate_floor,
        "pillar_c_soft_pass": pass_soft,
        "wall_s": round(wall, 1),
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    args.output.write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
