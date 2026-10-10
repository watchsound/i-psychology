"""Phase 8 Pillar E — higher-order ToM live-fire runner (BigToM + HiToM).

Scores a batch of items from the BigToM and/or HiToM provisioner shims
on BOTH the mentalistic-routed path and the bare-LLM baseline, grading
answers through the LLM FANToM judge (or lexical for CI determinism).

**Pillar E hard gate** (plan §2): HiToM order 2 at ``min_items=50``
clears ``accuracy_mentalistic ≥ 0.40`` AND
``accuracy_mentalistic − accuracy_bare_llm ≥ +0.30``.

**Soft criterion:** HiToM order 3 passes the same bar.

**Sanity cross-check:** BigToM order 1 should show a non-zero positive
delta (floor not formally gated; reported for close-out).

Usage::

    # The gate run (HiToM order 2, N=100)
    python benchmarks/scripts/phase8_e_higher_order_tom_run.py \
        --dataset hitom --order 2 --limit 100 \
        --mode cloud --tier simple --judge llm --responder both

    # BigToM order 1 sanity
    python benchmarks/scripts/phase8_e_higher_order_tom_run.py \
        --dataset bigtom --order 1 --limit 100

    # Full HiToM sweep (orders 1,2,3,4 at N=100 each)
    python benchmarks/scripts/phase8_e_higher_order_tom_run.py \
        --dataset hitom --order all --limit-per-order 100
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

from psychology_base.eval.benchmarks.public_datasets.bigtom import (  # noqa: E402
    BigToMScenarioCondition,
    bigtom_item_to_fantom_item,
    iter_bigtom_items,
)
from psychology_base.eval.benchmarks.public_datasets.fantom_tom_scorer import (  # noqa: E402
    LLMFANToMJudge,
    lexical_answer_judge,
)
from psychology_base.eval.benchmarks.public_datasets.higher_order_tom_scorer import (  # noqa: E402
    score_higher_order_tom,
)
from psychology_base.eval.benchmarks.public_datasets.hitom import (  # noqa: E402
    HiToMOrder,
    hitom_item_to_fantom_item,
    iter_hitom_items,
)
from psychology_base.llm.config import load_config  # noqa: E402
from psychology_base.llm.router import LLMRouter  # noqa: E402
from psychology_base.llm.types import ChatMessage, TaskTier  # noqa: E402


DEFAULT_BIGTOM_SHIM = REPO / ".data" / "bigtom" / "items.jsonl"
DEFAULT_HITOM_SHIM = REPO / ".data" / "hitom" / "items.jsonl"
DEFAULT_OUT = (
    REPO
    / "docs"
    / "experiments"
    / "phase-8-dataset-replay-expansion"
    / "pillar-e"
    / "higher-order-tom.json"
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


def _llm_responder_factory(router: LLMRouter, tier: TaskTier):
    def _call(prompt: str) -> str:
        try:
            completion = router.chat(
                [ChatMessage(role="user", content=prompt)], tier=tier
            )
            return completion.content
        except Exception as exc:  # pragma: no cover - defensive
            return f"[responder_error: {exc}]"

    return _call


_ORDER_CHOICES = ("0", "1", "2", "3", "4", "all")


def _parse_orders(spec: str) -> tuple[HiToMOrder, ...]:
    if spec == "all":
        return tuple(HiToMOrder)
    return (HiToMOrder.from_int(int(spec)),)


def _collect_items(args) -> tuple[list, dict[str, int]]:
    """Pool items from the requested dataset(s)/order(s)."""

    pooled = []
    stats: dict[str, int] = {}

    orders = _parse_orders(args.order)
    per_order_cap = args.limit_per_order if args.limit_per_order else args.limit

    if args.dataset in ("hitom", "both"):
        for order in orders:
            n = 0
            for hit in iter_hitom_items(args.hitom_shim, order=order):
                pooled.append(hitom_item_to_fantom_item(hit))
                n += 1
                if per_order_cap and n >= per_order_cap:
                    break
            stats[f"hitom_order_{int(order)}"] = n

    if args.dataset in ("bigtom", "both"):
        condition = (
            BigToMScenarioCondition.from_shim(args.bigtom_condition)
            if args.bigtom_condition
            else None
        )
        n = 0
        cap = per_order_cap or args.limit
        for bit in iter_bigtom_items(args.bigtom_shim, scenario_condition=condition):
            pooled.append(bigtom_item_to_fantom_item(bit))
            n += 1
            if cap and n >= cap:
                break
        stats[
            f"bigtom_order_1_{args.bigtom_condition or 'all_conditions'}"
        ] = n

    return pooled, stats


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="phase8_e_higher_order_tom_run")
    p.add_argument(
        "--dataset",
        choices=("bigtom", "hitom", "both"),
        default="hitom",
        help="Which sub-corpus to score. HiToM is the gated corpus.",
    )
    p.add_argument(
        "--order",
        choices=_ORDER_CHOICES,
        default="2",
        help="HiToM order to score (ignored for BigToM, which is always order 1).",
    )
    p.add_argument(
        "--bigtom-shim", type=Path, default=DEFAULT_BIGTOM_SHIM,
    )
    p.add_argument(
        "--hitom-shim", type=Path, default=DEFAULT_HITOM_SHIM,
    )
    p.add_argument(
        "--bigtom-condition",
        type=str,
        default=None,
        help="Restrict BigToM to one scenario_condition (e.g. "
        "forward_belief_unaware). Default: no filter.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Max items per order (HiToM) or total (BigToM single-order).",
    )
    p.add_argument(
        "--limit-per-order",
        type=int,
        default=None,
        help="Overrides --limit for multi-order runs (--order all). "
        "If unset, falls back to --limit.",
    )
    p.add_argument(
        "--mode",
        choices=("local", "cloud", "stub"),
        default="cloud",
    )
    p.add_argument(
        "--tier",
        choices=("simple", "advanced"),
        default="simple",
    )
    p.add_argument(
        "--responder",
        choices=("bare", "mentalistic", "both"),
        default="both",
        help="'both' = run both paths and report delta; "
        "'bare' or 'mentalistic' = single-path diagnostic.",
    )
    p.add_argument(
        "--judge",
        choices=("llm", "lexical"),
        default="llm",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUT,
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Reserved — not currently consumed by the Pillar E scorer.",
    )
    return p


def _paths_from_responder(responder_arg: str) -> tuple[str, ...]:
    if responder_arg == "bare":
        return ("bare_llm",)
    if responder_arg == "mentalistic":
        return ("mentalistic",)
    return ("mentalistic", "bare_llm")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tier = TaskTier.ADVANCED if args.tier == "advanced" else TaskTier.SIMPLE

    if args.dataset in ("hitom", "both") and not args.hitom_shim.exists():
        raise SystemExit(
            f"HiToM shim not found at {args.hitom_shim}; run provision_hitom.py first."
        )
    if args.dataset in ("bigtom", "both") and not args.bigtom_shim.exists():
        raise SystemExit(
            f"BigToM shim not found at {args.bigtom_shim}; run provision_bigtom.py first."
        )

    items, stats = _collect_items(args)
    print(f"Pooled {len(items)} items")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    if args.mode == "stub":
        responder = lambda prompt: "I do not know."  # noqa: E731
        judge = lexical_answer_judge
        routing_note = "stub (constant responder + lexical judge)"
        judge_name = "lexical"
        responder_name = "stub_constant"
    else:
        router, routing_note = _build_router(args.mode, tier)
        responder = _llm_responder_factory(router, tier)
        responder_name = f"{args.mode}/{args.tier}"
        if args.judge == "llm":
            judge = LLMFANToMJudge(router=router, tier=tier)
            judge_name = f"llm_{args.tier}"
        else:
            judge = lexical_answer_judge
            judge_name = "lexical"
    print(f"responder: {routing_note}")
    print(f"judge: {judge_name}")

    paths = _paths_from_responder(args.responder)
    dataset_tag = args.dataset

    t0 = time.time()
    report = score_higher_order_tom(
        items,
        responder=responder,
        judge=judge,
        judge_name=judge_name,
        responder_name=responder_name,
        paths=paths,
        dataset_tag=dataset_tag,
    )
    wall = time.time() - t0

    # Per-order summary (collect every order that appears in the pool)
    present_orders = sorted({v for v in report.cid_to_order.values() if v >= 0})

    def _summarise(order: int | None):
        return {
            "order": order if order is not None else "all",
            "n_mentalistic": report.item_count(path="mentalistic", order=order),
            "n_bare_llm": report.item_count(path="bare_llm", order=order),
            "accuracy_mentalistic": report.accuracy(path="mentalistic", order=order),
            "accuracy_bare_llm": report.accuracy(path="bare_llm", order=order),
            "architectural_delta": report.architectural_delta(order=order),
            "undecidable_rate_mentalistic": report.undecidable_rate(
                path="mentalistic", order=order
            ),
            "undecidable_rate_bare_llm": report.undecidable_rate(
                path="bare_llm", order=order
            ),
        }

    per_order = [_summarise(o) for o in present_orders] + [_summarise(None)]
    pillar_e_pass = report.meets_pillar_e_gate(order=2, min_items=50)

    print(f"\n=== Results ({wall:.1f}s wall) ===")
    for row in per_order:
        print(
            f"  [order={row['order']}] "
            f"n_mental={row['n_mentalistic']} n_bare={row['n_bare_llm']} "
            f"acc_mental={row['accuracy_mentalistic']:.3f} "
            f"acc_bare={row['accuracy_bare_llm']:.3f} "
            f"delta={row['architectural_delta']:+.3f} "
            f"undec={row['undecidable_rate_mentalistic']:.2f}/"
            f"{row['undecidable_rate_bare_llm']:.2f}"
        )
    print(
        f"\npillar_e_gate(acc_mental≥0.40 ∧ delta≥0.30 ∧ n≥50 at HiToM order 2) "
        f"= {pillar_e_pass}"
    )

    out = {
        "benchmark": "phase8_e_higher_order_tom",
        "dataset": args.dataset,
        "mode": args.mode,
        "tier": args.tier,
        "routing": routing_note,
        "bigtom_shim": str(args.bigtom_shim) if args.dataset != "hitom" else None,
        "hitom_shim": str(args.hitom_shim) if args.dataset != "bigtom" else None,
        "order_requested": args.order,
        "bigtom_condition_filter": args.bigtom_condition,
        "items_pooled": len(items),
        "per_dataset_counts": stats,
        "verdicts_total": len(report.base_report.verdicts),
        "per_order": per_order,
        "pillar_e_gate_pass": pillar_e_pass,
        "judge_name": judge_name,
        "responder_name": responder_name,
        "paths": list(paths),
        "wall_s": round(wall, 1),
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
