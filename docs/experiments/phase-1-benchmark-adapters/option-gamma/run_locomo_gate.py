"""Real-data Phase-1-exit gate for LoCoMo.

Companion to ``run_locomo_smoke.py``. The smoke proves the pipeline on
a 2-instance fixture; this proves the floor on the real 10-instance
snap-research/locomo locomo10.json shim at
``~/.cache/psychology_base/benchmarks/locomo/locomo_shim.jsonl``.

**Gate (per docs/roadmap.md + plan §5):** LoCoMo single-hop F1 ≥ 0.90.
Mem0's published number is 92.5% single-hop F1 on the same split. We
report α-direct and γ-RAG side-by-side; gate decision is on γ-RAG
single-hop (α-direct is a floor-check on substrate quality, not the
decision substrate).

**Cost shape.** Single-hop uses deterministic token F1 — no judge calls
on the gate category. γ-RAG runs 282 reader calls on the full set
(Tongyi qwen-max). Direct mode is substrate-only (no LLM). First-run
estimate: ~15-30 min wall time, ~$2-3 reader cost.

Flags:
  --instance-limit N   cap to first N instances (staged runs)
  --categories a,b,c   comma-separated subset (default: single-hop only;
                       pass "all" to include every category)
  --mode direct|rag|both  default: both

Run (gate):
    ALIBABA_TONGYI_API_KEY=... python docs/experiments/phase-1-benchmark-adapters/option-gamma/run_locomo_gate.py

Run (fast triage, direct-only, 2 instances):
    python docs/experiments/phase-1-benchmark-adapters/option-gamma/run_locomo_gate.py \\
        --mode direct --instance-limit 2
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from psychology_base.companion.system import CompanionSystem  # noqa: E402
from psychology_base.eval.benchmarks.judge import JudgmentCache, LLMJudge  # noqa: E402
from psychology_base.eval.benchmarks.locomo import (  # noqa: E402
    LoCoMoLoader,
    run_direct_mode,
    run_rag_mode,
)
from psychology_base.eval.benchmarks.reader import LLMReader  # noqa: E402
from psychology_base.llm.router import router_from_config  # noqa: E402
from psychology_base.llm.types import TaskTier  # noqa: E402


DEFAULT_SHIM = (
    Path.home() / ".cache" / "psychology_base" / "benchmarks" / "locomo" / "locomo_shim.jsonl"
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run_locomo_gate")
    p.add_argument("--shim-path", type=Path, default=DEFAULT_SHIM)
    p.add_argument(
        "--instance-limit",
        type=int,
        default=None,
        help="Cap to first N instances (staged runs). Default: all.",
    )
    p.add_argument(
        "--categories",
        default="single-hop",
        help=(
            "Comma-separated subset. Default 'single-hop' (gate category, "
            "F1-only, no judge cost). Pass 'all' to run every category."
        ),
    )
    p.add_argument(
        "--mode",
        choices=("direct", "rag", "both"),
        default="both",
    )
    p.add_argument(
        "--consolidation",
        action="store_true",
        help=(
            "Phase 1.5 Option 5.B — enable the Consolidator layer (facts "
            "extracted on session boundaries; recall dispatches facts-first "
            "with fragment back-off). Default: off (Phase 1 behavior)."
        ),
    )
    p.add_argument(
        "--instance-id",
        default=None,
        help=(
            "Filter to a specific instance_id (e.g. 'conv-26' for the E1 "
            "prototype). Default: no filter."
        ),
    )
    p.add_argument(
        "--gate-target",
        type=float,
        default=0.90,
        help=(
            "Gate pass threshold on the gate category. Phase 1.5 E1 "
            "softens this to 0.40 (conv-26 prototype) and E2 to 0.70 "
            "(full 10-instance Mem0-parity target)."
        ),
    )
    p.add_argument(
        "--embedder-ensemble",
        action="store_true",
        help=(
            "Phase 1.5-prime P2: dual-backend recall. Every candidate "
            "(fact + fragment) is scored by both local Ollama (mxbai) and "
            "the active cloud provider (DashScope text-embedding-v3), and "
            "the final cosine is max(local, cloud). Targets Bucket A "
            "proper-noun failures (Sweden/Single/Adoption) that mxbai "
            "alone buries. Requires consolidation."
        ),
    )
    p.add_argument(
        "--cross-encoder-rerank",
        action="store_true",
        help=(
            "Phase 1.5-prime P3: after cosine recall, rerank the top-K "
            "fragments through the active provider's cross-encoder "
            "(DashScope gte-rerank-v1) and feed the reranked top-N to the "
            "reader. Targets paraphrase-cue → gold-fragment matches that "
            "bi-encoder cosine alone misses."
        ),
    )
    p.add_argument(
        "--judge-single-hop",
        action="store_true",
        help=(
            "Phase 1.5-prime P1: force single-hop through LLMJudge instead "
            "of token F1. Captures paraphrase-correct answers that token "
            "F1 scores 0.0 (e.g. gold 'Roast marshmallows, tell stories' "
            "vs reader 'exploring, hiking' — reader has the context, "
            "scorer is strict). Costs ~32 ADVANCED-tier calls per "
            "instance; use with --instance-id for E1 cost control."
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=HERE / "locomo-gate.json",
    )
    return p


def _filter_instance(instance, allowed: set[str] | None):
    if allowed is None:
        return instance
    kept = tuple(q for q in instance.questions if q.question_type in allowed)
    if not kept:
        return None
    return instance.__class__(
        name=instance.name,
        instance_id=instance.instance_id,
        history=instance.history,
        questions=kept,
        metadata=instance.metadata,
    )


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _by_cat(per_instance) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for r in per_instance:
        for qr in r.per_question:
            buckets.setdefault(qr.category or "unknown", []).append(qr.score)
    return {c: _mean(xs) for c, xs in buckets.items()}


def _incorrect_sample(per_instance, limit: int = 20):
    out = []
    for r in per_instance:
        for qr in r.per_question:
            if qr.score < 0.5:
                out.append(
                    {
                        "instance": r.instance_id,
                        "question_id": qr.question_id,
                        "category": qr.category,
                        "gold": qr.gold,
                        "candidate": qr.candidate[:160],
                        "score": round(qr.score, 3),
                        "verdict": qr.verdict,
                    }
                )
                if len(out) >= limit:
                    return out
    return out


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.shim_path.exists():
        print(f"missing shim at {args.shim_path}", file=sys.stderr)
        print(
            "Run: python -m psychology_base.eval.scripts.convert_locomo "
            "--input <locomo10.json> --output <shim.jsonl>",
            file=sys.stderr,
        )
        return 1

    if args.categories.strip().lower() == "all":
        allowed = None
    else:
        allowed = {c.strip() for c in args.categories.split(",") if c.strip()}

    router = router_from_config()
    _, decision = router.route(TaskTier.ADVANCED)
    print(f"ADVANCED tier → {decision.provider}:{decision.model_id} ({decision.reason})")
    if args.consolidation:
        _, simple_decision = router.route(TaskTier.SIMPLE)
        print(
            f"SIMPLE tier → {simple_decision.provider}:{simple_decision.model_id} "
            f"({simple_decision.reason}) [consolidation ON]"
        )

    judge = LLMJudge(router=router)
    reader = LLMReader(router=router)
    judgment_cache = JudgmentCache()

    loader = LoCoMoLoader(shim_path=args.shim_path)
    instances_all = list(loader.iter_instances())
    if args.instance_id:
        instances_all = [i for i in instances_all if i.instance_id == args.instance_id]
        if not instances_all:
            print(f"no instance matched instance_id={args.instance_id!r}", file=sys.stderr)
            return 1
    if args.instance_limit:
        instances_all = instances_all[: args.instance_limit]

    filtered = [_filter_instance(i, allowed) for i in instances_all]
    instances = [i for i in filtered if i is not None]
    skipped = len(filtered) - len(instances)

    total_q = sum(len(i.questions) for i in instances)
    print(
        f"loaded {len(instances)}/{len(instances_all)} instances "
        f"(skipped {skipped} with no matching questions); "
        f"total {total_q} questions "
        f"(categories={sorted(allowed) if allowed else 'all'})"
    )

    # Per-instance audit log so a FAIL run can be post-mortemed from the
    # extracted facts without re-running. Path is deterministic across
    # runs; a re-run of the same instance_id overwrites in-place rather
    # than accruing stale logs. Only wired when consolidation is on.
    fact_log_dir = (
        Path.home() / ".cache" / "psychology_base" / "phase-1.5" / "facts"
        if args.consolidation
        else None
    )

    def factory(instance_id: str = "default") -> CompanionSystem:
        return CompanionSystem(
            seed=0,
            witness_active=False,
            consolidation_enabled=args.consolidation,
            consolidation_router=router if args.consolidation else None,
            consolidation_log_dir=fact_log_dir,
            consolidation_instance_id=instance_id,
            embedder_ensemble=args.embedder_ensemble,
            cross_encoder_rerank=args.cross_encoder_rerank,
        )

    per_direct = []
    per_rag = []
    t_start = time.time()
    for i, inst in enumerate(instances):
        t_inst = time.time()
        line = f"  [{i + 1}/{len(instances)}] {inst.instance_id}  q={len(inst.questions)}"
        # Bind the current instance_id so the Consolidator audit log
        # file name matches the LoCoMo instance (not "default").
        inst_factory = lambda inst_id=inst.instance_id: factory(inst_id)
        force_judge = {"single-hop"} if args.judge_single_hop else None
        if args.mode in ("direct", "both"):
            d = run_direct_mode(
                inst,
                system_factory=inst_factory,
                judge=judge,
                judgment_cache=judgment_cache,
                force_judge_categories=force_judge,
            )
            per_direct.append(d)
            line += f"  direct={d.mean_score:.3f}"
        if args.mode in ("rag", "both"):
            r = run_rag_mode(
                inst,
                system_factory=inst_factory,
                reader=reader,
                judge=judge,
                judgment_cache=judgment_cache,
                force_judge_categories=force_judge,
            )
            per_rag.append(r)
            line += f"  rag={r.mean_score:.3f}"
        line += f"  ({time.time() - t_inst:.1f}s)"
        print(line)

    wall = time.time() - t_start

    direct_scores = [qr.score for r in per_direct for qr in r.per_question]
    rag_scores = [qr.score for r in per_rag for qr in r.per_question]
    direct_by_cat = _by_cat(per_direct) if per_direct else {}
    rag_by_cat = _by_cat(per_rag) if per_rag else {}

    gate_target = args.gate_target
    rag_single_hop = rag_by_cat.get("single-hop")
    direct_single_hop = direct_by_cat.get("single-hop")
    gate_mode = "rag" if rag_single_hop is not None else "direct"
    gate_value = rag_single_hop if rag_single_hop is not None else direct_single_hop
    gate_pass = gate_value is not None and gate_value >= gate_target

    report = {
        "benchmark": "locomo",
        "split": "real-full" if args.instance_limit is None else f"real-first{args.instance_limit}",
        "shim_path": str(args.shim_path),
        "instance_count": len(instances),
        "instance_id_filter": args.instance_id,
        "question_count": total_q,
        "categories_filter": sorted(allowed) if allowed else "all",
        "mode": args.mode,
        "consolidation_enabled": args.consolidation,
        "judge_single_hop": args.judge_single_hop,
        "embedder_ensemble": args.embedder_ensemble,
        "cross_encoder_rerank": args.cross_encoder_rerank,
        "advanced_tier": {
            "provider": decision.provider,
            "model_id": decision.model_id,
            "reason": decision.reason,
        },
        "direct_mode": None if not per_direct else {
            "mean_score": _mean(direct_scores),
            "mean_by_category": direct_by_cat,
            "incorrect_sample": _incorrect_sample(per_direct),
        },
        "rag_mode": None if not per_rag else {
            "mean_score": _mean(rag_scores),
            "mean_by_category": rag_by_cat,
            "incorrect_sample": _incorrect_sample(per_rag),
        },
        "gate": {
            "target": gate_target,
            "mode": gate_mode,
            "category": "single-hop",
            "value": gate_value,
            "pass": gate_pass,
        },
        "judge": {
            "judge_id": judge.judge_id,
            "prompt_sha": judge.prompt_sha,
            "cache_hits": judgment_cache.hits,
            "cache_misses": judgment_cache.misses,
        },
        "wall_time_s": round(wall, 1),
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
    }

    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nwrote {args.output}")
    print(f"wall={wall:.1f}s  cache hits={judgment_cache.hits} misses={judgment_cache.misses}")
    if per_direct:
        print(f"direct  mean={_mean(direct_scores):.3f}  by_cat={direct_by_cat}")
    if per_rag:
        print(f"rag     mean={_mean(rag_scores):.3f}  by_cat={rag_by_cat}")
    print(
        f"gate ({gate_mode} single-hop ≥ {gate_target}): "
        f"{gate_value!r} → {'PASS' if gate_pass else 'FAIL'}"
    )
    return 0 if gate_pass else 2


if __name__ == "__main__":
    sys.exit(main())
