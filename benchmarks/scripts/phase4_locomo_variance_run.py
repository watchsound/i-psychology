"""Phase 4a §6d — LoCoMo monitor multi-trial variance study.

Converts the Phase 1.5 single-trial LoCoMo monitor (R5 F1 = 0.544 point
estimate) into a confidence-bounded floor by running N trials over the
same 3-instance gate (conv-26 / conv-30 / conv-41) and aggregating with
the generic :func:`run_multi_trial` harness.

Scope. Pilot reports need to distinguish *regressions* from *LLM-induced
noise* — a single-trial 0.544 can slide to 0.510 purely from reader
temperature, and the monitor must know that is noise, not a mechanism
regression. This script produces per-instance :class:`VarianceReport`
with 95% CI so the pilot team can quote "0.514 CI-lower is held" rather
than "0.544 single-trial is held".

Modes.
- ``--mode direct`` (default): no reader, substrate-only recall. The
  substrate is deterministic → variance will be zero. Useful as a
  wiring smoke; proves the harness composes without burning API $$.
- ``--mode rag``: reader-mediated runs through
  :class:`psychology_base.eval.benchmarks.reader.LLMReader`. Each trial
  rebuilds the ``CompanionSystem`` via ``system_factory`` and makes
  real LLM calls to the active cloud provider. **This burns tokens**:
  rough estimate 3 instances × ~94 single-hop Q × N trials × qwen-max.
  Cache is default-on (``configs/llm.yaml``) and SHA-keyed on the full
  prompt, so cache hits dominate re-runs with same inputs. First run
  at ``temperature>0`` will miss on every call.

Usage:
    # Wiring smoke (free):
    python benchmarks/scripts/phase4_locomo_variance_run.py \\
        --mode direct --n-trials 3

    # Real monitor update (needs ALIBABA_TONGYI_API_KEY):
    ALIBABA_TONGYI_API_KEY=... python benchmarks/scripts/phase4_locomo_variance_run.py \\
        --mode rag --n-trials 3 --consolidation --cross-encoder-rerank
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))

from psychology_base.companion.system import CompanionSystem  # noqa: E402
from psychology_base.eval.benchmarks.judge import JudgmentCache, LLMJudge  # noqa: E402
from psychology_base.eval.benchmarks.locomo import (  # noqa: E402
    LoCoMoLoader,
    run_direct_mode,
    run_rag_mode,
)
from psychology_base.eval.benchmarks.reader import LLMReader  # noqa: E402
from psychology_base.eval.benchmarks.variance_study import run_multi_trial  # noqa: E402
from psychology_base.llm.router import router_from_config  # noqa: E402
from psychology_base.llm.types import TaskTier  # noqa: E402


DEFAULT_SHIM = (
    Path.home() / ".cache" / "psychology_base" / "benchmarks" / "locomo" / "locomo_shim.jsonl"
)
GATE_INSTANCES = ("conv-26", "conv-30", "conv-41")
DEFAULT_OUT = REPO / "docs" / "experiments" / "phase-1-benchmark-adapters" / "option-gamma" / "locomo-variance.json"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="phase4_locomo_variance_run")
    p.add_argument("--shim-path", type=Path, default=DEFAULT_SHIM)
    p.add_argument(
        "--mode",
        choices=("direct", "rag"),
        default="direct",
        help=(
            "direct = substrate-only (no reader, no API cost, zero "
            "variance expected). rag = reader-mediated (real LLM calls, "
            "variance comes from reader temperature)."
        ),
    )
    p.add_argument(
        "--n-trials",
        type=int,
        default=3,
        help="Number of trials per instance. N<10 → CI warning surfaced in report.",
    )
    p.add_argument(
        "--instances",
        default=",".join(GATE_INSTANCES),
        help="Comma-separated instance_ids. Default = Phase 1.5 3-instance gate.",
    )
    p.add_argument(
        "--consolidation",
        action="store_true",
        help="Phase 1.5 Option 5.B — Consolidator layer (facts on session boundaries).",
    )
    p.add_argument(
        "--cross-encoder-rerank",
        action="store_true",
        help="Phase 1.5-prime P3 — rerank top-K through cross-encoder.",
    )
    p.add_argument(
        "--full-session-reader",
        action="store_true",
        help="Phase 2b Step F — reader sees the full session, not top-K.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Where to write the variance report. Default: {DEFAULT_OUT}",
    )
    return p


def _load_instances(shim_path: Path, instance_ids: tuple[str, ...]):
    loader = LoCoMoLoader(shim_path=shim_path)
    all_instances = {inst.instance_id: inst for inst in loader.iter_instances()}
    missing = [i for i in instance_ids if i not in all_instances]
    if missing:
        raise SystemExit(f"instances not found in shim: {missing}")
    instances = [all_instances[i] for i in instance_ids]
    # Filter to single-hop (the monitor category).
    filtered = []
    for inst in instances:
        kept = tuple(q for q in inst.questions if q.question_type == "single-hop")
        if not kept:
            continue
        filtered.append(
            inst.__class__(
                name=inst.name,
                instance_id=inst.instance_id,
                history=inst.history,
                questions=kept,
                metadata=inst.metadata,
            )
        )
    return filtered


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    instance_ids = tuple(s.strip() for s in args.instances.split(",") if s.strip())
    instances = _load_instances(args.shim_path, instance_ids)
    print(f"loaded {len(instances)} instances; single-hop Q={sum(len(i.questions) for i in instances)}")

    needs_cloud = args.mode == "rag"
    if needs_cloud and not os.environ.get("ALIBABA_TONGYI_API_KEY"):
        print(
            "ERROR: --mode rag requires ALIBABA_TONGYI_API_KEY. "
            "Set it or run --mode direct for a free wiring smoke."
        )
        return 2

    router = router_from_config()
    judge = LLMJudge(router=router)
    judgment_cache = JudgmentCache()
    reader = LLMReader(router=router) if args.mode == "rag" else None

    def system_factory():
        return CompanionSystem(
            seed=0,
            witness_active=False,
            consolidation_enabled=args.consolidation,
            consolidation_router=router if args.consolidation else None,
            cross_encoder_rerank=args.cross_encoder_rerank,
            full_session_reader=args.full_session_reader,
        )

    def run_fn_direct(*, instance, system_factory, **_):
        return run_direct_mode(
            instance,
            system_factory=system_factory,
            judge=judge,
            judgment_cache=judgment_cache,
        )

    def run_fn_rag(*, instance, system_factory, **_):
        return run_rag_mode(
            instance,
            system_factory=system_factory,
            reader=reader,
            judge=judge,
            judgment_cache=judgment_cache,
        )

    run_fn = run_fn_rag if args.mode == "rag" else run_fn_direct

    per_instance_reports = []
    t_start = time.time()
    for i, inst in enumerate(instances):
        t_inst = time.time()
        print(f"  [{i + 1}/{len(instances)}] {inst.instance_id}  Q={len(inst.questions)}  trials={args.n_trials}")
        report = run_multi_trial(
            instance=inst,
            system_factory=system_factory,
            run_fn=run_fn,
            n_trials=args.n_trials,
        )
        per_instance_reports.append(report)
        print(
            f"    mean={report.instance_mean:.3f}  "
            f"stderr={report.instance_stderr:.3f}  "
            f"CI95=[{report.ci95_lower:.3f}, {report.ci95_upper:.3f}]  "
            f"({time.time() - t_inst:.1f}s)"
        )

    wall = time.time() - t_start

    aggregate_mean = sum(r.instance_mean for r in per_instance_reports) / len(per_instance_reports)
    # Pool per-question across instances: average of per-instance mean (equal weight).
    # Lower-bound aggregate CI via the min of per-instance CI lower bounds (conservative).
    aggregate_ci_lower = min(r.ci95_lower for r in per_instance_reports)
    monitor_floor = 0.514
    meets_floor = aggregate_ci_lower >= monitor_floor

    report = {
        "benchmark": "locomo",
        "scope": "single-hop variance monitor",
        "mode": args.mode,
        "n_trials": args.n_trials,
        "instance_ids": list(instance_ids),
        "consolidation": args.consolidation,
        "cross_encoder_rerank": args.cross_encoder_rerank,
        "full_session_reader": args.full_session_reader,
        "per_instance": [
            {
                "instance_id": r.instance_id,
                "mean": round(r.instance_mean, 4),
                "stderr": round(r.instance_stderr, 4),
                "ci95_lower": round(r.ci95_lower, 4),
                "ci95_upper": round(r.ci95_upper, 4),
                "ci_warning": r.ci_warning,
                "per_category_mean": {k: round(v, 4) for k, v in r.per_category_mean.items()},
            }
            for r in per_instance_reports
        ],
        "aggregate": {
            "mean_of_means": round(aggregate_mean, 4),
            "conservative_ci95_lower": round(aggregate_ci_lower, 4),
            "monitor_floor": monitor_floor,
            "meets_floor": meets_floor,
        },
        "wall_time_s": round(wall, 1),
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")
    print(f"wall={wall:.1f}s  aggregate mean-of-means={aggregate_mean:.3f}  "
          f"conservative CI95-lower={aggregate_ci_lower:.3f}  meets_floor({monitor_floor})={meets_floor}")
    return 0 if meets_floor else 1


if __name__ == "__main__":
    raise SystemExit(main())
