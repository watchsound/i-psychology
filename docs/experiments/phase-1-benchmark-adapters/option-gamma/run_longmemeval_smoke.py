"""Fixture-scale smoke run for LongMemEval dual-mode (α-direct + γ-RAG).

This is NOT the step-3 gate check — the gate runs on the real HuggingFace
dataset at 50 instances. This smoke run exercises the full pipeline
end-to-end against the 3-instance test fixture so we have:

  (a) proof the pipeline wires through CompanionSystem → recall → reader →
      judge → cache without a crash on real-shaped data;
  (b) a baseline α-direct vs. γ-RAG delta on toy inputs that is
      diagnostic even at n=3 (if the reader can't lift α=0 to γ=1 on
      pathological reconstructions, something is wrong upstream);
  (c) a reproducible artifact (``longmemeval-fixture-smoke.json``) checked
      into the option-gamma directory, following the same convention as
      ``option-alpha/epbench-fixture-smoke.json``.

Run:
    ALIBABA_TONGYI_API_KEY=... python docs/experiments/phase-1-benchmark-adapters/option-gamma/run_longmemeval_smoke.py
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from psychology_base.companion.system import CompanionSystem  # noqa: E402
from psychology_base.eval.benchmarks.judge import JudgmentCache, LLMJudge  # noqa: E402
from psychology_base.eval.benchmarks.longmemeval import (  # noqa: E402
    LongMemEvalLoader,
    run_direct_mode,
    run_rag_mode,
)
from psychology_base.eval.benchmarks.reader import LLMReader  # noqa: E402
from psychology_base.llm.router import router_from_config  # noqa: E402
from psychology_base.llm.types import TaskTier  # noqa: E402


def main() -> int:
    fixture = REPO_ROOT / "tests" / "fixtures" / "longmemeval_mini" / "longmemeval_shim.jsonl"
    if not fixture.exists():
        print(f"missing fixture at {fixture}", file=sys.stderr)
        return 1

    router = router_from_config()
    _, decision = router.route(TaskTier.ADVANCED)
    print(f"ADVANCED tier → {decision.provider}:{decision.model_id} ({decision.reason})")

    judge = LLMJudge(router=router)
    reader = LLMReader(router=router)
    judgment_cache = JudgmentCache()

    loader = LongMemEvalLoader(shim_path=fixture)
    instances = list(loader.iter_instances())
    print(f"loaded {len(instances)} LongMemEval fixture instances")

    def factory() -> CompanionSystem:
        return CompanionSystem(seed=0, witness_active=False)

    per_instance_direct = []
    per_instance_rag = []
    for inst in instances:
        d = run_direct_mode(
            inst, system_factory=factory, judge=judge, judgment_cache=judgment_cache
        )
        r = run_rag_mode(
            inst,
            system_factory=factory,
            reader=reader,
            judge=judge,
            judgment_cache=judgment_cache,
        )
        per_instance_direct.append(d)
        per_instance_rag.append(r)
        print(
            f"  {inst.instance_id}  direct_mean={d.mean_score:.3f}  "
            f"rag_mean={r.mean_score:.3f}"
        )

    def mean(scores):
        return sum(scores) / len(scores) if scores else 0.0

    direct_scores = [qr.score for r in per_instance_direct for qr in r.per_question]
    rag_scores = [qr.score for r in per_instance_rag for qr in r.per_question]

    report = {
        "benchmark": "longmemeval",
        "split": "fixture-mini",
        "instance_count": len(instances),
        "question_count": len(direct_scores),
        "advanced_tier": {
            "provider": decision.provider,
            "model_id": decision.model_id,
            "reason": decision.reason,
        },
        "direct_mode": {
            "mean_score": mean(direct_scores),
            "per_question": [
                {
                    "instance": inst.instance_id,
                    "question_id": qr.question_id,
                    "question_type": qr.question_type,
                    "gold": qr.gold,
                    "candidate": qr.candidate,
                    "score": qr.score,
                    "verdict": qr.verdict,
                }
                for inst, r in zip(instances, per_instance_direct)
                for qr in r.per_question
            ],
        },
        "rag_mode": {
            "mean_score": mean(rag_scores),
            "per_question": [
                {
                    "instance": inst.instance_id,
                    "question_id": qr.question_id,
                    "question_type": qr.question_type,
                    "gold": qr.gold,
                    "candidate": qr.candidate,
                    "score": qr.score,
                    "verdict": qr.verdict,
                }
                for inst, r in zip(instances, per_instance_rag)
                for qr in r.per_question
            ],
        },
        "judge": {
            "judge_id": judge.judge_id,
            "prompt_sha": judge.prompt_sha,
            "cache_hits": judgment_cache.hits,
            "cache_misses": judgment_cache.misses,
        },
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
    }

    out_path = HERE / "longmemeval-fixture-smoke.json"
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nwrote {out_path}")
    print(
        f"direct_mean={report['direct_mode']['mean_score']:.3f}  "
        f"rag_mean={report['rag_mode']['mean_score']:.3f}  "
        f"delta={report['rag_mode']['mean_score'] - report['direct_mode']['mean_score']:+.3f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
