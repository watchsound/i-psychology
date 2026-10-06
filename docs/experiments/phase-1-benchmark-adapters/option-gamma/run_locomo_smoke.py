"""Fixture-scale smoke run for LoCoMo dual-mode (α-direct + γ-RAG).

Not the step-4 gate check — that targets Mem0's 92.5% single-hop F1 on
the real snap-research/locomo locomo10.json. This smoke exercises the
full pipeline on the 2-instance fixture so we have:

  (a) proof the LoCoMo pipeline wires through CompanionSystem → recall →
      (reader →) judge/F1 → cache without a crash on the two-speaker
      dialogue shape (which differs from LongMemEval's user/assistant
      roles);
  (b) a direct-vs-RAG delta on toy inputs, broken out by category — the
      single-hop subset is scored by F1 (no judge calls), adversarial
      routes through the LLM judge. If α=0 on single-hop with clean
      fragments, the F1 scorer or the normalizer is suspect; if
      adversarial lands wrong on both modes, the judge rubric is suspect.
  (c) a reproducible artifact (``locomo-fixture-smoke.json``) following
      the option-alpha/option-gamma convention.

Run:
    ALIBABA_TONGYI_API_KEY=... python docs/experiments/phase-1-benchmark-adapters/option-gamma/run_locomo_smoke.py
"""

from __future__ import annotations

import datetime as _dt
import json
import sys
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


def main() -> int:
    fixture = REPO_ROOT / "tests" / "fixtures" / "locomo_mini" / "locomo_shim.jsonl"
    if not fixture.exists():
        print(f"missing fixture at {fixture}", file=sys.stderr)
        return 1

    router = router_from_config()
    _, decision = router.route(TaskTier.ADVANCED)
    print(f"ADVANCED tier → {decision.provider}:{decision.model_id} ({decision.reason})")

    judge = LLMJudge(router=router)
    reader = LLMReader(router=router)
    judgment_cache = JudgmentCache()

    loader = LoCoMoLoader(shim_path=fixture)
    instances = list(loader.iter_instances())
    print(f"loaded {len(instances)} LoCoMo fixture instances")

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

    def by_cat(per_instance):
        buckets: dict[str, list[float]] = {}
        for r in per_instance:
            for qr in r.per_question:
                buckets.setdefault(qr.category or "unknown", []).append(qr.score)
        return {c: mean(xs) for c, xs in buckets.items()}

    direct_by_cat = by_cat(per_instance_direct)
    rag_by_cat = by_cat(per_instance_rag)

    report = {
        "benchmark": "locomo",
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
            "mean_by_category": direct_by_cat,
            "per_question": [
                {
                    "instance": inst.instance_id,
                    "question_id": qr.question_id,
                    "category": qr.category,
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
            "mean_by_category": rag_by_cat,
            "per_question": [
                {
                    "instance": inst.instance_id,
                    "question_id": qr.question_id,
                    "category": qr.category,
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

    out_path = HERE / "locomo-fixture-smoke.json"
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
    print(f"direct_by_category={direct_by_cat}")
    print(f"rag_by_category={rag_by_cat}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
