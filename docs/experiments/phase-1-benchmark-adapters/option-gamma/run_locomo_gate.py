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
        "--cross-encoder-rerank-all",
        action="store_true",
        help=(
            "Phase 2 R1: when cross-encoder rerank is on, bypass the "
            "top-150 cosine prefilter and rerank over ALL fragments in "
            "the store (bounded by the provider's per-call document cap). "
            "Targets 5 Bucket A failures where gold is cosine-ranked past "
            "150 so rerank never sees it. Requires --cross-encoder-rerank. "
            "KILLED 2026-10-06 (regressed to F1=0.391); flag retained as "
            "reproducible negative result."
        ),
    )
    p.add_argument(
        "--bm25-hybrid",
        action="store_true",
        help=(
            "Phase 2 R2: RRF-fuse BM25 (literal-token recall) with cosine "
            "(semantic recall) over all fragments; feed the fused top-K "
            "pool to the reader (or to --cross-encoder-rerank for a final "
            "trim). Targets Bucket A proper nouns (Sweden, Adoption, "
            "Single, dates) that cosine buries under paraphrase cues. "
            "Requires consolidation."
        ),
    )
    p.add_argument(
        "--within-session-rerank",
        action="store_true",
        help=(
            "Phase 2 R8-c: two-stage rerank. After cosine recall picks the "
            "top-150 fragment pool, group by session_id and pick the top-M "
            "sessions by max-cosine; pull ALL turns of those M sessions "
            "from the full fragment store and cross-encoder-rerank that "
            "scoped pool. Targets the 5 session_present_gold_absent rows "
            "where the right session reaches top-50 but the specific gold "
            "turn is cosine-ranked past 150 within its session. Requires "
            "--cross-encoder-rerank; mutually exclusive with "
            "--cross-encoder-rerank-all."
        ),
    )
    p.add_argument(
        "--within-session-top-m-sessions",
        type=int,
        default=3,
        help=(
            "Phase 2 R8-c: number of top sessions to pull into the "
            "within-session rerank pool. M=3 covers typical single-hop "
            "evidence (single session) with headroom. Default 3."
        ),
    )
    p.add_argument(
        "--fact-density-tie-breaker",
        action="store_true",
        help=(
            "Phase 2 R8-b: conservative post-rerank tie-breaker. Count "
            "proper nouns + years + ISO/DMY dates + bare integers per "
            "fragment; adjacent reranked pairs within score delta 0.03 "
            "swap if the lower-ranked fragment is >=2 density units "
            "denser. Targets fact-dense gold rows (q71 'Becoming Nicole', "
            "q75 '3', q76 '19 October 2023'). Requires "
            "--cross-encoder-rerank. Stackable with --within-session-rerank."
        ),
    )
    p.add_argument(
        "--fact-source-turn-lift",
        action="store_true",
        help=(
            "Phase 2b Step A: for each of the top-N fact hits, lift the "
            "turns named in source_fragment_ids directly into the DIALOGUE "
            "block ahead of cosine-ranked and substring-matched turns. "
            "The back-pointer is literal extractor-grounded evidence: the "
            "fact exists because the extractor used this turn. Targets "
            "the session_present_gold_absent bucket (71% of 3-instance "
            "incorrect) by surfacing gold turns the within-session cosine "
            "ranker buries. Requires --consolidation."
        ),
    )
    p.add_argument(
        "--fact-source-lift-top-facts",
        type=int,
        default=15,
        help=(
            "Phase 2b Step A: number of top fact hits to pull "
            "source_fragment_ids from. Default 15."
        ),
    )
    p.add_argument(
        "--fact-source-lift-fragment-cap",
        type=int,
        default=30,
        help=(
            "Phase 2b Step A: cap on lifted fragments prepended to the "
            "DIALOGUE block, protects context budget. Default 30."
        ),
    )
    p.add_argument(
        "--fact-hybrid-retrieval",
        action="store_true",
        help=(
            "Phase 2b Step B: RRF-fuse three channels over the fact "
            "store — BM25 (literal-token recall), dense cosine "
            "(semantic recall, existing signal), and entity-overlap "
            "(cue tokens matching fact.entity tokens). Diversifies "
            "fact retrieval so proper-noun / named-concept cues reach "
            "the fact tier even when cosine ranks them past top-K. "
            "Independent of --fact-source-turn-lift; stackable. "
            "Requires --consolidation."
        ),
    )
    p.add_argument(
        "--full-session-reader",
        action="store_true",
        help=(
            "Phase 2b Step F (ADDITIVE 2026-10-07): after rerank, UNION "
            "in every un-reranked turn of the top-1 reranked fragment's "
            "session at the tail (preserving all reranked top-K hits). "
            "Attacks `session_present_gold_absent`: the gold turn is "
            "visible to the reader even when it ranks past top-150 "
            "within session, without dropping cross-session rerank "
            "evidence. Requires --cross-encoder-rerank; mutex with "
            "--within-session-rerank."
        ),
    )
    p.add_argument(
        "--localize-filter",
        action="store_true",
        help=(
            "Phase 2b Step E: SIMPLE-tier LLM pass between cross-encoder "
            "rerank and the reader with a recall-biased prompt ('which "
            "of these COULD support answering Q — err on inclusion'). "
            "Attacks `gold_in_pool` reader-precision failures by "
            "re-priming the pool composition the reader sees. EMem "
            "ablation reports ~4.7 LoCoMo points for this stage alone. "
            "Requires --cross-encoder-rerank + --consolidation."
        ),
    )
    p.add_argument(
        "--reconstructive-retrieval",
        action="store_true",
        help=(
            "Phase 2b Step H: wire R1 spreading-activation over the "
            "Jaccard-seeded EdgeStore as a thesis-aligned fragment-recall "
            "channel; top-N by activation become a 'reconstructive_boost' "
            "tier appended to the fragment pool. Independent of embedder "
            "/BM25/entity-overlap signals - a structural recall path "
            "built from co-occurrence. No consolidation required."
        ),
    )
    p.add_argument(
        "--entity-graph-retrieval",
        action="store_true",
        help=(
            "Phase 2b Step C: build a fact-derived entity graph; probe "
            "with cue → 1-hop neighbors → fragments mentioning any "
            "expanded entity get appended to the fragment pool as a "
            "bounded top-N 'entity_boost' tier. Dedup'd against cosine "
            "/substring/source_lift so no double emission. Attacks "
            "`session_present_gold_absent` independent of per-fact "
            "provenance. Requires --consolidation."
        ),
    )
    p.add_argument(
        "--reader-evidence-rerank-first",
        action="store_true",
        help=(
            "Phase 2b reader-side ordering fix: reorder DIALOGUE + "
            "`fragments_used` so cross-encoder-reranked top-K comes "
            "FIRST (then substring, then source-lift, then facts' "
            "window-sources). Legacy ordering buries reranked top-K "
            "below facts' ~150+ window-wide sources, so the reader's "
            "formatted-list (truncated at 50) shows noise and DIALOGUE "
            "puts highest-precision evidence in the attention-weakest "
            "tail. This flag attacks prompt-invariant failures where "
            "gold is in-pool with substring+coverage=1.0 but the "
            "reader still refuses or over-generalizes (lost-in-the-"
            "middle). Requires --cross-encoder-rerank."
        ),
    )
    p.add_argument(
        "--reader-prompt-variant",
        default="v1",
        choices=("v1", "v2", "v3", "v4"),
        help=(
            "Reader system prompt variant. v1 = Phase 1.5 locked prompt "
            "(default; conservative refusal gate). v2 = Phase 2 R3 "
            "refusal-rebalanced (explicit scan-all + question-type "
            "extraction templates). v3 = Phase 2 R5 precision-tightened "
            "on top of v2 ('where' → proper-noun place; 'how many' → "
            "integer not frequency; single-word labels valid). v4 = "
            "Phase 2b reader-side ceiling attack — v3 + few-shot "
            "worked examples targeting the three residual failure "
            "modes (anti-refusal when gold is in-pool; proper name "
            "over generic descriptor; integer count over paraphrase). "
            "Independent of recall-channel toggles."
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


def _refusal_stats(per_instance) -> dict:
    """Count reader refusals (verbatim 'not stated' or close variants) so
    stacked ablations can show refusal rate dropping independently of F1.

    Phase 2 R3 diagnosis: reader-refusal is a serial downstream bottleneck
    for recall-side levers (R1, R2). Without this metric, upstream recall
    improvements are invisible when the reader absorbs them as refusals."""
    total = 0
    refusals = 0
    refusal_rows: list[dict] = []
    for r in per_instance:
        for qr in r.per_question:
            total += 1
            cand = (qr.candidate or "").strip().lower().rstrip(".")
            if cand == "not stated":
                refusals += 1
                refusal_rows.append(
                    {
                        "instance": r.instance_id,
                        "question_id": qr.question_id,
                        "category": qr.category,
                        "gold": qr.gold,
                        "score": round(qr.score, 3),
                    }
                )
    rate = refusals / total if total else 0.0
    return {
        "total": total,
        "refusals": refusals,
        "rate": round(rate, 3),
        "rows": refusal_rows,
    }


def _recall_stats(per_instance) -> dict:
    """Phase 2 R6 — classify every row by recall-diagnostic signal.

    Buckets each question into one of four states:
      - ``recall_missing``: evidence_session_in_top_k False AND
        gold_substring_in_top_k False — the pool genuinely never had the
        answer. Recall-channel is the binding constraint; reader cannot
        recover.
      - ``session_present_gold_absent``: evidence_session_in_top_k True
        but gold_substring_in_top_k False — right session surfaced,
        wrong turns. Fragment-ranking is the binding constraint.
      - ``gold_in_pool``: gold_substring_in_top_k True — the reader
        literally saw the normalized gold text. Any incorrect row in
        this bucket is a reader-precision failure, not a recall failure.
      - ``no_evidence_metadata``: the question didn't ship evidence
        session IDs; session-level signal unusable. Lexical check still
        reported.

    Also computes bucket breakdowns on the incorrect subset so stacked
    ablations can see where the ceiling is binding rather than only F1.
    """
    buckets = {
        "recall_missing": 0,
        "session_present_gold_absent": 0,
        "gold_in_pool": 0,
        "no_evidence_metadata": 0,
    }
    incorrect_buckets = {k: 0 for k in buckets}
    gold_in_pool_rows: list[dict] = []
    recall_missing_rows: list[dict] = []
    total = 0
    incorrect = 0
    for r in per_instance:
        for qr in r.per_question:
            total += 1
            diag = qr.recall_diagnostics or {}
            has_evidence = bool(diag.get("evidence_sessions"))
            gold_in_text = bool(diag.get("gold_substring_in_top_k"))
            evidence_hit = bool(diag.get("evidence_session_in_top_k"))
            if not has_evidence:
                bucket = "no_evidence_metadata"
            elif gold_in_text:
                bucket = "gold_in_pool"
            elif evidence_hit:
                bucket = "session_present_gold_absent"
            else:
                bucket = "recall_missing"
            buckets[bucket] += 1
            is_wrong = qr.score < 0.5
            if is_wrong:
                incorrect += 1
                incorrect_buckets[bucket] += 1
                row = {
                    "instance": r.instance_id,
                    "question_id": qr.question_id,
                    "category": qr.category,
                    "gold": qr.gold,
                    "candidate": (qr.candidate or "")[:160],
                    "score": round(qr.score, 3),
                    "evidence_sessions": list(diag.get("evidence_sessions") or ()),
                    "evidence_session_in_top_k": evidence_hit,
                    "gold_substring_in_top_k": gold_in_text,
                    "gold_token_coverage": diag.get("gold_token_coverage"),
                    "top_k_fragment_count": diag.get("top_k_fragment_count"),
                }
                if bucket == "gold_in_pool":
                    gold_in_pool_rows.append(row)
                elif bucket == "recall_missing":
                    recall_missing_rows.append(row)
    return {
        "total": total,
        "incorrect": incorrect,
        "buckets": buckets,
        "incorrect_buckets": incorrect_buckets,
        "incorrect_gold_in_pool_rows": gold_in_pool_rows,
        "incorrect_recall_missing_rows": recall_missing_rows,
    }


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
    reader = LLMReader(router=router, prompt_variant=args.reader_prompt_variant)
    judgment_cache = JudgmentCache()
    print(f"reader prompt variant → {args.reader_prompt_variant}")

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
            cross_encoder_rerank_all=args.cross_encoder_rerank_all,
            bm25_hybrid=args.bm25_hybrid,
            within_session_rerank=args.within_session_rerank,
            within_session_top_m_sessions=args.within_session_top_m_sessions,
            fact_density_tie_breaker=args.fact_density_tie_breaker,
            fact_source_turn_lift=args.fact_source_turn_lift,
            fact_source_lift_top_facts=args.fact_source_lift_top_facts,
            fact_source_lift_fragment_cap=args.fact_source_lift_fragment_cap,
            fact_hybrid_retrieval=args.fact_hybrid_retrieval,
            full_session_reader=args.full_session_reader,
            localize_filter=args.localize_filter,
            reader_evidence_rerank_first=args.reader_evidence_rerank_first,
            entity_graph_retrieval=args.entity_graph_retrieval,
            reconstructive_retrieval=args.reconstructive_retrieval,
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
        "cross_encoder_rerank_all": args.cross_encoder_rerank_all,
        "bm25_hybrid": args.bm25_hybrid,
        "within_session_rerank": args.within_session_rerank,
        "within_session_top_m_sessions": args.within_session_top_m_sessions,
        "fact_density_tie_breaker": args.fact_density_tie_breaker,
        "fact_source_turn_lift": args.fact_source_turn_lift,
        "fact_source_lift_top_facts": args.fact_source_lift_top_facts,
        "fact_source_lift_fragment_cap": args.fact_source_lift_fragment_cap,
        "fact_hybrid_retrieval": args.fact_hybrid_retrieval,
        "full_session_reader": args.full_session_reader,
        "localize_filter": args.localize_filter,
        "reader_evidence_rerank_first": args.reader_evidence_rerank_first,
        "entity_graph_retrieval": args.entity_graph_retrieval,
        "reconstructive_retrieval": args.reconstructive_retrieval,
        "reader_prompt_variant": args.reader_prompt_variant,
        "advanced_tier": {
            "provider": decision.provider,
            "model_id": decision.model_id,
            "reason": decision.reason,
        },
        "direct_mode": None if not per_direct else {
            "mean_score": _mean(direct_scores),
            "mean_by_category": direct_by_cat,
            "incorrect_sample": _incorrect_sample(per_direct),
            "recall_diagnostics": _recall_stats(per_direct),
        },
        "rag_mode": None if not per_rag else {
            "mean_score": _mean(rag_scores),
            "mean_by_category": rag_by_cat,
            "incorrect_sample": _incorrect_sample(per_rag),
            "reader_refusals": _refusal_stats(per_rag),
            "recall_diagnostics": _recall_stats(per_rag),
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
        refusal_stats = _refusal_stats(per_rag)
        print(
            f"reader  refusals={refusal_stats['refusals']}/"
            f"{refusal_stats['total']} ({refusal_stats['rate']:.1%})"
        )
        rstats = _recall_stats(per_rag)
        ib = rstats["incorrect_buckets"]
        print(
            f"recall  incorrect={rstats['incorrect']}/{rstats['total']}  "
            f"gold_in_pool={ib['gold_in_pool']}  "
            f"session_present_gold_absent={ib['session_present_gold_absent']}  "
            f"recall_missing={ib['recall_missing']}  "
            f"no_evidence_meta={ib['no_evidence_metadata']}"
        )
    print(
        f"gate ({gate_mode} single-hop ≥ {gate_target}): "
        f"{gate_value!r} → {'PASS' if gate_pass else 'FAIL'}"
    )
    return 0 if gate_pass else 2


if __name__ == "__main__":
    sys.exit(main())
