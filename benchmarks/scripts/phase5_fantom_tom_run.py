"""Phase 5 Track 1 FANToM ToM live-fire runner.

Scores a batch of FANToM items from the provisioned shim on BOTH the
mentalistic-routed path and the bare-LLM baseline, grading answers
through the LLM FANToM judge. Writes a numeric report under
``docs/experiments/phase-5-mentalistic/fantom-tom{,-advanced}.json``.

Phase 5 §2 kill criterion:
- accuracy ≥ 0.40 on BELIEF_INACCESSIBLE under the mentalistic path
- `accuracy_mentalistic − accuracy_bare_llm` ≥ 0.03 (architectural
  contribution floor)
- min 50 items on each path at the gated question type

Usage::

    python benchmarks/scripts/phase5_fantom_tom_run.py \
        --mode cloud --tier simple --limit 50
    python benchmarks/scripts/phase5_fantom_tom_run.py \
        --mode cloud --tier simple --question-types belief_inaccessible,belief_accessible

``--question-types`` is a comma-separated list of
``FANToMQuestionType`` enum values. Default is BELIEF_INACCESSIBLE +
BELIEF_ACCESSIBLE (the two gated-against sub-splits).
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

from psychology_base.eval.benchmarks.public_datasets.fantom import (  # noqa: E402
    FANToMQuestionType,
    iter_fantom_items,
)
from psychology_base.eval.benchmarks.public_datasets.fantom_tom_scorer import (  # noqa: E402
    LLMFANToMJudge,
    lexical_answer_judge,
    score_fantom_tom,
)
from psychology_base.llm.config import load_config  # noqa: E402
from psychology_base.llm.router import LLMRouter  # noqa: E402
from psychology_base.llm.types import ChatMessage, TaskTier  # noqa: E402


DEFAULT_SHIM = REPO / ".data" / "fantom" / "test.jsonl"
DEFAULT_OUT = REPO / "docs" / "experiments" / "phase-5-mentalistic" / "fantom-tom.json"
DEFAULT_OUT_ADVANCED = (
    REPO / "docs" / "experiments" / "phase-5-mentalistic" / "fantom-tom-advanced.json"
)


_QUESTION_TYPE_BY_NAME = {qt.value: qt for qt in FANToMQuestionType}


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


def _parse_question_types(spec: str | None) -> tuple[FANToMQuestionType, ...]:
    if not spec:
        return (
            FANToMQuestionType.BELIEF_INACCESSIBLE,
            FANToMQuestionType.BELIEF_ACCESSIBLE,
        )
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    out: list[FANToMQuestionType] = []
    for p in parts:
        if p not in _QUESTION_TYPE_BY_NAME:
            raise SystemExit(
                f"Unknown question type: {p}. "
                f"Valid: {sorted(_QUESTION_TYPE_BY_NAME)}"
            )
        out.append(_QUESTION_TYPE_BY_NAME[p])
    return tuple(out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="phase5_fantom_tom_run")
    p.add_argument("--shim", type=Path, default=DEFAULT_SHIM)
    p.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Max items (one per question) to pool across BOTH paths.",
    )
    p.add_argument(
        "--mode",
        choices=("local", "cloud", "stub"),
        default="cloud",
        help="LLM backend. 'stub' uses the lexical judge and a constant "
        "responder — CI determinism only, not a real measurement.",
    )
    p.add_argument(
        "--tier",
        choices=("simple", "advanced"),
        default="simple",
        help="simple=qwen-turbo-like, advanced=qwen-max-like (uses active provider).",
    )
    p.add_argument(
        "--question-types",
        type=str,
        default=None,
        help="Comma-separated FANToMQuestionType values; defaults to "
        "belief_inaccessible,belief_accessible.",
    )
    p.add_argument(
        "--judge",
        choices=("llm", "lexical"),
        default="llm",
        help="llm=LLMFANToMJudge via router; lexical=deterministic substring "
        "matcher (CI-only).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path; defaults by tier.",
    )
    return p


def _llm_responder_factory(router: LLMRouter, tier: TaskTier):
    """A responder closure that forwards one user-turn chat to the router.

    Temperature is left at provider-default (lowest for the simple tier)
    to minimise noise between the two paths' answers — the architectural
    delta we want to measure is cleaner on the deterministic end.
    """

    def _call(prompt: str) -> str:
        try:
            completion = router.chat(
                [ChatMessage(role="user", content=prompt)], tier=tier
            )
            return completion.content
        except Exception as exc:
            # Keep the run alive; a surfaced failure becomes undecidable
            # at the judge stage.
            return f"[responder_error: {exc}]"

    return _call


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tier = TaskTier.ADVANCED if args.tier == "advanced" else TaskTier.SIMPLE
    if args.output is None:
        args.output = (
            DEFAULT_OUT_ADVANCED if args.tier == "advanced" else DEFAULT_OUT
        )
    if not args.shim.exists():
        raise SystemExit(
            f"Shim not found at {args.shim}; run provision_fantom.py first."
        )

    question_types = _parse_question_types(args.question_types)
    print(
        f"question_types = {[qt.value for qt in question_types]}, "
        f"limit = {args.limit}"
    )

    items = []
    for item in iter_fantom_items(args.shim, question_types=question_types):
        items.append(item)
        if len(items) >= args.limit:
            break
    print(f"Pooled {len(items)} items from {args.shim}")

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

    t0 = time.time()
    report = score_fantom_tom(
        items,
        responder=responder,
        judge=judge,
        judge_name=judge_name,
        responder_name=responder_name,
    )
    wall = time.time() - t0

    # Per-question-type breakdown + the full gated metric.
    def _summarise(qt):
        return {
            "question_type": qt.value if qt is not None else "all",
            "n_mentalistic": report.item_count(path="mentalistic", question_type=qt),
            "n_bare_llm": report.item_count(path="bare_llm", question_type=qt),
            "accuracy_mentalistic": report.accuracy(
                path="mentalistic", question_type=qt
            ),
            "accuracy_bare_llm": report.accuracy(path="bare_llm", question_type=qt),
            "architectural_delta": report.architectural_delta(question_type=qt),
            "undecidable_rate_mentalistic": report.undecidable_rate(
                path="mentalistic", question_type=qt
            ),
            "undecidable_rate_bare_llm": report.undecidable_rate(
                path="bare_llm", question_type=qt
            ),
        }

    per_type = [_summarise(qt) for qt in question_types] + [_summarise(None)]
    pivot_pass = report.meets_pivot_gate()

    print(f"\n=== Results ({wall:.1f}s wall) ===")
    for row in per_type:
        print(
            f"  [{row['question_type']}] "
            f"n_mental={row['n_mentalistic']} n_bare={row['n_bare_llm']} "
            f"acc_mental={row['accuracy_mentalistic']:.3f} "
            f"acc_bare={row['accuracy_bare_llm']:.3f} "
            f"delta={row['architectural_delta']:+.3f} "
            f"undec={row['undecidable_rate_mentalistic']:.2f}/"
            f"{row['undecidable_rate_bare_llm']:.2f}"
        )
    print(
        f"\npivot_gate(acc≥0.40 ∧ delta≥0.03 ∧ n≥50 on BELIEF_INACCESSIBLE) "
        f"= {pivot_pass}"
    )

    out = {
        "benchmark": "phase5_fantom_tom",
        "mode": args.mode,
        "tier": args.tier,
        "routing": routing_note,
        "shim": str(args.shim),
        "question_types": [qt.value for qt in question_types],
        "items_pooled": len(items),
        "verdicts_total": len(report.verdicts),
        "per_type": per_type,
        "pivot_gate_pass": pivot_pass,
        "judge_name": judge_name,
        "responder_name": responder_name,
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
