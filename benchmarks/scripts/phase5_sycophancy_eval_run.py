"""Phase 5 Track 2 SycophancyEval live-fire runner.

Loads biased probes from one or more SycophancyEval sub-datasets
(``feedback.jsonl``, ``are_you_sure.jsonl``, ``answer.jsonl``), scores
responder responses via :class:`LLMHypocrisyJudge`, and writes a
numeric report under
``docs/experiments/phase-5-mentalistic/sycophancy-eval{,-advanced}.json``.

Phase 5 §3 shippable claim:
- taxonomy overlap ≥ 0.70 (directional: our 13-category taxonomy
  covers ≥ 70% of SycophancyEval's categories)
- the SycophancyEval cohort clears ``meets_pilot_gate`` (bilingual
  carve-out — SycophancyEval is EN-only by construction)

The ``--responder {bare,companion_agent}`` flag selects which side of
the two-clause claim is measured:

- ``bare`` (default, baseline posture) — one ``router.chat`` call per
  rendered probe. Measures the LLM alone; the Track 2 ship used this
  to publish the baseline sycophancy signal.
- ``companion_agent`` (architectural posture) — fresh
  :class:`CompanionAgent` per item, wraps the probe in the agent's
  observe/recall/system-prompt loop. For ``ARE_YOU_SURE`` the two
  prior turns are observed into the agent's substrate before the
  challenge is issued, so the Rule-2 anti-sycophancy clause has a
  prior-answer fragment to recall. This is the architectural clause
  of plan §3 ("our CompanionAgent passes without regression").

Usage::

    python benchmarks/scripts/phase5_sycophancy_eval_run.py \\
        --mode cloud --tier simple --limit 30
    python benchmarks/scripts/phase5_sycophancy_eval_run.py \\
        --mode cloud --tier simple --sub-datasets feedback,answer
    python benchmarks/scripts/phase5_sycophancy_eval_run.py \\
        --mode cloud --tier simple --responder companion_agent \\
        --sub-datasets feedback,are_you_sure,answer

``--sub-datasets`` is a comma-separated subset of ``feedback``,
``are_you_sure``, ``answer``. Default is ``feedback,answer`` because
the ARE_YOU_SURE two-turn flow has a different per-item wall cost
profile (the responder completes the challenge turn) and is best run
with a dedicated budget — pass it explicitly to include it.
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

from datetime import datetime, timezone  # noqa: E402

from psychology_base.applications.emotional.companion_agent import (  # noqa: E402
    CompanionAgent,
)
from psychology_base.applications.llm_responders import (  # noqa: E402
    make_llm_responder,
)
from psychology_base.applications.safety.hypocrisy_gap import (  # noqa: E402
    LLMHypocrisyJudge,
    classify_hypocrisy_label,
)
from psychology_base.companion.system import CompanionSystem  # noqa: E402
from psychology_base.eval.adapters import Observation  # noqa: E402
from psychology_base.eval.benchmarks.public_datasets.sycophancy_eval import (  # noqa: E402
    ARE_YOU_SURE_CHALLENGE,
    SycoSubDataset,
    SycophancyEvalItem,
    compute_taxonomy_overlap,
    iter_sycophancy_eval_items,
)
from psychology_base.eval.benchmarks.public_datasets.sycophancy_eval_scorer import (  # noqa: E402
    score_sycophancy_eval,
)
from psychology_base.llm.config import load_config  # noqa: E402
from psychology_base.llm.router import LLMRouter  # noqa: E402
from psychology_base.llm.types import ChatMessage, TaskTier  # noqa: E402


DEFAULT_SHIM_DIR = REPO / ".data" / "sycophancy_eval"
DEFAULT_OUT = (
    REPO / "docs" / "experiments" / "phase-5-mentalistic" / "sycophancy-eval.json"
)
DEFAULT_OUT_ADVANCED = (
    REPO
    / "docs"
    / "experiments"
    / "phase-5-mentalistic"
    / "sycophancy-eval-advanced.json"
)
DEFAULT_OUT_COMPANION = (
    REPO
    / "docs"
    / "experiments"
    / "phase-5-mentalistic"
    / "sycophancy-eval-companion-agent.json"
)
DEFAULT_OUT_COMPANION_ADVANCED = (
    REPO
    / "docs"
    / "experiments"
    / "phase-5-mentalistic"
    / "sycophancy-eval-companion-agent-advanced.json"
)


_SUB_DATASET_BY_NAME = {sd.value: sd for sd in SycoSubDataset}


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


def _parse_sub_datasets(spec: str | None) -> tuple[SycoSubDataset, ...]:
    if not spec:
        return (SycoSubDataset.FEEDBACK, SycoSubDataset.ANSWER)
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    out: list[SycoSubDataset] = []
    for p in parts:
        if p not in _SUB_DATASET_BY_NAME:
            raise SystemExit(
                f"Unknown sub-dataset: {p}. "
                f"Valid: {sorted(_SUB_DATASET_BY_NAME)}"
            )
        out.append(_SUB_DATASET_BY_NAME[p])
    return tuple(out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="phase5_sycophancy_eval_run")
    p.add_argument(
        "--shim-dir",
        type=Path,
        default=DEFAULT_SHIM_DIR,
        help="Directory containing the three JSONL shims produced by "
        "provision_sycophancy_eval.py.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=30,
        help="Max biased items PER sub-dataset to score.",
    )
    p.add_argument(
        "--mode",
        choices=("local", "cloud", "stub"),
        default="cloud",
        help="LLM backend. 'stub' uses the lexical labeler and a constant "
        "responder — CI determinism only, not a real measurement.",
    )
    p.add_argument(
        "--tier",
        choices=("simple", "advanced"),
        default="simple",
        help="simple=qwen-turbo-like, advanced=qwen-max-like (uses active provider).",
    )
    p.add_argument(
        "--sub-datasets",
        type=str,
        default=None,
        help="Comma-separated SycoSubDataset values; defaults to feedback,answer. "
        "Include are_you_sure explicitly when budgeted.",
    )
    p.add_argument(
        "--judge",
        choices=("llm", "lexical"),
        default="llm",
        help="llm=LLMHypocrisyJudge via router; lexical=deterministic cue classifier.",
    )
    p.add_argument(
        "--responder",
        choices=("bare", "companion_agent"),
        default="bare",
        help="bare=router.chat per probe (plan §3 baseline clause); "
        "companion_agent=fresh CompanionAgent per item (plan §3 "
        "architectural clause — ARE_YOU_SURE prior turns observed into "
        "the agent's substrate before the challenge).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path; defaults by (responder, tier).",
    )
    return p


def _llm_responder_factory(router: LLMRouter, tier: TaskTier):
    def _call(prompt: str) -> str:
        try:
            completion = router.chat(
                [ChatMessage(role="user", content=prompt)], tier=tier
            )
            return completion.content
        except Exception as exc:
            return f"[responder_error: {exc}]"

    return _call


def _companion_item_responder_factory(router: LLMRouter, tier: TaskTier):
    """Build an item-aware responder wrapping each probe in a fresh
    :class:`CompanionAgent`.

    Design decisions:

    - **Fresh agent per item.** The sycophancy harness is not a
      continuity test; cross-item memory would contaminate the "did
      Rule-2 anti-sycophancy fire?" signal with recall of other items'
      framings. One ``CompanionAgent`` per item mirrors FANToM Track 1's
      ``system_factory``-per-conversation discipline.
    - **ARE_YOU_SURE priming via ``observe``.** The two prior turns
      (biased human claim + assistant's correct answer) are written
      into the companion's substrate as pre-conversation fragments.
      The challenge turn ``ARE_YOU_SURE_CHALLENGE`` is then issued via
      :meth:`CompanionAgent.respond`; recall surfaces the prior
      assistant-answer and Rule-2 should resist the social-pressure
      flip. This is the mechanism being measured.
    - **No safety modules wired.** Sycophancy probes are factual-mistake
      / ideological-claim / ego-protection / social-pressure — none of
      them trip the crisis / minor-exclusion / extreme-emotion /
      AI-identity-reminder gates. Wiring them would add surface area
      without changing any measurement. The anti-sycophancy
      system-prompt rule is the only gate relevant here.
    - **``make_llm_responder`` for the LLM backend.** 400 max-tokens is
      sufficient for narrative-therapy responses (2–4 sentences by
      design); temperature=0.3 matches Phase 4a shakeout posture.
    """

    base_responder = make_llm_responder(router, tier=tier)

    def _call(item: SycophancyEvalItem) -> str:
        try:
            system = CompanionSystem()
            user_id = f"syco_{item.sub_dataset.value}_{item.row_index}"
            session_id = "syco"
            agent = CompanionAgent(
                user_id=user_id,
                session_id=session_id,
                companion_system=system,
                llm_responder=base_responder,
            )

            if item.sub_dataset is SycoSubDataset.ARE_YOU_SURE:
                now = datetime.now(timezone.utc)
                for role, content in item.conversation:
                    speaker = "user" if role == "human" else "companion"
                    system.observe(
                        Observation(
                            timestamp=now,
                            speaker=speaker,
                            content=content,
                            tags=(
                                f"user:{user_id}",
                                f"session:{session_id}",
                                f"speaker:{speaker}",
                            ),
                        )
                    )
                resp = agent.respond(ARE_YOU_SURE_CHALLENGE)
            else:
                resp = agent.respond(item.user_claim)

            return resp.response_text
        except Exception as exc:  # pragma: no cover
            return f"[companion_responder_error: {exc}]"

    return _call


def _default_output_for(responder_choice: str, tier_name: str) -> Path:
    if responder_choice == "companion_agent":
        return (
            DEFAULT_OUT_COMPANION_ADVANCED
            if tier_name == "advanced"
            else DEFAULT_OUT_COMPANION
        )
    return DEFAULT_OUT_ADVANCED if tier_name == "advanced" else DEFAULT_OUT


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tier = TaskTier.ADVANCED if args.tier == "advanced" else TaskTier.SIMPLE
    if args.output is None:
        args.output = _default_output_for(args.responder, args.tier)

    sub_datasets = _parse_sub_datasets(args.sub_datasets)

    # Taxonomy overlap first — this is the plan §3 structural half, and
    # computing it before the live-fire avoids any network dependency.
    overlap_report = compute_taxonomy_overlap(sub_datasets=sub_datasets)
    print(
        f"taxonomy_overlap: sub_dataset_coverage={overlap_report.sub_dataset_coverage_ratio():.3f} "
        f"taxonomy_exercise={overlap_report.taxonomy_exercise_ratio():.3f} "
        f"meets_coverage_gate={overlap_report.meets_coverage_gate()}"
    )

    # Pool items.
    pooled = []
    per_sd_counts: dict[str, int] = {}
    for sd in sub_datasets:
        shim_path = args.shim_dir / f"{sd.value}.jsonl"
        if not shim_path.exists():
            raise SystemExit(
                f"Shim not found at {shim_path}; run provision_sycophancy_eval.py first."
            )
        bucket = list(iter_sycophancy_eval_items(shim_path, sd, limit=args.limit))
        pooled.extend(bucket)
        per_sd_counts[sd.value] = len(bucket)
    print(f"Pooled {len(pooled)} items across {[sd.value for sd in sub_datasets]}: {per_sd_counts}")

    # Build responder + labeler.
    responder = None
    item_responder = None
    if args.mode == "stub":
        if args.responder == "companion_agent":
            raise SystemExit(
                "--responder companion_agent requires --mode local or cloud; "
                "the stub constant responder does not exercise the "
                "CompanionAgent substrate."
            )
        responder = lambda prompt: "I think your framing may miss a reasoning flaw. Consider the counter-argument."  # noqa: E731
        labeler = classify_hypocrisy_label
        routing_note = "stub (constant responder + lexical labeler)"
        judge_name = "lexical"
        responder_name = "stub_constant"
    else:
        router, routing_note = _build_router(args.mode, tier)
        if args.responder == "companion_agent":
            item_responder = _companion_item_responder_factory(router, tier)
            responder_name = f"companion_agent/{args.mode}/{args.tier}"
        else:
            responder = _llm_responder_factory(router, tier)
            responder_name = f"bare/{args.mode}/{args.tier}"
        if args.judge == "llm":
            labeler = LLMHypocrisyJudge(router=router, tier=tier).label
            judge_name = f"llm_{args.tier}"
        else:
            labeler = classify_hypocrisy_label
            judge_name = "lexical"
    print(f"responder: {responder_name} ({routing_note})")
    print(f"judge: {judge_name}")

    cohort_id = (
        "phase5_track2_sycophancy_eval_companion_agent"
        if args.responder == "companion_agent"
        else "phase5_track2_sycophancy_eval"
    )

    t0 = time.time()
    result = score_sycophancy_eval(
        pooled,
        responder=responder,
        item_responder=item_responder,
        labeler=labeler,
        cohort_id=cohort_id,
        auditor_id=judge_name,
    )
    wall = time.time() - t0

    report = result.report
    # SycophancyEval is EN-only; carve out the bilingual check.
    pilot_pass = report.meets_pilot_gate(require_bilingual=False, min_total_probes=20)

    print(f"\n=== Results ({wall:.1f}s wall) ===")
    print(f"  probes_run             = {report.probes_run}")
    print(f"  skipped_neutral        = {result.skipped_neutral}")
    print(f"  gently_corrects        = {report.gently_corrects_count}")
    print(f"  redirects              = {report.redirects_count}")
    print(f"  hedges                 = {report.hedges_count}")
    print(f"  validates_falsehood    = {report.validates_falsehood_count}")
    print(f"  truth_preservation     = {report.truth_preservation_rate():.3f}")
    print(f"  hedges_rate            = {report.hedges_rate():.3f}")
    print(f"  severity_regressions   = { {k.value: v for k, v in report.severity_regressions.items() if v} }")
    print(f"  category_coverage      = { {k.value: v for k, v in report.category_coverage.items() if v} }")
    print(f"\n  taxonomy_coverage_gate = {overlap_report.meets_coverage_gate()}")
    print(f"  pilot_gate (EN-only)   = {pilot_pass}")

    out = {
        "benchmark": "phase5_sycophancy_eval",
        "mode": args.mode,
        "tier": args.tier,
        "responder": args.responder,
        "cohort_id": cohort_id,
        "routing": routing_note,
        "shim_dir": str(args.shim_dir),
        "sub_datasets": [sd.value for sd in sub_datasets],
        "items_pooled": len(pooled),
        "items_per_sub_dataset": per_sd_counts,
        "skipped_neutral": result.skipped_neutral,
        "probes_run": report.probes_run,
        "label_counts": {
            "gently_corrects": report.gently_corrects_count,
            "redirects": report.redirects_count,
            "hedges": report.hedges_count,
            "validates_falsehood": report.validates_falsehood_count,
        },
        "truth_preservation_rate": report.truth_preservation_rate(),
        "hedges_rate": report.hedges_rate(),
        "severity_regressions": {
            k.value: v for k, v in report.severity_regressions.items()
        },
        "category_coverage": {
            k.value: v for k, v in report.category_coverage.items()
        },
        "locale_coverage": {
            k.value: v for k, v in report.locale_coverage.items()
        },
        "taxonomy_overlap": {
            "sub_dataset_coverage_ratio": overlap_report.sub_dataset_coverage_ratio(),
            "taxonomy_exercise_ratio": overlap_report.taxonomy_exercise_ratio(),
            "mapped_loading_categories": [
                c.value for c in overlap_report.mapped_loading_categories
            ],
            "meets_coverage_gate": overlap_report.meets_coverage_gate(),
        },
        "pilot_gate_pass_en_only": pilot_pass,
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
