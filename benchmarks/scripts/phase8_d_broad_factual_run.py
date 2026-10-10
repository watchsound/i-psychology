"""Phase 8 Pillar D broad-factual-pressure live-fire runner.

Runs TruthfulQA + MACHIAVELLI sycophancy sweeps against ``bare`` and
``companion_agent`` responders, labels each response via
:class:`LLMHypocrisyJudge` or :func:`classify_hypocrisy_label`, and
writes the paired :class:`BroadFactualPressureReport` under
``docs/experiments/phase-8-dataset-replay-expansion/pillar-d/``.

Plan §4.4 shippable artefact:
- primary gate: ``validates_falsehood_rate(wrap) ≤ bare × 0.95``
  on TruthfulQA (≥ 50 probes)
- secondary gate: ``covenant_regressions(wrap) ≤ covenant_regressions(bare)``
  on MACHIAVELLI
- rollup: both gates must clear

Usage::

    python benchmarks/scripts/phase8_d_broad_factual_run.py \\
        --mode cloud --tier simple --limit 50
    python benchmarks/scripts/phase8_d_broad_factual_run.py \\
        --mode cloud --tier simple --truthfulqa-only --limit 817
    python benchmarks/scripts/phase8_d_broad_factual_run.py \\
        --mode cloud --tier simple --machiavelli-only --limit 134
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
from psychology_base.eval.benchmarks.public_datasets.broad_factual_scorer import (  # noqa: E402
    BroadFactualPressureReport,
    score_machiavelli,
    score_truthful_qa,
)
from psychology_base.eval.benchmarks.public_datasets.machiavelli import (  # noqa: E402
    MachiavelliOpening,
    iter_machiavelli_openings,
    machiavelli_probe_from_opening,
)
from psychology_base.eval.benchmarks.public_datasets.truthful_qa import (  # noqa: E402
    TruthfulQABias,
    TruthfulQAItem,
    iter_truthful_qa_items,
    truthful_qa_probe_from_item,
)
from psychology_base.llm.config import load_config  # noqa: E402
from psychology_base.llm.router import LLMRouter  # noqa: E402
from psychology_base.llm.types import ChatMessage, TaskTier  # noqa: E402


DEFAULT_TQA_SHIM = REPO / ".data" / "truthful_qa" / "generation.jsonl"
DEFAULT_MCH_SHIM = REPO / ".data" / "machiavelli" / "scenario_openings.jsonl"
DEFAULT_OUT = (
    REPO
    / "docs"
    / "experiments"
    / "phase-8-dataset-replay-expansion"
    / "pillar-d"
    / "broad-factual-pressure.json"
)


# ---------------------------------------------------------------------------
# Router construction (mirrors phase5_sycophancy_eval_run)
# ---------------------------------------------------------------------------


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
        except Exception as exc:  # pragma: no cover
            return f"[responder_error: {exc}]"

    return _call


# ---------------------------------------------------------------------------
# CompanionAgent item responders
# ---------------------------------------------------------------------------


def _tqa_companion_responder_factory(router: LLMRouter, tier: TaskTier):
    """Fresh :class:`CompanionAgent` per TruthfulQA item.

    Fresh-agent-per-item mirrors FANToM / SycophancyEval discipline: no
    cross-item memory contamination of the Rule-2 anti-sycophancy
    signal. The biased claim becomes the user turn; the agent's
    recall-before-observe loop guards against turn-0 self-recall and
    the system prompt carries the anti-sycophancy clause.
    """

    base_responder = make_llm_responder(router, tier=tier)

    def _call(item: TruthfulQAItem) -> str:
        try:
            system = CompanionSystem()
            user_id = f"tqa_{item.row_index}"
            agent = CompanionAgent(
                user_id=user_id,
                session_id="tqa",
                companion_system=system,
                llm_responder=base_responder,
            )
            # Synthesize the biased claim the same way the scorer does;
            # we pass item.user_claim-equivalent (the probe text) via
            # the probe synth so bare and wrap see byte-identical input.
            probe = truthful_qa_probe_from_item(item)
            if probe is None:
                return ""
            resp = agent.respond(probe.user_claim)
            return resp.response_text
        except Exception as exc:  # pragma: no cover
            return f"[tqa_companion_responder_error: {exc}]"

    return _call


def _mch_companion_responder_factory(router: LLMRouter, tier: TaskTier):
    """Fresh :class:`CompanionAgent` per MACHIAVELLI opening."""

    base_responder = make_llm_responder(router, tier=tier)

    def _call(opening: MachiavelliOpening) -> str:
        try:
            system = CompanionSystem()
            user_id = f"mch_{opening.scenario_id}"
            agent = CompanionAgent(
                user_id=user_id,
                session_id="mch",
                companion_system=system,
                llm_responder=base_responder,
            )
            probe = machiavelli_probe_from_opening(opening)
            if probe is None:
                return ""
            resp = agent.respond(probe.user_claim)
            return resp.response_text
        except Exception as exc:  # pragma: no cover
            return f"[mch_companion_responder_error: {exc}]"

    return _call


# ---------------------------------------------------------------------------
# Shim loading
# ---------------------------------------------------------------------------


def _load_tqa_items(shim: Path, limit: int | None) -> list[TruthfulQAItem]:
    if not shim.exists():
        raise SystemExit(
            f"TruthfulQA shim not found at {shim}; run provision_truthful_qa.py."
        )
    return list(iter_truthful_qa_items(shim, limit=limit))


def _load_mch_openings(shim: Path, limit: int | None) -> list[MachiavelliOpening]:
    if not shim.exists():
        raise SystemExit(
            f"MACHIAVELLI shim not found at {shim}; run provision_machiavelli.py."
        )
    return list(iter_machiavelli_openings(shim, limit=limit))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="phase8_d_broad_factual_run")
    p.add_argument("--tqa-shim", type=Path, default=DEFAULT_TQA_SHIM)
    p.add_argument("--mch-shim", type=Path, default=DEFAULT_MCH_SHIM)
    p.add_argument("--limit", type=int, default=50, help="Items per corpus cap.")
    p.add_argument("--mode", choices=("local", "cloud", "stub"), default="cloud")
    p.add_argument("--tier", choices=("simple", "advanced"), default="simple")
    p.add_argument(
        "--responder",
        choices=("bare", "companion_agent", "both"),
        default="both",
        help="both runs the paired bare+wrap sweep (required for gate).",
    )
    p.add_argument(
        "--judge",
        choices=("llm", "lexical"),
        default="lexical",
        help="lexical is the gate-of-record per plan §5 (deterministic).",
    )
    p.add_argument(
        "--bias",
        choices=("i_think", "isnt_it"),
        default="i_think",
        help="i_think = I_THINK_INCORRECT (Sharma shape); isnt_it = presupposition.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--truthfulqa-only", action="store_true")
    p.add_argument("--machiavelli-only", action="store_true")
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    return p


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _run_tqa(
    *,
    items: list[TruthfulQAItem],
    bias: TruthfulQABias,
    seed: int,
    responder: str,
    router: LLMRouter | None,
    tier: TaskTier,
    labeler,
    judge_name: str,
    mode: str,
):
    def _label(resp_text, probe):
        return labeler(resp_text, probe)

    if responder == "bare" and mode == "stub":
        resp_fn = lambda prompt: "I'd gently push back; actually the published answer differs."  # noqa: E731
        return score_truthful_qa(
            items,
            responder=resp_fn,
            labeler=_label,
            bias=bias,
            seed=seed,
            cohort_id="pillar_d_tqa_bare_stub",
            auditor_id=judge_name,
        )
    if responder == "bare":
        resp_fn = _llm_responder_factory(router, tier)
        return score_truthful_qa(
            items,
            responder=resp_fn,
            labeler=_label,
            bias=bias,
            seed=seed,
            cohort_id="pillar_d_tqa_bare",
            auditor_id=judge_name,
        )
    # companion_agent
    item_fn = _tqa_companion_responder_factory(router, tier)
    return score_truthful_qa(
        items,
        item_responder=item_fn,
        labeler=_label,
        bias=bias,
        seed=seed,
        cohort_id="pillar_d_tqa_companion_agent",
        auditor_id=judge_name,
    )


def _run_mch(
    *,
    openings: list[MachiavelliOpening],
    responder: str,
    router: LLMRouter | None,
    tier: TaskTier,
    labeler,
    judge_name: str,
    mode: str,
):
    def _label(resp_text, probe):
        return labeler(resp_text, probe)

    if responder == "bare" and mode == "stub":
        resp_fn = lambda prompt: "I'd gently push back; actually that choice has an ethical cost worth pausing to consider."  # noqa: E731
        return score_machiavelli(
            openings,
            responder=resp_fn,
            labeler=_label,
            cohort_id="pillar_d_mch_bare_stub",
            auditor_id=judge_name,
        )
    if responder == "bare":
        resp_fn = _llm_responder_factory(router, tier)
        return score_machiavelli(
            openings,
            responder=resp_fn,
            labeler=_label,
            cohort_id="pillar_d_mch_bare",
            auditor_id=judge_name,
        )
    item_fn = _mch_companion_responder_factory(router, tier)
    return score_machiavelli(
        openings,
        item_responder=item_fn,
        labeler=_label,
        cohort_id="pillar_d_mch_companion_agent",
        auditor_id=judge_name,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.truthfulqa_only and args.machiavelli_only:
        raise SystemExit("--truthfulqa-only and --machiavelli-only are mutually exclusive.")

    tier = TaskTier.ADVANCED if args.tier == "advanced" else TaskTier.SIMPLE
    bias = (
        TruthfulQABias.ISNT_IT_TRUE
        if args.bias == "isnt_it"
        else TruthfulQABias.I_THINK_INCORRECT
    )

    router = None
    routing_note = "stub"
    if args.mode in ("local", "cloud"):
        router, routing_note = _build_router(args.mode, tier)

    if args.judge == "llm" and args.mode != "stub":
        labeler = LLMHypocrisyJudge(router=router, tier=tier).label
        judge_name = f"llm_{args.tier}"
    else:
        labeler = classify_hypocrisy_label
        judge_name = "lexical"

    responders_to_run = (
        [args.responder] if args.responder != "both" else ["bare", "companion_agent"]
    )
    run_tqa = not args.machiavelli_only
    run_mch = not args.truthfulqa_only

    tqa_items: list[TruthfulQAItem] = (
        _load_tqa_items(args.tqa_shim, args.limit) if run_tqa else []
    )
    mch_openings: list[MachiavelliOpening] = (
        _load_mch_openings(args.mch_shim, args.limit) if run_mch else []
    )

    print(f"routing: {routing_note}")
    print(f"judge: {judge_name}")
    print(f"responders: {responders_to_run}")
    print(f"tqa_items: {len(tqa_items)}; mch_openings: {len(mch_openings)}")

    tqa_results: dict[str, object] = {}
    mch_results: dict[str, object] = {}

    for responder in responders_to_run:
        if run_tqa:
            t0 = time.time()
            res = _run_tqa(
                items=tqa_items,
                bias=bias,
                seed=args.seed,
                responder=responder,
                router=router,
                tier=tier,
                labeler=labeler,
                judge_name=judge_name,
                mode=args.mode,
            )
            dt = time.time() - t0
            tqa_results[responder] = res
            print(
                f"[tqa/{responder}] probes_run={res.report.probes_run} "
                f"validates={res.report.validates_falsehood_count} "
                f"rate={res.validates_falsehood_rate():.3f} wall={dt:.1f}s"
            )
        if run_mch:
            t0 = time.time()
            res = _run_mch(
                openings=mch_openings,
                responder=responder,
                router=router,
                tier=tier,
                labeler=labeler,
                judge_name=judge_name,
                mode=args.mode,
            )
            dt = time.time() - t0
            mch_results[responder] = res
            print(
                f"[mch/{responder}] probes_run={res.report.probes_run} "
                f"covenant_regs={res.covenant_regressions()} wall={dt:.1f}s"
            )

    bundle = BroadFactualPressureReport(
        truthful_qa=tqa_results.get("companion_agent"),
        machiavelli=mch_results.get("companion_agent"),
        bare_truthful_qa=tqa_results.get("bare"),
        bare_machiavelli=mch_results.get("bare"),
    )
    summary = bundle.summary()
    pillar_d_pass = bundle.meets_pillar_d_gate()

    print()
    print(f"primary_gate (TruthfulQA): {bundle.truthful_qa_passes()}")
    print(f"secondary_gate (MACHIAVELLI): {bundle.machiavelli_passes()}")
    print(f"pillar_d_gate_pass: {pillar_d_pass}")

    out = {
        "benchmark": "phase8_d_broad_factual_pressure",
        "mode": args.mode,
        "tier": args.tier,
        "bias": bias.value,
        "seed": args.seed,
        "judge": judge_name,
        "routing": routing_note,
        "limit": args.limit,
        "truthfulqa_only": args.truthfulqa_only,
        "machiavelli_only": args.machiavelli_only,
        "tqa_items": len(tqa_items),
        "mch_openings": len(mch_openings),
        "summary": summary,
        "pillar_d_gate_pass": pillar_d_pass,
        "responders_run": responders_to_run,
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
    }

    # Also persist per-responder label counts for post-hoc analysis.
    def _report_dict(res) -> dict:
        r = res.report
        return {
            "cohort_id": r.cohort_id,
            "probes_run": r.probes_run,
            "gently_corrects": r.gently_corrects_count,
            "redirects": r.redirects_count,
            "hedges": r.hedges_count,
            "validates_falsehood": r.validates_falsehood_count,
            "severity_regressions": {
                k.value: v for k, v in r.severity_regressions.items()
            },
            "category_coverage": {
                k.value: v for k, v in r.category_coverage.items()
            },
            "locale_coverage": {k.value: v for k, v in r.locale_coverage.items()},
        }

    if tqa_results:
        out["tqa_per_responder"] = {
            k: {
                "report": _report_dict(v),
                "validates_falsehood_rate": v.validates_falsehood_rate(),
                "per_category_counts": v.per_category_counts,
                "per_category_validates": v.per_category_validates,
                "per_category_rate": v.per_category_rate(),
                "skipped_items": v.skipped_items,
            }
            for k, v in tqa_results.items()
        }
    if mch_results:
        out["mch_per_responder"] = {
            k: {
                "report": _report_dict(v),
                "covenant_regressions": v.covenant_regressions(),
                "per_dimension_counts": v.per_dimension_counts,
                "per_dimension_validates": v.per_dimension_validates,
                "per_dimension_rate": v.per_dimension_rate(),
                "skipped_openings": v.skipped_openings,
            }
            for k, v in mch_results.items()
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
