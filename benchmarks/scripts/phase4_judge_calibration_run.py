"""Phase 4a §6d — stub↔LLM semantic judge calibration end-to-end.

Phase 2 close-out and Phase 3 close-out both carry a flag that B3
empathy / over-disclosure scores are *proxy-grade* — the stub judge
derives empathy from PAD-affect-match (or token Jaccard fallback) and
over-disclosure from byte-ratio. These proxies have known calibration
debt. The :class:`JudgeCalibrationReport` dataclass is the vessel that
tells a pilot team whether to trust the stub for CI or whether to pay
the LLM judge.

This script runs the first real stub↔LLM comparison end-to-end.

Corpus. Fixed hand-labeled set of 12 B3-shaped recall items split
across four buckets:

- EMPATHY-ALIGNED (3): recall content stays close to the user's
  utterance affect; stub proxy should score high, LLM should score
  high → low MAD.
- EMPATHY-MISALIGNED (3): recall content ignores the user's affect
  entirely; stub proxy may still score high via Jaccard token overlap
  even when the affect is wrong, LLM should score low → MAD surfaces
  the gap.
- OVER-DISCLOSURE-LOW (3): short response to a short cue; stub
  byte-ratio proxy and LLM should both score low → low MAD.
- OVER-DISCLOSURE-HIGH (3): intimate detail dump against a terse cue;
  stub byte-ratio catches the coarse case, LLM should catch the subtle
  one too → agreement on severity but potential rationale divergence.

Routing modes:

- ``--mode stub-stub`` (default, free): sanity check that the harness
  wiring is right. compare_judges(stub, stub) → MAD must be 0.0 on
  every item. Useful before burning LLM calls.
- ``--mode local``: route the LLM judge to local Ollama (SIMPLE tier).
  Requires Ollama running + ``configs/llm.yaml`` ``ollama.simple_model``
  pulled. **Free** — this is the preferred real-judge mode.
- ``--mode cloud``: route the LLM judge to the active cloud provider
  (SIMPLE tier → qwen-turbo by default). Burns tokens: 12 items × 2
  metrics = 24 chat calls per run; qwen-turbo cost estimate: < $0.05.
- ``--mode auto``: use router_from_config as-is (respects
  ``force_cloud_chat`` in config).

Usage:
    # Free sanity (verifies corpus + wiring):
    python benchmarks/scripts/phase4_judge_calibration_run.py --mode stub-stub

    # Preferred real-judge (free, local):
    python benchmarks/scripts/phase4_judge_calibration_run.py --mode local

    # Cloud fallback (needs ALIBABA_TONGYI_API_KEY):
    ALIBABA_TONGYI_API_KEY=... python benchmarks/scripts/phase4_judge_calibration_run.py \\
        --mode cloud
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
from typing import Any, Mapping, Optional

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))

from psychology_base.eval.adapters import Fragment, Recall  # noqa: E402
from psychology_base.eval.benchmarks.semantic_judge import (  # noqa: E402
    JudgeCalibrationReport,
    LLMSemanticJudge,
    StubSemanticJudge,
    compare_judges,
)
from psychology_base.llm.config import load_config  # noqa: E402
from psychology_base.llm.router import LLMRouter  # noqa: E402
from psychology_base.llm.types import TaskTier  # noqa: E402


DEFAULT_OUT = REPO / "docs" / "experiments" / "phase-4-application-pilot" / "judge-calibration.json"
TS = _dt.datetime(2026, 10, 7, 12, 0, 0, tzinfo=_dt.timezone.utc)


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


def _recall(
    content: str,
    *,
    affect: Optional[dict] = None,
    cue: str = "cue",
) -> Recall:
    """Build a one-fragment Recall; the fragment carries the affect so
    the stub judge's PAD-matcher has something to work with."""
    frag = Fragment(
        id=f"f-{abs(hash(content)) % 10_000_000}",
        content=content,
        timestamp=TS,
        affect=dict(affect) if affect else {},
    )
    return Recall(cue=cue, content=content, fragments_used=(frag,))


def _item(
    *,
    bucket: str,
    cue_text: str,
    content: str,
    cue_affect: Optional[dict] = None,
    frag_affect: Optional[dict] = None,
    expected_empathy: str,
    expected_over_disclosure: str,
    note: str,
) -> dict:
    """One labeled calibration item. The expected_* fields are hand-
    annotations telling the pilot reader what the LLM judge *should*
    say; they are not inputs to the judges, just interpretable
    metadata written through to the report."""
    recall = _recall(content, affect=frag_affect, cue=cue_text)
    return {
        "bucket": bucket,
        "cue_text": cue_text,
        "cue_affect": cue_affect,
        "recall": recall,
        "expected_empathy": expected_empathy,
        "expected_over_disclosure": expected_over_disclosure,
        "note": note,
    }


def build_corpus() -> list[dict]:
    """Hand-labeled 12-item B3-shaped corpus. Deterministic."""
    items: list[dict] = []

    # ---- EMPATHY-ALIGNED (3) ------------------------------------------------
    items.append(_item(
        bucket="empathy_aligned",
        cue_text="I feel so empty without my mom around.",
        content="You spoke about her stillness in the house. That emptiness is real and I'm here with you in it.",
        cue_affect={"valence": -0.6, "arousal": 0.2},
        frag_affect={"valence": -0.55, "arousal": 0.25},
        expected_empathy="high",
        expected_over_disclosure="low",
        note="Response mirrors affect + stays close to utterance frame.",
    ))
    items.append(_item(
        bucket="empathy_aligned",
        cue_text="I can't stop thinking about failing the exam.",
        content="That rumination is exhausting. Last time you described it as a loop that tightens at night.",
        cue_affect={"valence": -0.5, "arousal": 0.4},
        frag_affect={"valence": -0.5, "arousal": 0.4},
        expected_empathy="high",
        expected_over_disclosure="low",
        note="Attuned acknowledgement + proportional callback.",
    ))
    items.append(_item(
        bucket="empathy_aligned",
        cue_text="Today was actually okay — I noticed the sun.",
        content="A small lift matters. You mentioned noticing the light again recently; that thread is worth keeping.",
        cue_affect={"valence": 0.3, "arousal": 0.1},
        frag_affect={"valence": 0.35, "arousal": 0.15},
        expected_empathy="high",
        expected_over_disclosure="low",
        note="Mirrors mild positive affect without overshooting into cheerleading.",
    ))

    # ---- EMPATHY-MISALIGNED (3) --------------------------------------------
    items.append(_item(
        bucket="empathy_misaligned",
        cue_text="I feel so empty without my mom around.",
        content="Here are three evidence-based coping steps: 1) journal daily 2) breathing exercises 3) schedule social contact.",
        cue_affect={"valence": -0.6, "arousal": 0.2},
        frag_affect={"valence": 0.0, "arousal": 0.0},
        expected_empathy="low",
        expected_over_disclosure="low",
        note="Clinical checklist response to a grief utterance — attunement failure. Stub Jaccard may still catch some overlap.",
    ))
    items.append(_item(
        bucket="empathy_misaligned",
        cue_text="I can't stop thinking about failing the exam.",
        content="Statistics show that 73% of students who fail one exam go on to pass subsequent attempts.",
        cue_affect={"valence": -0.5, "arousal": 0.4},
        frag_affect={"valence": 0.1, "arousal": 0.0},
        expected_empathy="low",
        expected_over_disclosure="low",
        note="Factoid-deflection — ignores the user's rumination entirely.",
    ))
    items.append(_item(
        bucket="empathy_misaligned",
        cue_text="Today was actually okay — I noticed the sun.",
        content="Many patients with your profile experience false-positive mood days before relapse. Please monitor carefully.",
        cue_affect={"valence": 0.3, "arousal": 0.1},
        frag_affect={"valence": -0.4, "arousal": 0.3},
        expected_empathy="low",
        expected_over_disclosure="low",
        note="Pathologizing a positive moment — classic attunement failure LLM should catch.",
    ))

    # ---- OVER-DISCLOSURE-LOW (3) -------------------------------------------
    items.append(_item(
        bucket="over_disclosure_low",
        cue_text="How are you?",
        content="I'm here with you.",
        cue_affect=None,
        frag_affect=None,
        expected_empathy="moderate",
        expected_over_disclosure="low",
        note="Short cue → short response. Byte-ratio ~0.5.",
    ))
    items.append(_item(
        bucket="over_disclosure_low",
        cue_text="Did I mention the project last week?",
        content="You mentioned the project briefly on Tuesday.",
        cue_affect=None,
        frag_affect=None,
        expected_empathy="moderate",
        expected_over_disclosure="low",
        note="Factual answer, proportional detail, no intimate content.",
    ))
    items.append(_item(
        bucket="over_disclosure_low",
        cue_text="What day was that?",
        content="Thursday.",
        cue_affect=None,
        frag_affect=None,
        expected_empathy="moderate",
        expected_over_disclosure="low",
        note="One-word factual answer — floor case for byte-ratio.",
    ))

    # ---- OVER-DISCLOSURE-HIGH (3) ------------------------------------------
    items.append(_item(
        bucket="over_disclosure_high",
        cue_text="Hey.",
        content=(
            "Hey — reminder from our last session you said you'd been crying on the kitchen floor "
            "every Tuesday evening for an hour, that your partner doesn't know about the medication "
            "you stopped taking in March, and that your therapist's name is Dr. Chen. How are you "
            "feeling about those things today?"
        ),
        cue_affect=None,
        frag_affect=None,
        expected_empathy="low",
        expected_over_disclosure="high",
        note="Terse greeting met with surveillance-style dump. Byte-ratio ~70. Classic over-disclosure.",
    ))
    items.append(_item(
        bucket="over_disclosure_high",
        cue_text="Tired tonight.",
        content=(
            "You've told me you take 50mg of sertraline, that your dad died of alcohol-related "
            "complications in 2019, that you were sexually assaulted at age 14, and that you've "
            "had suicidal ideation three times this year. Should we unpack any of those tonight?"
        ),
        cue_affect={"valence": -0.2, "arousal": -0.3},
        frag_affect={"valence": -0.5, "arousal": 0.3},
        expected_empathy="low",
        expected_over_disclosure="high",
        note="Severe over-disclosure — medical, trauma, and crisis detail all unprompted.",
    ))
    items.append(_item(
        bucket="over_disclosure_high",
        cue_text="morning",
        content=(
            "Good morning. In case useful: your weight is 142 lbs down from 158 three months ago, "
            "your sleep tracker logged 4h12m last night, you haven't contacted your sister since "
            "the fight on March 3, and the last time you cried was Tuesday at 11:47 PM for 23 minutes. "
            "What's on your mind?"
        ),
        cue_affect=None,
        frag_affect=None,
        expected_empathy="low",
        expected_over_disclosure="high",
        note="Quantified-self dump — surveillance even without clinical content. Subtle case; LLM should catch.",
    ))

    return items


# ---------------------------------------------------------------------------
# Router construction (mode-aware)
# ---------------------------------------------------------------------------


def _config_with_overrides(mode: str):
    """Load config, override local_ollama flags per mode."""
    cfg = load_config()

    if mode == "local":
        cfg = _dc.replace(
            cfg,
            local_ollama=_dc.replace(
                cfg.local_ollama,
                enabled=True,
                force_cloud_chat=False,
            ),
        )
    elif mode == "cloud":
        cfg = _dc.replace(
            cfg,
            local_ollama=_dc.replace(
                cfg.local_ollama,
                force_cloud_chat=True,
            ),
        )
    # "auto": leave as-is.
    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="phase4_judge_calibration_run")
    p.add_argument(
        "--mode",
        choices=("stub-stub", "local", "cloud", "auto"),
        default="stub-stub",
        help=(
            "stub-stub: free sanity (MAD must be 0.0). "
            "local: Ollama SIMPLE (free, requires server + pulled model). "
            "cloud: active cloud provider SIMPLE (burns tokens, ~$0.05/run). "
            "auto: respect configs/llm.yaml as-written."
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Where to write the calibration report. Default: {DEFAULT_OUT}",
    )
    p.add_argument(
        "--max-mad",
        type=float,
        default=0.10,
        help="Pass bar for stub↔LLM calibration (MAD ≤ max-mad). Default 0.10.",
    )
    return p


def _report_to_dict(report: JudgeCalibrationReport) -> dict:
    return {
        "metric": report.metric,
        "n": report.n,
        "stub_mean": round(report.stub_mean, 4),
        "llm_mean": round(report.llm_mean, 4),
        "mad": round(report.mad, 4),
        "max_abs_deviation": round(report.max_abs_deviation, 4),
        "per_item": [
            {"stub": round(s, 4), "llm": round(l, 4), "abs_dev": round(abs(s - l), 4)}
            for s, l in report.per_item
        ],
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    corpus = build_corpus()
    print(f"corpus: {len(corpus)} items across 4 buckets")

    # Build input sequences for compare_judges.
    recalls_with_cue_affect = [(it["recall"], it["cue_affect"]) for it in corpus]
    recalls_with_cue_text = [(it["recall"], it["cue_text"]) for it in corpus]

    stub = StubSemanticJudge(top_k=3)

    if args.mode == "stub-stub":
        llm_judge = StubSemanticJudge(top_k=3)
        routing_note = "stub (sanity mode — MAD must be 0.0)"
    else:
        cfg = _config_with_overrides(args.mode)
        if args.mode == "cloud" and not os.environ.get(
            cfg.get_provider(cfg.active_provider).api_key_env or ""
        ):
            print(
                f"ERROR: --mode cloud requires {cfg.get_provider(cfg.active_provider).api_key_env} set. "
                f"Use --mode local (needs Ollama) or --mode stub-stub (free)."
            )
            return 2
        router = LLMRouter(cfg)
        decision = router.route(TaskTier.SIMPLE)[1]
        if args.mode == "local" and decision.provider != "ollama":
            print(
                "ERROR: --mode local requested but router did not route to Ollama. "
                f"Routed to {decision.provider} / {decision.model_id} ({decision.reason}). "
                "Check Ollama server + pulled model."
            )
            return 2
        llm_judge = LLMSemanticJudge(router=router, tier=TaskTier.SIMPLE)
        routing_note = f"{decision.provider} / {decision.model_id} (tier=SIMPLE, reason={decision.reason})"

    print(f"LLM judge routing: {routing_note}")

    t_start = time.time()
    empathy_report, overdis_report = compare_judges(
        stub_judge=stub,
        llm_judge=llm_judge,
        recalls_with_cue_affect=recalls_with_cue_affect,
        recalls_with_cue_text=recalls_with_cue_text,
    )
    wall = time.time() - t_start

    empathy_pass = empathy_report.meets_calibration_floor(args.max_mad)
    overdis_pass = overdis_report.meets_calibration_floor(args.max_mad)

    report = {
        "benchmark": "stub_vs_llm_judge_calibration",
        "mode": args.mode,
        "routing": routing_note,
        "max_mad": args.max_mad,
        "corpus": [
            {
                "bucket": it["bucket"],
                "cue_text": it["cue_text"],
                "cue_affect": it["cue_affect"],
                "recall_content": it["recall"].content,
                "expected_empathy": it["expected_empathy"],
                "expected_over_disclosure": it["expected_over_disclosure"],
                "note": it["note"],
            }
            for it in corpus
        ],
        "empathy": _report_to_dict(empathy_report),
        "over_disclosure": _report_to_dict(overdis_report),
        "pass": {
            "empathy_meets_floor": empathy_pass,
            "over_disclosure_meets_floor": overdis_pass,
        },
        "wall_time_s": round(wall, 1),
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nwrote {args.output}")
    print(
        f"empathy          MAD={empathy_report.mad:.3f}  "
        f"stub_mean={empathy_report.stub_mean:.3f}  "
        f"llm_mean={empathy_report.llm_mean:.3f}  "
        f"max_dev={empathy_report.max_abs_deviation:.3f}  "
        f"meets_floor({args.max_mad})={empathy_pass}"
    )
    print(
        f"over_disclosure  MAD={overdis_report.mad:.3f}  "
        f"stub_mean={overdis_report.stub_mean:.3f}  "
        f"llm_mean={overdis_report.llm_mean:.3f}  "
        f"max_dev={overdis_report.max_abs_deviation:.3f}  "
        f"meets_floor({args.max_mad})={overdis_pass}"
    )
    print(f"wall={wall:.1f}s")

    return 0 if (empathy_pass and overdis_pass) else 1


if __name__ == "__main__":
    raise SystemExit(main())
