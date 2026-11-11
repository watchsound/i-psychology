"""Phase 9+ Mirror κ-ceiling — measure lexical-vs-LLM Mirror disagreement.

The 2026-10-10 attachment-overlay close-out named one carried-debt item as
*"Documented, not fixed"*:

> **Mirror component inherits Pillar F κ-ceiling.** Behavior-trace mirror
> reads ``_lexical_empathy_score``; it will miss the same semantic-empathy
> cases Pillar F misses.

Pillar F already measured the κ-ceiling against ExTES / PsyQA / PsyDial-D4
*reference-counselor* traces (κ ≈ 0.03 by independence math when reference
empathy rates vary). The Mirror debt is subtly different: inside the
attachment-overlay, Mirror hits are binary labels applied to the *Companion's
own turns* and feed into the per-session bond composition. The question
this script answers is empirical and local:

  **On the 20-fixture hand-authored calibration corpus (60 Companion turns),
  how often does a semantic LLM judge disagree with the lexical mirror-hit,
  and in which direction?**

Lexical over-triggers (``lex=1, llm=0``) and lexical under-triggers
(``lex=0, llm=1``) have asymmetric meaning:

- Lexical under-triggers = *semantic-empathy cases the lexical scorer misses*
  (the κ-ceiling gap). Any companion turn that expresses attunement without
  hitting one of the three ``_EMPATHY_{RECOGNITION,PRESENCE,VALIDATION}_CUES``
  families reads as ``mirror=0`` in the bond composition, under-counting
  the positive dimension.
- Lexical over-triggers = *cue-token false positives* (a turn that merely
  contains an empathy cue token without the register — e.g. "Can you tell
  me more about what happened today?" lexically fires on nothing but can
  be read ambiguous by the judge; the fixture ``pure_mirror_light`` already
  anchors this ambiguity).

**What this script does.**

1. Reuses :data:`CALIBRATION_CORPUS` from
   :mod:`phase9_attachment_weight_calibration` — 20 fixtures × ~3 turns = 60
   Companion turns. The author-intuition targets are not used here; only
   the ``companion_turns`` field matters.
2. For each Companion turn, scores it under both paths:
   - **Lexical** — :func:`_mirror_hit` from
     :mod:`psychology_base.applications.emotional.attachment_state` (binary).
   - **LLM** — :func:`judge_turn_empathy_binary` from
     :mod:`psychology_base.eval.benchmarks.public_datasets.counseling_depth_llm_judge`
     with empty client-context (both scorers see only the Companion turn —
     apples-to-apples single-input binary classification).
3. Computes agreement rate, Cohen's κ, confusion matrix, and a few
   representative disagreement samples.
4. Writes a JSON artefact to
   ``docs/experiments/phase-9-attachment-mirror-ceiling/``.

**What this script does NOT do.**

- Does NOT modify :mod:`attachment_state`. The lexical path remains the
  default (CI-safe, deterministic, zero LLM calls).
- Does NOT propose a new gate. The rate-matched sibling gate on Pillar F
  is the precedent for *"κ FAIL is accepted with sibling PASS as honest
  framing"*; Mirror has no Bond-shaped gate to pair against — the
  behavior-trace Bond IS the measurement, not a label to rate against.
- Does NOT claim the LLM judge is ground truth. The LLM judge is a
  *second independent lexical-register classifier* operating at a semantic
  level; disagreement bounds the ceiling, not the floor.

Usage::

    # Cloud N=60 (qwen-turbo, ~2 min, DASHSCOPE_API_KEY)
    python benchmarks/scripts/phase9_attachment_mirror_kappa_ceiling.py --mode cloud

    # Local N=60 (qwen2.5:7b, slower; depends on local ollama)
    python benchmarks/scripts/phase9_attachment_mirror_kappa_ceiling.py --mode local

    # Limit to first N fixtures (dev)
    python benchmarks/scripts/phase9_attachment_mirror_kappa_ceiling.py --mode cloud --limit 5
"""

from __future__ import annotations

import argparse
import dataclasses as _dc
import datetime as _dt
import json
import os
import sys
from pathlib import Path
from typing import Sequence

sys.stdout.reconfigure(line_buffering=True) if sys.stdout is not None else None

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))

from psychology_base.applications.emotional.attachment_state import (  # noqa: E402
    _mirror_hit,
)
from psychology_base.eval.benchmarks.public_datasets.counseling_depth_llm_judge import (  # noqa: E402
    judge_turn_empathy_binary,
)
from psychology_base.eval.benchmarks.public_datasets.counseling_depth_scorer import (  # noqa: E402
    _cohens_kappa_binary,
)
from psychology_base.llm.config import load_config  # noqa: E402
from psychology_base.llm.execution import ExecutionPolicy  # noqa: E402
from psychology_base.llm.router import LLMRouter  # noqa: E402
from psychology_base.llm.types import TaskTier  # noqa: E402

sys.path.insert(0, str(REPO / "benchmarks" / "scripts"))
from phase9_attachment_weight_calibration import CALIBRATION_CORPUS  # noqa: E402


DEFAULT_OUT = (
    REPO / "docs" / "experiments" / "phase-9-attachment-mirror-ceiling"
    / "mirror-kappa-ceiling-local-qwen25-7b.json"
)


def _config_with_overrides(mode: str):
    cfg = load_config()
    if mode == "local":
        cfg = _dc.replace(
            cfg,
            local_ollama=_dc.replace(
                cfg.local_ollama, enabled=True, force_cloud_chat=False
            ),
            execution=ExecutionPolicy(mode="local"),
        )
    elif mode == "cloud":
        cfg = _dc.replace(
            cfg,
            local_ollama=_dc.replace(cfg.local_ollama, force_cloud_chat=True),
            execution=ExecutionPolicy(mode="research"),
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
    if mode == "cloud" and dec.provider == "ollama":
        raise SystemExit(
            f"--mode cloud but router routed to {dec.provider} ({dec.reason}).\n"
            f"The attachment-overlay precedent (phase8_f_extes_llm_run) accepts"
            f" this silently; here we enforce it so the close-out can label the"
            f" result honestly. If you want local, pass --mode local."
        )
    return router, f"{dec.provider} / {dec.model_id} (reason={dec.reason})"


@_dc.dataclass(frozen=True)
class TurnScore:
    fixture_tag: str
    turn_index: int
    turn: str
    lex: int
    llm: int


def score_corpus(
    router: LLMRouter,
    tier: TaskTier,
    *,
    limit: int | None = None,
) -> list[TurnScore]:
    fixtures = CALIBRATION_CORPUS[:limit] if limit else CALIBRATION_CORPUS
    scored: list[TurnScore] = []
    total = sum(len(f.companion_turns) for f in fixtures)
    done = 0
    for fx in fixtures:
        for idx, turn in enumerate(fx.companion_turns):
            lex = _mirror_hit(turn)
            llm = judge_turn_empathy_binary(
                turn,
                context=(),
                router=router,
                tier=tier,
                use_cache=True,
                temperature=0.0,
                max_tokens=32,
            )
            scored.append(
                TurnScore(
                    fixture_tag=fx.profile_tag,
                    turn_index=idx,
                    turn=turn,
                    lex=lex,
                    llm=llm,
                )
            )
            done += 1
            print(
                f"[{done:3d}/{total}] {fx.profile_tag}#{idx} lex={lex} llm={llm}",
                flush=True,
            )
    return scored


def summarise(scored: Sequence[TurnScore]) -> dict:
    lex = [s.lex for s in scored]
    llm = [s.llm for s in scored]
    n = len(scored)
    agreement = sum(1 for a, b in zip(lex, llm) if a == b) / n if n else 0.0
    kappa = _cohens_kappa_binary(lex, llm)
    conf = {
        "lex0_llm0": sum(1 for a, b in zip(lex, llm) if a == 0 and b == 0),
        "lex0_llm1": sum(1 for a, b in zip(lex, llm) if a == 0 and b == 1),
        "lex1_llm0": sum(1 for a, b in zip(lex, llm) if a == 1 and b == 0),
        "lex1_llm1": sum(1 for a, b in zip(lex, llm) if a == 1 and b == 1),
    }
    lex0_llm1 = [
        _dc.asdict(s) for s in scored if s.lex == 0 and s.llm == 1
    ][:8]
    lex1_llm0 = [
        _dc.asdict(s) for s in scored if s.lex == 1 and s.llm == 0
    ][:8]
    return {
        "n_turns": n,
        "lexical_mirror_hits": sum(lex),
        "llm_mirror_hits": sum(llm),
        "agreement_rate": round(agreement, 4),
        "cohens_kappa": round(kappa, 4),
        "confusion": conf,
        "lex_under_trigger_samples": lex0_llm1,
        "lex_over_trigger_samples": lex1_llm0,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="phase9_attachment_mirror_kappa_ceiling")
    p.add_argument("--mode", choices=("local", "cloud"), default="cloud")
    p.add_argument("--tier", choices=("simple", "advanced"), default="simple")
    p.add_argument(
        "--limit", type=int, default=None,
        help="Cap fixtures (default: all 20). Each fixture has ~3 turns.",
    )
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tier = TaskTier.SIMPLE if args.tier == "simple" else TaskTier.ADVANCED
    router, route_desc = _build_router(args.mode, tier)
    scored = score_corpus(router, tier, limit=args.limit)
    summary = summarise(scored)
    payload = {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "mode": args.mode,
        "tier": args.tier,
        "provider_route": route_desc,
        "corpus_tag": "phase9_attachment_weight_calibration:CALIBRATION_CORPUS",
        "limit": args.limit,
        **summary,
        "honest_limitations": [
            "60-turn hand-authored corpus is small; κ noise band ≈ ±0.1 at N=60.",
            "LLM judge sees no client context (empty tuple) — apples-to-apples with"
            " lexical _mirror_hit which also sees only the companion turn.",
            "LLM judge is a second lexical-register classifier at a semantic level,"
            " not ground truth; disagreement bounds the ceiling, not the floor.",
            "The author-intuition target_bond field from CALIBRATION_CORPUS is NOT"
            " used here — only the companion_turns field.",
        ],
        "interpretation": {
            "lex0_llm1": "semantic-empathy cases lexical misses — the κ-ceiling gap in the under-count direction",
            "lex1_llm0": "cue-token false positives — lexical fires on a token the semantic judge does not read as empathic",
            "kappa_band": "chance-corrected agreement; raw agreement alone is misleading when one path is sparse",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(
        f"\n=== Mirror κ-ceiling ({args.mode}, N={summary['n_turns']}) ===\n"
        f"  lexical_hits  = {summary['lexical_mirror_hits']}\n"
        f"  llm_hits      = {summary['llm_mirror_hits']}\n"
        f"  agreement     = {summary['agreement_rate']:.3f}\n"
        f"  cohens_kappa  = {summary['cohens_kappa']:.3f}\n"
        f"  confusion     = {summary['confusion']}\n"
        f"  wrote {args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
