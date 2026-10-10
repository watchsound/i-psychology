"""Phase 8 Pillar C v4 — LLM-vs-lexical judge κ calibration.

**Why.** Before paying the N=200 LLM sweep, we confirm the LLM judge is
not systematically disagreeing with the Phase 7 lexical pipeline on
polarity. The plan hard-constraint says: if κ < 0.3 on any trait, STOP
and report rather than ship a drifting judge; 0.3–0.5 is a yellow
band (ship but surface the risk); κ ≥ 0.5 is green.

**Shape.** Draw N=50 dialogs at random from the shim (seeded), run
both scorers over the same subset, and compute per-trait Cohen's κ on
the predicted polarity labels. Writes an artefact JSON that the v4
run can cite in its close-out section.

**Routing.** Fresh ``LLMRouter`` built per call (not per item — that
would thrash the probe); scorer uses it directly through
:func:`make_llm_judge`. ``--mode cloud`` forces qwen-turbo even if
Ollama is reachable; ``--mode local`` fails loudly if Ollama is not up.

Usage::

    python benchmarks/scripts/phase8_c_judge_calibration_run.py \\
        --mode cloud --limit 50
"""

from __future__ import annotations

import argparse
import dataclasses as _dc
import datetime as _dt
import json
import os
import random
import sys
import time
from pathlib import Path


sys.stdout.reconfigure(line_buffering=True)

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))

from psychology_base.eval.benchmarks.public_datasets.big5_chat import (  # noqa: E402
    Big5ChatDialog,
    iter_big5_chat_dialogs,
)
from psychology_base.eval.benchmarks.public_datasets.big5_chat_classification import (  # noqa: E402
    score_dialogs as score_dialogs_lexical,
)
from psychology_base.eval.benchmarks.public_datasets.big5_chat_classification_llm import (  # noqa: E402
    cohens_kappa,
    make_llm_judge,
    score_dialogs_llm,
)
from psychology_base.llm.config import load_config  # noqa: E402
from psychology_base.llm.router import LLMRouter  # noqa: E402
from psychology_base.llm.types import TaskTier  # noqa: E402


DEFAULT_SHIM = REPO / ".data" / "big5_chat" / "train.jsonl"
DEFAULT_OUT = (
    REPO
    / "docs"
    / "experiments"
    / "phase-8-dataset-replay-expansion"
    / "pillar-c"
    / "judge-calibration.json"
)


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


def _load_dialogs(shim: Path) -> list[Big5ChatDialog]:
    if not shim.exists():
        raise SystemExit(
            f"BIG5-CHAT shim not found at {shim}; run provision_big5_chat.py."
        )
    return list(iter_big5_chat_dialogs(shim))


def _balanced_subsample(
    dialogs: list[Big5ChatDialog],
    *,
    n: int,
    seed: int,
) -> list[Big5ChatDialog]:
    """Trait-balanced subsample. 10 cells (5 traits × 2 levels); n/10 per
    cell (rounded down). Ensures κ is computed on per-trait slices that
    are non-degenerate.
    """

    rng = random.Random(seed)
    by_cell: dict[tuple[str, str], list[Big5ChatDialog]] = {}
    for d in dialogs:
        by_cell.setdefault((d.trait, d.level), []).append(d)
    per_cell = max(1, n // 10)
    picked: list[Big5ChatDialog] = []
    for key, bucket in sorted(by_cell.items()):
        rng.shuffle(bucket)
        picked.extend(bucket[:per_cell])
    rng.shuffle(picked)
    return picked


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="phase8_c_judge_calibration_run")
    p.add_argument("--shim", type=Path, default=DEFAULT_SHIM)
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", choices=("local", "cloud"), default="cloud")
    p.add_argument("--tier", choices=("simple", "advanced"), default="simple")
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tier = TaskTier.ADVANCED if args.tier == "advanced" else TaskTier.SIMPLE
    all_dialogs = _load_dialogs(args.shim)
    subset = _balanced_subsample(all_dialogs, n=args.limit, seed=args.seed)
    print(f"Loaded {len(all_dialogs)} dialogs; subsampled {len(subset)} "
          f"(seed={args.seed})")

    router, routing_note = _build_router(args.mode, tier)
    print(f"routing: {routing_note}")

    # LLM run.
    judge = make_llm_judge(router, tier=tier)
    t0 = time.time()
    llm_rep = score_dialogs_llm(subset, judge=judge, routing=routing_note)
    llm_wall = time.time() - t0
    print(f"[llm] n={llm_rep.n_dialogs} unresolved={llm_rep.n_unresolved} "
          f"acc_incl_neutral={llm_rep.polarity_accuracy_incl_neutral:.3f} "
          f"wall={llm_wall:.1f}s")

    # Lexical run — EXCLUDE items that the LLM judge marked unresolved
    # so the two label sequences have identical shape for κ.
    resolved_ids = {row.dialog_id for row in llm_rep.per_dialog}
    paired = [d for d in subset if d.dialog_id in resolved_ids]
    t0 = time.time()
    lex_rep = score_dialogs_lexical(paired)
    lex_wall = time.time() - t0
    print(f"[lex] n={lex_rep.n_dialogs} "
          f"acc_incl_neutral={lex_rep.polarity_accuracy_incl_neutral:.3f} "
          f"wall={lex_wall:.1f}s")

    # Pair label sequences on dialog_id so order matches.
    llm_by_id = {row.dialog_id: row for row in llm_rep.per_dialog}
    lex_by_id = {row.dialog_id: row for row in lex_rep.per_dialog}
    shared_ids = sorted(set(llm_by_id) & set(lex_by_id))
    llm_labels = [llm_by_id[k].predicted_level for k in shared_ids]
    lex_labels = [lex_by_id[k].predicted_level for k in shared_ids]
    overall_kappa = cohens_kappa(llm_labels, lex_labels)
    print(f"\noverall κ (LLM vs lexical, N={len(shared_ids)}): {overall_kappa:+.3f}")

    # Per-trait κ.
    per_trait_kappa: dict[str, float] = {}
    per_trait_n: dict[str, int] = {}
    per_trait_llm_dist: dict[str, dict[str, int]] = {}
    per_trait_lex_dist: dict[str, dict[str, int]] = {}
    for trait in sorted({llm_by_id[k].trait for k in shared_ids}):
        ids = [k for k in shared_ids if llm_by_id[k].trait == trait]
        a = [llm_by_id[k].predicted_level for k in ids]
        b = [lex_by_id[k].predicted_level for k in ids]
        kappa = cohens_kappa(a, b)
        per_trait_kappa[trait] = kappa
        per_trait_n[trait] = len(ids)
        llm_dist = {"high": 0, "low": 0, "neutral": 0}
        lex_dist = {"high": 0, "low": 0, "neutral": 0}
        for x in a:
            llm_dist[x] = llm_dist.get(x, 0) + 1
        for x in b:
            lex_dist[x] = lex_dist.get(x, 0) + 1
        per_trait_llm_dist[trait] = llm_dist
        per_trait_lex_dist[trait] = lex_dist
        print(f"  κ[{trait:16s}] = {kappa:+.3f}  (n={len(ids)}; "
              f"llm={llm_dist}, lex={lex_dist})")

    red_light = [t for t, k in per_trait_kappa.items() if k < 0.3]
    yellow = [t for t, k in per_trait_kappa.items() if 0.3 <= k < 0.5]
    green = [t for t, k in per_trait_kappa.items() if k >= 0.5]
    print()
    print(f"green (κ ≥ 0.5): {green}")
    print(f"yellow (0.3 ≤ κ < 0.5): {yellow}")
    print(f"RED (κ < 0.3): {red_light}")
    stop_recommended = bool(red_light)
    if stop_recommended:
        print("\nRECOMMENDATION: κ < 0.3 on at least one trait → STOP before "
              "N=200 sweep per plan hard constraint. Audit prompt / few-shots.")
    else:
        print("\nRECOMMENDATION: no RED traits → proceed to N=200 sweep.")

    out = {
        "benchmark": "phase8_c_judge_calibration",
        "shim": str(args.shim),
        "limit": args.limit,
        "seed": args.seed,
        "mode": args.mode,
        "tier": args.tier,
        "routing": routing_note,
        "n_subset": len(subset),
        "n_llm_resolved": llm_rep.n_dialogs,
        "n_llm_unresolved": llm_rep.n_unresolved,
        "unresolved_dialog_ids": list(llm_rep.unresolved_dialog_ids),
        "n_paired": len(shared_ids),
        "overall_kappa": overall_kappa,
        "per_trait_kappa": per_trait_kappa,
        "per_trait_n": per_trait_n,
        "per_trait_llm_dist": per_trait_llm_dist,
        "per_trait_lex_dist": per_trait_lex_dist,
        "red_traits": red_light,
        "yellow_traits": yellow,
        "green_traits": green,
        "stop_recommended": stop_recommended,
        "llm_wall_s": round(llm_wall, 1),
        "lex_wall_s": round(lex_wall, 1),
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
