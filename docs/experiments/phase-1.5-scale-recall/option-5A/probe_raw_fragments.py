"""Option 5.A diagnostic probe — raw-fragment cosine retrieval, no extraction.

E1 v1-v4 (Option 5.B, session-level LLM fact extraction) plateaued at
single-hop F1 ≈ 0.03 on conv-26. Hypothesis: session-level extraction is
structurally lossy for trivia-style QA — a 7B SIMPLE-tier LLM over a 19-
session dialogue produces ~100 ADDs that miss specific proper nouns the
LoCoMo evaluator checks (Sweden, pottery, sunset, "Charlotte's Web").

This probe bypasses CompanionSystem entirely:
  1. Load conv-26 via LoCoMoLoader (unchanged shim).
  2. Each turn's content becomes a 'fragment' (no byte_cap, no purge).
  3. Embed every fragment with mxbai-embed-large (router.embed_batch).
  4. For each question: embed cue, cosine-rank fragments, take top-K.
  5. Join top-K content, score token F1 against gold.

If this clears F1 ≥ 0.40, Option 5.A is the right architecture: raw
fragments + retrieval-tuned embeddings + cosine re-rank. If it also
fails, the problem is deeper than extraction (dialogue references like
'4 years ago' → 'Sweden' aren't surface-matchable), and we need
Option 5.C (hierarchical tiering) or a reader-based path.

Not a product surface — a diagnostic. Does not touch CompanionSystem
or the configured consolidation layer.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from psychology_base.eval.benchmarks.locomo import (  # noqa: E402
    LoCoMoLoader,
    token_f1,
)
from psychology_base.llm.router import router_from_config  # noqa: E402


DEFAULT_SHIM = (
    Path.home() / ".cache" / "psychology_base" / "benchmarks" / "locomo" / "locomo_shim.jsonl"
)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="probe_raw_fragments")
    p.add_argument("--shim-path", type=Path, default=DEFAULT_SHIM)
    p.add_argument("--instance-id", default="conv-26")
    p.add_argument("--category", default="single-hop")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument(
        "--output",
        type=Path,
        default=HERE / "probe-raw-fragments.json",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    router = router_from_config()

    loader = LoCoMoLoader(shim_path=args.shim_path)
    instances = [
        i for i in loader.iter_instances() if i.instance_id == args.instance_id
    ]
    if not instances:
        print(f"no instance_id={args.instance_id!r}", file=sys.stderr)
        return 1
    instance = instances[0]

    questions = [q for q in instance.questions if q.question_type == args.category]
    print(
        f"instance={instance.instance_id}  turns={len(instance.history)}  "
        f"questions[{args.category}]={len(questions)}  top_k={args.top_k}"
    )

    t_embed = time.time()
    frag_texts = [
        f"{turn.speaker}: {turn.content}" if turn.speaker else turn.content
        for turn in instance.history
    ]
    frag_embeds = router.embed_batch(frag_texts, max_workers=4)
    print(f"embedded {len(frag_embeds)} fragments in {time.time() - t_embed:.1f}s")

    t_q = time.time()
    per_question = []
    f1s: list[float] = []
    for q in questions:
        cue_emb = router.embed(q.cue)
        scored = [
            (_cosine(cue_emb, fe), i)
            for i, fe in enumerate(frag_embeds)
        ]
        scored.sort(key=lambda x: -x[0])
        top = scored[: args.top_k]
        content = " ".join(frag_texts[i] for _, i in top)
        f1 = token_f1(q.gold, content)
        f1s.append(f1)
        per_question.append(
            {
                "question_id": q.id,
                "cue": q.cue,
                "gold": q.gold,
                "top_cos": round(top[0][0], 3) if top else 0.0,
                "top_content_head": content[:200],
                "f1": round(f1, 3),
            }
        )
    print(f"scored {len(questions)} questions in {time.time() - t_q:.1f}s")

    mean_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    report = {
        "benchmark": "locomo",
        "instance_id": instance.instance_id,
        "category": args.category,
        "top_k": args.top_k,
        "mean_f1": mean_f1,
        "per_question": per_question,
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    print(f"mean f1 = {mean_f1:.4f}  (gate 0.40 = {'PASS' if mean_f1 >= 0.40 else 'FAIL'})")
    return 0 if mean_f1 >= 0.40 else 2


if __name__ == "__main__":
    sys.exit(main())
