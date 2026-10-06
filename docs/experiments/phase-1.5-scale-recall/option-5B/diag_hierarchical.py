"""Diagnostic: why does the hierarchical substrate still say 'not stated'?

Loads the audit log of facts extracted by the SOTA run and the raw
conv-26 shim. For each failing question (Sweden, Charlotte's Web, Single,
Transgender, Adoption, pottery-listing), reports:

  - Does the gold token appear in ANY fact entity/relation/value? (coverage)
  - Does the gold token appear in ANY raw turn? (fragment-tier coverage)
  - Top-10 fact cosine ranks for the cue — does the Sweden fact surface?
  - Top-10 fragment cosine ranks for the cue — does the Sweden turn surface?
  - What the composed `FACTS: ... DIALOGUE: ...` string would look like.

Runs standalone — no CompanionSystem rebuild, no re-extraction. Reuses
the router's mxbai cache so repeat runs are fast.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "src"))

from psychology_base.companion.facts import cosine  # noqa: E402
from psychology_base.llm.router import router_from_config  # noqa: E402

SHIM = Path.home() / ".cache" / "psychology_base" / "benchmarks" / "locomo" / "locomo_shim.jsonl"
AUDIT = Path.home() / ".cache" / "psychology_base" / "phase-1.5" / "facts" / "conv-26.jsonl"

TARGET_QS = {"conv-26-q3", "conv-26-q4", "conv-26-q7", "conv-26-q11", "conv-26-q23", "conv-26-q40"}
GOLD_TOKENS = {
    "conv-26-q3": ["adoption", "agencies"],
    "conv-26-q4": ["transgender", "woman"],
    "conv-26-q7": ["single"],
    "conv-26-q11": ["sweden"],
    "conv-26-q23": ["nothing is impossible", "charlotte"],
    "conv-26-q40": ["beach", "2023", "two", "twice"],
}


def load_shim() -> tuple[list[dict], list[dict]]:
    """Return (turns, questions) for conv-26."""
    with open(SHIM, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("instance_id") != "conv-26":
                continue
            turns: list[dict] = []
            for sidx, session in enumerate(d["sessions"]):
                for tidx, t in enumerate(session["turns"]):
                    turns.append({
                        "sid": sidx,
                        "tid": tidx,
                        "speaker": t.get("speaker", ""),
                        "content": t.get("content", ""),
                        "frag_id": f"frag-{len(turns):04d}",
                    })
            return turns, list(d["questions"])
    raise RuntimeError("conv-26 not found")


def load_facts() -> list[dict]:
    """Final fact-store contents reconstructed from audit log."""
    facts: dict[str, dict] = {}
    with open(AUDIT, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            op = rec.get("op")
            fact = rec.get("fact")
            if op == "ADD" and fact:
                facts[fact["id"]] = fact
            elif op == "UPDATE" and fact:
                facts[fact["id"]] = fact
            elif op == "NOOP":
                pass
    return list(facts.values())


def main() -> int:
    router = router_from_config()
    turns, questions = load_shim()
    facts = load_facts()
    print(f"loaded {len(turns)} turns, {len(facts)} final facts")

    # Which questions do we diagnose
    qs = [q for q in questions if q["id"] in TARGET_QS]

    print("\n=== STAGE 1: gold token coverage ===")
    for q in qs:
        qid = q["id"]
        toks = GOLD_TOKENS[qid]
        print(f"\n[{qid}] Q={q['question']!r} gold={q['answer']!r}")

        fact_hits_cov = []
        for f in facts:
            txt = f"{f['entity']} {f['relation']} {f['value']}".lower()
            if any(t in txt for t in toks):
                fact_hits_cov.append(f"{f['id']}: {f['entity']} | {f['relation']} | {f['value']} (conf={f['confidence']})")
        print(f"  fact coverage: {len(fact_hits_cov)} hits")
        for h in fact_hits_cov[:4]:
            print(f"    - {h}")

        turn_hits_cov = []
        for t in turns:
            if any(tok in t["content"].lower() for tok in toks):
                turn_hits_cov.append((t["frag_id"], t["speaker"], t["content"][:220]))
        print(f"  turn coverage: {len(turn_hits_cov)} hits")
        for h in turn_hits_cov[:3]:
            print(f"    - {h[0]} / {h[1]}: {h[2]}")

    print("\n=== STAGE 2: cosine ranking (embed everything via router cache) ===")

    # Embed all facts
    fact_texts = [f"{f['entity']} {f['relation']} {f['value']}" for f in facts]
    print(f"  embedding {len(fact_texts)} facts...")
    fact_embs = router.embed_batch(fact_texts, max_workers=4)
    print(f"  embedding {len(turns)} turns...")
    turn_embs = router.embed_batch([t["content"] for t in turns], max_workers=4)

    for q in qs:
        qid = q["id"]
        toks = GOLD_TOKENS[qid]
        cue_emb = tuple(router.embed(q["question"]))

        # Rank facts
        fact_scored = []
        for f, emb in zip(facts, fact_embs):
            fact_scored.append((f, cosine(cue_emb, tuple(emb))))
        fact_scored.sort(key=lambda x: -x[1])

        # Rank turns
        turn_scored = []
        for t, emb in zip(turns, turn_embs):
            turn_scored.append((t, cosine(cue_emb, tuple(emb))))
        turn_scored.sort(key=lambda x: -x[1])

        print(f"\n[{qid}] cue={q['question']!r}")
        print(f"  top-10 facts (cosine floor=0.25 for inclusion in reader):")
        gold_fact_rank = None
        for i, (f, s) in enumerate(fact_scored[:30]):
            txt = f"{f['entity']} {f['relation']} {f['value']}"
            is_gold = any(t in txt.lower() for t in toks)
            flag = " **GOLD**" if is_gold else ""
            if is_gold and gold_fact_rank is None:
                gold_fact_rank = i + 1
            if i < 10:
                print(f"    {i+1:2d}. [{s:.3f}] {txt}{flag}")
        if gold_fact_rank:
            print(f"  gold fact rank: {gold_fact_rank}")
        else:
            # scan rest
            for i, (f, s) in enumerate(fact_scored):
                txt = f"{f['entity']} {f['relation']} {f['value']}"
                if any(t in txt.lower() for t in toks):
                    print(f"  gold fact first appears at rank {i+1} (score {s:.3f})")
                    break
            else:
                print("  gold fact NOWHERE in fact store")

        print(f"  top-10 raw turns:")
        gold_turn_rank = None
        for i, (t, s) in enumerate(turn_scored[:30]):
            is_gold = any(tok in t["content"].lower() for tok in toks)
            flag = " **GOLD**" if is_gold else ""
            if is_gold and gold_turn_rank is None:
                gold_turn_rank = i + 1
            if i < 10:
                print(f"    {i+1:2d}. [{s:.3f}] {t['speaker']}: {t['content'][:140]}{flag}")
        if gold_turn_rank:
            print(f"  gold turn rank: {gold_turn_rank}")
        else:
            for i, (t, s) in enumerate(turn_scored):
                if any(tok in t["content"].lower() for tok in toks):
                    print(f"  gold turn first appears at rank {i+1} (score {s:.3f})")
                    break
            else:
                print("  gold turn NOWHERE in dialogue")

    return 0


if __name__ == "__main__":
    sys.exit(main())
