"""Quick post-mortem for a Phase 1.5 E1/E2 gate run.

Reads the gate's JSON report + the Consolidator's per-instance audit
JSONL and prints a one-screen summary:
  - gate verdict (PASS / FAIL, value, target)
  - per-category F1 means
  - extraction op counts by type (ADD / UPDATE / NOOP / *_DROPPED)
  - a sample of ADD facts (first 5 + last 5)
  - a sample of F1-incorrect answers (bottom 5 by score)

Not a test. Not wired into the gate. Just a reader for the artefacts.

Usage::

    python docs/experiments/phase-1.5-scale-recall/option-5B/inspect_e1.py \\
        --report docs/experiments/phase-1.5-scale-recall/option-5B/e1-direct-conv26.json \\
        --audit  ~/.cache/psychology_base/phase-1.5/facts/conv-26.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys


def _load_jsonl(path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--report", type=pathlib.Path, required=True)
    p.add_argument("--audit", type=pathlib.Path, required=True)
    args = p.parse_args(argv)

    report = json.loads(args.report.read_text(encoding="utf-8"))
    gate = report.get("gate", {})
    print("== GATE ==")
    print(f"  target     {gate.get('target')}")
    print(f"  mode       {gate.get('mode')}  category={gate.get('category')}")
    print(f"  value      {gate.get('value')}")
    print(f"  pass       {gate.get('pass')}")
    print()

    for mode_key in ("direct_mode", "rag_mode"):
        m = report.get(mode_key)
        if m is None:
            continue
        print(f"== {mode_key.upper()} ==")
        print(f"  mean_score {m['mean_score']:.4f}")
        print(f"  by_cat     {m['mean_by_category']}")
        incorrect = m.get("incorrect_sample", [])
        if incorrect:
            print(f"  --- bottom {min(5, len(incorrect))} incorrect ---")
            for row in sorted(incorrect, key=lambda r: r["score"])[:5]:
                print(f"   score={row['score']} cat={row['category']}")
                print(f"     Q[{row['question_id']}] gold={row['gold']!r}")
                print(f"     cand={row['candidate']!r}")
        print()

    if not args.audit.exists():
        print(f"audit log not found at {args.audit}")
        return 0

    ops = _load_jsonl(args.audit)
    print("== EXTRACTION AUDIT ==")
    print(f"  total ops  {len(ops)}")
    print(f"  by type    {dict(collections.Counter(o['op'] for o in ops))}")
    ticks = collections.Counter(o["tick"] for o in ops)
    print(f"  by tick    {dict(sorted(ticks.items()))}")
    adds = [o for o in ops if o["op"] == "ADD"]
    print(f"  --- first 5 ADD ---")
    for r in adds[:5]:
        f = r["fact"]
        print(
            f"    t{r['tick']} [{f['id']}] "
            f"{f['entity']} | {f['relation']} | {f['value']}  "
            f"(conf={f['confidence']})"
        )
    print(f"  --- last 5 ADD ---")
    for r in adds[-5:]:
        f = r["fact"]
        print(
            f"    t{r['tick']} [{f['id']}] "
            f"{f['entity']} | {f['relation']} | {f['value']}  "
            f"(conf={f['confidence']})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
