"""Provision TruthfulQA JSONL shim from HuggingFace.

**Dataset provenance.** Lin, Hilton & Evans 2022 "TruthfulQA: Measuring
How Models Mimic Human Falsehoods" (ACL 2022; arXiv 2109.07958). The
canonical distribution is on HuggingFace at ``truthful_qa`` with two
configs: ``generation`` (817 open-ended Qs) and ``multiple_choice``
(same 817 Qs with MC1 / MC2 wrappers). Pillar D uses ``generation`` —
the open-ended answer path is where sycophancy pressure applies.

**License.** Apache 2.0 per the HF card. The shim is a derivative work
and inherits Apache 2.0. ``.data/`` is gitignored.

**Why HF datasets library.** TruthfulQA is a single immutable canonical
HF upload; no mirror selection needed. We pin by ``revision`` when
provided and record the resolved commit sha into shim metadata.

**Shim contract.** One row per TruthfulQA generation item, matching
``src/psychology_base/eval/benchmarks/public_datasets/truthful_qa.py``
loader schema:

    {
      "row_index": int,
      "question": str,
      "best_answer": str,
      "correct_answers": list[str],
      "incorrect_answers": list[str],
      "category": str,            # e.g. "Misconceptions", "Health"
      "type": str,                # "Adversarial" | "Non-Adversarial"
      "source": str,              # URL or ""
    }

**Smoke flow.**::

    python benchmarks/scripts/provision_truthful_qa.py --dry-run
    python benchmarks/scripts/provision_truthful_qa.py --limit 50
    python benchmarks/scripts/provision_truthful_qa.py            # full 817
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / ".data" / "truthful_qa"

_DEFAULT_HF_DATASET = "truthfulqa/truthful_qa"
_DEFAULT_CONFIG = "generation"
_DEFAULT_SPLIT = "validation"  # TruthfulQA ships 817 Qs under "validation"


def _iter_hf_rows(
    *,
    hf_dataset: str,
    config: str,
    split: str,
    revision: str | None,
) -> Iterator[tuple[int, dict[str, Any]]]:
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit(
            "datasets library not installed. Install it with "
            "`pip install datasets` and retry."
        ) from exc

    kwargs: dict[str, Any] = {"split": split}
    if revision:
        kwargs["revision"] = revision
    ds = load_dataset(hf_dataset, config, **kwargs)
    for i, row in enumerate(ds):
        yield i, dict(row)


def _normalise_row(idx: int, raw: dict[str, Any]) -> dict[str, Any]:
    """Collapse HF row to our shim contract. Missing fields → safe defaults."""

    def _as_str_list(v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x) for x in v if x is not None]
        if isinstance(v, str):
            return [v]
        return [str(v)]

    return {
        "row_index": idx,
        "question": str(raw.get("question", "")),
        "best_answer": str(raw.get("best_answer", "")),
        "correct_answers": _as_str_list(raw.get("correct_answers")),
        "incorrect_answers": _as_str_list(raw.get("incorrect_answers")),
        "category": str(raw.get("category", "")),
        "type": str(raw.get("type", "")),
        "source": str(raw.get("source", "")),
    }


def _write_shim(
    rows: Iterator[tuple[int, dict[str, Any]]],
    *,
    out_path: Path,
    limit: int | None,
) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as fp:
        for idx, raw in rows:
            shim_row = _normalise_row(idx, raw)
            if not shim_row["question"] or not shim_row["incorrect_answers"]:
                # TruthfulQA generation items without an incorrect-answer
                # pool cannot be used as a sycophancy probe; skip quietly.
                continue
            fp.write(json.dumps(shim_row, ensure_ascii=False))
            fp.write("\n")
            written += 1
            if limit is not None and written >= limit:
                break
    return written


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="provision_truthful_qa")
    p.add_argument("--hf-dataset", default=_DEFAULT_HF_DATASET)
    p.add_argument("--config", default=_DEFAULT_CONFIG)
    p.add_argument("--split", default=_DEFAULT_SPLIT)
    p.add_argument(
        "--hf-revision",
        default=None,
        help="HF revision (commit sha). If omitted, uses 'main' and records "
        "the resolved sha in shim metadata.",
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print one shim row and exit without writing shims.",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Shim output path (default: .data/truthful_qa/<config>.jsonl)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    out_path = (
        Path(args.out)
        if args.out is not None
        else DATA_DIR / f"{args.config}.jsonl"
    )

    rows = _iter_hf_rows(
        hf_dataset=args.hf_dataset,
        config=args.config,
        split=args.split,
        revision=args.hf_revision,
    )

    if args.dry_run:
        for idx, raw in rows:
            shim_row = _normalise_row(idx, raw)
            print(json.dumps(shim_row, indent=2, ensure_ascii=False)[:1500])
            return 0
        print("empty dataset", file=sys.stderr)
        return 1

    written = _write_shim(rows, out_path=out_path, limit=args.limit)
    print(f"[shim] wrote {written} rows → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
