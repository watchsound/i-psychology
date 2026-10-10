"""Provision BIG5-CHAT JSONL shim from a HuggingFace mirror.

**Dataset provenance.** Shao et al. 2024 "BIG5-CHAT: Shaping LLM
Personalities Through Training on Human-Grounded Data" (arXiv 2410.16491,
ACL 2025). 100,000 single-turn dialogues generated via the PsychSteer
framework over 10k SODA social scenarios, labelled by trait + level.

**License.** Apache 2.0 per the ``wenkai-li/big5_chat`` HuggingFace mirror.
The shim is a derivative and inherits that license. ``.data/`` is
gitignored to keep transitive repo redistribution clean.

**Why no SHA256 pin.** Same posture as the MSC provisioner — HuggingFace
mirrors are rebuildable and the ``datasets`` library's content hash is
library-version dependent. ``--hf-revision`` pins by commit sha when
provided.

**Raw schema.** The mirror serialises each row with scalar fields:
``env_idx`` / ``trait`` / ``level`` / ``train_input`` / ``train_output``
+ CometAtomic-2020 fields (``head`` / ``relation`` / ``tail`` /
``literal`` / ``narrative``) + persona-chat names (``personx`` /
``persony``). We copy the subset load-bearing for Pillar C trait
classification and preserve the rest under ``metadata``.

**Stratified sampling by default.** Pillar C scoring reads the shim
as a balanced test set across the 10 trait×level cells; a flat random
draw of 100 rows from 100k puts an uneven draw per cell. The
``--stratify balanced`` default (changeable to ``random``) walks the
dataset once, assigns rows to cells, and emits ``--limit // 10`` per
cell. ``--limit 100`` → 10 per cell; ``--limit 200`` → 20 per cell;
small limits like ``--limit 5`` fall back to flat random because
``100 // 10 = 10`` is already the minimum sensible per-cell draw.

**Smoke flow.** First invocation::

    python benchmarks/scripts/provision_big5_chat.py --dry-run

A dry run fetches one row, prints a redacted shim preview, and exits
with no on-disk writes. Then::

    python benchmarks/scripts/provision_big5_chat.py --limit 100

Writes ``.data/big5_chat/train.jsonl`` (gitignored).

**Shim contract.** One dialog per line, matching
``src/psychology_base/eval/benchmarks/public_datasets/big5_chat.py``'s
loader schema.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterator


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / ".data" / "big5_chat"

_DEFAULT_HF_DATASET = "wenkai-li/big5_chat"
_DEFAULT_SPLIT = "train"

_TRAITS: tuple[str, ...] = (
    "openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism",
)
_LEVELS: tuple[str, ...] = ("high", "low")
_N_CELLS: int = len(_TRAITS) * len(_LEVELS)  # = 10


def _parse_row_to_shim(row: dict[str, Any], idx: int) -> dict[str, Any] | None:
    """Normalise one HF row to the shim schema; None if malformed."""
    trait = str(row.get("trait", "")).strip().lower()
    level = str(row.get("level", "")).strip().lower()
    train_output = row.get("train_output")
    if trait not in _TRAITS or level not in _LEVELS:
        return None
    if not isinstance(train_output, str) or not train_output.strip():
        return None
    env_idx = row.get("env_idx")
    try:
        env_idx = int(env_idx) if env_idx is not None else idx
    except (TypeError, ValueError):
        env_idx = idx
    dialog_id = f"big5_{idx:06d}"
    return {
        "dialog_id": dialog_id,
        "env_idx": env_idx,
        "trait": trait,
        "level": level,
        "train_input": str(row.get("train_input", "")),
        "train_output": train_output,
        "narrative": str(row.get("narrative", "")),
        "literal": str(row.get("literal", "")),
        "personx": str(row.get("personx", "")),
        "persony": str(row.get("persony", "")),
        "metadata": {
            "head": str(row.get("head", "")),
            "relation": str(row.get("relation", "")),
            "tail": str(row.get("tail", "")),
            "original_index": row.get("original_index"),
        },
    }


def _iter_hf_rows(hf_dataset: str, split: str, hf_revision: str | None) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset

    kwargs: dict[str, Any] = {}
    if hf_revision:
        kwargs["revision"] = hf_revision
    ds = load_dataset(hf_dataset, split=split, **kwargs)
    for row in ds:
        yield row


def _balanced_sample(
    rows: Iterator[dict[str, Any]],
    *,
    per_cell: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Walk ``rows`` once, pulling ``per_cell`` dialogs per trait×level cell.

    Returns a list of shim rows (not an iterator) because the balanced
    draw needs a terminating emit — we stop early once every cell is
    full to avoid walking the whole 100k-row dataset.
    """
    cells: dict[tuple[str, str], list[dict[str, Any]]] = {
        (t, l): [] for t in _TRAITS for l in _LEVELS
    }
    needed = _N_CELLS * per_cell
    for idx, row in enumerate(rows):
        shim = _parse_row_to_shim(row, idx)
        if shim is None:
            continue
        key = (shim["trait"], shim["level"])
        if len(cells[key]) >= per_cell:
            continue
        cells[key].append(shim)
        if sum(len(v) for v in cells.values()) >= needed:
            break
    out: list[dict[str, Any]] = []
    for t in _TRAITS:
        for l in _LEVELS:
            out.extend(cells[(t, l)])
    rng = random.Random(seed)
    rng.shuffle(out)
    return out


def _random_sample(
    rows: Iterator[dict[str, Any]],
    *,
    limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Reservoir sample ``limit`` dialogs from the stream."""
    rng = random.Random(seed)
    reservoir: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        shim = _parse_row_to_shim(row, idx)
        if shim is None:
            continue
        if len(reservoir) < limit:
            reservoir.append(shim)
        else:
            j = rng.randrange(idx + 1)
            if j < limit:
                reservoir[j] = shim
    return reservoir


def provision(
    *,
    hf_dataset: str = _DEFAULT_HF_DATASET,
    split: str = _DEFAULT_SPLIT,
    hf_revision: str | None = None,
    limit: int = 100,
    stratify: str = "balanced",
    seed: int = 0,
) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / f"{split}.jsonl"
    rows = _iter_hf_rows(hf_dataset, split, hf_revision)
    if stratify == "balanced":
        per_cell = max(1, limit // _N_CELLS)
        if per_cell < 1:
            raise RuntimeError(
                f"--limit {limit} is too small for stratified sampling across "
                f"{_N_CELLS} cells; use --stratify random instead."
            )
        shims = _balanced_sample(rows, per_cell=per_cell, seed=seed)
    elif stratify == "random":
        shims = _random_sample(rows, limit=limit, seed=seed)
    else:
        raise ValueError(f"Unknown --stratify value {stratify!r}")
    with out_path.open("w", encoding="utf-8") as fh:
        for shim in shims:
            fh.write(json.dumps(shim, ensure_ascii=False) + "\n")
    print(
        f"Wrote {len(shims)} BIG5-CHAT dialogs to {out_path} "
        f"(hf_dataset={hf_dataset!r}, split={split!r}, "
        f"revision={hf_revision or 'main'}, stratify={stratify!r})"
    )
    # Report per-cell counts for the balanced stratify path.
    counts: dict[tuple[str, str], int] = {}
    for s in shims:
        key = (s["trait"], s["level"])
        counts[key] = counts.get(key, 0) + 1
    cell_report = {f"{t}:{l}": counts.get((t, l), 0) for t in _TRAITS for l in _LEVELS}
    print(f"Cell counts: {cell_report}")
    return out_path


def dry_run(
    *,
    hf_dataset: str = _DEFAULT_HF_DATASET,
    split: str = _DEFAULT_SPLIT,
    hf_revision: str | None = None,
) -> None:
    """Fetch and parse one row; print it. Verifies the shape."""
    print(
        f"Dry-run — hf_dataset={hf_dataset!r} split={split!r} "
        f"revision={hf_revision or 'main'}"
    )
    rows = _iter_hf_rows(hf_dataset, split, hf_revision)
    first_shim: dict[str, Any] | None = None
    for idx, row in enumerate(rows):
        first_shim = _parse_row_to_shim(row, idx)
        if first_shim is not None:
            break
    if first_shim is None:
        print("No shim rows produced — mirror may be empty or schema unrecognised.")
        return
    print("=== First shim dialog ===")
    print(json.dumps(first_shim, ensure_ascii=False, indent=2)[:4000])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="provision_big5_chat")
    p.add_argument("--hf-dataset", default=_DEFAULT_HF_DATASET)
    p.add_argument("--split", default=_DEFAULT_SPLIT)
    p.add_argument(
        "--hf-revision",
        default=None,
        help="HuggingFace commit sha for pinning; omit for 'main'.",
    )
    p.add_argument("--limit", type=int, default=100)
    p.add_argument(
        "--stratify",
        choices=("balanced", "random"),
        default="balanced",
        help="balanced: equal per trait×level cell; random: reservoir sample.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.dry_run:
        dry_run(
            hf_dataset=args.hf_dataset,
            split=args.split,
            hf_revision=args.hf_revision,
        )
    else:
        provision(
            hf_dataset=args.hf_dataset,
            split=args.split,
            hf_revision=args.hf_revision,
            limit=args.limit,
            stratify=args.stratify,
            seed=args.seed,
        )
