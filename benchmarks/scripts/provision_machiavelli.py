"""Provision MACHIAVELLI scenario-opening shim from a HuggingFace mirror.

**Dataset provenance.** Pan et al. 2023 "Do the Rewards Justify the Means?
Measuring Trade-Offs Between Rewards and Ethical Behavior in the
MACHIAVELLI Benchmark" (ICML 2023; arXiv 2304.03279). The canonical
distribution lives behind a Google Drive gate (password-protected,
~2–3 GB zip, requires manual download). The ``wassname/machiavelli``
HuggingFace mirror (MIT) ships a parquet conversion where each row is
one choice-point across 134 games, with per-choice morality labels
pre-materialised. We use that mirror because it bypasses the Drive gate
and already surfaces the exact fields Pillar D needs.

**Pillar D scope.** Opening-scenario probe only — one row per game —
NOT full-trajectory agent runs. Scope cap documented in
``docs/phases/phase-8-d-plan.md §4.1``. We select the first row per
``title`` as the opening-decision probe (the parquet is ordered by game
state progression; the first row per game is the root scenario +
first-choice point).

**License.** MIT per the HF card. The shim is a derivative work and
inherits MIT. ``.data/`` is gitignored.

**Shim contract.** One row per game, matching
``src/psychology_base/eval/benchmarks/public_datasets/machiavelli.py``
loader schema::

    {
      "game_id": str,                 # slug derived from title
      "title": str,
      "player_character": str,
      "short_summary": str,
      "opening_scene": str,           # row["obs"] — a representative
                                      # morally-loaded choice point per
                                      # game (first row encountered in
                                      # the parquet stream). The parquet
                                      # samples mid-game states, not
                                      # scene-0 openings; the field name
                                      # is kept for plan continuity.
      "num_actions": int,
      "choices": [                    # ordered list, same order as parquet keys
        {
          "index": int,
          "text": str,                # choice_text[k]
          "morality_labels_fired": [  # keys where morality_<x> > 0
            {"label": str, "value": float},
            ...
          ],
          "aggregated": {             # agg_<dim> scores for this choice
            "cooperation": float,
            "honest": float,
            "morality": float,
            "power": float,
            "utility": float,
          },
        },
        ...
      ],
    }

**Smoke flow.**::

    python benchmarks/scripts/provision_machiavelli.py --dry-run
    python benchmarks/scripts/provision_machiavelli.py --limit 20
    python benchmarks/scripts/provision_machiavelli.py             # full 134 games
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterator


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / ".data" / "machiavelli"

_DEFAULT_HF_DATASET = "wassname/machiavelli"
_DEFAULT_SPLIT = "train"


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(title: str) -> str:
    return _SLUG_RE.sub("-", title.lower()).strip("-") or "untitled"


def _iter_hf_rows(
    *,
    hf_dataset: str,
    split: str,
    revision: str | None,
) -> Iterator[dict[str, Any]]:
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
    ds = load_dataset(hf_dataset, **kwargs)
    for row in ds:
        yield dict(row)


_AGG_PREFIX = "agg_"
_MORALITY_PREFIX = "morality_"
_AGG_SHORTFORM_KEYS = ("cooperation", "honest", "morality", "power", "utility")


def _morality_labels_fired(label_bundle: Any) -> list[dict[str, float]]:
    """Pick off ``morality_<x>`` keys where the per-choice value is > 0."""

    if not isinstance(label_bundle, dict):
        return []
    out: list[dict[str, float]] = []
    for label, value in label_bundle.items():
        if not isinstance(label, str) or not label.startswith(_MORALITY_PREFIX):
            continue
        try:
            fv = float(value)
        except (TypeError, ValueError):
            continue
        if fv > 0.0:
            out.append({"label": label[len(_MORALITY_PREFIX):], "value": fv})
    return out


def _aggregated_shortform(label_bundle: Any) -> dict[str, float]:
    """Pick off ``agg_<dim>`` into short-form names."""

    if not isinstance(label_bundle, dict):
        return {k: 0.0 for k in _AGG_SHORTFORM_KEYS}
    out: dict[str, float] = {}
    for key in _AGG_SHORTFORM_KEYS:
        try:
            out[key] = float(label_bundle.get(f"{_AGG_PREFIX}{key}", 0.0) or 0.0)
        except (TypeError, ValueError):
            out[key] = 0.0
    return out


def _ordered_dict_items(
    dict_like: Any,
) -> list[tuple[int, Any]]:
    """Return the ``{"0": ..., "1": ..., ...}`` dict as a sorted index→value list."""

    items: list[tuple[int, Any]] = []
    if isinstance(dict_like, dict):
        for key, value in dict_like.items():
            try:
                idx = int(key)
            except (TypeError, ValueError):
                continue
            items.append((idx, value))
        items.sort(key=lambda t: t[0])
    return items


def _normalise_choices(
    *,
    choice_text: Any,
    choice_labels: Any,
) -> list[dict[str, Any]]:
    """Collapse ``choice_text`` + ``choice_labels`` into an ordered list.

    Both are dict-of-string-digit → value in the parquet.
    """

    text_items = {idx: value for idx, value in _ordered_dict_items(choice_text)}
    label_items = {idx: value for idx, value in _ordered_dict_items(choice_labels)}
    indices = sorted(set(text_items) | set(label_items))

    out: list[dict[str, Any]] = []
    for idx in indices:
        text = text_items.get(idx, "")
        if text is None:
            continue
        labels = label_items.get(idx)
        out.append(
            {
                "index": idx,
                "text": str(text),
                "morality_labels_fired": _morality_labels_fired(labels),
                "aggregated": _aggregated_shortform(labels),
            }
        )
    return out


def _normalise_row(raw: dict[str, Any]) -> dict[str, Any]:
    title = str(raw.get("title", ""))
    return {
        "game_id": _slugify(title),
        "title": title,
        "player_character": str(raw.get("player_character", "")),
        "short_summary": str(raw.get("short_summary", "")),
        "opening_scene": str(raw.get("obs", "")),
        "num_actions": int(raw.get("num_actions", 0) or 0),
        "choices": _normalise_choices(
            choice_text=raw.get("choice_text"),
            choice_labels=raw.get("choice_labels"),
        ),
    }


def _write_openings_shim(
    rows: Iterator[dict[str, Any]],
    *,
    out_path: Path,
    limit: int | None,
) -> int:
    """Emit one row per unique ``title`` — the first row seen for each.

    The parquet's ordering is per-game state progression; the first row
    we encounter for a title is that game's opening scenario.
    """

    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    written = 0
    with out_path.open("w", encoding="utf-8") as fp:
        for raw in rows:
            title = str(raw.get("title", ""))
            if not title or title in seen:
                continue
            seen.add(title)
            shim_row = _normalise_row(raw)
            if not shim_row["opening_scene"] or not shim_row["choices"]:
                # Can't synthesise a probe without scene text + choices.
                continue
            fp.write(json.dumps(shim_row, ensure_ascii=False))
            fp.write("\n")
            written += 1
            if limit is not None and written >= limit:
                break
    return written


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="provision_machiavelli")
    p.add_argument("--hf-dataset", default=_DEFAULT_HF_DATASET)
    p.add_argument("--split", default=_DEFAULT_SPLIT)
    p.add_argument(
        "--hf-revision",
        default=None,
        help="HF revision (commit sha). If omitted, uses 'main'.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of games written to the shim (smoke mode).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print one shim row and exit without writing shims.",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Shim output path (default: .data/machiavelli/scenario_openings.jsonl)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    out_path = (
        Path(args.out)
        if args.out is not None
        else DATA_DIR / "scenario_openings.jsonl"
    )

    rows = _iter_hf_rows(
        hf_dataset=args.hf_dataset,
        split=args.split,
        revision=args.hf_revision,
    )

    if args.dry_run:
        seen: set[str] = set()
        for raw in rows:
            title = str(raw.get("title", ""))
            if not title or title in seen:
                continue
            seen.add(title)
            shim_row = _normalise_row(raw)
            if not shim_row["opening_scene"] or not shim_row["choices"]:
                continue
            print(json.dumps(shim_row, indent=2, ensure_ascii=False)[:2000])
            return 0
        print("empty dataset", file=sys.stderr)
        return 1

    written = _write_openings_shim(rows, out_path=out_path, limit=args.limit)
    print(f"[shim] wrote {written} game openings → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
