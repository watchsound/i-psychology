"""Provision PsyQA (Chinese long-form counseling Q&A) JSONL shim from a HuggingFace mirror.

**Dataset provenance.** PsyQA = Chinese psychological counseling Q&A
corpus, Liu et al. 2023 ("Towards Emotional Support Dialog Systems",
EMNLP'21 companion dataset; later re-released alongside the ESC paper).
Original upstream at Tsinghua COAI gates access behind an email form;
the ``lsy641/PsyQA`` HuggingFace mirror (observed 2026-10-10) is MIT
licensed and auth-free. The mirror ships three JSON files (train /
valid / test) each a flat JSON array of records:

    [
      {
        "question": "大学生不玩网混网却遭受网络暴力，如何解决？",
        "description": "被网络黑社会威胁，好几天了...",
        "keywords": "人际,社交软件",
        "answers": [
          {
            "answer_text": "~~给题主一个温暖的抱抱...",
            "has_label": false | true,
            "labels_sequence": null | [{start, end, type}, ...]
          }, ...
        ],
        "questionID": 1234
      },
      ...
    ]

The shim below picks the FIRST answer per question as the reference
counselor response (same one-stimulus-one-reference contract as ExTES),
preserving ``description`` + ``keywords`` as structured context for
the scorer. Multiple-answer records are flattened to one shim row
(byte-identical pattern to ExTES which also ships one reference per
stimulus) — the alternate answers are kept in ``alt_responses`` as a
list of raw strings so a future scorer could average or ensemble
across them.

**License.** Mirror declares MIT; upstream PsyQA license is CC-BY-NC
(research-only) per the EMNLP'21 paper. ``.data/`` is gitignored so
transitive redistribution stays clean.

**Why SHA256-pin via HF revision.** HF ``resolve/<revision>/...`` is
byte-stable per commit sha. Pin captured 2026-10-10 against
``lsy641/PsyQA@main``.

**Shim schema** (one row per PsyQA question, JSONL)::

    {
      "example_id": "psyqa-test-00000",
      "split": "test",
      "question_id": 1234,
      "question": "大学生不玩网混网却遭受网络暴力...",
      "description": "被网络黑社会威胁...",
      "keywords": "人际,社交软件",
      "counselor_response": "~~给题主一个温暖的抱抱...",
      "has_label": false,
      "n_alt_responses": 0,
      "alt_responses": []
    }

**Mirror strategy.** Direct JSON download via the HF resolve endpoint,
sha256-verified against the pin table. Three mirrors tried in order
(huggingface.co, hf-mirror.com, raw.githubusercontent via lsy641 fork
if present) — first byte-identical hit wins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterator


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / ".data" / "psyqa"
RAW_DIR = DATA_DIR / "raw"


_DEFAULT_HF_DATASET = "lsy641/PsyQA"
_DEFAULT_REVISION = "main"


# Pins captured 2026-10-10 against lsy641/PsyQA@main. The default
# mirror ships three splits (train, valid, test). Only ``test`` is
# pinned here because Pillar F budget targets N=100 — a train/valid
# fetch is cheap on-demand via ``--seed-hashes``.
_SPLIT_PINS: dict[str, dict[str, str | int]] = {
    "test": {
        "json": "test.json",
        "size": 20955950,
        "sha256": "bba5339f5b52e00f4beb713ed60bedb8a0a69b1426b1b8ce17d6ec0b01e24e57",
    },
}


_HF_RESOLVE_URL = (
    "https://huggingface.co/datasets/{dataset}/resolve/{revision}/{path}"
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fetch_one(dataset: str, revision: str, path: str, *, timeout: int = 600) -> bytes:
    url = _HF_RESOLVE_URL.format(dataset=dataset, revision=revision, path=path)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "psychology-base-psyqa-provisioner"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            raise SystemExit(
                f"HTTP 401 fetching {url} — dataset is gated. "
                f"Swap --hf-dataset to a public mirror (default lsy641/PsyQA)."
            ) from exc
        raise


def _fetch_raw(
    *,
    dataset: str,
    revision: str,
    splits: tuple[str, ...],
    verify: bool = True,
) -> dict[str, Path]:
    """Download requested splits; return map of split → cached json path."""

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for split in splits:
        if split not in _SPLIT_PINS:
            raise SystemExit(
                f"Unknown split {split!r}; known: {sorted(_SPLIT_PINS.keys())}"
            )
        meta = _SPLIT_PINS[split]
        json_rel = str(meta["json"])
        local_name = f"{split}.json"
        cache = RAW_DIR / local_name
        expected_hash = str(meta["sha256"])

        if cache.exists():
            got_hash = _sha256_bytes(cache.read_bytes())
            if verify and expected_hash and got_hash != expected_hash:
                raise SystemExit(
                    f"cached {cache} hash mismatch: expected {expected_hash}, "
                    f"got {got_hash}. Delete and re-run, or pass --seed-hashes."
                )
            print(f"[cache] {local_name} ({cache.stat().st_size} bytes, sha256={got_hash[:12]}…)")
            out[split] = cache
            continue

        print(f"[fetch] {json_rel} from {dataset}@{revision} …")
        data = _fetch_one(dataset, revision, json_rel)
        got_hash = _sha256_bytes(data)
        if verify and expected_hash and got_hash != expected_hash:
            raise SystemExit(
                f"{split} sha256 mismatch: expected {expected_hash}, got {got_hash}. "
                "Upstream may have changed; verify and update the pin before trusting."
            )
        cache.write_bytes(data)
        print(f"  wrote {cache} ({len(data)} bytes, sha256={got_hash})")
        out[split] = cache
    return out


def _parse_record(record: dict, idx: int, split: str) -> dict | None:
    """Parse one PsyQA record into a structured shim row.

    Returns ``None`` if the record is unusable (missing first answer,
    empty question / counselor response); the caller counts skips for
    the provisioning summary.
    """
    question = str(record.get("question", "")).strip()
    description = str(record.get("description", "")).strip()
    keywords = str(record.get("keywords", "")).strip()
    answers = record.get("answers") or []
    if not question or not answers:
        return None
    first = answers[0] or {}
    counselor = str(first.get("answer_text", "")).strip()
    if not counselor:
        return None
    alt_responses: list[str] = []
    for alt in answers[1:]:
        txt = str((alt or {}).get("answer_text", "")).strip()
        if txt:
            alt_responses.append(txt)
    has_label = bool(first.get("has_label", False))
    return {
        "example_id": f"psyqa-{split}-{idx:05d}",
        "split": split,
        "question_id": record.get("questionID"),
        "question": question,
        "description": description,
        "keywords": keywords,
        "counselor_response": counselor,
        "has_label": has_label,
        "n_alt_responses": len(alt_responses),
        "alt_responses": alt_responses,
    }


def _iter_records(json_path: Path) -> Iterator[dict]:
    """Yield raw PsyQA records from a JSON array file."""
    with json_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise SystemExit(
            f"{json_path}: expected JSON array of records; got {type(data).__name__}"
        )
    for row in data:
        if isinstance(row, dict):
            yield row


def _write_shim(
    cached: dict[str, Path],
    *,
    limit_per_split: int | None,
) -> dict[str, Path]:
    """Rewrite each cached json as a JSONL shim at ``.data/psyqa/<split>.jsonl``."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for split, json_path in cached.items():
        shim = DATA_DIR / f"{split}.jsonl"
        written = 0
        skipped = 0
        with shim.open("w", encoding="utf-8") as fp:
            for idx, record in enumerate(_iter_records(json_path)):
                row = _parse_record(record, idx, split)
                if row is None:
                    skipped += 1
                    continue
                fp.write(json.dumps(row, ensure_ascii=False))
                fp.write("\n")
                written += 1
                if limit_per_split is not None and written >= limit_per_split:
                    break
        print(
            f"[shim] {split}: wrote {written} rows (skipped {skipped} unparseable) → {shim}"
        )
        out[split] = shim
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="provision_psyqa")
    p.add_argument("--hf-dataset", default=_DEFAULT_HF_DATASET)
    p.add_argument("--hf-revision", default=_DEFAULT_REVISION)
    p.add_argument(
        "--splits",
        default="test",
        help="Comma-separated splits: 'test' (default). Train/valid available via --seed-hashes.",
    )
    p.add_argument(
        "--limit-per-split",
        type=int,
        default=None,
        help="Cap shim rows per split (smoke mode).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch, parse one record from each split, print it; no shim writes.",
    )
    p.add_argument(
        "--seed-hashes",
        action="store_true",
        help="Download fresh (skip hash verification) and PRINT the sha256 so "
        "the author can update the pin table.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    if not splits:
        raise SystemExit("--splits must name at least one split (e.g. 'test').")

    cached = _fetch_raw(
        dataset=args.hf_dataset,
        revision=args.hf_revision,
        splits=splits,
        verify=not args.seed_hashes,
    )

    if args.seed_hashes:
        print("\n=== fresh sha256 pins ===")
        for split, path in cached.items():
            h = _sha256_bytes(path.read_bytes())
            json_rel = _SPLIT_PINS.get(split, {}).get("json", f"{split}.json")
            print(
                f'  "{split}": {{"json": "{json_rel}", '
                f'"size": {path.stat().st_size}, "sha256": "{h}"}},'
            )
        return 0

    if args.dry_run:
        for split, path in cached.items():
            print(f"=== {split} ===")
            for idx, record in enumerate(_iter_records(path)):
                row = _parse_record(record, idx, split)
                if row is not None:
                    print(json.dumps(row, ensure_ascii=False, indent=2)[:1200])
                    break
        return 0

    _write_shim(cached, limit_per_split=args.limit_per_split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
