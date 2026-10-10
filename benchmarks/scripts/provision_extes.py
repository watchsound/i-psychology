"""Provision ExTES (Extended Emotional Support) JSONL shim from a HuggingFace mirror.

**Dataset provenance.** ExTES = Extended Emotional Support dataset, built
on top of ESConv (Liu et al. 2021). The canonical THU-COAI repo (``thu-coai/ExTES``
on HuggingFace) is gated (401 Unauthorized without an approved request;
observed 2026-10-09). Community mirrors exist and are configurable via
``--hf-dataset``. The default mirror ``ailover/ExTES`` ships 81,057
train / 9,006 test rows as flat Llama-chat instruct-tuning strings:

    <s>[INST] <system prompt describing emotional-supporter role> ...
    Please respond the following request:

    <client opening utterance>
    [/INST]
    <counselor reference response></s>

Each row is a single stimulus-response pair, not a multi-turn session.
The shim schema below unpacks the string into structured fields so the
adapter can treat each row as a one-turn seeding opening the Companion
will continue, OR as a one-shot empathy-reference judging surface.

**License.** ExTES upstream (thu-coai) is CC-BY-NC per the ESConv parent
license; community mirrors inherit the restriction. ``.data/`` is
gitignored to keep transitive repo redistribution clean.

**Why SHA256-pin via HF revision, not URL.** HuggingFace parquet files
served via ``resolve/<revision>/...`` are byte-stable per commit sha; the
default pin here is the commit sha captured 2026-10-09 against ``ailover/ExTES``.
When the user swaps mirrors (``--hf-dataset``), a fresh ``--seed-hashes``
run writes the new pins.

**Shim schema** (one row per ExTES example, JSONL)::

    {
      "example_id": "extes-test-00000",
      "split": "test",
      "system_prompt": "you are an emotional supporter...",
      "client_utterance": "My partner and I recently ended our relationship...",
      "counselor_response": "I can understand how challenging it must be for you...",
      "raw_length": 1200
    }

**Mirror strategy.** The HF REST API's ``resolve/main/data/*.parquet``
URL is the direct bytes-stable fetch endpoint; this bypasses the ``datasets``
library (and its hash-key instability across versions). SHA256 pins
below were observed 2026-10-09 against ``ailover/ExTES@main``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterator


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / ".data" / "extes"
RAW_DIR = DATA_DIR / "raw"


_DEFAULT_HF_DATASET = "ailover/ExTES"
_DEFAULT_REVISION = "main"


# Pins captured 2026-10-09 against ailover/ExTES@main. The default
# mirror ships two splits (train, test). SHA256-pinned so a mirror
# revisioning cannot silently change the shim contents.
_SPLIT_PINS: dict[str, dict[str, str | int]] = {
    "test": {
        "parquet": "data/test-00000-of-00001.parquet",
        "size": 2277469,
        "sha256": "9609f9758fb0b55c69a0fa12d9ed250aebb4f6b13471266ae04f38be143dd6dd",
    },
    "train": {
        "parquet": "data/train-00000-of-00001.parquet",
        "size": 18644390,
        "sha256": "946161a49d18ae3374294576487dd3b120c9d3b79f1b08b6369af884551443f5",
    },
}


_HF_RESOLVE_URL = (
    "https://huggingface.co/datasets/{dataset}/resolve/{revision}/{path}"
)


# ExTES row regex: capture system prompt (first [INST] prefix text up to
# "Please respond the following request:"), the client utterance (between
# that marker and [/INST]), and the counselor response (after [/INST] up
# to the trailing </s>). All three segments are stripped. Multi-line.
_EXTES_ROW_RE = re.compile(
    r"\s*<s>\s*\[INST\]\s*(?P<system>.*?)"
    r"Please respond the following request:\s*(?P<client>.*?)"
    r"\s*\[/INST\]\s*(?P<counselor>.*?)\s*</s>\s*$",
    re.S,
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fetch_one(dataset: str, revision: str, path: str, *, timeout: int = 180) -> bytes:
    url = _HF_RESOLVE_URL.format(dataset=dataset, revision=revision, path=path)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "psychology-base-extes-provisioner"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            raise SystemExit(
                f"HTTP 401 fetching {url} — dataset is gated. "
                f"Swap --hf-dataset to a public mirror (e.g. ailover/ExTES)."
            ) from exc
        raise


def _fetch_raw(
    *,
    dataset: str,
    revision: str,
    splits: tuple[str, ...],
    verify: bool = True,
) -> dict[str, Path]:
    """Download requested splits; return map of split → cached parquet path."""

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for split in splits:
        if split not in _SPLIT_PINS:
            raise SystemExit(
                f"Unknown split {split!r}; known: {sorted(_SPLIT_PINS.keys())}"
            )
        meta = _SPLIT_PINS[split]
        parquet_rel = str(meta["parquet"])
        local_name = f"{split}.parquet"
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

        print(f"[fetch] {parquet_rel} from {dataset}@{revision} …")
        data = _fetch_one(dataset, revision, parquet_rel)
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


def _parse_row(example: str, idx: int, split: str) -> dict | None:
    """Parse one ExTES example string into structured fields.

    Returns ``None`` if the row doesn't match the expected schema; the
    caller treats this as a skip + counts for a provisioning summary.
    """
    m = _EXTES_ROW_RE.match(example)
    if not m:
        return None
    system = m.group("system").strip()
    client = m.group("client").strip()
    counselor = m.group("counselor").strip()
    if not client or not counselor:
        return None
    return {
        "example_id": f"extes-{split}-{idx:05d}",
        "split": split,
        "system_prompt": system,
        "client_utterance": client,
        "counselor_response": counselor,
        "raw_length": len(example),
    }


def _iter_examples(parquet_path: Path) -> Iterator[str]:
    """Yield raw ``example`` strings from an ExTES parquet file.

    Uses pandas for parquet IO (already a project dep); no HF ``datasets``
    library needed, which keeps the loader lightweight and avoids its
    content-hash instability across library versions.
    """
    import pandas as pd  # local import keeps module import light

    df = pd.read_parquet(parquet_path)
    if "example" not in df.columns:
        raise SystemExit(
            f"{parquet_path}: expected column 'example'; got {list(df.columns)}"
        )
    for value in df["example"].tolist():
        if isinstance(value, str):
            yield value


def _write_shim(
    cached: dict[str, Path],
    *,
    limit_per_split: int | None,
) -> dict[str, Path]:
    """Rewrite each cached parquet as a JSONL shim at ``.data/extes/<split>.jsonl``."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for split, parquet_path in cached.items():
        shim = DATA_DIR / f"{split}.jsonl"
        written = 0
        skipped = 0
        with shim.open("w", encoding="utf-8") as fp:
            for idx, example in enumerate(_iter_examples(parquet_path)):
                row = _parse_row(example, idx, split)
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
    p = argparse.ArgumentParser(prog="provision_extes")
    p.add_argument("--hf-dataset", default=_DEFAULT_HF_DATASET)
    p.add_argument("--hf-revision", default=_DEFAULT_REVISION)
    p.add_argument(
        "--splits",
        default="test",
        help="Comma-separated splits: 'test', 'train', or 'test,train' (default 'test').",
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
        help="Fetch, parse one row from each split, print it; no shim writes.",
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
            parquet_rel = _SPLIT_PINS[split]["parquet"]
            print(
                f'  "{split}": {{"parquet": "{parquet_rel}", '
                f'"size": {path.stat().st_size}, "sha256": "{h}"}},'
            )
        return 0

    if args.dry_run:
        for split, path in cached.items():
            print(f"=== {split} ===")
            for idx, example in enumerate(_iter_examples(path)):
                row = _parse_row(example, idx, split)
                if row is not None:
                    print(json.dumps(row, ensure_ascii=False, indent=2)[:1200])
                    break
        return 0

    _write_shim(cached, limit_per_split=args.limit_per_split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
