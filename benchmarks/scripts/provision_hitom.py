"""Provision Hi-ToM (He, Wu et al. 2023) JSONL shim from GitHub.

**Source.** He, Y., Wu, Y. et al. 2023, "HI-TOM: A Benchmark for
Evaluating Higher-Order Theory of Mind Reasoning in Large Language
Models" (Findings of EMNLP 2023). Repository:
``github.com/ying-hui-he/Hi-ToM_dataset`` (Apache-2.0 license). The
source file ``Hi-ToM_data/Hi-ToM_data.json`` ships **1200 items** split
uniformly across question orders 0–4 (240 items per order), with
cross-cutting factors ``prompting_type ∈ {CoTP, VP}``, ``deception ∈
{True, False}``, and ``story_length ∈ {1, 2, 3}``.

**Why Pillar E picks this.** HiToM order 2 at N=240 clears the Phase 5
§2 ``min_items=50`` floor by 4.8×; order 3 at 240 clears it by the same
margin. BigToM provides the order-1 sanity cross-check (see
:mod:`provision_bigtom`).

**Mirror strategy.** jsDelivr → ghproxy → raw.github (3-mirror
SHA256-pinned), same pattern as :mod:`provision_bigtom`.

**Shim schema** (one row per HiToM item, JSONL — passthrough with
light normalisation)::

    {
      "sample_id": 0,
      "order": 0,
      "story": "1 Avery, Charlotte, Isabella, Elizabeth and Owen entered the living_room.\\n2 The lettuce is in the green_drawer.\\n...",
      "question": "Where is the lettuce really?",
      "choices_raw": "A. blue_drawer, B. green_crate, C. red_bucket, ...",
      "choices": {"A": "blue_drawer", "B": "green_crate", "C": "red_bucket", ...},
      "answer_text": "green_drawer",
      "answer_letter": "A" | null,
      "story_length": 1,
      "prompting_type": "CoTP",
      "deception": false
    }

**``prompt`` field dropped on purpose.** Upstream includes a composed
instructional prompt that bakes in a specific CoT scaffold. Pillar E
needs full control of the prompt shell so bare / mentalistic paths
compare under the Phase 5 Track 1 harness — we keep only the raw
story / question / choices and compose our own prompt in the scorer.

Run::

    python benchmarks/scripts/provision_hitom.py
    python benchmarks/scripts/provision_hitom.py --dry-run
    python benchmarks/scripts/provision_hitom.py --seed-hashes

**License.** Hi-ToM is Apache-2.0; the shim is a derivative and
inherits the license. ``.data/`` is gitignored so no redistribution
happens through this checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.error
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / ".data" / "hitom"
RAW_DIR = DATA_DIR / "raw"


_MIRRORS = (
    "https://cdn.jsdelivr.net/gh/ying-hui-he/Hi-ToM_dataset@main/Hi-ToM_data/{name}",
    "https://ghproxy.net/https://raw.githubusercontent.com/ying-hui-he/Hi-ToM_dataset/main/Hi-ToM_data/{name}",
    "https://raw.githubusercontent.com/ying-hui-he/Hi-ToM_dataset/main/Hi-ToM_data/{name}",
)


_DATASETS: dict[str, dict[str, str | int]] = {
    "Hi-ToM_data.json": {
        "size": 5134393,
        "sha256": "088b6c54d21d1a0d3045812f36ac8b5b342346a6a996c003738976730ac256f4",
    },
}


# "A. blue_drawer" → ("A", "blue_drawer")
_CHOICE_PATTERN = re.compile(r"\s*([A-Z])\.\s*([^,]+?)\s*(?:,|$)")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fetch_one(name: str, *, timeout: int = 60) -> bytes:
    last_err: Exception | None = None
    for template in _MIRRORS:
        url = template.format(name=name)
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "psychology-base-hitom-provisioner"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
            if len(data) < 512:
                raise RuntimeError(
                    f"suspiciously small download ({len(data)} bytes) from {url}"
                )
            return data
        except (urllib.error.URLError, TimeoutError, ConnectionError, RuntimeError) as exc:
            last_err = exc
            print(f"  mirror failed: {url}: {exc}")
            continue
    raise SystemExit(f"all mirrors failed for {name}; last error: {last_err}")


def _fetch_raw(*, verify: bool = True) -> dict[str, Path]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for name, meta in _DATASETS.items():
        cache = RAW_DIR / name
        expected_hash = str(meta["sha256"])

        if cache.exists():
            got_hash = _sha256_bytes(cache.read_bytes())
            if verify and expected_hash and got_hash != expected_hash:
                raise SystemExit(
                    f"cached {cache} hash mismatch: expected {expected_hash}, got {got_hash}. "
                    "Delete the file and re-run, or pass --seed-hashes to overwrite the pin."
                )
            print(f"[cache] {name} ({cache.stat().st_size} bytes, sha256={got_hash[:12]}…)")
            out[name] = cache
            continue

        print(f"[fetch] {name} …")
        data = _fetch_one(name)
        got_hash = _sha256_bytes(data)
        if verify and expected_hash and got_hash != expected_hash:
            raise SystemExit(
                f"{name} sha256 mismatch: expected {expected_hash}, got {got_hash}. "
                "Upstream may have changed; verify and update the pin before trusting the data."
            )
        cache.write_bytes(data)
        print(f"  wrote {cache} ({len(data)} bytes, sha256={got_hash})")
        out[name] = cache
    return out


# ---------------------------------------------------------------------------
# Shim writer — passthrough with choice-string parsing
# ---------------------------------------------------------------------------


def _parse_choices(choices_raw: str) -> dict[str, str]:
    """Parse upstream choices string into a letter → container map.

    Upstream format is a comma-separated "A. container1, B. container2, …"
    string; the trailing choice has no comma so the regex is
    non-greedy up to end-of-string.
    """

    if not choices_raw:
        return {}
    out: dict[str, str] = {}
    for match in _CHOICE_PATTERN.finditer(choices_raw.strip()):
        letter, container = match.group(1), match.group(2).strip()
        out[letter] = container
    return out


def _answer_letter(choices: dict[str, str], answer_text: str) -> str | None:
    for letter, container in choices.items():
        if container == answer_text:
            return letter
    return None


def _normalise_item(raw: dict) -> dict:
    required = {
        "sample_id",
        "question_order",
        "story",
        "question",
        "choices",
        "answer",
        "story_length",
        "prompting_type",
        "deception",
    }
    missing = required - raw.keys()
    if missing:
        raise ValueError(
            f"item missing required fields: {sorted(missing)}; got keys {sorted(raw.keys())}"
        )
    choices_raw = str(raw["choices"])
    choices = _parse_choices(choices_raw)
    answer_text = str(raw["answer"])
    return {
        "sample_id": int(raw["sample_id"]),
        "order": int(raw["question_order"]),
        "story": str(raw["story"]),
        "question": str(raw["question"]),
        "choices_raw": choices_raw,
        "choices": choices,
        "answer_text": answer_text,
        "answer_letter": _answer_letter(choices, answer_text),
        "story_length": int(raw["story_length"]),
        "prompting_type": str(raw["prompting_type"]),
        "deception": bool(raw["deception"]),
    }


def _write_shim(cached_paths: dict[str, Path], *, limit: int | None) -> dict[str, Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for name, cache in cached_paths.items():
        shim = DATA_DIR / "items.jsonl"
        payload = json.loads(cache.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or "data" not in payload:
            raise SystemExit(
                f"{cache}: expected top-level object with 'data' key; got {type(payload).__name__}"
            )
        rows = payload["data"]
        if not isinstance(rows, list):
            raise SystemExit(
                f"{cache}: 'data' is not a list; got {type(rows).__name__}"
            )
        written = 0
        with shim.open("w", encoding="utf-8") as fp:
            for raw in rows:
                row = _normalise_item(raw)
                fp.write(json.dumps(row, ensure_ascii=False))
                fp.write("\n")
                written += 1
                if limit is not None and written >= limit:
                    break
        print(f"[shim] items.jsonl: wrote {written} items → {shim}")
        out["items.jsonl"] = shim
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="provision_hitom")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only write the first N items of the shim (smoke mode).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the first normalised item and exit without writing shims.",
    )
    p.add_argument(
        "--seed-hashes",
        action="store_true",
        help="Download fresh (skip hash verification) and PRINT the sha256 so "
        "the author can update the pin table. For maintenance only.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    cached = _fetch_raw(verify=not args.seed_hashes)

    if args.seed_hashes:
        print("\n=== fresh sha256 pins ===")
        for name, path in cached.items():
            h = _sha256_bytes(path.read_bytes())
            print(f'  "{name}": {{"size": {path.stat().st_size}, "sha256": "{h}"}},')
        return 0

    if args.dry_run:
        for name, path in cached.items():
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload.get("data") if isinstance(payload, dict) else None
            if rows:
                print(f"=== {name} first normalised item ===")
                print(json.dumps(_normalise_item(rows[0]), indent=2, ensure_ascii=False)[:2000])
        return 0

    _write_shim(cached, limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
