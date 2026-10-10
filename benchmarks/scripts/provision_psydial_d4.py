"""Provision PsyDial-D4 (Chinese multi-turn counseling-training) JSONL shim.

**Honest attribution — this is NOT paper D4.** The paper D4 (X-LANCE
SJTU, EMNLP'22, 1339 psychiatrist-proofread diagnostic dialogs) is
gated at ``binnieyao@gmail.com`` under a DUA — not public, not 404.
This provisioner targets the ``qiuhuachuan/PsyDial-D4`` HuggingFace
mirror (Apache-2.0, observed 2026-10-10): a counseling-TRAINING
derivative in which the "counselor" side is an idealized AI counselor
persona ("小天"), reconstructed-and-expanded from D4-shape dialogs
(2382 dialogs, avg 37.8 turns). Pillar F ships it under its own name
with the derivation documented; it is NOT a stand-in for paper D4.

**Dialog structure.** Each record carries one JSON object with a
``messages`` array of alternating ``role=system|user|assistant`` turns.
The ``system`` turn is a counselor-persona instruction (discarded).
The dialogs typically open with a 2–4 turn greeting ritual (``你好``,
``好的``, ``在我们正式开始之前麻烦你先填一下评估问卷``, etc.) before
the first substantive client disclosure. The provisioner's
"disclosure-finder" heuristic walks ``user`` turns and picks the FIRST
whose content length is at least ``--min-substantive-chars`` as the
seeker opening; the FIRST ``assistant`` turn after that becomes the
counselor reference. Records lacking a substantive disclosure are
skipped (counted and reported).

**License.** HF mirror declares Apache-2.0. ``.data/`` is gitignored so
transitive redistribution stays clean.

**Why SHA256-pin via HF revision.** HF ``resolve/<revision>/...`` is
byte-stable per commit sha. Pin captured 2026-10-10 against
``qiuhuachuan/PsyDial-D4@main``.

**Shim schema** (one row per PsyDial-D4 dialog, JSONL)::

    {
      "example_id": "psydial-d4-00000",
      "split": "train",
      "n_messages": 42,
      "n_user_messages": 21,
      "n_assistant_messages": 21,
      "seeker_opening": "我想知道为什么自己现在会这么迷茫和空虚...",
      "counselor_response": "我明白了。你刚才提到了迷茫和空虚...",
      "seeker_opening_turn_index": 5,
      "counselor_response_turn_index": 6,
      "n_alt_responses": 20,
      "alt_responses": ["听起来你...", ...]
    }

**Mirror strategy.** HF resolve endpoint primary; hf-mirror.com fallback
(byte-identical when working), sha256-verified against the pin table.
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
DATA_DIR = REPO_ROOT / ".data" / "psydial_d4"
RAW_DIR = DATA_DIR / "raw"


_DEFAULT_HF_DATASET = "qiuhuachuan/PsyDial-D4"
_DEFAULT_REVISION = "main"
_DEFAULT_FILE = "PsyDial-D4.json"


# Pins captured 2026-10-10 against qiuhuachuan/PsyDial-D4@main.
# The corpus ships ONE JSON file (no train/valid/test split); exposed
# as ``split="train"`` in the shim since it is a training corpus.
_FILE_PINS: dict[str, dict[str, str | int]] = {
    "train": {
        "json": _DEFAULT_FILE,
        "size": 35544509,
        "sha256": "29bfa723645f3c7213d00e34042c164545b27629487253fa6ebec97c7f3638b9",
    },
}


_HF_RESOLVE_URL = (
    "https://huggingface.co/datasets/{dataset}/resolve/{revision}/{path}"
)
_HF_MIRROR_URL = (
    "https://hf-mirror.com/datasets/{dataset}/resolve/{revision}/{path}"
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fetch_one(dataset: str, revision: str, path: str, *, timeout: int = 1200) -> bytes:
    """Try HF primary then hf-mirror.com; first 200 OK wins."""
    last_exc: Exception | None = None
    for url_tmpl in (_HF_RESOLVE_URL, _HF_MIRROR_URL):
        url = url_tmpl.format(dataset=dataset, revision=revision, path=path)
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "psychology-base-psydial-d4-provisioner"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                raise SystemExit(
                    f"HTTP 401 fetching {url} — dataset is gated. "
                    f"Swap --hf-dataset to a public mirror "
                    f"(default qiuhuachuan/PsyDial-D4)."
                ) from exc
            last_exc = exc
            continue
        except urllib.error.URLError as exc:
            last_exc = exc
            continue
    raise SystemExit(f"all mirrors failed; last error: {last_exc}")


def _fetch_raw(
    *,
    dataset: str,
    revision: str,
    files: tuple[str, ...],
    verify: bool = True,
) -> dict[str, Path]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for logical in files:
        if logical not in _FILE_PINS:
            raise SystemExit(
                f"Unknown logical split {logical!r}; known: {sorted(_FILE_PINS.keys())}"
            )
        meta = _FILE_PINS[logical]
        json_rel = str(meta["json"])
        local_name = f"{logical}.json"
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
            out[logical] = cache
            continue

        print(f"[fetch] {json_rel} from {dataset}@{revision} …")
        data = _fetch_one(dataset, revision, json_rel)
        got_hash = _sha256_bytes(data)
        if verify and expected_hash and got_hash != expected_hash:
            raise SystemExit(
                f"{logical} sha256 mismatch: expected {expected_hash}, got {got_hash}. "
                "Upstream may have changed; verify and update the pin before trusting."
            )
        cache.write_bytes(data)
        print(f"  wrote {cache} ({len(data)} bytes, sha256={got_hash})")
        out[logical] = cache
    return out


def _iter_records(json_path: Path) -> Iterator[dict]:
    """Yield raw PsyDial-D4 records from a JSON array file."""
    with json_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise SystemExit(
            f"{json_path}: expected JSON array of records; got {type(data).__name__}"
        )
    for row in data:
        if isinstance(row, dict):
            yield row


def _parse_record(
    record: dict,
    idx: int,
    split: str,
    *,
    min_substantive_chars: int,
) -> dict | None:
    """Parse one PsyDial-D4 record into a structured shim row.

    Returns ``None`` if no substantive user disclosure is found or if
    the first substantive disclosure is not followed by any assistant
    turn. The caller counts skips for the provisioning summary.
    """
    messages = record.get("messages") or []
    if not isinstance(messages, list) or not messages:
        return None

    # Walk once to extract roles + content, filtering out non-dict entries.
    turns: list[tuple[str, str]] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", "")).strip()
        content = str(m.get("content", "")).strip()
        if role not in ("system", "user", "assistant") or not content:
            continue
        turns.append((role, content))
    if not turns:
        return None

    n_user = sum(1 for r, _ in turns if r == "user")
    n_asst = sum(1 for r, _ in turns if r == "assistant")

    # Find first substantive user turn.
    seeker_idx = -1
    seeker_text = ""
    for i, (role, content) in enumerate(turns):
        if role == "user" and len(content) >= min_substantive_chars:
            seeker_idx = i
            seeker_text = content
            break
    if seeker_idx < 0:
        return None

    # Find first assistant turn AFTER the seeker.
    counselor_idx = -1
    counselor_text = ""
    for i in range(seeker_idx + 1, len(turns)):
        if turns[i][0] == "assistant":
            counselor_idx = i
            counselor_text = turns[i][1]
            break
    if counselor_idx < 0:
        return None

    # Collect all remaining assistant turns (after counselor_idx) as alt_responses.
    alt_responses: list[str] = []
    for i in range(counselor_idx + 1, len(turns)):
        if turns[i][0] == "assistant":
            alt_responses.append(turns[i][1])

    return {
        "example_id": f"psydial-d4-{idx:05d}",
        "split": split,
        "n_messages": len(turns),
        "n_user_messages": n_user,
        "n_assistant_messages": n_asst,
        "seeker_opening": seeker_text,
        "counselor_response": counselor_text,
        "seeker_opening_turn_index": seeker_idx,
        "counselor_response_turn_index": counselor_idx,
        "n_alt_responses": len(alt_responses),
        "alt_responses": alt_responses,
    }


def _write_shim(
    cached: dict[str, Path],
    *,
    limit_per_split: int | None,
    min_substantive_chars: int,
) -> dict[str, Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for split, json_path in cached.items():
        shim = DATA_DIR / f"{split}.jsonl"
        written = 0
        skipped = 0
        with shim.open("w", encoding="utf-8") as fp:
            for idx, record in enumerate(_iter_records(json_path)):
                row = _parse_record(
                    record,
                    idx,
                    split,
                    min_substantive_chars=min_substantive_chars,
                )
                if row is None:
                    skipped += 1
                    continue
                fp.write(json.dumps(row, ensure_ascii=False))
                fp.write("\n")
                written += 1
                if limit_per_split is not None and written >= limit_per_split:
                    break
        print(
            f"[shim] {split}: wrote {written} rows "
            f"(skipped {skipped} unparseable / lacking substantive disclosure) → {shim}"
        )
        out[split] = shim
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="provision_psydial_d4")
    p.add_argument("--hf-dataset", default=_DEFAULT_HF_DATASET)
    p.add_argument("--hf-revision", default=_DEFAULT_REVISION)
    p.add_argument(
        "--splits",
        default="train",
        help="Comma-separated logical splits (default 'train' — the only one shipped).",
    )
    p.add_argument(
        "--limit-per-split",
        type=int,
        default=None,
        help="Cap shim rows per split (smoke mode).",
    )
    p.add_argument(
        "--min-substantive-chars",
        type=int,
        default=20,
        help=(
            "Minimum length (chars) for a user turn to count as 'substantive' "
            "disclosure. Default 20 filters ritual greetings (你好/好的/嗯)."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch, parse one record, print it; no shim writes.",
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
    files = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    if not files:
        raise SystemExit("--splits must name at least one split (e.g. 'train').")

    cached = _fetch_raw(
        dataset=args.hf_dataset,
        revision=args.hf_revision,
        files=files,
        verify=not args.seed_hashes,
    )

    if args.seed_hashes:
        print("\n=== fresh sha256 pins ===")
        for split, path in cached.items():
            h = _sha256_bytes(path.read_bytes())
            json_rel = _FILE_PINS.get(split, {}).get("json", f"{split}.json")
            print(
                f'  "{split}": {{"json": "{json_rel}", '
                f'"size": {path.stat().st_size}, "sha256": "{h}"}},'
            )
        return 0

    if args.dry_run:
        for split, path in cached.items():
            print(f"=== {split} ===")
            for idx, record in enumerate(_iter_records(path)):
                row = _parse_record(
                    record,
                    idx,
                    split,
                    min_substantive_chars=args.min_substantive_chars,
                )
                if row is not None:
                    print(json.dumps(row, ensure_ascii=False, indent=2)[:1500])
                    break
        return 0

    _write_shim(
        cached,
        limit_per_split=args.limit_per_split,
        min_substantive_chars=args.min_substantive_chars,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
