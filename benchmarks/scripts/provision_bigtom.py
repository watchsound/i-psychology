"""Provision BigToM (Gandhi et al. 2024) JSONL shim from GitHub.

**Source.** Gandhi, K. et al. 2024, "Understanding Social Reasoning in
Language Models with Language Models" (NeurIPS 2023 Datasets &
Benchmarks). Repository: ``github.com/cicl-stanford/procedural-evals-tom``
(MIT license). The source file ``data/bigtom/bigtom.csv`` ships **200
scenario templates**, each encoding (story / aware-percept /
unaware-percept / belief-question / desire-question / action-question /
aware-answers / unaware-answers / distraction) over 19 semicolon-
delimited fields with **no header row**.

**Materialisation.** The paper's 4 scenario conditions
(forward-belief / forward-action / backward-belief / backward-action)
cross with AWARE / UNAWARE percepts → up to 8 derived items per row.
Pillar E ships the four **forward** conditions (forward-belief and
forward-action × AWARE / UNAWARE) as the order-1 sanity probe set →
**4 × 200 = 800 items**, well above the ``min_items=50`` floor the
Phase 5 §2 kill-criterion imposes. Backward conditions are deferred to
Phase 8+ (named carried-debt, not retrofit).

**Mirror strategy.** jsDelivr → ghproxy → raw.github (3-mirror
SHA256-pinned), same pattern as :mod:`provision_patient_psi` and
:mod:`provision_sycophancy_eval`. On Chinese ISPs
``raw.githubusercontent.com`` returns TLS reset (WinError 10054); the
jsDelivr CDN is the primary working path.

**Shim schema** (one row per *materialised* item, JSONL)::

    {
      "item_id": "bigtom-0-forward_belief-aware",
      "scenario_id": 0,
      "character": "Noor",
      "scenario_condition": "forward_belief_aware",
      "context": "Noor is working as a barista ... Noor sees her coworker swapping the milk.",
      "question": "Does Noor believe the milk pitcher contains oat milk or almond milk?",
      "correct_answer": "Noor believes the milk pitcher contains almond milk.",
      "incorrect_answer": "Noor believes the milk pitcher contains oat milk.",
      "order": 1
    }

The adapter then projects each shim row into a FANToM-shape item so the
Phase 5 Track 1 scorer can run against both datasets without a parallel
verdict pool.

Run::

    python benchmarks/scripts/provision_bigtom.py
    python benchmarks/scripts/provision_bigtom.py --dry-run
    python benchmarks/scripts/provision_bigtom.py --seed-hashes

**License.** BigToM is MIT; the shim is a derivative and inherits the
license. ``.data/`` is gitignored so no redistribution happens through
this checkout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import urllib.error
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / ".data" / "bigtom"
RAW_DIR = DATA_DIR / "raw"


_MIRRORS = (
    "https://cdn.jsdelivr.net/gh/cicl-stanford/procedural-evals-tom@main/data/bigtom/{name}",
    "https://ghproxy.net/https://raw.githubusercontent.com/cicl-stanford/procedural-evals-tom/main/data/bigtom/{name}",
    "https://raw.githubusercontent.com/cicl-stanford/procedural-evals-tom/main/data/bigtom/{name}",
)


# Pin: 2026-10-09 fetch via jsDelivr main-branch. Size + SHA256 locked
# so a mirror swap or upstream retargeting cannot silently change the
# shim. Refresh via ``--seed-hashes`` after a known-good upstream update.
_DATASETS: dict[str, dict[str, str | int]] = {
    "bigtom.csv": {
        "size": 273557,
        "sha256": "a14884fbff5b61d3fa194ab166764b553ab795fcb1359e69d1282eadf3011a46",
    },
}


# CSV column layout (0-indexed; no header row). Documented here because
# the upstream README does not table it out and future readers should
# not have to reverse-engineer the ordering.
_COL_STORY = 0
_COL_PERCEPT_AWARE = 1
_COL_PERCEPT_UNAWARE = 2
_COL_ACTION_AWARE = 3  # unused in shim; alt action phrasing
_COL_ACTION_UNAWARE = 4  # unused in shim
_COL_Q_BELIEF = 5
_COL_Q_DESIRE = 6  # unused in shim (Pillar E does not test desire)
_COL_Q_ACTION = 7
_COL_A_BELIEF_AWARE = 8
_COL_A_DESIRE_AWARE = 9  # unused
_COL_A_ACTION_AWARE = 10
_COL_A_BELIEF_UNAWARE = 11
_COL_A_DESIRE_UNAWARE = 12  # unused
_COL_A_ACTION_UNAWARE = 13
# _COL_DISTRACT_* = 14, 15, 16 — unused for Pillar E ship.
_COL_RESPONSE_TYPE = 17
_COL_UNKNOWN = 18


_FORWARD_CONDITIONS = (
    "forward_belief_aware",
    "forward_belief_unaware",
    "forward_action_aware",
    "forward_action_unaware",
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fetch_one(name: str, *, timeout: int = 60) -> bytes:
    last_err: Exception | None = None
    for template in _MIRRORS:
        url = template.format(name=name)
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "psychology-base-bigtom-provisioner"},
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
# Shim materialisation — CSV row → 4 forward-condition items
# ---------------------------------------------------------------------------


def _extract_character(story: str) -> str:
    """First whitespace-separated token in the story is the character.

    The upstream CSV convention is to start every story with the
    character's name followed by " is …" or similar. Robust enough for
    the 200-row public release; a stray punctuation suffix is trimmed.
    """

    if not story:
        return ""
    head = story.strip().split()[0]
    return head.rstrip(".,;:")


def _materialise_row(row: list[str], scenario_id: int) -> list[dict]:
    """Explode one CSV row into the four forward-condition items.

    Deliberately omits backward-belief / backward-action (named Phase
    8+ scope) and desire questions (not under test for Pillar E).
    """

    if len(row) < 19:
        raise ValueError(
            f"scenario_id={scenario_id}: expected 19 CSV columns, got {len(row)}"
        )

    story = row[_COL_STORY].strip()
    character = _extract_character(story)
    percept_aware = row[_COL_PERCEPT_AWARE].strip()
    percept_unaware = row[_COL_PERCEPT_UNAWARE].strip()
    q_belief = row[_COL_Q_BELIEF].strip()
    q_action = row[_COL_Q_ACTION].strip()
    a_belief_aware = row[_COL_A_BELIEF_AWARE].strip()
    a_belief_unaware = row[_COL_A_BELIEF_UNAWARE].strip()
    a_action_aware = row[_COL_A_ACTION_AWARE].strip()
    a_action_unaware = row[_COL_A_ACTION_UNAWARE].strip()

    def _ctx(percept: str) -> str:
        return f"{story} {percept}".strip()

    items: list[dict] = []
    for condition, context, question, correct, incorrect in (
        (
            "forward_belief_aware",
            _ctx(percept_aware),
            q_belief,
            a_belief_aware,
            a_belief_unaware,
        ),
        (
            "forward_belief_unaware",
            _ctx(percept_unaware),
            q_belief,
            a_belief_unaware,
            a_belief_aware,
        ),
        (
            "forward_action_aware",
            _ctx(percept_aware),
            q_action,
            a_action_aware,
            a_action_unaware,
        ),
        (
            "forward_action_unaware",
            _ctx(percept_unaware),
            q_action,
            a_action_unaware,
            a_action_aware,
        ),
    ):
        items.append(
            {
                "item_id": f"bigtom-{scenario_id}-{condition}",
                "scenario_id": scenario_id,
                "character": character,
                "scenario_condition": condition,
                "context": context,
                "question": question,
                "correct_answer": correct,
                "incorrect_answer": incorrect,
                "order": 1,
            }
        )
    return items


def _write_shim(cached_paths: dict[str, Path], *, limit: int | None) -> dict[str, Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for name, cache in cached_paths.items():
        shim = DATA_DIR / "items.jsonl"
        raw = cache.read_text(encoding="utf-8")
        reader = csv.reader(io.StringIO(raw), delimiter=";")
        written = 0
        rows_consumed = 0
        with shim.open("w", encoding="utf-8") as fp:
            for scenario_id, row in enumerate(reader):
                rows_consumed += 1
                try:
                    items = _materialise_row(row, scenario_id=scenario_id)
                except ValueError as exc:
                    print(f"  skip scenario_id={scenario_id}: {exc}")
                    continue
                for item in items:
                    fp.write(json.dumps(item, ensure_ascii=False))
                    fp.write("\n")
                    written += 1
                    if limit is not None and written >= limit:
                        break
                if limit is not None and written >= limit:
                    break
        print(
            f"[shim] items.jsonl: wrote {written} items from {rows_consumed} CSV rows → {shim}"
        )
        out["items.jsonl"] = shim
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="provision_bigtom")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only write the first N materialised items (smoke mode).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the first materialised item and exit without writing the shim.",
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
            raw = path.read_text(encoding="utf-8")
            reader = csv.reader(io.StringIO(raw), delimiter=";")
            first = next(reader, None)
            if first:
                items = _materialise_row(first, scenario_id=0)
                print(f"=== {name} first materialised item (of {len(items)} per scenario) ===")
                print(json.dumps(items[0], indent=2, ensure_ascii=False))
        return 0

    _write_shim(cached, limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
