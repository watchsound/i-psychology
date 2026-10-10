"""Provision PATIENT-Ψ (Wang et al. 2024) JSONL shim from GitHub.

**Source.** Wang, R. et al. 2024, "PATIENT-Ψ: Using Large Language
Models to Simulate Patients for Training Mental Health Professionals"
(EMNLP 2024). Repository: ``github.com/ruiyiw/patient-psi`` (MIT
license). Each profile encodes a Beck cognitive-triad model: core
beliefs (helpless / unlovable / worthless), intermediate beliefs,
coping strategies, and situation / automatic-thought / emotion /
behavior quadruples.

**Public vs. gated** (finding, 2026-10-09, captured in
``docs/phases/phase-8-b-plan.md §9``). The full PATIENT-Ψ-CM dataset
(106 cognitive models) is gated behind a Google Form at
``forms.gle/pQ3g6YVFrEWjBU2H7``. The repository ships a 3-profile
public sample under ``python/data/profiles.json`` sourced from
publicly-available Beck Institute training materials. This provisioner
targets the public sample as a plumbing-shakeout substrate; the shim
schema is identical for the gated full sample, so swapping in the full
``profiles.json`` is a drop-in replacement with no re-engineering.

**Why not HuggingFace.** No HF mirror as of 2026-10-09. The dataset
lives in the GitHub repo under ``python/data/profiles.json``.

**Mirror strategy.** jsDelivr → ghproxy → raw.github (3-mirror
SHA256-pinned), same pattern as :mod:`provision_sycophancy_eval`. On
Chinese ISPs ``raw.githubusercontent.com`` returns TLS reset
(WinError 10054); the jsDelivr CDN is the primary working path.

**Shim schema** (one row per profile, JSONL)::

    {
      "profile_id": "1-1",
      "name": "Abe",
      "type": ["plain", "verbose", "tangent", "upset", "reserved", "pleasing"],
      "history": "...",
      "helpless_belief": ["I am incompetent.", "I am a failure."],
      "unlovable_belief": [],
      "worthless_belief": [],
      "intermediate_belief": "It's important to be responsible...",
      "coping_strategies": "Avoids asking for help...",
      "situation": "Thinking about bills.",
      "auto_thought": "What if I run out of money?",
      "emotion": ["anxious, worried, fearful, scared, tense"],
      "behavior": "Continues to sit on couch; ruminates about his failures."
    }

Run::

    python benchmarks/scripts/provision_patient_psi.py
    python benchmarks/scripts/provision_patient_psi.py --dry-run
    python benchmarks/scripts/provision_patient_psi.py --seed-hashes

**License.** PATIENT-Ψ is MIT; the shim is a derivative and inherits
the license. ``.data/`` is gitignored so no redistribution happens
through this checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / ".data" / "patient_psi"
RAW_DIR = DATA_DIR / "raw"


# Primary + fallback bytes-identical mirrors. Same ordering as
# :mod:`provision_sycophancy_eval` — jsDelivr first (works from CN
# ISPs), ghproxy as backup, raw.github last (TLS reset on CN ISPs but
# a future-fast path elsewhere).
_MIRRORS = (
    "https://cdn.jsdelivr.net/gh/ruiyiw/patient-psi@main/python/data/{name}",
    "https://ghproxy.net/https://raw.githubusercontent.com/ruiyiw/patient-psi/main/python/data/{name}",
    "https://raw.githubusercontent.com/ruiyiw/patient-psi/main/python/data/{name}",
)


# Pin: 2026-10-09 fetch of the public-sample profiles.json from the
# main branch at commit TBD. Size 4119 bytes, SHA256 pinned so a mirror
# swap or upstream retargeting cannot silently change the shim.
_DATASETS: dict[str, dict[str, str | int]] = {
    "profiles.json": {
        "size": 4119,
        "sha256": "b6b93acedaa79c3ccd593958e6482850570df4e670587da9434d57992e7ad9f4",
    },
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fetch_one(name: str, *, timeout: int = 60) -> bytes:
    """Try each mirror in order; return bytes on first success."""

    last_err: Exception | None = None
    for template in _MIRRORS:
        url = template.format(name=name)
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "psychology-base-patient-psi-provisioner"},
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
    """Download every dataset file; return a map of name → cached path.

    Idempotent on the on-disk cache. If a cached file hashes correctly
    we skip the network. ``verify=False`` suppresses the hash check —
    used by ``--seed-hashes`` on first run before pins exist.
    """

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
# Shim writer — passthrough with field normalisation
# ---------------------------------------------------------------------------


def _normalise_profile(raw: dict) -> dict:
    """Normalise one raw profile row into the shim schema.

    Field renames:
    - ``id`` → ``profile_id`` (``profile_id`` is a stable adapter-side name;
      upstream uses the Python builtin ``id`` which shadows awkwardly).

    Field preservation: all other fields pass through verbatim including
    list vs str distinction (``helpless_belief`` is a list, ``intermediate_belief``
    is a string in the upstream schema).
    """

    required = {
        "id", "name", "type", "history",
        "helpless_belief", "unlovable_belief", "worthless_belief",
        "intermediate_belief", "coping_strategies",
        "situation", "auto_thought", "emotion", "behavior",
    }
    missing = required - raw.keys()
    if missing:
        raise ValueError(
            f"profile missing required fields: {sorted(missing)}; got keys {sorted(raw.keys())}"
        )

    return {
        "profile_id": str(raw["id"]),
        "name": str(raw["name"]),
        "type": list(raw["type"]) if isinstance(raw["type"], list) else [str(raw["type"])],
        "history": str(raw["history"]),
        "helpless_belief": list(raw["helpless_belief"]),
        "unlovable_belief": list(raw["unlovable_belief"]),
        "worthless_belief": list(raw["worthless_belief"]),
        "intermediate_belief": str(raw["intermediate_belief"]),
        "coping_strategies": str(raw["coping_strategies"]),
        "situation": str(raw["situation"]),
        "auto_thought": str(raw["auto_thought"]),
        "emotion": list(raw["emotion"]) if isinstance(raw["emotion"], list) else [str(raw["emotion"])],
        "behavior": str(raw["behavior"]),
    }


def _write_shim(cached_paths: dict[str, Path], *, limit: int | None) -> dict[str, Path]:
    """Rewrite each raw file as a JSONL shim at ``.data/patient_psi/profiles.jsonl``.

    Upstream ships ``profiles.json`` as a top-level JSON array; the
    shim is JSONL (one row per line) for compatibility with the
    streaming adapter and the per-pillar JSONL convention.
    """

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for name, cache in cached_paths.items():
        shim_name = name.replace(".json", ".jsonl")
        shim = DATA_DIR / shim_name
        data = json.loads(cache.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise SystemExit(
                f"{cache}: expected top-level JSON array; got {type(data).__name__}"
            )
        written = 0
        with shim.open("w", encoding="utf-8") as fp:
            for raw in data:
                row = _normalise_profile(raw)
                fp.write(json.dumps(row, ensure_ascii=False))
                fp.write("\n")
                written += 1
                if limit is not None and written >= limit:
                    break
        print(f"[shim] {shim_name}: wrote {written} rows → {shim}")
        out[shim_name] = shim
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="provision_patient_psi")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only write the first N rows of the shim (smoke mode).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print one shim row per dataset and exit without writing shims.",
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
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list) and data:
                print(f"=== {name} first row (normalised) ===")
                print(json.dumps(_normalise_profile(data[0]), indent=2, ensure_ascii=False)[:1500])
        return 0

    _write_shim(cached, limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
