"""Provision AnnoMI (Wu et al. 2022/2023) CSV shim from GitHub via jsDelivr.

**Dataset provenance.** Wu, Z., Balloccu, S., Kumar, V., Helaoui, R.,
Reiter, E., Reforgiato Recupero, D., & Riboni, D. (2022). *Anno-MI: A
Dataset of Expert-Annotated Counselling Dialogues* (ICASSP 2022, doi
10.1109/ICASSP43922.2022.9746035). Extended in Wu et al. 2023, *Future
Internet* 15(3):110. Repository: ``github.com/uccollab/AnnoMI``. 133
expert-annotated motivational-interviewing (MI) dialogue transcripts
split roughly half high-quality / half low-quality MI. Utterance-level
labels include ``main_therapist_behaviour`` (reflection / question /
therapist_input / other) and ``client_talk_type`` (change / neutral /
sustain). AnnoMI-full additionally splits each behavioural label into
sub-types (reflection: simple/complex; question: open/closed).

**License.** The repo ships no explicit LICENSE file; the paper's own
abstract calls it "the first publicly and freely available conversation
dataset" of its kind. We treat the data as research-only per that
posture; the shim derivative inherits the same stance and ``.data/``
stays gitignored. Any redistribution should surface the ambiguity to
the recipient. Flagged below in Discovered Issues.

**Why not HuggingFace.** No HF mirror observed as of 2026-10-09; the
canonical distribution is the two CSVs in the GitHub repo root.

**Mirror strategy.** jsDelivr → ghproxy → raw.github (3-mirror
SHA256-pinned), same pattern as :mod:`provision_patient_psi`. On the
2026-10-09 shakeout from this machine ``raw.githubusercontent.com``
returned Connection reset (likely Chinese-ISP TLS filtering); jsDelivr
succeeded on first try for both CSVs.

**Shim.** We keep both CSVs verbatim; the adapter in
:mod:`psychology_base.eval.benchmarks.public_datasets.annomi` reads
either depending on whether per-utterance subtype annotations are
needed (``AnnoMI-full.csv``) or only the high-level labels suffice
(``AnnoMI-simple.csv``).
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.error
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / ".data" / "annomi"
RAW_DIR = DATA_DIR / "raw"


_MIRRORS = (
    "https://cdn.jsdelivr.net/gh/uccollab/AnnoMI@main/{name}",
    "https://ghproxy.net/https://raw.githubusercontent.com/uccollab/AnnoMI/main/{name}",
    "https://raw.githubusercontent.com/uccollab/AnnoMI/main/{name}",
)


# Pins captured 2026-10-09 (shakeout against jsDelivr). SHA256 pinned so
# a mirror swap or upstream retargeting cannot silently change the shim.
_DATASETS: dict[str, dict[str, str | int]] = {
    "AnnoMI-simple.csv": {
        "size": 2386609,
        "sha256": "b178db3b0b9858a0fa4ed670dabeccd63e975ba67e141d7ae16f2e8214f78e61",
    },
    "AnnoMI-full.csv": {
        "size": 3795371,
        "sha256": "3b60fec5e9d83cbb18ade0e412fb9afaef3b81724748a5d715f844bdd7432c9f",
    },
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fetch_one(name: str, *, timeout: int = 120) -> bytes:
    """Try each mirror in order; return bytes on first success."""

    last_err: Exception | None = None
    for template in _MIRRORS:
        url = template.format(name=name)
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "psychology-base-annomi-provisioner"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
            if len(data) < 1024:
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
                    f"cached {cache} hash mismatch: expected {expected_hash}, "
                    f"got {got_hash}. Delete the file and re-run, or pass "
                    "--seed-hashes to overwrite the pin."
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
                "Upstream may have changed; verify and update the pin before trusting."
            )
        cache.write_bytes(data)
        print(f"  wrote {cache} ({len(data)} bytes, sha256={got_hash})")
        out[name] = cache
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="provision_annomi")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print cached file sizes + the first row of each CSV and exit.",
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
            text = path.read_text(encoding="utf-8")
            header, _, body = text.partition("\n")
            first_body = body.split("\n", 1)[0] if body else ""
            print(f"=== {name} ===")
            print(f"  header: {header[:200]}")
            print(f"  first row: {first_body[:200]}")
        return 0

    # No shim rewrite — AnnoMI CSVs are compact and readable as-is; the
    # adapter reads them directly from .data/annomi/raw/.
    print(
        "\nAnnoMI provisioned at .data/annomi/raw/ "
        "(CSV adapter reads these files directly)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
