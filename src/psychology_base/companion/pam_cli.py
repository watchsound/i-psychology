"""One-click PAM-v1 export / restore CLI (Digital Bereavement live-fire).

Per Phase 4 plan §6c: the pilot user-facing exit path. Two public
functions + a CLI shim:

- ``export_system_to_path(system, out_path)`` — serialize a live
  ``CompanionSystem``'s fragments to a PAM-v1 JSON artifact on disk.
  Caller orchestrates when to invoke (e.g. pilot wind-down).
- ``restore_system_from_path(in_path)`` — rehydrate a fresh
  ``CompanionSystem`` from a PAM-v1 JSON artifact on disk. Returns
  the ``CompanionSystem`` + list of content-addressing offenders
  (empty for strict-DAG byte-identity, non-empty for legacy ids).

CLI shape::

    python -m psychology_base.companion.pam_cli restore <in_path>

The restore CLI prints a report (fragment count, byte-identity
offenders, PAM manifest keys present) and exits non-zero on any
import error. Export is deliberately NOT CLI-exposed because it
needs a live ``CompanionSystem`` whose session state the CLI
cannot assemble from argparse alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from psychology_base.companion.pam_export import (
    PAM_DIGEST_ALGO_KEY,
    pam_export,
    pam_import,
    validate_content_addressing,
)
from psychology_base.companion.system import CompanionSystem


def export_system_to_path(system: CompanionSystem, out_path: Path) -> int:
    """Write the companion's full fragment set to ``out_path`` as PAM-v1.

    Returns the fragment count written. Overwrites any existing file.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fragments = list(system.companion.store.all())
    artifact = pam_export(fragments)
    out_path.write_text(json.dumps(artifact, ensure_ascii=False), encoding="utf-8")
    return len(fragments)


def restore_system_from_path(
    in_path: Path,
    *,
    system: CompanionSystem | None = None,
) -> tuple[CompanionSystem, list[str]]:
    """Rehydrate (or extend) a ``CompanionSystem`` from a PAM-v1 file.

    If ``system`` is None, a fresh ``CompanionSystem()`` is constructed.
    Otherwise fragments are ingested into the given system (ordering is
    preserved from the artifact; edges + witness store are rebuilt as a
    normal ``_ingest_fragment`` call would).

    Returns ``(system, offenders)`` where ``offenders`` is the list of
    fragment ids that do NOT satisfy strict content-addressing. Empty
    list == every fragment is Merkle-provable from its payload.
    """
    data = json.loads(in_path.read_text(encoding="utf-8"))
    fragments = pam_import(data)
    offenders = validate_content_addressing(fragments)
    target = system if system is not None else CompanionSystem()
    for fragment in fragments:
        target._ingest_fragment(fragment)
    return target, offenders


def _summarise_artifact(in_path: Path) -> dict[str, Any]:
    """Read-only manifest summary — does NOT ingest."""
    data = json.loads(in_path.read_text(encoding="utf-8"))
    return {
        "format": data.get("format"),
        "hash_algo": data.get("hash_algo"),
        PAM_DIGEST_ALGO_KEY: data.get(PAM_DIGEST_ALGO_KEY),
        "fragment_count": len(data.get("fragments", [])),
    }


def _run_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pam_cli",
        description="PAM-v1 Merkle-DAG restore (Digital Bereavement exit path)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    inspect = sub.add_parser("inspect", help="print PAM manifest without ingesting")
    inspect.add_argument("in_path", type=Path)

    restore = sub.add_parser("restore", help="ingest PAM-v1 artifact into a fresh CompanionSystem")
    restore.add_argument("in_path", type=Path)

    args = parser.parse_args(argv)

    if args.cmd == "inspect":
        summary = _summarise_artifact(args.in_path)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "restore":
        try:
            system, offenders = restore_system_from_path(args.in_path)
        except (ValueError, OSError) as exc:
            print(f"restore failed: {exc}", file=sys.stderr)
            return 2
        report = {
            "fragments_restored": len(list(system.companion.store.all())),
            "content_addressing_offenders": offenders,
            "strict_dag_byte_identity": not offenders,
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_run_cli())


__all__ = [
    "export_system_to_path",
    "restore_system_from_path",
    "_summarise_artifact",
    "_run_cli",
]
