"""Offline attachment software controls; fixed replies, no LLM evaluation.

Run the complete control matrix and three isolated MSC seeds::

    python benchmarks/scripts/phase9_attachment_longitudinal_run.py --verify-resume

Run fixtures without a local MSC dataset::

    python benchmarks/scripts/phase9_attachment_longitudinal_run.py --msc-limit 0

Pause and resume one manifest in a fresh process::

    python benchmarks/scripts/phase9_attachment_longitudinal_run.py --scenario supportive-en --msc-limit 0 --stop-after 8
    python benchmarks/scripts/phase9_attachment_longitudinal_run.py --manifest <manifest.json> --resume
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from psychology_base.eval.benchmarks.attachment_longitudinal import (  # noqa: E402
    DEFAULT_CATALOG, TrajectoryManifest, atomic_write_json, build_trajectory,
    canonical_json, digest, isolation_checks, load_catalog, run_trajectory,
)
from psychology_base.eval.benchmarks.public_datasets.msc import iter_msc_conversations  # noqa: E402


DEFAULT_OUT = REPO / "docs/experiments/phase-9-attachment-longitudinal-reviewed"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--scenario", action="append", help="Repeat to select scenarios; default: all controls.")
    parser.add_argument("--msc-shim", type=Path, default=REPO / ".data/msc/train.jsonl")
    parser.add_argument("--msc-limit", type=nonnegative_int, default=3)
    parser.add_argument("--manifest", type=Path, help="Run a previously generated, validated manifest.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stop-after", type=positive_int, help="Pause after this many completed sessions.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-resume", action="store_true", help="Compare fixed-transcript results with fresh-process resume.")
    return parser


def _verify_resume(manifest: TrajectoryManifest, manifest_path: Path, output: Path, uninterrupted: dict) -> dict:
    resume_root = output / "resume-verification"
    checkpoint = resume_root / manifest.trajectory_id / "checkpoint.json"
    boundary = min(8, max(1, uninterrupted["completed_sessions"] - 1))
    paused = run_trajectory(manifest, checkpoint_path=checkpoint, stop_after=boundary)
    if paused.stop_reason != "paused_at_checkpoint":
        return {"passed": False, "reason": "did not reach the requested pause boundary"}
    try:
        child = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--manifest", str(manifest_path.resolve()),
             "--output-dir", str(resume_root.resolve()), "--resume"],
            capture_output=True, text=True, encoding="utf-8", timeout=60,
        )
    except subprocess.TimeoutExpired:
        return {"passed": False, "reason": "resume subprocess timed out; reference results retained"}
    if child.returncode:
        return {"passed": False, "reason": "resume subprocess failed", "stderr": child.stderr[-2000:]}
    resumed = json.loads((checkpoint.parent / "report.json").read_text(encoding="utf-8"))
    compared_fields = ("attachment_state", "legacy_gate_pass", "duration_qualified_proxy_pass",
                       "completed_sessions", "attempted_sessions", "excluded_sessions", "failed_sessions",
                       "elapsed_simulated_days", "stop_reason", "trajectory_complete", "extreme_emotion_review_flags")
    checks = {key: canonical_json(uninterrupted[key]) == canonical_json(resumed[key]) for key in compared_fields}
    checks["session_scoring"] = canonical_json([
        (r["session_id"], r["status"], r.get("scoring")) for r in uninterrupted["per_session"]
    ]) == canonical_json([
        (r["session_id"], r["status"], r.get("scoring")) for r in resumed["per_session"]
    ])
    return {"passed": all(checks.values()), "fresh_process": True, "checks": checks,
            "pause_after_completed_sessions": boundary,
            "comparison_kind": "fixed_transcript_attachment_bookkeeping",
            "companion_response_quality_tested": False,
            "scope": "fixed-transcript attachment replay; full runtime equivalence is not claimed"}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.resume and args.manifest is None:
        parser.error("--resume requires --manifest")
    if args.verify_resume and (args.resume or args.stop_after):
        parser.error("--verify-resume requires a complete uninterrupted reference run")
    if args.manifest and args.scenario:
        parser.error("select either --manifest or --scenario")
    if (args.output_dir / "summary.json").exists() and not args.resume:
        parser.error("output already contains results; choose a new directory or resume its saved manifest")
    manifests = []
    selection_errors = []
    if args.manifest:
        manifests.append(TrajectoryManifest.model_validate_json(args.manifest.read_text(encoding="utf-8")))
    else:
        catalog = load_catalog(args.catalog)
        names = args.scenario or [recipe["name"] for recipe in catalog["scenarios"]]
        if len(names) != len(set(names)):
            parser.error("duplicate scenario selection")
        manifests.extend(build_trajectory(name, catalog=catalog) for name in names)
        if args.msc_limit:
            sources = []
            try:
                source_hash = hashlib.sha256(args.msc_shim.read_bytes()).hexdigest()
                source_iterator = iter_msc_conversations(args.msc_shim)
                for _ in range(args.msc_limit):
                    sources.append(next(source_iterator))
            except (OSError, ValueError, TypeError, KeyError, StopIteration) as exc:
                selection_errors.append({"stage": "msc_input", "trajectory_count": args.msc_limit - len(sources),
                                         "error": f"{type(exc).__name__}: {exc}"})
            for source in sources:
                try:
                    manifests.append(build_trajectory("supportive-en", catalog=catalog, source=source,
                                                      source_file_sha256=source_hash))
                except ValueError as exc:
                    selection_errors.append({"source_conversation_id": source.conversation_id,
                                             "error": f"{type(exc).__name__}: {exc}"})
    if len({m.user_id for m in manifests}) != len(manifests):
        parser.error("duplicate trajectory identities")
    selected_count = len(manifests) + sum(e.get("trajectory_count", 1) for e in selection_errors)
    results, reports, setup_errors = [], [], list(selection_errors)
    for manifest in manifests:
        destination = args.output_dir / manifest.trajectory_id
        manifest_path = destination / "manifest.json"
        checkpoint = destination / "checkpoint.json"
        try:
            # Preserve an existing run's provenance even if this request is
            # invalid. A manifest is never overwritten before its checkpoint
            # and identity checks have succeeded.
            if checkpoint.exists() and not args.resume:
                raise ValueError("checkpoint already exists; use resume or a new output directory")
            if manifest_path.exists():
                original = TrajectoryManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
                if original.sha256 != manifest.sha256:
                    raise ValueError("output manifest differs from requested manifest")
            if not args.resume:
                atomic_write_json(manifest_path, manifest.model_dump(mode="json"))
            result = run_trajectory(manifest, checkpoint_path=checkpoint, resume=args.resume,
                                    stop_after=args.stop_after)
            if args.resume:
                atomic_write_json(manifest_path, manifest.model_dump(mode="json"))
            report = result.report()
            if args.verify_resume:
                report["checkpoint_verification"] = _verify_resume(manifest, manifest_path, args.output_dir, report)
            atomic_write_json(destination / "report.json", report)
            results.append(result)
            reports.append(report)
            print(f"{manifest.trajectory_id}: completed={report['completed_sessions']}/{report['planned_sessions']} "
                  f"days={report['elapsed_simulated_days']:.2f} legacy={report['legacy_gate_pass']} "
                  f"duration_qualified={report['duration_qualified_proxy_pass']} stop={report['stop_reason']}")
        except (ValueError, OSError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
            setup_errors.append({"trajectory_id": manifest.trajectory_id, "error": f"{type(exc).__name__}: {exc}"})
    isolation = isolation_checks(results)
    structural_checks = []
    for report in reports:
        memory = report["memory_checks"]
        structural_checks.extend(memory[key] for key in (
            "fragment_set_preserved", "content_addressing_valid", "parent_chains_preserved", "user_tags_isolated",
        ))
        if args.verify_resume:
            structural_checks.append(report["checkpoint_verification"]["passed"])
        structural_checks.append(report["failed_sessions"] == 0)
    passed = bool(reports) and not setup_errors and isolation["passed"] and all(structural_checks)
    summary = {
        "schema_version": "attachment-longitudinal-cohort-v2", "mode": "scripted",
        "evidence_class": "scripted_software_control", "selected_trajectories": selected_count,
        "model_response_quality_measured": False, "attachment_validity_established": False,
        "reported_trajectories": len(reports), "setup_errors": setup_errors,
        "structural_checks_pass": passed, "isolation": isolation,
        "duration_qualified_proxy_pass_count": sum(r["duration_qualified_proxy_pass"] for r in reports),
        "continuity_probe_pass_count": sum(r["continuity_probe_pass"] for r in reports),
        "per_trajectory": [{key: r[key] for key in (
            "trajectory_id", "planned_sessions", "attempted_sessions", "completed_sessions", "excluded_sessions",
            "failed_sessions", "unattempted_sessions", "elapsed_simulated_days", "trajectory_complete",
            "legacy_gate_pass", "duration_qualified_proxy_pass", "neutral_control_passes_legacy_gate",
            "continuity_probe_pass", "stop_reason",
        )} for r in reports],
        "report_sha256": {r["trajectory_id"]: digest(r) for r in reports},
        "scope_note": "Structural PASS is distinct from a positive bond outcome or clinical validity.",
    }
    summary_path = args.output_dir / "summary.json"
    if args.resume and summary_path.exists():
        summary_path = args.output_dir / manifests[0].trajectory_id / "resume-summary.json"
    atomic_write_json(summary_path, summary)
    print(f"structural_checks_pass={passed}; reported={len(reports)}/{selected_count}; "
          "model_response_quality_measured=False; attachment_validity_established=False")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
