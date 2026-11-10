"""Overlay runner CLI regression locks.

The overlay runner at ``benchmarks/scripts/phase9_attachment_overlay_run.py``
is a CLI-only script; its behavioral contract is enforced here via
subprocess invocation. These tests do not launch an LLM — scripted mode
is used so CI stays free of Ollama / cloud dependencies.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "benchmarks/scripts/phase9_attachment_overlay_run.py"
SHIM = REPO / ".data/msc/train.jsonl"


def _invoke(*extra: str, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--mode",
            "scripted",
            "--limit",
            "2",
            "--max-turns-per-session",
            "2",
            "--output",
            str(output),
            *extra,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )


def test_session_scoped_and_recall_first_are_mutually_exclusive(tmp_path: Path) -> None:
    result = _invoke(
        "--session-scoped-wai-sr",
        "--recall-first-wai-sr",
        output=tmp_path / "mutex.json",
    )
    assert result.returncode != 0
    assert "mutually exclusive" in result.stdout + result.stderr


def test_baseline_variant_recorded_in_artefact(tmp_path: Path) -> None:
    out = tmp_path / "baseline.json"
    result = _invoke(output=out)
    assert result.returncode == 0, result.stderr
    artefact = json.loads(out.read_text(encoding="utf-8"))
    assert artefact["scope_hint_variant"] == "baseline"
    assert artefact["scope_hint"] is None
    assert artefact["session_scoped_wai_sr"] is False
    assert artefact["recall_first_wai_sr"] is False


def test_session_scoped_variant_recorded_in_artefact(tmp_path: Path) -> None:
    out = tmp_path / "session_scoped.json"
    result = _invoke("--session-scoped-wai-sr", output=out)
    assert result.returncode == 0, result.stderr
    artefact = json.loads(out.read_text(encoding="utf-8"))
    assert artefact["scope_hint_variant"] == "session_scoped"
    assert artefact["scope_hint"] is not None
    assert "THIS most recent session" in artefact["scope_hint"]
    assert artefact["session_scoped_wai_sr"] is True
    assert artefact["recall_first_wai_sr"] is False


def test_recall_first_variant_recorded_in_artefact(tmp_path: Path) -> None:
    out = tmp_path / "recall_first.json"
    result = _invoke("--recall-first-wai-sr", output=out)
    assert result.returncode == 0, result.stderr
    artefact = json.loads(out.read_text(encoding="utf-8"))
    assert artefact["scope_hint_variant"] == "recall_first"
    assert artefact["scope_hint"] is not None
    assert "recall ONE concrete moment" in artefact["scope_hint"]
    assert artefact["session_scoped_wai_sr"] is False
    assert artefact["recall_first_wai_sr"] is True


def test_artefact_records_within_user_variance_stats(tmp_path: Path) -> None:
    out = tmp_path / "variance.json"
    result = _invoke(output=out)
    assert result.returncode == 0, result.stderr
    artefact = json.loads(out.read_text(encoding="utf-8"))
    cohort = artefact["cohort"]
    assert "mean_llm_bond_std" in cohort
    assert "max_llm_bond_std" in cohort
    assert "mean_behavior_bond_std" in cohort
    for run in artefact["per_user_runs"]:
        assert "llm_bond_std" in run
        assert "behavior_bond_std" in run
