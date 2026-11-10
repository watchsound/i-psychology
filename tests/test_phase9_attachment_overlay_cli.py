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


def test_per_session_seeker_flag_recorded_in_artefact(tmp_path: Path) -> None:
    out = tmp_path / "per_session_seeker.json"
    result = _invoke("--per-session-seeker", output=out)
    assert result.returncode == 0, result.stderr
    artefact = json.loads(out.read_text(encoding="utf-8"))
    assert artefact["per_session_seeker"] is True
    # Composable with scope hints — not a mutex.
    assert artefact["scope_hint_variant"] == "baseline"


def test_per_session_seeker_default_off(tmp_path: Path) -> None:
    out = tmp_path / "default.json"
    result = _invoke(output=out)
    assert result.returncode == 0, result.stderr
    artefact = json.loads(out.read_text(encoding="utf-8"))
    assert artefact["per_session_seeker"] is False


def test_per_session_seeker_composes_with_recall_first(tmp_path: Path) -> None:
    out = tmp_path / "composed.json"
    result = _invoke(
        "--per-session-seeker", "--recall-first-wai-sr", output=out
    )
    assert result.returncode == 0, result.stderr
    artefact = json.loads(out.read_text(encoding="utf-8"))
    assert artefact["per_session_seeker"] is True
    assert artefact["scope_hint_variant"] == "recall_first"
    assert artefact["recall_first_wai_sr"] is True


def test_persona_reroll_flag_recorded_in_artefact(tmp_path: Path) -> None:
    out = tmp_path / "persona_reroll.json"
    result = _invoke("--persona-reroll-per-session", output=out)
    assert result.returncode == 0, result.stderr
    artefact = json.loads(out.read_text(encoding="utf-8"))
    assert artefact["persona_reroll_per_session"] is True
    # Composable with scope hints — not a mutex.
    assert artefact["scope_hint_variant"] == "baseline"
    # Mutex partner stays off.
    assert artefact["per_session_seeker"] is False


def test_persona_reroll_default_off(tmp_path: Path) -> None:
    out = tmp_path / "default.json"
    result = _invoke(output=out)
    assert result.returncode == 0, result.stderr
    artefact = json.loads(out.read_text(encoding="utf-8"))
    assert artefact["persona_reroll_per_session"] is False


def test_persona_reroll_and_per_session_seeker_are_mutually_exclusive(
    tmp_path: Path,
) -> None:
    result = _invoke(
        "--persona-reroll-per-session",
        "--per-session-seeker",
        output=tmp_path / "mutex.json",
    )
    assert result.returncode != 0
    assert "mutually exclusive" in result.stdout + result.stderr


def test_persona_reroll_composes_with_recall_first(tmp_path: Path) -> None:
    out = tmp_path / "composed.json"
    result = _invoke(
        "--persona-reroll-per-session", "--recall-first-wai-sr", output=out
    )
    assert result.returncode == 0, result.stderr
    artefact = json.loads(out.read_text(encoding="utf-8"))
    assert artefact["persona_reroll_per_session"] is True
    assert artefact["scope_hint_variant"] == "recall_first"
    assert artefact["recall_first_wai_sr"] is True


def test_wai_sr_temperature_default_off(tmp_path: Path) -> None:
    out = tmp_path / "default.json"
    result = _invoke(output=out)
    assert result.returncode == 0, result.stderr
    artefact = json.loads(out.read_text(encoding="utf-8"))
    assert artefact["wai_sr_temperature"] is None


def test_wai_sr_temperature_flag_recorded_in_artefact(tmp_path: Path) -> None:
    out = tmp_path / "temp.json"
    result = _invoke("--wai-sr-temperature", "1.0", output=out)
    assert result.returncode == 0, result.stderr
    artefact = json.loads(out.read_text(encoding="utf-8"))
    assert artefact["wai_sr_temperature"] == 1.0
    # Composable with scope hints and persona-reroll — not a mutex.
    assert artefact["scope_hint_variant"] == "baseline"
    assert artefact["persona_reroll_per_session"] is False


def test_wai_sr_temperature_composes_with_persona_reroll(tmp_path: Path) -> None:
    out = tmp_path / "composed.json"
    result = _invoke(
        "--wai-sr-temperature",
        "1.0",
        "--persona-reroll-per-session",
        "--recall-first-wai-sr",
        output=out,
    )
    assert result.returncode == 0, result.stderr
    artefact = json.loads(out.read_text(encoding="utf-8"))
    assert artefact["wai_sr_temperature"] == 1.0
    assert artefact["persona_reroll_per_session"] is True
    assert artefact["recall_first_wai_sr"] is True
    assert artefact["scope_hint_variant"] == "recall_first"


def test_wai_sr_samples_default_off(tmp_path: Path) -> None:
    out = tmp_path / "default.json"
    result = _invoke(output=out)
    assert result.returncode == 0, result.stderr
    artefact = json.loads(out.read_text(encoding="utf-8"))
    assert artefact["wai_sr_samples"] == 1
    # Cohort rollup absent when N=1 (no within-sample dispersion defined).
    assert artefact["cohort"].get("mean_within_sample_std") is None
    assert artefact["cohort"].get("max_within_sample_std") is None
    # Per-session records omit the samples side-channel at N=1.
    for run in artefact["per_user_runs"]:
        assert run.get("mean_within_sample_std") is None
        for s in run["per_session"]:
            assert "wai_sr_item_samples" not in s


def test_wai_sr_samples_flag_recorded_in_artefact(tmp_path: Path) -> None:
    out = tmp_path / "samples.json"
    result = _invoke("--wai-sr-samples", "3", output=out)
    assert result.returncode == 0, result.stderr
    artefact = json.loads(out.read_text(encoding="utf-8"))
    assert artefact["wai_sr_samples"] == 3
    # Composable with every other flag — not a mutex.
    assert artefact["scope_hint_variant"] == "baseline"
    assert artefact["persona_reroll_per_session"] is False
    assert artefact["wai_sr_temperature"] is None
    # Per-session records carry the raw sample arrays when N>1.
    scripted_sample_count_seen = False
    for run in artefact["per_user_runs"]:
        for s in run["per_session"]:
            samples = s.get("wai_sr_item_samples")
            if samples is None:
                continue
            scripted_sample_count_seen = True
            # 5 items, N=3 samples each — every value is an int in [1,7].
            assert set(samples.keys()) == {"1", "2", "3", "4", "5"}
            for arr in samples.values():
                assert len(arr) == 3
                assert all(isinstance(x, int) and 1 <= x <= 7 for x in arr)
    assert scripted_sample_count_seen


def test_wai_sr_samples_rejects_zero(tmp_path: Path) -> None:
    result = _invoke("--wai-sr-samples", "0", output=tmp_path / "bad.json")
    assert result.returncode != 0
    assert ">= 1" in result.stdout + result.stderr
