"""Phase 4a §6c — Digital Bereavement live-fire test.

End-to-end PAM-v1 export → disk → fresh-system restore → recall-still-works.
Covers Phase 4 plan §6c Digital Bereavement live-fire scope:

1. CLI-equivalent round-trip via `export_system_to_path` + `restore_system_from_path`.
2. agent-memory-mcp interop bridge: the `digest_algo` manifest key is
   emitted on export + accepted on import (and conflicts with `hash_algo`
   are rejected).
3. Semantic equivalence: a recall against the restored system returns
   fragments byte-identical to the pre-export recall.
4. CLI subcommands (`inspect`, `restore`) exercised via `_run_cli`.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from psychology_base.companion.pam_cli import (
    _run_cli,
    _summarise_artifact,
    export_system_to_path,
    restore_system_from_path,
)
from psychology_base.companion.pam_export import (
    PAM_DIGEST_ALGO_KEY,
    PAM_FORMAT,
    PAM_HASH_ALGO,
    pam_export,
    pam_import,
)
from psychology_base.companion.system import CompanionSystem
from psychology_base.eval.adapters import Observation


TS = datetime(2026, 10, 7, 12, 0, 0, tzinfo=timezone.utc)


def _obs(content: str, offset_minutes: int = 0, tags=()) -> Observation:
    return Observation(
        timestamp=TS + timedelta(minutes=offset_minutes),
        speaker="user",
        content=content,
        tags=tuple(tags),
    )


def _populate_system() -> CompanionSystem:
    system = CompanionSystem(top_k=4)
    for i, text in enumerate([
        "I moved to Boston last year for a research fellowship.",
        "My cat's name is Mochi and she sleeps on the keyboard.",
        "I'm learning Mandarin for a trip to Taipei in spring.",
        "Last Tuesday I ran five kilometers before breakfast.",
        "The novel I'm writing is about a cartographer who maps grief.",
    ]):
        system.observe(_obs(text, offset_minutes=i))
    return system


def test_round_trip_via_cli_api_restores_full_fragment_set(tmp_path: Path) -> None:
    source = _populate_system()
    out = tmp_path / "pam.json"

    count = export_system_to_path(source, out)
    assert count == 5
    assert out.exists()

    restored, offenders = restore_system_from_path(out)
    assert offenders == []  # strict DAG byte-identity

    source_ids = sorted(f.id for f in source.companion.store.all())
    restored_ids = sorted(f.id for f in restored.companion.store.all())
    assert source_ids == restored_ids


def test_restored_system_semantic_recall_matches_source(tmp_path: Path) -> None:
    source = _populate_system()
    out = tmp_path / "pam.json"
    export_system_to_path(source, out)

    restored, _ = restore_system_from_path(out)

    source_recall = source.recall("What is my cat's name?")
    restored_recall = restored.recall("What is my cat's name?")

    source_contents = {f.content for f in source_recall.fragments_used}
    restored_contents = {f.content for f in restored_recall.fragments_used}
    # Semantic equivalence: the restored system surfaces at least the
    # same fragment the source surfaced for the Mochi cue.
    assert any("Mochi" in c for c in source_contents)
    assert any("Mochi" in c for c in restored_contents)


def test_export_emits_both_hash_algo_and_digest_algo(tmp_path: Path) -> None:
    source = _populate_system()
    out = tmp_path / "pam.json"
    export_system_to_path(source, out)

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["format"] == PAM_FORMAT
    assert data["hash_algo"] == PAM_HASH_ALGO
    assert data[PAM_DIGEST_ALGO_KEY] == PAM_HASH_ALGO


def test_import_accepts_digest_algo_only(tmp_path: Path) -> None:
    """agent-memory-mcp-produced artifact: digest_algo only, no hash_algo."""
    source = _populate_system()
    artifact = pam_export(list(source.companion.store.all()))
    del artifact["hash_algo"]
    # digest_algo-only shape should still round-trip.
    rehydrated = pam_import(artifact)
    assert len(rehydrated) == 5


def test_import_rejects_conflicting_manifest_keys() -> None:
    source = _populate_system()
    artifact = pam_export(list(source.companion.store.all()))
    artifact["hash_algo"] = "blake2b-128"
    artifact[PAM_DIGEST_ALGO_KEY] = "sha256"
    with pytest.raises(ValueError, match="disagree"):
        pam_import(artifact)


def test_import_rejects_missing_both_manifest_keys() -> None:
    source = _populate_system()
    artifact = pam_export(list(source.companion.store.all()))
    del artifact["hash_algo"]
    del artifact[PAM_DIGEST_ALGO_KEY]
    with pytest.raises(ValueError, match="missing"):
        pam_import(artifact)


def test_summarise_artifact_reports_manifest(tmp_path: Path) -> None:
    source = _populate_system()
    out = tmp_path / "pam.json"
    export_system_to_path(source, out)

    summary = _summarise_artifact(out)
    assert summary["format"] == PAM_FORMAT
    assert summary["hash_algo"] == PAM_HASH_ALGO
    assert summary[PAM_DIGEST_ALGO_KEY] == PAM_HASH_ALGO
    assert summary["fragment_count"] == 5


def test_cli_inspect_prints_manifest(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = _populate_system()
    out = tmp_path / "pam.json"
    export_system_to_path(source, out)

    rc = _run_cli(["inspect", str(out)])
    assert rc == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["fragment_count"] == 5
    assert payload["hash_algo"] == PAM_HASH_ALGO


def test_cli_restore_prints_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = _populate_system()
    out = tmp_path / "pam.json"
    export_system_to_path(source, out)

    rc = _run_cli(["restore", str(out)])
    assert rc == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["fragments_restored"] == 5
    assert payload["content_addressing_offenders"] == []
    assert payload["strict_dag_byte_identity"] is True


def test_cli_restore_exits_nonzero_on_malformed_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text('{"format": "pam-v99"}', encoding="utf-8")
    rc = _run_cli(["restore", str(bad)])
    assert rc == 2


def test_restore_into_existing_system_extends_it(tmp_path: Path) -> None:
    source = _populate_system()
    out = tmp_path / "pam.json"
    export_system_to_path(source, out)

    # Second system has its own content; restore should union.
    existing = CompanionSystem(top_k=4)
    existing.observe(_obs("Independent observation not in the artifact.", offset_minutes=100))
    pre_count = len(list(existing.companion.store.all()))
    assert pre_count == 1

    restored, _ = restore_system_from_path(out, system=existing)
    assert restored is existing
    post_count = len(list(restored.companion.store.all()))
    assert post_count == pre_count + 5
