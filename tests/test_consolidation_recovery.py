"""Real PAM/checkpoint recovery with inert extraction/embedding controls."""
import json
from pathlib import Path

import pytest

from psychology_base.companion.pam_cli import export_system_to_path, restore_system_from_path
from psychology_base.companion.system import CompanionSystem
from psychology_base.llm.execution import ExecutionDeferred
from tests.test_consolidation import _frag, _router


def durable(tmp_path, *, response="ADD | A | likes | tea | 0.9\n"):
    router, provider = _router(tmp_path, response)
    path = tmp_path/"state.pam-v1.json"
    def fresh():
        return CompanionSystem(consolidation_enabled=True, consolidation_router=router,
                               consolidation_checkpoint_path=path)
    return fresh(), fresh, router, provider, path


def test_extraction_refusal_recovers_original_operation_and_clock(tmp_path, monkeypatch):
    system, fresh, router, _, path = durable(tmp_path)
    system._ingest_fragment(_frag("f1", "A likes tea"))
    system.mark_session_boundary()
    def refuse(*args, **kwargs):
        raise ExecutionDeferred("no allowance")
    with monkeypatch.context() as patch:
        patch.setattr(router, "chat", refuse)
        with pytest.raises(ExecutionDeferred):
            system.advance_virtual_time(1)
    operation = system.consolidator._operation_id
    resumed, _ = restore_system_from_path(path, system=fresh())
    assert resumed.consolidator._operation_id == operation
    assert resumed.heartbeat.sleep_ticks == 1
    resumed.tick()
    assert len(resumed.fact_store) == 1
    assert resumed.fact_store.all()[0].updated_tick == 2
    assert not resumed.consolidator._pending


def test_later_embedding_failure_never_persists_partial_facts(tmp_path, monkeypatch):
    system, fresh, router, _, path = durable(tmp_path,
        response="ADD | A | likes | tea | 0.9\nADD | B | likes | coffee | 0.8\n")
    system._ingest_fragment(_frag("f1", "A likes tea; B likes coffee"))
    system.mark_session_boundary()
    original, count = router.embed, []
    def fail_second(*args, **kwargs):
        count.append(1)
        if len(count) == 2:
            raise OSError("embedding interrupted")
        return original(*args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(router, "embed", fail_second)
        with pytest.raises(OSError):
            system.tick()
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert not saved["consolidation"]["facts"]["facts"]
    resumed, _ = restore_system_from_path(path, system=fresh())
    assert not len(resumed.fact_store)
    resumed.tick()
    assert [f.id for f in resumed.fact_store.all()] == ["fact-0001", "fact-0002"]
    committed, _ = restore_system_from_path(path, system=fresh())
    assert [f.id for f in committed.fact_store.all()] == ["fact-0001", "fact-0002"]
    assert committed.consolidator.stats.adds == 2


def test_atomic_replace_failure_retains_previous_valid_window(tmp_path, monkeypatch):
    from psychology_base.companion import pam_export
    system, fresh, _, _, path = durable(tmp_path)
    system._ingest_fragment(_frag("f1", "A likes tea"))
    system.mark_session_boundary()
    original = pam_export.os.replace
    def fail_commit(source, destination):
        if Path(destination) == path:
            artifact = json.loads(Path(source).read_text(encoding="utf-8"))
            if artifact["consolidation"]["facts"]["facts"]:
                raise OSError("replace failed")
        return original(source, destination)
    with monkeypatch.context() as patch:
        patch.setattr(pam_export.os, "replace", fail_commit)
        with pytest.raises(ExecutionDeferred):
            system.tick()
    assert not len(system.fact_store)
    assert not list(tmp_path.glob("*.tmp"))
    resumed, _ = restore_system_from_path(path, system=fresh())
    assert len(resumed.consolidator._pending) == 1
    resumed.tick()
    assert len(resumed.fact_store) == 1


def test_crash_after_durable_commit_before_memory_publication_does_not_reapply(tmp_path, monkeypatch):
    system, fresh, router, provider, path = durable(tmp_path)
    system._ingest_fragment(_frag("f1", "A likes tea"))
    system.mark_session_boundary()
    def crash(staged):
        raise SystemExit("process stopped after disk commit")
    with monkeypatch.context() as patch:
        patch.setattr(system.fact_store, "commit_from", crash)
        with pytest.raises(SystemExit):
            system.tick()
    calls = len(provider.calls)
    resumed, _ = restore_system_from_path(path, system=fresh())
    assert len(resumed.fact_store) == 1 and not resumed.consolidator._pending
    resumed.tick()
    assert len(provider.calls) == calls


def test_observation_storage_failure_keeps_in_process_registration(tmp_path, monkeypatch):
    from psychology_base.companion import pam_export
    system, _, _, _, path = durable(tmp_path)
    def unavailable(*args, **kwargs):
        raise OSError("disk unavailable")
    with monkeypatch.context() as patch:
        patch.setattr(pam_export.os, "replace", unavailable)
        with pytest.raises(ExecutionDeferred):
            system._ingest_fragment(_frag("f1", "A likes tea"))
        with pytest.raises(ExecutionDeferred):
            system.mark_session_boundary()
    assert len(system.consolidator._pending) == 1
    assert system.consolidator._boundary_pending
    system.tick()
    assert path.exists() and len(system.fact_store) == 1


@pytest.mark.parametrize("mutation", ["source", "sequence", "stats", "operation", "embedding", "dimensions"])
def test_invalid_recovery_state_is_refused_before_ingestion(tmp_path, mutation):
    system, fresh, _, _, path = durable(tmp_path)
    system._ingest_fragment(_frag("f1", "A likes tea"))
    system.mark_session_boundary()
    system.tick()
    payload = json.loads(path.read_text(encoding="utf-8"))
    state = payload["consolidation"]
    if mutation == "source":
        payload["fragments"][0]["content"] = "changed source"
    elif mutation == "sequence":
        state["facts"]["sequence"] = 0
    elif mutation == "stats":
        state["stats"]["adds"] = -1
    elif mutation == "operation":
        state["operation_id"] = "stale-paid-operation"
    elif mutation == "embedding":
        state["embedding_config"]["model"] = "different-model"
    else:
        first = state["facts"]["facts"][0]
        state["facts"]["facts"].append({**first, "id": "fact-0002", "embedding": [1.]})
        state["facts"]["sequence"] = 2
    path.write_text(json.dumps(payload), encoding="utf-8")
    before = path.read_bytes()
    target = fresh()
    with pytest.raises(ValueError):
        restore_system_from_path(path, system=target)
    assert not target.companion.store.all() and not len(target.fact_store)
    assert path.read_bytes() == before


def test_retry_keeps_original_window_and_later_boundary(tmp_path, monkeypatch):
    from psychology_base.companion import consolidation
    system, fresh, _, _, path = durable(tmp_path)
    system._ingest_fragment(_frag("f1", "first"))
    system.mark_session_boundary()
    windows = []
    def fail(window, **kwargs):
        windows.append([f.id for f in window])
        system._ingest_fragment(_frag("f2", "second"))
        system.mark_session_boundary()
        raise ExecutionDeferred("interrupted")
    with monkeypatch.context() as patch:
        patch.setattr(consolidation, "extract_facts", fail)
        with pytest.raises(ExecutionDeferred):
            system.tick()
    resumed, _ = restore_system_from_path(path, system=fresh())
    def capture(window, **kwargs):
        windows.append([f.id for f in window])
        return []
    with monkeypatch.context() as patch:
        patch.setattr(consolidation, "extract_facts", capture)
        resumed.tick()
        assert [f.id for f in resumed.consolidator._pending] == ["f2"]
        assert resumed.consolidator._boundary_pending
        resumed.tick()
    assert windows == [["f1"], ["f1"], ["f2"]]


def test_plain_pam_export_keeps_legacy_shape_and_consolidation_export_is_portable(tmp_path):
    system, _, _, _, _ = durable(tmp_path)
    system._ingest_fragment(_frag("f1", "A likes tea"))
    path = tmp_path/"export.pam-v1.json"
    export_system_to_path(system, path)
    restored, _ = restore_system_from_path(path)
    assert [f.id for f in restored.companion.store.all()] == ["f1"]
    plain = tmp_path/"plain.pam-v1.json"
    export_system_to_path(restored, plain)
    assert set(json.loads(plain.read_text(encoding="utf-8"))) == {"format", "hash_algo", "digest_algo", "fragments"}


def test_missing_original_source_refuses_checkpoint_without_blocking_portable_export(tmp_path):
    system, _, _, _, path = durable(tmp_path)
    system._ingest_fragment(_frag("f1", "A likes tea"))
    saved = path.read_bytes()
    # Simulate source removal through the current store owner; no derived copy
    # may silently become authoritative after the original is absent.
    system.companion.store._fragments.clear()
    with pytest.raises(ExecutionDeferred):
        system.mark_session_boundary()
    assert path.read_bytes() == saved
    portable = tmp_path/"portable.pam-v1.json"
    export_system_to_path(system, portable)
    payload = json.loads(portable.read_text(encoding="utf-8"))
    assert payload["fragments"] == [] and "consolidation" not in payload


def test_fresh_instance_cannot_overwrite_existing_recovery_without_restoring(tmp_path):
    original, fresh, _, _, path = durable(tmp_path)
    original._ingest_fragment(_frag("f1", "A likes tea"))
    original.mark_session_boundary()
    before = path.read_bytes()
    restarted = fresh()
    with pytest.raises(ExecutionDeferred):
        restarted._ingest_fragment(_frag("f2", "new input before restore"))
    with pytest.raises(ExecutionDeferred):
        restarted.tick()
    with pytest.raises(ExecutionDeferred):
        export_system_to_path(restarted, path)
    assert path.read_bytes() == before
    resumed, _ = restore_system_from_path(path, system=restarted)
    assert [f.id for f in resumed.consolidator._pending] == ["f1", "f2"]


def test_legacy_pam_can_be_restored_into_durable_consolidation(tmp_path):
    from psychology_base.companion.pam_export import write_pam
    system, fresh, _, _, path = durable(tmp_path)
    write_pam(path, [_frag("f1", "A likes tea")])
    resumed, _ = restore_system_from_path(path, system=fresh())
    assert [f.id for f in resumed.consolidator._pending] == ["f1"]
    resumed.mark_session_boundary()
    resumed.tick()
    assert len(resumed.fact_store) == 1
