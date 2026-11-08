"""Offline acceptance cases for serving defaults, recovery and deadlines."""
from dataclasses import replace
import json
import sqlite3
import threading
import time

import pytest

from psychology_base.llm.accounting import RuntimeBudget
from psychology_base.llm.config import CacheConfig, LLMConfig, LocalOllamaConfig, ProviderConfig, default_config
from psychology_base.llm.execution import ExecutionDeferred, ExecutionPolicy
from psychology_base.llm.providers import AlibabaTongyiProvider
from psychology_base.llm.router import LLMRouter
from psychology_base.llm.types import ChatMessage
from tests.test_runtime_budget import allowance, paid_router, call


def _reserve_in_process(root, crash=False, ready=None):
    """Real child-process SQLite admission; no provider object is constructed."""
    from pathlib import Path
    import os
    if ready is not None:
        ready.set()
    policy = allowance(Path(root), max_calls=7)
    budget = RuntimeBudget(policy)
    admitted = 0
    for index in range(10):
        try:
            budget.reserve(".001", session="s", operation=f"process-{os.getpid()}", purpose="offline-test",
                           provider="control", model="none", kind="chat")
            admitted += 1
            if crash:
                os._exit(0)
        except ExecutionDeferred:
            break
    return admitted


def test_programmatic_configuration_defaults_to_local():
    assert ExecutionPolicy().mode == "local"
    assert LLMConfig("tongyi").execution.mode == "local"
    assert default_config().execution.mode == "local"


def test_programmatic_remote_provider_cannot_bypass_admission(monkeypatch):
    provider = AlibabaTongyiProvider()
    def forbidden(*args, **kwargs):
        raise AssertionError("Unadmitted cloud dispatch")
    monkeypatch.setattr(provider, "chat", forbidden)
    cfg = LLMConfig("tongyi", {"tongyi": ProviderConfig("tongyi", "small", "large")},
                    local_ollama=LocalOllamaConfig(enabled=False), cache=CacheConfig(False))
    router = LLMRouter(cfg, providers={"tongyi": provider})
    with pytest.raises(ExecutionDeferred):
        router.chat([ChatMessage("user", "hello")], max_tokens=50)


def test_corrupt_accounting_store_defers_without_replacing_it(tmp_path):
    policy = allowance(tmp_path)
    policy.ledger.write_bytes(b"not a sqlite database")
    with pytest.raises(ExecutionDeferred):
        RuntimeBudget(policy)
    assert policy.ledger.read_bytes() == b"not a sqlite database"


def test_accounting_limits_are_shared_across_real_processes(tmp_path):
    import multiprocessing
    policy = allowance(tmp_path, max_calls=7)
    RuntimeBudget(policy)
    with multiprocessing.get_context("spawn").Pool(4) as pool:
        admitted = pool.map(_reserve_in_process, [str(tmp_path)] * 4)
    assert sum(admitted) == 7
    assert RuntimeBudget(policy).summary()["provider_attempts"] == 7


def test_killed_process_cannot_recover_reserved_allowance(tmp_path):
    import multiprocessing
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    process = context.Process(target=_reserve_in_process, args=(str(tmp_path), True, ready))
    process.start()
    try:
        # Cold spawn imports can exceed 15 seconds on a shared Windows host.
        # Keep the original reservation deadline, starting after imports finish.
        if not ready.wait(60):
            pytest.fail("Accounting child failed to initialize")
        process.join(15)
        if process.is_alive():
            pytest.fail("Accounting child reservation failed to finish")
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
    assert process.exitcode == 0
    summary = RuntimeBudget(allowance(tmp_path, max_calls=7)).summary()
    assert summary["provider_attempts"] == 1 and summary["usage_uncertain"]


def test_locked_accounting_storage_defers_before_dispatch(tmp_path, monkeypatch):
    router, calls = paid_router(tmp_path, monkeypatch)
    lock = sqlite3.connect(router.budget.path)
    try:
        lock.execute("BEGIN IMMEDIATE")
        with router.operation("locked", timeout_s=.05):
            with pytest.raises(ExecutionDeferred):
                router.chat([ChatMessage("user", "hello")], max_tokens=50)
    finally:
        lock.rollback()
        lock.close()
    assert not calls
    assert router.budget.summary()["provider_attempts"] == 0


def test_operation_scopes_restore_context_and_nested_deadlines_cannot_extend(tmp_path, monkeypatch):
    from psychology_base.llm.execution import current_operation
    router, _ = paid_router(tmp_path, monkeypatch)
    with router.operation("outer", timeout_s=.1) as operation_id:
        deadline = current_operation.get().deadline
        with router.operation("inner", timeout_s=30) as nested_id:
            assert nested_id == operation_id
            assert current_operation.get().deadline == deadline
    assert current_operation.get() is None


def test_caller_cannot_extend_configured_operation_limit(tmp_path, monkeypatch):
    from psychology_base.llm.execution import current_operation
    router, _ = paid_router(tmp_path, monkeypatch, policy=allowance(tmp_path, operation_timeout_s=.1))
    started = time.monotonic()
    with router.operation("bounded", timeout_s=300):
        assert current_operation.get().deadline <= started + .11


def test_provider_timeout_is_distinct_from_operation_deadline(tmp_path, monkeypatch):
    from psychology_base.llm.execution import current_operation
    router, _ = paid_router(tmp_path, monkeypatch)
    original = router.providers["tongyi"].chat
    def fail(*args, **kwargs):
        raise TimeoutError("request timeout before overall deadline")
    with router.operation("request-failure", timeout_s=2):
        with monkeypatch.context() as patch:
            patch.setattr(router.providers["tongyi"], "chat", fail)
            with pytest.raises(ExecutionDeferred):
                router.chat([ChatMessage("user", "first")], max_tokens=50)
        assert not current_operation.get().cancelled.is_set()
        router.chat([ChatMessage("user", "explicit second request")], max_tokens=50)
    assert router.budget.summary()["provider_attempts"] == 2


def test_exhausted_paid_transport_slots_do_not_block_local_responses(tmp_path, monkeypatch):
    from contextlib import ExitStack
    from psychology_base.llm.execution import request_slot
    from tests.test_runtime_execution import local_router
    local, _ = local_router(monkeypatch, tmp_path)
    with local.operation("transport-isolation", timeout_s=1), ExitStack() as stack:
        for _ in range(4):
            stack.enter_context(request_slot("paid"))
        assert local.chat([ChatMessage("user", "hello")]).provider == "ollama"


def test_expired_operation_admits_no_paid_request(tmp_path, monkeypatch):
    router, calls = paid_router(tmp_path, monkeypatch)
    with router.operation("expired", timeout_s=.01):
        time.sleep(.02)
        with pytest.raises(ExecutionDeferred):
            router.chat([ChatMessage("user", "hello")], max_tokens=50)
    assert not calls
    assert router.budget.summary()["provider_attempts"] == 0


def test_deadline_returns_without_waiting_for_stuck_provider(tmp_path, monkeypatch):
    from concurrent.futures import Future
    from types import SimpleNamespace
    from psychology_base.llm import execution
    router, calls = paid_router(tmp_path, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    original = router.providers["tongyi"].chat
    monotonic = time.monotonic
    clock, waits = [None], []
    # This test covers a transport already in flight. Real SQLite admission and
    # worker scheduling need not finish within 50 ms on a shared Windows host.
    # Freeze only the execution clock during admission; Future.result still
    # enforces the original 50 ms real wait after bounded worker readiness.
    monkeypatch.setattr(execution, "time", SimpleNamespace(
        monotonic=lambda: monotonic() if clock[0] is None else clock[0]))
    class ReadyFuture(Future):
        def result(self, timeout=None):
            assert entered.wait(5), "Provider worker failed to initialize"
            started = monotonic()
            try:
                return super().result(timeout=timeout)
            finally:
                waits.append((timeout, monotonic()-started))
    monkeypatch.setattr(execution, "Future", ReadyFuture)
    def blocked(*args, **kwargs):
        entered.set()
        release.wait(2)
        return original(*args, **kwargs)
    monkeypatch.setattr(router.providers["tongyi"], "chat", blocked)
    try:
        with router.operation("deadline", timeout_s=.05):
            clock[0] = execution.current_operation.get().deadline - .05
            with pytest.raises(ExecutionDeferred):
                router.chat([ChatMessage("user", "hello")], max_tokens=50)
            assert execution.current_operation.get().cancelled.is_set()
        assert entered.is_set()
        assert len(waits) == 1 and waits[0][0] == pytest.approx(.05)
        assert waits[0][1] < .7
        assert router.budget.summary()["usage_uncertain"]
    finally:
        release.set()


def test_deadline_expiring_after_reservation_prevents_provider_entry(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from psychology_base.llm import execution
    router, calls = paid_router(tmp_path, monkeypatch)
    original = router.budget.reserve
    clock = [None]
    monotonic = time.monotonic
    monkeypatch.setattr(execution, "time", SimpleNamespace(
        monotonic=lambda: monotonic() if clock[0] is None else clock[0]))
    def expire_after_real_reservation(*args, **kwargs):
        reservation = original(*args, **kwargs)
        clock[0] = execution.current_operation.get().deadline + 1
        return reservation
    monkeypatch.setattr(router.budget, "reserve", expire_after_real_reservation)
    with router.operation("admission-deadline", timeout_s=.05):
        with pytest.raises(ExecutionDeferred):
            router.chat([ChatMessage("user", "hello")], max_tokens=50)
    assert not calls
    summary = router.budget.summary()
    assert summary['provider_attempts'] == 1 and summary['usage_uncertain']


def test_local_metadata_cannot_start_inference_after_deadline(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from psychology_base.llm import execution
    from tests.test_runtime_execution import local_router
    router, _ = local_router(monkeypatch, tmp_path)
    # Cross the real operation's deadline inside the metadata boundary. Do not
    # require Windows to schedule a worker within an arbitrary 50 ms window.
    clock = [None]
    monotonic = time.monotonic
    monkeypatch.setattr(execution, "time", SimpleNamespace(
        monotonic=lambda: monotonic() if clock[0] is None else clock[0]))
    calls = []
    def slow_metadata(*args, **kwargs):
        clock[0] = execution.current_operation.get().deadline + 1
    monkeypatch.setattr(router.providers["ollama"], "require_local_model", slow_metadata)
    monkeypatch.setattr(router.providers["ollama"], "chat", lambda *args, **kwargs: calls.append("late-inference"))
    with pytest.raises(ExecutionDeferred):
        router.chat([ChatMessage("user", "hello")])
    assert clock[0] is not None
    assert not calls


@pytest.mark.parametrize("backend", ["openai", "anthropic"])
def test_sdk_request_uses_remaining_operation_time(tmp_path, monkeypatch, backend):
    from types import SimpleNamespace
    from psychology_base.llm.providers import OpenAIProvider, AnthropicProvider
    router, _ = paid_router(tmp_path, monkeypatch)
    captured = []
    def create(**kwargs):
        captured.append(kwargs)
        if backend == "openai":
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="control"), finish_reason="stop")], usage=None)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="control")], usage=None)
    if backend == "openai":
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        provider = OpenAIProvider(_client=client, request_timeout_s=60)
    else:
        client = SimpleNamespace(messages=SimpleNamespace(create=create))
        provider = AnthropicProvider(_client=client, request_timeout_s=60)
    with router.operation("wire-control", timeout_s=1):
        provider.chat([ChatMessage("user", "control")], max_tokens=10)
    assert 0 < captured[0]["timeout"] <= 1


def test_consolidation_checkpoint_recovers_pending_work_and_committed_facts(tmp_path):
    from psychology_base.companion.system import CompanionSystem
    from psychology_base.companion.pam_cli import restore_system_from_path
    from psychology_base.eval.adapters import Observation
    from tests.test_consolidation import _router, _frag
    router, _ = _router(tmp_path, "ADD | A | likes | tea | 0.9\n")
    path = tmp_path/"checkpoint.pam-v1.json"
    system = CompanionSystem(consolidation_enabled=True, consolidation_router=router,
                             consolidation_checkpoint_path=path)
    fragment = _frag("f1", "A likes tea")
    system._ingest_fragment(fragment)
    system.mark_session_boundary()
    restored, _ = restore_system_from_path(path, system=CompanionSystem(
        consolidation_enabled=True, consolidation_router=router, consolidation_checkpoint_path=path))
    assert [f.id for f in restored.consolidator._pending] == [fragment.id]
    assert restored.consolidator._boundary_pending
    restored.tick()
    assert len(restored.fact_store) == 1
    again, _ = restore_system_from_path(path, system=CompanionSystem(
        consolidation_enabled=True, consolidation_router=router, consolidation_checkpoint_path=path))
    assert len(again.fact_store) == 1
    assert not again.consolidator._pending
    again.tick()
    assert len(again.fact_store) == 1
