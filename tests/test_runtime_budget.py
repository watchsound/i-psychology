"""Offline controls for actual router admission, not simulated model accuracy."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from decimal import Decimal
import sqlite3

import pytest

from psychology_base.llm.accounting import Budget, RuntimeBudget
from psychology_base.llm.cache import LLMCache
from psychology_base.llm.config import CacheConfig, LLMConfig, LocalOllamaConfig
from psychology_base.llm.execution import ExecutionDeferred, ExecutionPolicy, PaidRate
from psychology_base.llm.providers import AlibabaTongyiProvider, AnthropicProvider, OpenAIProvider
from psychology_base.llm.router import LLMRouter
from psychology_base.llm.types import ChatMessage, Completion, TaskTier


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import socket
    def forbidden(*args, **kwargs):
        raise AssertionError("Runtime budget tests must never use the network")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def allowance(tmp_path, **kwargs):
    rates = {f"tongyi/{kind}/{model}": PaidRate("1", "2", 4000, 100)
             for kind, model in (("chat", "small"), ("chat", "large"), ("embed", "vectors"), ("rerank", "ranker"))}
    return replace(ExecutionPolicy(mode="paid", ledger=tmp_path/"budget.sqlite3", budget_id="test-only",
        max_calls=30, max_usd="1", session_max_calls=20, session_max_usd=".5",
        operation_max_calls=10, operation_max_usd=".1", rates=rates), **kwargs)


def paid_router(tmp_path, monkeypatch, *, policy=None, session="session", usage=(10, 5), cache=False, user=None):
    policy = policy or allowance(tmp_path)
    provider = AlibabaTongyiProvider(simple_model="small", advanced_model="large", embedding_model="vectors", rerank_model="ranker")
    calls = []

    def chat(messages, *, tier, **kwargs):
        # The durable reservation must exist before the transport boundary.
        with sqlite3.connect(policy.ledger) as db:
            assert db.execute("SELECT COUNT(*) FROM calls WHERE state='reserved'").fetchone()[0] > 0
        calls.append("chat")
        return Completion("accounting control", "tongyi", provider.model_for(tier), tier,
                          prompt_tokens=usage[0], completion_tokens=usage[1])

    def embed(text, **kwargs):
        calls.append("embed")
        return [1.0, 2.0]

    def rerank(query, documents, **kwargs):
        calls.append("rerank")
        return [(0, .5)]

    monkeypatch.setattr(provider, "chat", chat)
    monkeypatch.setattr(provider, "embed", embed)
    monkeypatch.setattr(provider, "rerank", rerank)
    cfg = LLMConfig("tongyi", local_ollama=LocalOllamaConfig(enabled=False), cache=CacheConfig(False), execution=policy)
    router = LLMRouter(cfg, providers={"tongyi": provider}, session_id=session,
                       user_id=user,
                       cache=LLMCache(tmp_path/"cache") if cache else None)
    return router, calls


def call(router, text="hello", *, operation=None):
    with router.operation("test_accounting", operation_id=operation):
        return router.chat([ChatMessage("user", text)], max_tokens=50)


def test_no_allowance_or_purpose_means_no_dispatch(tmp_path, monkeypatch):
    with pytest.raises(ExecutionDeferred):
        RuntimeBudget(ExecutionPolicy(mode="paid"))
    router, calls = paid_router(tmp_path, monkeypatch)
    with pytest.raises(ExecutionDeferred):
        router.chat([ChatMessage("user", "hello")], max_tokens=50)
    assert not calls
    assert router.budget.summary()["provider_attempts"] == 0


def test_cache_hit_is_free_even_after_call_allowance_exhausted(tmp_path, monkeypatch):
    router, calls = paid_router(tmp_path, monkeypatch, policy=allowance(tmp_path, max_calls=1), cache=True)
    assert call(router).content == call(router).content
    assert calls == ["chat"]
    assert router.budget.summary()["cache_hits"] == 1
    assert Decimal(router.budget.summary()["accounted_usd"]) == Decimal(".000020")
    with pytest.raises(ExecutionDeferred):
        call(router, "uncached")


def test_shared_admission_under_concurrent_multiple_routers(tmp_path, monkeypatch):
    policy = allowance(tmp_path, max_calls=5)
    routers = [paid_router(tmp_path, monkeypatch, policy=policy, session=f"session-{i}", usage=(0, 0)) for i in range(4)]
    def attempt(index):
        try:
            call(routers[index % 4][0], str(index))
            return True
        except ExecutionDeferred:
            return False
    with ThreadPoolExecutor(max_workers=12) as pool:
        accepted = list(pool.map(attempt, range(40)))
    assert sum(accepted) == 5
    assert sum(len(calls) for _, calls in routers) == 5
    assert RuntimeBudget(policy).summary()["provider_attempts"] == 5
    assert RuntimeBudget(policy).summary()["usage_uncertain"]


@pytest.mark.parametrize("scope", ["session", "operation"])
def test_shared_scope_limits(tmp_path, monkeypatch, scope):
    policy = allowance(tmp_path, **{scope + "_max_calls": 1})
    first, _ = paid_router(tmp_path, monkeypatch, policy=policy)
    second, calls = paid_router(tmp_path, monkeypatch, policy=policy)
    call(first, operation="same-operation")
    with pytest.raises(ExecutionDeferred):
        call(second, operation="same-operation")
    assert not calls


def test_atomic_spend_limit_and_unfinished_reservation_survive_restart(tmp_path):
    policy = allowance(tmp_path, max_usd=".001")
    first = RuntimeBudget(policy)
    args = dict(session="s", operation="o", purpose="test", provider="tongyi", model="m", kind="chat")
    first.reserve(Decimal(".0008"), **args)
    restarted = RuntimeBudget(policy)
    with pytest.raises(ExecutionDeferred):
        restarted.reserve(Decimal(".0003"), **args)
    assert restarted.summary()["provider_attempts"] == 1
    assert restarted.summary()["usage_uncertain"]
    assert Decimal(restarted.summary()["accounted_usd"]) == Decimal(".0008")


def test_contract_cannot_silently_enlarge_existing_allowance(tmp_path):
    policy = allowance(tmp_path)
    RuntimeBudget(policy)
    with pytest.raises(ExecutionDeferred):
        RuntimeBudget(replace(policy, max_calls=31))


def test_failed_request_keeps_reservation_and_never_retries(tmp_path, monkeypatch):
    router, calls = paid_router(tmp_path, monkeypatch)
    def failed(*args, **kwargs):
        calls.append("failure")
        raise TimeoutError("unknown billed outcome")
    monkeypatch.setattr(router.providers["tongyi"], "chat", failed)
    with pytest.raises(ExecutionDeferred):
        call(router)
    assert calls == ["failure"]
    assert router.budget.summary()["provider_attempts"] == 1
    assert router.budget.summary()["usage_uncertain"]
    assert Decimal(router.budget.summary()["accounted_usd"]) > 0


def test_usage_overrun_holds_budget_before_next_dispatch(tmp_path, monkeypatch):
    router, calls = paid_router(tmp_path, monkeypatch, usage=(10, 51))
    with pytest.raises(ExecutionDeferred):
        call(router)
    with pytest.raises(ExecutionDeferred):
        call(router, "next")
    assert calls == ["chat"]
    assert router.budget.summary()["usage_uncertain"]


@pytest.mark.parametrize("kind", ["chat", "embed", "rerank"])
def test_every_remote_operation_is_admitted_and_counted(tmp_path, monkeypatch, kind):
    router, calls = paid_router(tmp_path, monkeypatch, policy=allowance(tmp_path, max_calls=1))
    def dispatch():
        with router.operation("test_" + kind):
            if kind == "embed":
                return router.embed("a", backend="cloud", use_cache=False)
            if kind == "rerank":
                return router.rerank("q", ["a", "b"])
            return router.chat([ChatMessage("user", "a")], max_tokens=50, use_cache=False)
    dispatch()
    with pytest.raises(ExecutionDeferred):
        dispatch()
    assert calls == [kind]
    assert router.budget.summary()["provider_attempts"] == 1
    assert router.budget.summary()["usage_uncertain"] is (kind != "chat")


@pytest.mark.parametrize("kind", ["chat", "embed"])
def test_parallel_batches_share_operation_limit(tmp_path, monkeypatch, kind):
    router, calls = paid_router(tmp_path, monkeypatch, policy=allowance(tmp_path, operation_max_calls=2))
    with pytest.raises(ExecutionDeferred):
        if kind == "chat":
            router.chat_batch([[ChatMessage("user", str(i))] for i in range(10)], max_tokens=50, use_cache=False)
        else:
            router.embed_batch([str(i) for i in range(10)], backend="cloud", use_cache=False)
    assert len(calls) == 2
    assert router.budget.summary()["provider_attempts"] == 2


@pytest.mark.parametrize("condition", ["input", "output", "missing-rate", "missing-output"])
def test_invalid_request_is_refused_before_dispatch(tmp_path, monkeypatch, condition):
    router, calls = paid_router(tmp_path, monkeypatch)
    text = "x" * 5000 if condition == "input" else "hello"
    max_tokens = None if condition == "missing-output" else 101 if condition == "output" else 50
    tier = TaskTier.ADVANCED if condition == "missing-rate" else TaskTier.SIMPLE
    if condition == "missing-rate":
        router.providers["tongyi"].advanced_model = "unconfigured"
    with router.operation("test"):
        with pytest.raises(ExecutionDeferred):
            router.chat([ChatMessage("user", text)], max_tokens=max_tokens, tier=tier)
    assert not calls


@pytest.mark.parametrize("provider_type", [OpenAIProvider, AnthropicProvider])
def test_sdk_dispatch_has_no_hidden_retries_and_has_timeout(tmp_path, monkeypatch, provider_type):
    policy = allowance(tmp_path)
    provider = provider_type(simple_model="small", advanced_model="large", max_retries=3)
    policy = replace(policy, rates={f"{provider.name}/chat/small": PaidRate("1", "2", 4000, 100)})
    cfg = LLMConfig(provider.name, cache=CacheConfig(False), execution=policy)
    seen = []
    def chat(*args, **kwargs):
        seen.append((provider.max_retries, provider.request_timeout_s, provider.follow_redirects))
        return Completion("control", provider.name, "small", TaskTier.SIMPLE, 10, 5)
    monkeypatch.setattr(provider, "chat", chat)
    router = LLMRouter(cfg, providers={provider.name: provider}, session_id="sdk-test")
    call(router)
    call(router, "second")
    assert seen == [(0, 60.0, False)] * 2


def test_historical_budget_import_is_the_same_owner():
    from psychology_base.eval.benchmarks.application_calls import Budget as LegacyBudget
    assert LegacyBudget is Budget


def test_missing_local_embedding_does_not_fall_back_even_in_paid_mode(tmp_path, monkeypatch):
    router, calls = paid_router(tmp_path, monkeypatch)
    with router.operation("routine_embedding"):
        with pytest.raises(ExecutionDeferred):
            router.embed("text")
    assert not calls


def test_runtime_rates_are_snapshotted_at_budget_registration(tmp_path, monkeypatch):
    router, calls = paid_router(tmp_path, monkeypatch)
    router.execution.rates["tongyi/chat/small"] = PaidRate("0", "0", 4000, 100)
    call(router)
    assert Decimal(router.budget.summary()["accounted_usd"]) == Decimal(".000020")
    assert calls == ["chat"]


def test_wrong_completion_identity_is_not_cached(tmp_path, monkeypatch):
    router, _ = paid_router(tmp_path, monkeypatch, cache=True)
    provider = router.providers["tongyi"]
    monkeypatch.setattr(provider, "chat", lambda *args, **kwargs:
        Completion("wrong model", "tongyi", "different", TaskTier.SIMPLE, 10, 5))
    with pytest.raises(ExecutionDeferred):
        call(router)
    assert not list((tmp_path/"cache").rglob("*.json"))
    assert router.budget.summary()["usage_uncertain"]


def test_budget_settlement_is_unique_and_unknown_usage_never_refunds(tmp_path):
    budget = RuntimeBudget(allowance(tmp_path))
    dispatch = budget.reserve(Decimal(".001"), session="s", operation="o", purpose="p", provider="p", model="m", kind="chat")
    budget.settle(dispatch)
    assert Decimal(budget.summary()["accounted_usd"]) == Decimal(".001")
    with pytest.raises(ValueError):
        budget.settle(dispatch, actual=Decimal(0))


def test_execution_config_parsing_without_credentials():
    from psychology_base.llm.config import _parse_config
    cfg = _parse_config({"active_provider": "ollama", "providers": {"ollama": {
        "simple_model": "small", "advanced_model": "large"}}, "execution": {"mode": "local"}})
    assert cfg.execution.mode == "local"
    without_execution = _parse_config({"active_provider": "ollama", "providers": {"ollama": {
        "simple_model": "small", "advanced_model": "large"}}})
    assert without_execution.execution.mode == "local"
    for invalid in ({"mode": "automatic-cloud"}, {"max_calls": True}, {"max_usd": "NaN"}, {"request_timeout_s": float("inf")}):
        with pytest.raises(ValueError):
            ExecutionPolicy.from_mapping(invalid)


def test_fractional_ledger_unit_cannot_enlarge_spend_limit(tmp_path):
    budget = RuntimeBudget(allowance(tmp_path, max_usd=".0000000000001"))
    with pytest.raises(ExecutionDeferred):
        budget.reserve(Decimal(".0000000000001"), session="s", operation="o", purpose="p", provider="p", model="m", kind="chat")


@pytest.mark.parametrize("provider_type,module,constructor", [
    (OpenAIProvider, "openai", "OpenAI"), (AnthropicProvider, "anthropic", "Anthropic")])
def test_bounded_sdk_options_reach_actual_client_construction(tmp_path, monkeypatch, provider_type, module, constructor):
    import sys
    from types import SimpleNamespace
    seen = []
    transports = []
    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(Client=lambda **kwargs:
        transports.append(kwargs) or object()))
    def create(**kwargs):
        seen.append(kwargs)
        return SimpleNamespace(max_retries=kwargs["max_retries"])
    monkeypatch.setitem(sys.modules, module, SimpleNamespace(**{constructor: create}))
    router, _ = paid_router(tmp_path, monkeypatch)
    provider = provider_type()
    router._bound_paid_provider(provider, "chat")
    first = provider._ensure_client()
    assert provider._ensure_client() is first
    assert len(seen) == 1
    assert seen[0]["max_retries"] == 0 and seen[0]["timeout"] == 60.0
    assert transports == [{"follow_redirects": False}]
