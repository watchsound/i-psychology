"""Offline admission controls; prescribed transports do not measure model quality."""
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import asdict, replace
import sqlite3

import pytest

from psychology_base.applications.llm_responders import make_llm_responder
from psychology_base.llm.accounting import request_input_bound
from psychology_base.llm.cache import LLMCache
from psychology_base.llm.config import _parse_config
from psychology_base.llm.execution import (
    ConversationLimits, ExecutionDeferred, ExecutionPolicy, InferenceLimit, current_operation,
)
from psychology_base.llm.router import LLMRouter
from psychology_base.llm.types import ChatMessage, Completion, TaskTier
from tests.test_runtime_execution import local_router
from tests.test_runtime_budget import paid_router
from tests.test_local_embedding_identity import transport


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import socket
    def reject(*args, **kwargs):
        raise AssertionError("Conversation budget tests must not contact a model")
    monkeypatch.setattr(socket.socket, "connect", reject)
    monkeypatch.setattr(socket, "create_connection", reject)


MESSAGES = [ChatMessage("user", "hello")]


def aux(router, text="hello", max_tokens=50):
    return router.chat([ChatMessage("user", text)], max_tokens=max_tokens, use_cache=False)


def test_default_denies_aux_before_metadata_and_preserves_reply(monkeypatch, tmp_path):
    router, state = local_router(monkeypatch, tmp_path)
    with router.conversation_turn() as budget:
        with pytest.raises(ExecutionDeferred, match="auxiliary"):
            aux(router)
        assert state["requests"] == []
        assert make_llm_responder(router)("system", "", "hello") == "transport control"
        with pytest.raises(ExecutionDeferred, match="response"):
            make_llm_responder(router)("system", "", "different")
    usage = budget.snapshot()["usage"]
    assert usage["response"]["attempts"] == 1
    assert usage["auxiliary"]["attempts"] == 0
    assert usage["auxiliary"]["denied"] == 1
    assert sum(url.endswith("/api/chat") for url in state["requests"]) == 1
    assert current_operation.get() is None


def test_nested_routers_and_threads_share_one_atomic_allowance(monkeypatch, tmp_path):
    router, state = local_router(monkeypatch, tmp_path)
    other = LLMRouter(router.config, providers=router.providers)
    limits = ConversationLimits(auxiliary=InferenceLimit(3, 10000))
    with router.conversation_turn(limits=limits) as budget:
        original = current_operation.get()
        with other.operation("cannot_reset", operation_id="other", timeout_s=300):
            assert current_operation.get().id == original.id
            assert current_operation.get().deadline <= original.deadline
            with other.conversation_turn() as nested:
                assert nested is budget
            with pytest.raises(ExecutionDeferred, match="cannot change"):
                with other.conversation_turn(limits=ConversationLimits(auxiliary=InferenceLimit(10, 50000))):
                    pytest.fail("Nested scope enlarged the allowance")
            context = copy_context()
            def attempt(index):
                try:
                    context.copy().run(aux, other, str(index))
                    return True
                except ExecutionDeferred:
                    return False
            with ThreadPoolExecutor(max_workers=8) as pool:
                assert sum(pool.map(attempt, range(20))) == 3
        assert budget.snapshot()["usage"]["auxiliary"]["attempts"] == 3
        assert make_llm_responder(router)("system", "", "reply")
    assert sum(url.endswith("/api/chat") for url in state["requests"]) == 4


@pytest.mark.parametrize("value", [None, True, 0, -1, 1.5])
def test_serving_rejects_unbounded_output_before_transport(monkeypatch, tmp_path, value):
    router, state = local_router(monkeypatch, tmp_path)
    limits = ConversationLimits(auxiliary=InferenceLimit(1, 10000))
    with router.conversation_turn(limits=limits):
        with pytest.raises(ExecutionDeferred, match="finite"):
            aux(router, max_tokens=value)
    assert state["requests"] == []


def test_schema_and_utf8_input_share_token_bound(monkeypatch, tmp_path):
    router, state = local_router(monkeypatch, tmp_path)
    provider = router.providers["ollama"]
    provider.response_schema = {"type": "object", "description": "结构" * 100}
    messages = [ChatMessage("user", "你好")]
    bound = request_input_bound(messages, response_schema=provider.response_schema) + 50
    limits = ConversationLimits(auxiliary=InferenceLimit(2, bound))
    with router.conversation_turn(limits=limits) as budget:
        router.chat(messages, max_tokens=50)
        with pytest.raises(ExecutionDeferred):
            router.chat(messages, max_tokens=50, use_cache=False)
    assert budget.snapshot()["usage"]["auxiliary"]["token_bound"] == bound
    assert sum(url.endswith("/api/chat") for url in state["requests"]) == 1


def test_cache_hits_do_not_consume_attempt_or_bound(monkeypatch, tmp_path):
    router, state = local_router(monkeypatch, tmp_path)
    router.cache = LLMCache(tmp_path / "cache")
    responder = make_llm_responder(router)
    with router.conversation_turn() as budget:
        assert responder("system", "", "hello") == responder("system", "", "hello")
    usage = budget.snapshot()["usage"]["response"]
    assert usage["attempts"] == 1 and usage["cache_hits"] == 1
    assert sum(url.endswith("/api/chat") for url in state["requests"]) == 1


def test_failure_retains_bound_and_does_not_retry(monkeypatch, tmp_path):
    router, state = local_router(monkeypatch, tmp_path)
    state["up"] = False
    with router.conversation_turn() as budget:
        responder = make_llm_responder(router)
        with pytest.raises(ExecutionDeferred):
            responder("system", "", "hello")
        before = len(state["requests"])
        state["up"] = True
        with pytest.raises(ExecutionDeferred, match="response"):
            responder("system", "", "hello")
        assert len(state["requests"]) == before
    usage = budget.snapshot()["usage"]["response"]
    assert usage["attempts"] == usage["failed"] == 1
    assert usage["token_bound"] > 400


@pytest.mark.parametrize("usage", [(0, 0), (None, None), (-1, 1), (True, 1), (10, 5)])
def test_usage_never_refunds_the_conservative_turn_bound(tmp_path, monkeypatch, usage):
    router, calls = paid_router(tmp_path, monkeypatch, usage=usage)
    with router.conversation_turn() as budget:
        assert make_llm_responder(router, max_tokens=50)("system", "", "hello")
    expected = request_input_bound([ChatMessage("system", "system"), *MESSAGES]) + 50
    assert budget.snapshot()["usage"]["response"]["token_bound"] == expected
    assert calls == ["chat"]


def test_overrun_blocks_all_later_inference(tmp_path, monkeypatch):
    router, calls = paid_router(tmp_path, monkeypatch, usage=(10, 51))
    with router.conversation_turn() as budget:
        with pytest.raises(ExecutionDeferred, match="exceeded"):
            make_llm_responder(router, max_tokens=50)("system", "", "hello")
        with pytest.raises(ExecutionDeferred):
            router.embed("fragment", backend="cloud")
    assert calls == ["chat"]
    assert budget.snapshot()["blocked"]
    assert budget.snapshot()["usage"]["response"]["failed"] == 1


def test_turn_denial_does_not_reserve_money_or_change_paid_contract(tmp_path, monkeypatch):
    router, calls = paid_router(tmp_path, monkeypatch)
    contract = asdict(router.execution)
    with sqlite3.connect(router.execution.ledger) as db:
        before = db.execute("SELECT * FROM policies").fetchall()
    with router.conversation_turn():
        with pytest.raises(ExecutionDeferred, match="auxiliary"):
            aux(router)
        assert router.budget.summary()["provider_attempts"] == 0
        assert make_llm_responder(router, max_tokens=50)("system", "", "hello")
    with sqlite3.connect(router.execution.ledger) as db:
        assert db.execute("SELECT * FROM policies").fetchall() == before
    assert asdict(router.execution) == contract
    assert calls == ["chat"]


def test_research_profile_unchanged_but_cannot_bypass_active_turn(monkeypatch, tmp_path):
    router, state = local_router(monkeypatch, tmp_path)
    research = LLMRouter(replace(router.config, active_provider="ollama", execution=ExecutionPolicy(mode="research")),
                         providers=router.providers)
    state["requests"].clear()
    with research.conversation_turn() as budget:
        assert budget is None
        assert aux(research, max_tokens=None)
    state["requests"].clear()
    with router.conversation_turn():
        with pytest.raises(ExecutionDeferred, match="Research"):
            aux(research)
    assert not state["requests"]


@pytest.mark.parametrize("use_cache", [False, True])
def test_pinned_embedding_chunks_use_same_admission_without_extra_verification(transport, use_cache):
    provider, old, state = transport
    router = LLMRouter(replace(old.config, execution=ExecutionPolicy(mode="local")),
                       providers={"ollama": provider}, cache=old.cache)
    state["calls"].clear()
    texts = [f"fragment-{i}" for i in range(35)]
    limits = ConversationLimits(non_generation=InferenceLimit(2, 65536))
    with router.conversation_turn(limits=limits) as budget:
        assert len(router.embed_batch(texts, backend="local", use_cache=use_cache)) == 35
        if use_cache:
            assert len(router.embed_batch(texts, backend="local")) == 35
            assert budget.snapshot()["usage"]["non_generation"]["cache_hits"] == 35
        with pytest.raises(ExecutionDeferred):
            router.embed("another", backend="local")
    assert sum(endpoint == "embed" for endpoint, _ in state["calls"]) == 2
    assert budget.snapshot()["usage"]["non_generation"]["attempts"] == 2
    # One pre/post verification per batch; admission adds no metadata or slots.
    assert sum(endpoint in ("tags", "show") for endpoint, _ in state["calls"]) == (10 if use_cache else 6)


def test_embedding_later_chunk_denial_does_not_publish_partial_cache(transport):
    provider, old, state = transport
    router = LLMRouter(replace(old.config, execution=ExecutionPolicy(mode="local")),
                       providers={"ollama": provider}, cache=old.cache)
    limits = ConversationLimits(non_generation=InferenceLimit(1, 65536))
    with router.conversation_turn(limits=limits):
        with pytest.raises(ExecutionDeferred):
            router.embed_batch([f"fragment-{i}" for i in range(35)], backend="local")
    assert sum(endpoint == "embed" for endpoint, _ in state["calls"]) == 1
    assert not list(router.cache.cache_dir.rglob("*.json"))


@pytest.mark.parametrize("raw", [None, [], {"unknown": {}}, {"response": {}},
    {"response": {"max_calls": 1, "max_tokens": True}},
    {"auxiliary": {"max_calls": 1, "max_tokens": 0}},
    {"auxiliary": {"max_calls": -1, "max_tokens": 1}}])
def test_invalid_config_fails_before_router(raw):
    with pytest.raises(ValueError):
        _parse_config({"active_provider": "ollama", "providers": {"ollama": {
            "simple_model": "small", "advanced_model": "large"}}, "conversation": raw})


def test_config_uses_one_owner_for_defaults_and_explicit_aux_allowance():
    cfg = _parse_config({"active_provider": "ollama", "providers": {"ollama": {
        "simple_model": "small", "advanced_model": "large"}}, "conversation": {
        "auxiliary": {"max_calls": 2, "max_tokens": 4096}}})
    assert cfg.conversation == ConversationLimits(auxiliary=InferenceLimit(2, 4096))
    assert cfg.execution.mode == "local" and cfg.execution.max_calls == 0


@pytest.mark.parametrize("module_name", ["learning", "emotional"])
def test_actual_agents_scope_recall_and_one_generation_per_turn(monkeypatch, tmp_path, module_name, caplog):
    from tests.test_learning_tutor_agent import _build as tutor
    from tests.test_emotional_companion_agent import _build as companion
    router, state = local_router(monkeypatch, tmp_path)
    agent = (tutor if module_name == "learning" else companion)(responder=make_llm_responder(router))
    original = agent.companion_system.recall
    counters = []
    def recall(*args, **kwargs):
        operation = current_operation.get()
        assert operation.inference_role == "auxiliary"
        counters.append(operation.conversation)
        with pytest.raises(ExecutionDeferred, match="auxiliary"):
            aux(router, "helper")
        return original(*args, **kwargs)
    monkeypatch.setattr(agent.companion_system, "recall", recall)
    caplog.set_level("INFO", logger="psychology_base.llm.router")
    for index in range(20):
        assert agent.respond(f"daily ordinary turn {index}").response_text == "transport control"
    assert len({id(value) for value in counters}) == 20
    assert all(value.snapshot()["closed"] for value in counters)
    assert sum(url.endswith("/api/chat") for url in state["requests"]) == 20
    assert all(value.snapshot()["usage"]["auxiliary"]["attempts"] == 0 for value in counters)
    records = [record for record in caplog.records if hasattr(record, "conversation")]
    assert len(records) == 20
    assert "daily ordinary" not in repr([record.conversation for record in records])


def test_crisis_handoff_needs_no_generation(monkeypatch, tmp_path):
    from tests.test_emotional_companion_agent import _build
    from psychology_base.applications.safety import CrisisEscalation
    router, state = local_router(monkeypatch, tmp_path)
    agent = _build(responder=make_llm_responder(router), crisis=CrisisEscalation(handoff_sink=lambda incident: None))
    response = agent.respond("tonight i will end everything")
    assert response.handoff_triggered
    assert state["requests"] == []


def test_chat_batch_copied_contexts_cannot_multiply_allowance(monkeypatch, tmp_path):
    router, state = local_router(monkeypatch, tmp_path)
    with router.conversation_turn(limits=ConversationLimits(auxiliary=InferenceLimit(2, 10000))) as budget:
        with pytest.raises(ExecutionDeferred):
            router.chat_batch([[ChatMessage("user", str(i))] for i in range(8)],
                              max_tokens=50, use_cache=False)
    assert sum(url.endswith("/api/chat") for url in state["requests"]) == 2
    assert budget.snapshot()["usage"]["auxiliary"]["attempts"] == 2


def test_local_identity_mismatch_is_not_cached_or_refunded(monkeypatch, tmp_path):
    router, _ = local_router(monkeypatch, tmp_path)
    router.cache = LLMCache(tmp_path / "cache")
    provider = router.providers["ollama"]
    monkeypatch.setattr(provider, "chat", lambda *a, **kw:
        Completion("wrong identity", "other", "small", TaskTier.SIMPLE))
    with router.conversation_turn() as budget:
        with pytest.raises(ExecutionDeferred, match="identity"):
            make_llm_responder(router)("system", "", "hello")
    assert budget.snapshot()["usage"]["response"]["failed"] == 1
    assert not list(router.cache.cache_dir.rglob("*.json"))


def test_closed_scope_cannot_dispatch_from_a_copied_context(monkeypatch, tmp_path):
    router, state = local_router(monkeypatch, tmp_path)
    with router.conversation_turn(limits=ConversationLimits(auxiliary=InferenceLimit(1, 10000))) as budget:
        context = copy_context()
    with pytest.raises(ExecutionDeferred):
        context.run(aux, router)
    assert state["requests"] == []
    assert budget.snapshot()["closed"]


def test_shared_deadline_bounds_recall_and_reply(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from psychology_base.llm import execution
    router, state = local_router(monkeypatch, tmp_path)
    clock = [None]
    monotonic = execution.time.monotonic
    monkeypatch.setattr(execution, "time", SimpleNamespace(
        monotonic=lambda: monotonic() if clock[0] is None else clock[0]))
    with router.conversation_turn() as budget:
        # Deterministic expiry, without relying on CPU scheduling of this host.
        clock[0] = current_operation.get().deadline + 1
        with pytest.raises(ExecutionDeferred, match="deadline"):
            make_llm_responder(router)("system", "", "hello")
    assert state["requests"] == []
    assert budget.snapshot()["usage"]["response"]["attempts"] == 0


@pytest.mark.parametrize("kind", ["rerank", "classify_pairs"])
def test_local_pair_adapters_account_for_all_input_without_generation(kind):
    from tests.test_local_pair import inert_scorer, router as pair_router
    from tests.test_local_pair_classifier import inert_classifier
    scorer, calls = inert_scorer() if kind == "rerank" else inert_classifier([[1, 0, -1]])
    router = pair_router(scorer)
    query, document = "query", "document"
    bound = request_input_bound([ChatMessage("user", query + "\n" + document)])
    limits = ConversationLimits(non_generation=InferenceLimit(2, bound))
    with router.conversation_turn(limits=limits) as budget:
        if kind == "rerank":
            assert router.rerank(query, [document])
            assert router.rerank(query, []) == []
        else:
            assert router.classify_pairs([(query, document)])
            assert router.classify_pairs([]) == []
        with pytest.raises(ExecutionDeferred):
            if kind == "rerank":
                router.rerank(query, [document])
            else:
                router.classify_pairs([(query, document)])
    assert len(calls) == 1
    assert budget.snapshot()["usage"]["non_generation"]["token_bound"] == bound
    assert budget.snapshot()["usage"]["auxiliary"]["attempts"] == 0


@pytest.mark.parametrize("application", ["learning", "emotional"])
def test_recall_refusal_propagates_without_fabricated_reply_or_memory(monkeypatch, tmp_path, application):
    from tests.test_learning_tutor_agent import _build as tutor
    from tests.test_emotional_companion_agent import _build as companion
    router, state = local_router(monkeypatch, tmp_path)
    agent = (tutor if application == "learning" else companion)(responder=make_llm_responder(router))
    def refused_recall(*args, **kwargs):
        aux(router, "unqualified helper")
        pytest.fail("Refusal must not become a successful judgment")
    monkeypatch.setattr(agent.companion_system, "recall", refused_recall)
    with pytest.raises(ExecutionDeferred, match="auxiliary"):
        agent.respond("ordinary input")
    assert state["requests"] == []
    assert list(agent.companion_system.companion.store.all()) == []
