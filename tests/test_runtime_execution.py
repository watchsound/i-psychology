"""Offline transport/accounting regressions, not model-quality evidence."""
import io
import json
from dataclasses import replace

import pytest

from psychology_base.llm.config import CacheConfig, LLMConfig, ProviderConfig
from psychology_base.llm.router import LLMRouter
from psychology_base.llm.providers.ollama import OllamaProvider
from psychology_base.llm.types import ChatMessage, TaskTier


def local_router(monkeypatch, tmp_path):
    from psychology_base.llm.execution import ExecutionPolicy
    cfg = LLMConfig(active_provider="tongyi", providers={
        "tongyi": ProviderConfig("tongyi", "cloud-small", "cloud-big"),
        "ollama": ProviderConfig("ollama", "small", "large", embedding_model="vectors"),
    }, cache=CacheConfig(enabled=False), execution=ExecutionPolicy(mode="local"))
    local = OllamaProvider(simple_model="small", advanced_model="large", embedding_model="vectors")
    state = {"up": True, "forwarded": False, "requests": [], "models": ("small", "large", "vectors")}

    def transport(self, request, *, timeout):
        state["requests"].append(request.full_url)
        if not state["up"]:
            raise OSError("offline")
        if request.full_url.endswith("/api/tags"):
            body = {"models": [{"name": name + ":latest", "size": 10} for name in state["models"]]}
        elif request.full_url.endswith("/api/show"):
            body = {"capabilities": ["completion", "embedding"], "remote_host": "remote" if state["forwarded"] else ""}
        elif request.full_url.endswith("/api/chat"):
            body = {"message": {"content": "transport control"}}
        else:
            body = {"embedding": [1.0, 2.0]}
        return io.BytesIO(json.dumps(body).encode())

    monkeypatch.setattr(OllamaProvider, "_open", transport)
    # No cloud transport exists in this control: any attempted fallback fails.
    router = LLMRouter(cfg, providers={"ollama": local})
    return router, state


@pytest.mark.parametrize("tier", list(TaskTier))
def test_local_never_selects_cloud_even_when_forced(monkeypatch, tmp_path, tier):
    router, state = local_router(monkeypatch, tmp_path)
    router.config = replace(router.config, local_ollama=replace(router.config.local_ollama,
        force_cloud_chat=True, force_cloud_embeddings=True))
    assert router.chat([ChatMessage("user", "hello")], tier=tier).provider == "ollama"
    assert router.embed("fragment") == [1.0, 2.0]
    assert all(url.startswith("http://127.0.0.1:11434/") for url in state["requests"])


@pytest.mark.parametrize("force_flag", ["force_cloud_chat", "force_cloud_embeddings"])
def test_local_mode_warns_when_cloud_force_flags_set(monkeypatch, tmp_path, caplog, force_flag):
    """Airgap suppression is intentional; the WARN log is the caller's
    heads-up that a mis-set config was silently honored-then-ignored.
    Benchmark scripts that intend cloud routing must also set
    ``execution=ExecutionPolicy(mode="research")``."""
    import logging
    from psychology_base.llm.config import CacheConfig, LLMConfig, LocalOllamaConfig, ProviderConfig
    from psychology_base.llm.execution import ExecutionPolicy
    from psychology_base.llm.providers.ollama import OllamaProvider
    local = OllamaProvider(simple_model="small", advanced_model="large", embedding_model="vectors")
    cfg = LLMConfig(
        active_provider="tongyi",
        providers={
            "tongyi": ProviderConfig("tongyi", "cloud-small", "cloud-big"),
            "ollama": ProviderConfig("ollama", "small", "large", embedding_model="vectors"),
        },
        cache=CacheConfig(enabled=False),
        execution=ExecutionPolicy(mode="local"),
        local_ollama=LocalOllamaConfig(**{force_flag: True}),
    )
    caplog.set_level(logging.WARNING, logger="psychology_base.llm.router")
    LLMRouter(cfg, providers={"ollama": local})
    messages = [rec.message for rec in caplog.records
                if rec.name == "psychology_base.llm.router" and rec.levelno == logging.WARNING]
    assert any(force_flag in msg and "execution.mode=local" in msg for msg in messages)
    assert any("ExecutionPolicy(mode=\"research\")" in msg for msg in messages)


@pytest.mark.parametrize("kind", ["chat", "embed_batch"])
def test_local_outage_recovers_without_cloud_or_new_router(monkeypatch, tmp_path, kind):
    from psychology_base.llm.execution import ExecutionDeferred
    router, state = local_router(monkeypatch, tmp_path)
    state["up"] = False
    def dispatch():
        return (router.chat([ChatMessage("user", "hello")]) if kind == "chat" else
                router.embed_batch(["fragment", "another"]))
    with pytest.raises(ExecutionDeferred):
        dispatch()
    state["up"] = True
    assert dispatch()


def test_embedding_readiness_is_independent_of_chat_model(monkeypatch, tmp_path):
    from psychology_base.llm.execution import ExecutionDeferred
    router, state = local_router(monkeypatch, tmp_path)
    state["models"] = ("vectors",)
    with pytest.raises(ExecutionDeferred):
        router.chat([ChatMessage("user", "hello")])
    assert router.embed("fragment") == [1.0, 2.0]


@pytest.mark.parametrize("endpoint", ["http://example.com:11434", "https://127.0.0.1:11434",
    "http://user:pass@127.0.0.1:11434", "http://127.0.0.1:11434/proxy", "http://127.0.0.1:11434?url=remote"])
def test_runtime_local_endpoint_guard_rejects_nonlocal_transports(endpoint):
    with pytest.raises(ValueError):
        OllamaProvider(base_url=endpoint).enforce_loopback()


def test_local_guard_disables_proxies_and_redirects(monkeypatch):
    import urllib.request
    from psychology_base.llm.providers.ollama import _RejectRedirects
    provider = OllamaProvider()
    provider.enforce_loopback()
    seen = []
    class Opener:
        def open(self, request, **kwargs):
            seen.append(request.full_url)
            return "response"
    def build(*handlers):
        assert any(isinstance(handler, urllib.request.ProxyHandler) and handler.proxies == {} for handler in handlers)
        redirect = next(handler for handler in handlers if isinstance(handler, _RejectRedirects))
        with pytest.raises(ValueError):
            redirect.redirect_request(None, None, 302, "", {}, "https://remote")
        return Opener()
    monkeypatch.setattr(urllib.request, "build_opener", build)
    assert provider._open(urllib.request.Request("http://127.0.0.1:11434/api/tags"), timeout=1) == "response"
    assert seen == ["http://127.0.0.1:11434/api/tags"]


def test_local_rejects_forwarding_and_remote_features(monkeypatch, tmp_path):
    from psychology_base.llm.execution import ExecutionDeferred
    router, state = local_router(monkeypatch, tmp_path)
    state["forwarded"] = True
    with pytest.raises(ExecutionDeferred):
        router.embed_batch(["a", "b"])
    with pytest.raises(ExecutionDeferred):
        router.embed("a", backend="cloud")
    with pytest.raises(ExecutionDeferred):
        router.rerank("q", ["a"])


def test_both_cli_factories_enforce_local_policy(monkeypatch, tmp_path):
    from psychology_base.applications.learning import tutor_cli
    from psychology_base.applications.emotional import companion_cli
    from psychology_base.llm import config, router as router_module
    original, _ = local_router(monkeypatch, tmp_path)
    monkeypatch.setattr(config, "load_config", lambda: original.config)
    instances = []

    def build(cfg, **kwargs):
        result = LLMRouter(cfg, providers=original.providers, **kwargs)
        instances.append(result)
        return result

    monkeypatch.setattr(router_module, "LLMRouter", build)
    for module in (tutor_cli, companion_cli):
        assert module._build_llm_responder("local")("system", "", "hello") == "transport control"
    assert all(instance.execution.mode == "local" for instance in instances)


def test_local_construction_never_resolves_cloud_credentials(monkeypatch, tmp_path):
    original, _ = local_router(monkeypatch, tmp_path)
    def forbidden(self):
        raise AssertionError("Local execution must not build a cloud provider")
    monkeypatch.setattr(ProviderConfig, "api_key", forbidden)
    router = LLMRouter(original.config)
    assert set(router.providers) == {"ollama"}
    assert router.providers["ollama"].loopback_only
    assert router.providers["ollama"].request_timeout_s <= original.config.execution.request_timeout_s


def test_failed_consolidation_preserves_window_and_no_partial_facts(tmp_path):
    from psychology_base.companion.consolidation import Consolidator
    from psychology_base.companion.facts import FactStore
    from tests.test_consolidation import _frag, _router
    router, _ = _router(tmp_path, "ADD | A | likes | tea | 0.9\nADD | B | likes | coffee | 0.9\n")
    consolidator = Consolidator(router, FactStore())
    fragment = _frag("f1", "A likes tea. B likes coffee.")
    consolidator.observe(fragment)
    consolidator.mark_session_boundary()
    original = router.embed
    calls = []

    def fail_second(text, **kwargs):
        calls.append(text)
        if len(calls) == 2:
            raise OSError("second embedding failed")
        return original(text, **kwargs)

    router.embed = fail_second
    with pytest.raises(OSError):
        consolidator.maybe_consolidate(1)
    assert consolidator._pending == [fragment]
    assert consolidator._boundary_pending
    assert len(consolidator.store) == 0
    assert consolidator.stats.total() == 0
    router.embed = original
    assert consolidator.maybe_consolidate(2).adds == 2
    assert len(consolidator.store) == 2
    assert consolidator.maybe_consolidate(3).total() == 0


def test_consolidation_keeps_new_observations_and_boundaries(monkeypatch):
    from psychology_base.companion import consolidation
    from psychology_base.companion.facts import FactStore
    from tests.test_consolidation import _frag
    consolidator = consolidation.Consolidator(object(), FactStore())
    first, second = _frag("f1", "first"), _frag("f2", "second")
    consolidator.observe(first)
    consolidator.mark_session_boundary()
    def extract(window, **kwargs):
        assert window == [first]
        # A reentrant tick cannot apply the same pending window twice.
        assert consolidator.maybe_consolidate(1).total() == 0
        consolidator.observe(second)
        consolidator.mark_session_boundary()
        return []
    monkeypatch.setattr(consolidation, "extract_facts", extract)
    consolidator.maybe_consolidate(1)
    assert consolidator._pending == [second]
    assert consolidator._boundary_pending


def test_consolidation_refusal_preserves_operation_allowance(monkeypatch, tmp_path):
    from psychology_base.companion import consolidation
    from psychology_base.companion.facts import FactStore
    from psychology_base.llm.execution import ExecutionDeferred
    from tests.test_consolidation import _frag, _router
    router, _ = _router(tmp_path, "")
    pending = consolidation.Consolidator(router, FactStore())
    pending.observe(_frag("f1", "one"))
    pending.mark_session_boundary()
    def refused(*args, **kwargs):
        raise ExecutionDeferred("no allowance")
    monkeypatch.setattr(consolidation, "extract_facts", refused)
    with pytest.raises(ExecutionDeferred):
        pending.maybe_consolidate(1)
    operation = pending._operation_id
    with pytest.raises(ExecutionDeferred):
        pending.maybe_consolidate(2)
    assert pending._operation_id == operation
    assert len(pending._pending) == 1
    assert pending._boundary_pending


@pytest.mark.parametrize("application", ["tutor", "companion"])
def test_real_cli_defer_preserves_user_memory_without_assistant_reply(monkeypatch, tmp_path, application):
    from psychology_base.llm import config
    from psychology_base.applications.pilot.layout import pam_latest_path, session_log_path
    if application == "tutor":
        from tests.test_tutor_cli import _run_cli, _enrolled_layout
        user = "u01"
    else:
        from tests.test_companion_cli import _run_cli, _enrolled_layout
        user = "c01"
    router, state = local_router(monkeypatch, tmp_path)
    monkeypatch.setattr(config, "load_config", lambda: router.config)
    layout = _enrolled_layout(tmp_path)
    state["up"] = False
    monkeypatch.setattr("sys.stdin", io.StringIO("first ordinary question\nsecond ordinary question\n/quit\n"))
    # Omit --mode: both applications must now default to enforced local execution.
    rc, out, err = _run_cli(["--pilot-root", str(tmp_path), "session", "--user", user])
    assert rc == 0, err
    assert out.count("Reply deferred") == 2
    log = [json.loads(line) for line in session_log_path(layout, user, 1).read_text(encoding="utf-8").splitlines()]
    assert len([event for event in log if event["event"] == "turn_deferred"]) == 2
    assert [event["turn_index"] for event in log if event["event"] == "user_turn"] == [0, 1]
    assert [event["turn_index"] for event in log if event["event"] == "turn_deferred"] == [0, 1]
    assert not [event for event in log if event["event"] == "agent_turn"]
    fragments = json.loads(pam_latest_path(layout, user).read_text(encoding="utf-8"))["fragments"]
    assert [fragment["content"] for fragment in fragments] == ["first ordinary question", "second ordinary question"]


def test_companion_crisis_handoff_still_works_during_local_outage(monkeypatch, tmp_path):
    from tests.test_companion_cli import _run_cli, _enrolled_layout
    from psychology_base.llm import config
    from psychology_base.applications.emotional.companion_agent import HANDOFF_SCRIPT
    router, state = local_router(monkeypatch, tmp_path)
    monkeypatch.setattr(config, "load_config", lambda: router.config)
    _enrolled_layout(tmp_path)
    state["up"] = False
    monkeypatch.setattr("sys.stdin", io.StringIO("i want to kill myself\n/quit\n"))
    before = len(state["requests"])
    rc, out, err = _run_cli(["--pilot-root", str(tmp_path), "session", "--user", "c01"])
    assert rc == 0, err
    assert HANDOFF_SCRIPT in out
    assert len(state["requests"]) == before
