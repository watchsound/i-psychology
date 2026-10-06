"""Tests for the LLM router + config + providers + cache.

No vendor SDKs required. OllamaProvider.probe is monkeypatched via
``urllib.request.urlopen`` so no local Ollama server is needed either.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from psychology_base.llm import (
    ChatMessage,
    LLMCache,
    LLMConfig,
    LLMRouter,
    LocalOllamaConfig,
    ProviderConfig,
    TaskTier,
    default_config,
    load_config,
)
from psychology_base.llm.cache import chat_cache_key, embedding_cache_key
from psychology_base.llm.providers import (
    AlibabaTongyiProvider,
    MockProvider,
    OllamaProvider,
)


# ---------- config ----------


def _write_yaml(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "llm.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


def test_load_config_round_trips(tmp_path):
    data = {
        "active_provider": "anthropic",
        "providers": {
            "anthropic": {
                "simple_model": "claude-haiku-4-5-20251001",
                "advanced_model": "claude-opus-4-7",
                "api_key_env": "ANTHROPIC_API_KEY",
            },
            "ollama": {
                "base_url": "http://localhost:11434",
                "simple_model": "qwen2.5:7b",
                "advanced_model": "qwen2.5:32b",
            },
        },
        "local_ollama": {"enabled": True, "probe_timeout_s": 0.5},
        "cache": {"enabled": True},
    }
    cfg = load_config(_write_yaml(tmp_path, data))
    assert cfg.active_provider == "anthropic"
    assert set(cfg.providers) == {"anthropic", "ollama"}
    assert cfg.providers["anthropic"].simple_model == "claude-haiku-4-5-20251001"
    assert cfg.local_ollama.enabled is True
    assert cfg.local_ollama.probe_timeout_s == 0.5
    assert cfg.cache.enabled is True


def test_load_config_rejects_unknown_active_provider(tmp_path):
    data = {
        "active_provider": "made-up",
        "providers": {
            "anthropic": {
                "simple_model": "a",
                "advanced_model": "b",
            },
        },
    }
    with pytest.raises(ValueError, match="active_provider"):
        load_config(_write_yaml(tmp_path, data))


def test_load_config_rejects_provider_without_tiers(tmp_path):
    data = {
        "active_provider": "anthropic",
        "providers": {
            "anthropic": {"simple_model": "only-simple"},  # advanced_model missing
        },
    }
    with pytest.raises(ValueError, match="advanced_model"):
        load_config(_write_yaml(tmp_path, data))


def test_load_config_rejects_empty_providers(tmp_path):
    data = {"active_provider": "x", "providers": {}}
    with pytest.raises(ValueError, match="non-empty"):
        load_config(_write_yaml(tmp_path, data))


def test_provider_config_api_key_resolution(monkeypatch):
    monkeypatch.setenv("MY_KEY_ENV", "sk-test-123")
    cfg = ProviderConfig(
        name="openai",
        simple_model="s",
        advanced_model="a",
        api_key_env="MY_KEY_ENV",
    )
    assert cfg.api_key() == "sk-test-123"

    monkeypatch.delenv("MY_KEY_ENV")
    assert cfg.api_key() is None


def test_cache_config_resolved_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("PSYCHOLOGY_BASE_LLM_CACHE_DIR", str(tmp_path))
    cfg = default_config()
    assert cfg.cache.resolved_dir() == tmp_path


def test_default_config_is_ollama_only():
    cfg = default_config()
    assert cfg.active_provider == "ollama"
    assert set(cfg.providers) == {"ollama"}


# ---------- cache ----------


def test_cache_key_sensitive_to_every_field():
    base = dict(
        provider="p",
        model_id="m",
        tier=TaskTier.SIMPLE,
        messages=[ChatMessage(role="user", content="hi")],
        temperature=0.0,
        max_tokens=100,
    )
    k0 = chat_cache_key(**base)
    for override in [
        {"provider": "other"},
        {"model_id": "other"},
        {"tier": TaskTier.ADVANCED},
        {"messages": [ChatMessage(role="user", content="hello")]},
        {"temperature": 0.1},
        {"max_tokens": 101},
    ]:
        new = {**base, **override}
        assert chat_cache_key(**new) != k0, f"collision when {override} changed"


def test_llm_cache_round_trip_chat(tmp_path):
    cache = LLMCache(cache_dir=tmp_path)
    key = chat_cache_key(
        provider="mock",
        model_id="mock-simple",
        tier=TaskTier.SIMPLE,
        messages=[ChatMessage(role="user", content="hi")],
        temperature=None,
        max_tokens=None,
    )
    assert cache.get_chat(key) is None
    assert cache.misses == 1

    from psychology_base.llm.types import Completion
    completion = Completion(
        content="hello back",
        provider="mock",
        model_id="mock-simple",
        tier=TaskTier.SIMPLE,
    )
    cache.put_chat(key, completion)
    recalled = cache.get_chat(key)
    assert recalled is not None
    assert recalled.content == "hello back"
    assert recalled.tier == TaskTier.SIMPLE
    assert cache.hits == 1


def test_llm_cache_round_trip_embedding(tmp_path):
    cache = LLMCache(cache_dir=tmp_path)
    key = embedding_cache_key(provider="mock", model_id="mock-emb", text="hello")
    assert cache.get_embedding(key) is None

    cache.put_embedding(key, [0.1, 0.2, 0.3], model_id="mock-emb")
    recalled = cache.get_embedding(key)
    assert recalled == [0.1, 0.2, 0.3]


def test_llm_cache_corrupt_entry_self_heals(tmp_path):
    """A corrupt JSON file in the cache dir must miss-and-delete, not crash."""
    cache = LLMCache(cache_dir=tmp_path)
    key = "0" * 64
    path = cache._path_for(key, kind="chat")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not valid json {", encoding="utf-8")
    assert cache.get_chat(key) is None
    assert not path.exists(), "corrupt entry should be deleted"


# ---------- router routing + mock provider ----------


def _config_with_mock(tmp_path, *, ollama_enabled: bool) -> LLMConfig:
    """Build a minimal config pointing at 'mock' as active, with Ollama
    optionally present and toggled."""
    return LLMConfig(
        active_provider="mock",
        providers={
            "mock": ProviderConfig(
                name="mock",
                simple_model="mock-simple",
                advanced_model="mock-advanced",
            ),
            "ollama": ProviderConfig(
                name="ollama",
                simple_model="qwen2.5:7b",
                advanced_model="qwen2.5:32b",
                base_url="http://localhost:11434",
            ),
        },
        local_ollama=LocalOllamaConfig(enabled=ollama_enabled, probe_timeout_s=0.1),
    )


def test_router_routes_simple_to_cloud_when_ollama_disabled(tmp_path):
    cfg = _config_with_mock(tmp_path, ollama_enabled=False)
    mock = MockProvider(name="mock")
    router = LLMRouter(
        cfg,
        providers={"mock": mock},
        cache=LLMCache(cache_dir=tmp_path / "cache"),
    )

    _, decision = router.route(TaskTier.SIMPLE)
    assert decision.provider == "mock"
    assert decision.model_id == "mock-simple"
    assert "advanced_tier" not in decision.reason

    _, decision_adv = router.route(TaskTier.ADVANCED)
    assert decision_adv.provider == "mock"
    assert decision_adv.model_id == "mock-advanced"


def test_router_routes_simple_to_ollama_when_probe_passes(tmp_path):
    cfg = _config_with_mock(tmp_path, ollama_enabled=True)
    mock_cloud = MockProvider(name="mock")
    mock_ollama = MockProvider(name="ollama", simple_model="qwen2.5:7b")
    # Force the probe success manually — the fake ollama provider has no
    # probe method, so inject readiness after construction.
    router = LLMRouter(
        cfg,
        providers={"mock": mock_cloud, "ollama": mock_ollama},
        cache=LLMCache(cache_dir=tmp_path / "cache"),
    )
    router._ollama_ready = True

    provider, decision = router.route(TaskTier.SIMPLE)
    assert decision.provider == "ollama"
    assert decision.model_id == "qwen2.5:7b"
    assert provider is mock_ollama


def test_router_advanced_tier_never_routes_to_ollama(tmp_path):
    cfg = _config_with_mock(tmp_path, ollama_enabled=True)
    mock_cloud = MockProvider(name="mock")
    mock_ollama = MockProvider(name="ollama")
    router = LLMRouter(
        cfg,
        providers={"mock": mock_cloud, "ollama": mock_ollama},
        cache=LLMCache(cache_dir=tmp_path / "cache"),
    )
    router._ollama_ready = True  # even when Ollama is up:

    _, decision = router.route(TaskTier.ADVANCED)
    assert decision.provider == "mock", "ADVANCED must stay on cloud"


# ---------- router chat + cache ----------


def test_router_chat_round_trips_through_cache(tmp_path):
    cfg = _config_with_mock(tmp_path, ollama_enabled=False)
    mock = MockProvider(name="mock")
    router = LLMRouter(
        cfg,
        providers={"mock": mock},
        cache=LLMCache(cache_dir=tmp_path / "cache"),
    )
    msgs = [ChatMessage(role="user", content="what is 2+2?")]

    first = router.chat(msgs, tier=TaskTier.SIMPLE)
    assert first.content == "echo: what is 2+2?"
    assert len(mock.calls) == 1

    # Second call with same messages must hit cache, not re-invoke provider.
    second = router.chat(msgs, tier=TaskTier.SIMPLE)
    assert second.content == first.content
    assert len(mock.calls) == 1, "cache hit should not re-invoke provider"


def test_router_chat_use_cache_false_bypasses_cache(tmp_path):
    cfg = _config_with_mock(tmp_path, ollama_enabled=False)
    mock = MockProvider(name="mock")
    router = LLMRouter(
        cfg,
        providers={"mock": mock},
        cache=LLMCache(cache_dir=tmp_path / "cache"),
    )
    msgs = [ChatMessage(role="user", content="hi")]
    router.chat(msgs, use_cache=False)
    router.chat(msgs, use_cache=False)
    assert len(mock.calls) == 2


def test_router_chat_batch_runs_all_prompts(tmp_path):
    cfg = _config_with_mock(tmp_path, ollama_enabled=False)
    mock = MockProvider(name="mock")
    router = LLMRouter(
        cfg,
        providers={"mock": mock},
        cache=None,
    )
    prompts = [
        [ChatMessage(role="user", content=f"q{i}")] for i in range(5)
    ]
    results = router.chat_batch(prompts, max_workers=3)
    assert len(results) == 5
    assert {r.content for r in results} == {f"echo: q{i}" for i in range(5)}


def test_router_chat_batch_respects_cache(tmp_path):
    cfg = _config_with_mock(tmp_path, ollama_enabled=False)
    mock = MockProvider(name="mock")
    router = LLMRouter(
        cfg,
        providers={"mock": mock},
        cache=LLMCache(cache_dir=tmp_path / "cache"),
    )
    prompts = [
        [ChatMessage(role="user", content="dup")],
        [ChatMessage(role="user", content="dup")],
        [ChatMessage(role="user", content="unique")],
    ]
    router.chat_batch(prompts)
    # First invocation: 'dup' called once (second is cache hit from first put),
    # 'unique' called once. But concurrent execution means both 'dup' prompts
    # may race before the first put completes, so we assert the UPPER bound
    # (3 calls — no cache) is not exceeded and the lower bound (2 unique
    # prompts) is reached. The hard contract: the SECOND round sees no new
    # provider calls at all.
    first_round_calls = len(mock.calls)
    assert 2 <= first_round_calls <= 3

    router.chat_batch(prompts)
    assert len(mock.calls) == first_round_calls, (
        "second batch over identical prompts must be 100% cache hits"
    )


# ---------- ollama provider probe (mocked urllib) ----------


class _FakeResp:
    def __init__(self, body: dict):
        self._body = json.dumps(body).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_ollama_probe_returns_true_when_simple_model_pulled():
    provider = OllamaProvider(simple_model="qwen2.5:7b")
    body = {"models": [{"name": "qwen2.5:7b"}, {"name": "nomic-embed-text"}]}
    with patch("urllib.request.urlopen", return_value=_FakeResp(body)):
        assert provider.probe(timeout_s=0.1) is True
    assert provider._server_up is True
    assert "qwen2.5:7b" in provider._available_models


def test_ollama_probe_returns_false_when_simple_model_missing():
    """Server up, but configured model isn't pulled — this is the exact
    silent-failure mode the reference project learned to guard against."""
    provider = OllamaProvider(simple_model="qwen2.5:7b")
    body = {"models": [{"name": "llama3.2:3b"}]}  # different model
    with patch("urllib.request.urlopen", return_value=_FakeResp(body)):
        assert provider.probe(timeout_s=0.1) is False
    assert provider._server_up is True, "server IS up, just missing the model"


def test_ollama_probe_returns_false_when_server_unreachable():
    import urllib.error

    provider = OllamaProvider(simple_model="qwen2.5:7b")
    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.URLError("connection refused"),
    ):
        assert provider.probe(timeout_s=0.1) is False
    assert provider._server_up is False


# ---------- router + mock embedding path ----------


def test_router_embed_round_trips_through_cache(tmp_path):
    cfg = _config_with_mock(tmp_path, ollama_enabled=False)
    mock = MockProvider(name="mock")
    router = LLMRouter(
        cfg,
        providers={"mock": mock},
        cache=LLMCache(cache_dir=tmp_path / "cache"),
    )
    v1 = router.embed("hello")
    v2 = router.embed("hello")
    assert v1 == v2
    assert len(mock.embed_calls) == 1, "second embed call should hit cache"


def test_router_embed_batch(tmp_path):
    cfg = _config_with_mock(tmp_path, ollama_enabled=False)
    mock = MockProvider(name="mock")
    router = LLMRouter(cfg, providers={"mock": mock}, cache=None)
    out = router.embed_batch(["a", "b", "c"], max_workers=2)
    assert len(out) == 3
    assert all(len(v) == mock.simple_embed_dim for v in out)


# ---------- alibaba tongyi provider (mocked urllib) ----------


def _tongyi_response_body(content: str, request_id: str = "req-1") -> dict:
    return {
        "output": {
            "choices": [
                {
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ]
        },
        "usage": {"input_tokens": 7, "output_tokens": 11, "total_tokens": 18},
        "request_id": request_id,
    }


def test_tongyi_provider_model_slots():
    p = AlibabaTongyiProvider(api_key="sk-fake")
    assert p.model_for(TaskTier.SIMPLE) == "qwen-turbo"
    assert p.model_for(TaskTier.ADVANCED) == "qwen-max"


def test_tongyi_provider_requires_api_key():
    p = AlibabaTongyiProvider(api_key=None)
    with pytest.raises(RuntimeError, match="requires an API key"):
        p.chat([ChatMessage(role="user", content="hi")])


def test_tongyi_provider_chat_round_trip():
    p = AlibabaTongyiProvider(api_key="sk-fake")
    captured: dict = {}

    def _fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.headers)
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResp(_tongyi_response_body("你好,世界", request_id="req-xyz"))

    with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        result = p.chat(
            [ChatMessage(role="user", content="说你好")],
            tier=TaskTier.ADVANCED,
            max_tokens=256,
            temperature=0.3,
        )

    # Wire shape
    assert "dashscope.aliyuncs.com" in captured["url"]
    assert captured["headers"]["Authorization"] == "Bearer sk-fake"
    assert captured["body"]["model"] == "qwen-max"
    assert captured["body"]["input"]["messages"] == [
        {"role": "user", "content": "说你好"}
    ]
    assert captured["body"]["parameters"]["result_format"] == "message"
    assert captured["body"]["parameters"]["max_tokens"] == 256
    assert captured["body"]["parameters"]["temperature"] == 0.3

    # Completion shape
    assert result.content == "你好,世界"
    assert result.provider == "tongyi"
    assert result.model_id == "qwen-max"
    assert result.tier == TaskTier.ADVANCED
    assert result.prompt_tokens == 7
    assert result.completion_tokens == 11
    assert result.metadata["request_id"] == "req-xyz"


def test_tongyi_provider_raises_on_empty_content():
    """Empty content is a known silent-failure mode on DashScope — the
    reference project at Stock-AI learned to guard against it. Match."""
    p = AlibabaTongyiProvider(api_key="sk-fake")
    body = {
        "output": {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]},
        "usage": {"input_tokens": 0, "output_tokens": 0},
        "request_id": "req-empty",
    }
    with patch("urllib.request.urlopen", return_value=_FakeResp(body)):
        with pytest.raises(RuntimeError, match="empty content"):
            p.chat([ChatMessage(role="user", content="x")])


def test_tongyi_provider_raises_on_http_error():
    import urllib.error

    p = AlibabaTongyiProvider(api_key="sk-fake")
    err = urllib.error.HTTPError(
        url="x", code=401, msg="Unauthorized",
        hdrs=None, fp=io.BytesIO(b'{"message":"Invalid API-key"}')
    )
    with patch("urllib.request.urlopen", side_effect=err):
        with pytest.raises(RuntimeError, match="Tongyi HTTP 401"):
            p.chat([ChatMessage(role="user", content="x")])


def test_router_builds_tongyi_from_config(tmp_path, monkeypatch):
    """End-to-end: config file names tongyi as active_provider, router
    constructs the provider with the env-sourced API key."""
    monkeypatch.setenv("ALIBABA_TONGYI_API_KEY", "sk-env-provided")
    cfg = LLMConfig(
        active_provider="tongyi",
        providers={
            "tongyi": ProviderConfig(
                name="tongyi",
                simple_model="qwen-turbo",
                advanced_model="qwen-max",
                api_key_env="ALIBABA_TONGYI_API_KEY",
            ),
        },
        local_ollama=LocalOllamaConfig(enabled=False),
    )
    router = LLMRouter(cfg, cache=None)
    provider = router.providers["tongyi"]
    assert isinstance(provider, AlibabaTongyiProvider)
    assert provider.api_key == "sk-env-provided"

    with patch(
        "urllib.request.urlopen",
        return_value=_FakeResp(_tongyi_response_body("ok")),
    ):
        result = router.chat(
            [ChatMessage(role="user", content="ping")],
            tier=TaskTier.SIMPLE,
            use_cache=False,
        )
    assert result.content == "ok"
    assert result.model_id == "qwen-turbo"
