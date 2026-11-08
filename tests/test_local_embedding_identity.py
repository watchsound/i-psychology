"""HTTP/cache contract controls, with prescribed vectors, never quality evidence."""

from io import BytesIO
from dataclasses import replace
import json

import pytest

from psychology_base.llm.cache import LLMCache, embedding_cache_key
from psychology_base.llm.config import default_config
from psychology_base.llm.execution import ExecutionPolicy
from psychology_base.llm.providers.ollama import OllamaProvider
from psychology_base.llm.router import LLMRouter
from psychology_base.llm.types import TaskTier


@pytest.fixture
def transport(monkeypatch, tmp_path):
    provider = OllamaProvider(base_url="http://127.0.0.1:11434", simple_model="missing-chat",
        embedding_model="embed:latest", local_only=True, expected_digest="f" * 64,
        context_length=512)
    state = {"digest": "f" * 64, "capabilities": ["embedding"], "vector": [.2, .8],
             "response_model": "embed:latest", "calls": []}

    def open_control(request, *, timeout):
        body = json.loads(request.data) if request.data else None
        state["calls"].append((request.full_url.rsplit("/", 1)[-1], body))
        if request.full_url.endswith("/api/tags"):
            value = {"models": [{"name": "embed:latest", "digest": state["digest"], "size": 100}]}
        elif request.full_url.endswith("/api/show"):
            value = {"capabilities": state["capabilities"], "model_info": {
                "bert.context_length": 512, "bert.embedding_length": 2}}
        elif request.full_url.endswith("/api/embed"):
            inputs = body['input'] if isinstance(body['input'], list) else [body['input']]
            rows = [state.get('vectors', {}).get(text, state['vector']) for text in inputs]
            value = {"model": state["response_model"], "embeddings": state.get('response_rows', rows)}
        elif request.full_url.endswith("/api/embeddings"):
            value = {"embedding": state["vector"]}
        else:
            pytest.fail(f"Unexpected transport endpoint: {request.full_url}")
        return BytesIO(json.dumps(value).encode())

    monkeypatch.setattr(provider, "_open", open_control)
    cfg = replace(default_config(), execution=ExecutionPolicy(mode="research"))
    router = LLMRouter(cfg, providers={"ollama": provider}, cache=LLMCache(tmp_path))
    return provider, router, state


def test_embedding_capability_can_be_attested_without_completion(transport):
    provider, _, _ = transport
    result = provider.attest_model("embed:latest", capability="embedding")
    assert result["dimension"] == 2 and result["context_length"] == 512
    with pytest.raises(ValueError, match="completion"):
        provider.attest_model("embed:latest")


def test_serving_policy_retains_verified_native_batch_contract(transport):
    provider, old, state = transport
    router = LLMRouter(replace(old.config, execution=ExecutionPolicy()),
                       providers={"ollama": provider}, cache=old.cache)
    before = len(state["calls"])
    assert router.embed_batch(["one", "two", "one"], backend="local") == [[.2, .8]] * 3
    recent = state["calls"][before:]
    assert sum(endpoint == "embed" for endpoint, _ in recent) == 1
    assert sum(endpoint == "tags" for endpoint, _ in recent) == 2
    assert sum(endpoint == "show" for endpoint, _ in recent) == 2


def test_pinned_local_embedding_does_not_require_a_chat_model(transport):
    provider, router, state = transport
    assert not router._ollama_ready
    assert router.embed("A frozen input.", backend="local") == [.2, .8]
    assert not router._ollama_ready
    _, decision = router.route(TaskTier.SIMPLE)
    assert decision.reason == "ollama_unavailable_fallback_to_cloud"
    body = next(body for endpoint, body in state["calls"] if endpoint == "embed")
    assert body["truncate"] is False and body["input"] == "A frozen input."


def test_pinned_route_does_not_accept_legacy_cache_entries(transport):
    _, router, state = transport
    key = embedding_cache_key(provider="ollama", model_id="embed:latest", text="A frozen input.")
    router.cache.put_embedding(key, [.9, .1], model_id="embed:latest")
    assert router.embed("A frozen input.", backend="local") == [.2, .8]
    assert sum(endpoint == "embed" for endpoint, _ in state["calls"]) == 1


def test_same_name_model_replacement_cannot_reuse_old_vector(transport):
    provider, router, state = transport
    assert router.embed("A frozen input.", backend="local") == [.2, .8]
    provider.expected_digest = state["digest"] = "e" * 64
    state["vector"] = [.8, .2]
    assert router.embed("A frozen input.", backend="local") == [.8, .2]
    assert sum(endpoint == "embed" for endpoint, _ in state["calls"]) == 2


def test_cached_vector_requires_fresh_attestation(transport):
    _, router, state = transport
    router.embed("A frozen input.", backend="local")
    state["digest"] = "e" * 64
    with pytest.raises(ValueError, match="digest"):
        router.embed("A frozen input.", backend="local")
    assert sum(endpoint == "embed" for endpoint, _ in state["calls"]) == 1


@pytest.mark.parametrize("vector", [[], [0., 0.], [float("nan"), 1.], [float("inf"), 1.], [1.], [True, 1.], ["1", 2.]])
def test_bad_vector_rejects_before_caching(transport, vector):
    _, router, state = transport
    state["vector"] = vector
    with pytest.raises(ValueError, match="[Vv]ector|[Ee]mbedding|[Dd]imension"):
        router.embed("A frozen input.", backend="local")
    assert not list(router.cache.cache_dir.rglob("*.json"))


def test_wrong_response_model_is_rejected(transport):
    _, router, state = transport
    state["response_model"] = "other-model"
    with pytest.raises(ValueError, match="model"):
        router.embed("A frozen input.", backend="local")


def test_cache_identity_is_optional_and_model_sensitive():
    base = dict(provider="ollama", model_id="embed:latest", text="A frozen input.")
    legacy = embedding_cache_key(**base)
    assert legacy == "6cbfe01e27b2704ec12d723d0c3d72b32628c779334e8c43a38297a41c3ab761"
    identity = {"digest": "f" * 64, "input_policy": "raw-v1", "dimension": 2}
    assert embedding_cache_key(**base, identity=None) == legacy
    assert embedding_cache_key(**base, identity=identity) != legacy
    assert embedding_cache_key(**base, identity={**identity, "digest": "e" * 64}) != embedding_cache_key(**base, identity=identity)


def test_valid_cache_hit_still_attests_but_does_not_repeat_inference(transport):
    _, router, state = transport
    first = router.embed("A frozen input.", backend="local")
    count = len(state["calls"])
    assert router.embed("A frozen input.", backend="local") == first
    assert len(state["calls"]) > count
    assert sum(endpoint == "embed" for endpoint, _ in state["calls"]) == 1


def test_cached_invalid_vector_is_rejected(transport):
    provider, router, _ = transport
    identity = provider.embedding_identity("embed:latest")
    key = embedding_cache_key(provider="ollama", model_id="embed:latest", text="A frozen input.", identity=identity)
    router.cache.put_embedding(key, [0., 0.], model_id="embed:latest")
    with pytest.raises(ValueError, match="norm"):
        router.embed("A frozen input.", backend="local")


def test_completion_capability_does_not_qualify_as_embedding(transport):
    _, router, state = transport
    state["capabilities"] = ["completion"]
    with pytest.raises(ValueError, match="embedding"):
        router.embed("A frozen input.", backend="local")
    assert not any(endpoint == "embed" for endpoint, _ in state["calls"])


def test_unused_multilingual_slot_cannot_qualify_a_chat_fallback(transport):
    provider, router, _ = transport
    provider.embedding_model = None
    provider.embedding_model_multilingual = "embed:latest"
    with pytest.raises(ValueError, match="dedicated"):
        router.embed("A frozen input.", backend="local")


def test_research_binding_encodes_only_the_eligible_pool_and_leaves_witness_plain(transport):
    from datetime import datetime, timedelta, timezone
    from psychology_base.companion.system import CompanionSystem
    from psychology_base.eval.adapters import Observation
    from psychology_base.eval.benchmarks.graph_priors import bind_navigation
    provider, router, state = transport
    identity = provider.embedding_identity('embed:latest')
    system = bind_navigation(CompanionSystem(), 'semantic', router=router, floor=.4,
                             model_identity=identity, query_prefix='Query: ')
    stamp = datetime(2026, 10, 23, tzinfo=timezone.utc)
    for i, (subject, text) in enumerate([('self','Own original experience.'), ('other','Foreign experience.')]):
        system.observe(Observation(stamp+timedelta(seconds=i), 'user', text,
                                   subject_id=subject, entity_id='memory'))
    result = system.recall('New wording', scope=('self','memory'))
    assert [f.content for f in result.fragments_used] == ['Own original experience.']
    inputs = [body['input'] for endpoint,body in state['calls'] if endpoint=='embed']
    assert inputs == [['Query: New wording','Own original experience.']]
    assert system.activation.recall_count == system.witness_state.activation.recall_count == 1
    before = system.inspect()
    state['digest'] = 'e'*64
    with pytest.raises(ValueError, match='digest'):
        system.recall('Next query', scope=('self','memory'))
    assert system.inspect() == before


@pytest.mark.parametrize('kind', ['missing', 'modified'])
def test_diagnostic_scope_rejects_before_transport(tmp_path, monkeypatch, kind):
    from types import SimpleNamespace
    from psychology_base.eval.benchmarks import application_context as study
    monkeypatch.setattr(OllamaProvider, '_open', lambda *a, **k: pytest.fail('Invalid scope reached transport'))
    path = None
    if kind == 'modified':
        value = study.read(study.FIXTURES/'memory-navigation-local-diagnostic-v1.json')
        value['hosted_provider_requests'] = 1
        path = tmp_path/'modified.json'
        path.write_text(json.dumps(value), encoding='utf-8')
    with pytest.raises(ValueError):
        study.navigation_diagnostic(SimpleNamespace(navigation_plan=path, output=tmp_path/'out.json'))
    assert not (tmp_path/'out').exists()


def test_ordinary_explicit_local_embedding_probes_its_own_model(transport):
    provider, router, state = transport
    provider.local_only = False
    assert not router._ollama_ready
    assert router.embed('A frozen input.', backend='local') == [.2,.8]
    assert not router._ollama_ready
    assert any(endpoint=='embeddings' for endpoint,_ in state['calls'])
    assert not any(endpoint=='embed' for endpoint,_ in state['calls'])


def test_oversized_semantic_view_falls_back_before_inference(transport, monkeypatch):
    from datetime import datetime, timezone
    from psychology_base.companion import navigation
    from psychology_base.companion.system import CompanionSystem
    from psychology_base.eval.adapters import Observation
    from psychology_base.eval.benchmarks.graph_priors import bind_navigation
    provider, router, state = transport
    identity = provider.embedding_identity('embed:latest')
    system = bind_navigation(CompanionSystem(), 'semantic', router=router, floor=.4, model_identity=identity)
    system.observe(Observation(datetime(2026,10,23,tzinfo=timezone.utc),'user','Original memory.'))
    monkeypatch.setattr(navigation,'MAX_BYTES',1)
    result = system.recall('Original memory')
    assert result.fragments_used and result.metadata['fragment_navigation']['reason']=='resource-bytes'
    assert not any(endpoint=='embed' for endpoint,_ in state['calls'])
    assert system.activation.recall_count == 1

