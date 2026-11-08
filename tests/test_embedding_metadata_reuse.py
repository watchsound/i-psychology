"""Metadata freshness protocol controls, not evidence of encoder quality."""
import json

import pytest

from psychology_base.llm.providers import ollama
from tests.test_local_embedding_identity import transport


def enable(provider, monkeypatch):
    provider.embedding_metadata_ttl_s = 5.0
    clock = [100.0]
    monkeypatch.setattr(ollama.time, 'monotonic', lambda: clock[0])
    return clock


def test_warm_reuse_checks_digest_at_both_boundaries(transport, monkeypatch):
    provider, router, state = transport
    enable(provider, monkeypatch)
    router.embed_batch(['alpha', 'beta'], backend='local')
    state['calls'].clear()
    assert router.embed_batch(['beta', 'alpha'], backend='local') == [[.2,.8],[.2,.8]]
    assert [endpoint for endpoint,_ in state['calls']] == ['tags', 'tags']


def test_expiry_refreshes_capability_and_never_serves_stale_on_failure(transport, monkeypatch):
    provider, router, state = transport
    clock = enable(provider, monkeypatch)
    router.embed('alpha', backend='local')
    clock[0] += 5.0
    state['capabilities'] = ['completion']
    for _ in range(2):
        with pytest.raises(ValueError, match='embedding'):
            router.embed('alpha', backend='local')


def test_reuse_does_not_extend_original_expiry(transport, monkeypatch):
    provider, router, state = transport
    clock = enable(provider, monkeypatch)
    router.embed('alpha', backend='local')
    clock[0] += 4
    router.embed('alpha', backend='local')
    state['calls'].clear()
    clock[0] += 1
    router.embed('alpha', backend='local')
    assert any(endpoint == 'show' for endpoint,_ in state['calls'])


def test_changed_digest_rejects_warm_cache_and_clears_metadata(transport, monkeypatch):
    provider, router, state = transport
    enable(provider, monkeypatch)
    router.embed('alpha', backend='local')
    state['digest'] = 'e' * 64
    with pytest.raises(ValueError, match='digest'):
        router.embed('alpha', backend='local')
    state['digest'] = 'f' * 64
    state['capabilities'] = []
    with pytest.raises(ValueError, match='embedding'):
        router.embed('alpha', backend='local')


def test_same_digest_tag_change_refreshes_metadata(transport, monkeypatch):
    provider, router, _ = transport
    enable(provider, monkeypatch)
    router.embed('alpha', backend='local')
    opened = provider._open
    from io import BytesIO
    def changed(request, **kwargs):
        response = opened(request, **kwargs)
        if request.full_url.endswith('/api/tags'):
            data = json.load(response)
            data['models'][0]['modified_at'] = '2026-10-24T01:00:00Z'
            return BytesIO(json.dumps(data).encode())
        if request.full_url.endswith('/api/show'):
            return BytesIO(json.dumps({'capabilities': []}).encode())
        return response
    provider._open = changed
    with pytest.raises(ValueError, match='embedding'):
        router.embed('alpha', backend='local')


def test_new_inference_forces_full_final_metadata_and_is_atomic(transport, monkeypatch):
    provider, router, state = transport
    enable(provider, monkeypatch)
    router.embed('alpha', backend='local')
    opened = provider._open
    def changed(request, **kwargs):
        response = opened(request, **kwargs)
        if request.full_url.endswith('/api/embed'):
            state['capabilities'] = []
        return response
    provider._open = changed
    before = {p.name for p in router.cache.cache_dir.rglob('*.json')}
    with pytest.raises(ValueError, match='embedding'):
        router.embed_batch(['alpha', 'beta'], backend='local')
    assert {p.name for p in router.cache.cache_dir.rglob('*.json')} == before


def test_changed_context_cannot_reuse_metadata(transport, monkeypatch):
    provider, router, _ = transport
    enable(provider, monkeypatch)
    router.embed('alpha', backend='local')
    provider.context_length = 1024
    with pytest.raises(ValueError, match='context'):
        router.embed('alpha', backend='local')


@pytest.mark.parametrize('ttl', [-1, 61, float('inf'), float('nan'), True, '5', None])
def test_invalid_reuse_bound_fails_before_transport(transport, ttl):
    provider, router, state = transport
    provider.embedding_metadata_ttl_s = ttl
    state['calls'].clear()
    with pytest.raises(ValueError, match='TTL'):
        router.embed('alpha', backend='local')
    assert not state['calls']


def test_reuse_is_disabled_by_default():
    assert ollama.OllamaProvider().embedding_metadata_ttl_s == 0


def test_slow_tags_cannot_extend_metadata_expiry(transport, monkeypatch):
    provider, router, state = transport
    clock = enable(provider, monkeypatch)
    router.embed('alpha', backend='local')
    clock[0] += 4
    opened = provider._open
    def slow(request, **kwargs):
        response = opened(request, **kwargs)
        if request.full_url.endswith('/api/tags'):
            clock[0] += 2
        return response
    provider._open = slow
    state['capabilities'] = []
    with pytest.raises(ValueError, match='embedding'):
        router.embed('alpha', backend='local')


def test_metadata_transport_failure_invalidates_warm_entry(transport, monkeypatch):
    provider, router, state = transport
    enable(provider, monkeypatch)
    router.embed('alpha', backend='local')
    opened = provider._open
    def fail(request, **kwargs):
        raise TimeoutError('metadata failure')
    provider._open = fail
    with pytest.raises(TimeoutError):
        router.embed('alpha', backend='local')
    provider._open = opened
    state['capabilities'] = []
    with pytest.raises(ValueError, match='embedding'):
        router.embed('alpha', backend='local')


def test_post_boundary_replacement_rejects_even_without_inference(transport, monkeypatch):
    provider, router, state = transport
    enable(provider, monkeypatch)
    router.embed('alpha', backend='local')
    opened = provider._open
    calls = 0
    def replace(request, **kwargs):
        nonlocal calls
        if request.full_url.endswith('/api/tags'):
            calls += 1
            if calls == 2:
                state['digest'] = 'e' * 64
        return opened(request, **kwargs)
    provider._open = replace
    with pytest.raises(ValueError, match='digest'):
        router.embed('alpha', backend='local')


def test_full_attestation_bypasses_reuse_and_copies_results(transport, monkeypatch):
    provider, router, state = transport
    enable(provider, monkeypatch)
    router.embed('alpha', backend='local')
    first = provider.embedding_identity('embed:latest')
    first['dimension'] = 99
    assert provider.embedding_identity('embed:latest')['dimension'] == 2
    state['capabilities'] = []
    with pytest.raises(ValueError, match='embedding'):
        provider.embedding_identity('embed:latest', refresh=True)
