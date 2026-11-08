"""Batch transport controls; real-model discrimination is measured separately."""
import pytest

from tests.test_local_embedding_identity import transport


def test_native_batch_deduplicates_inputs_preserves_order_and_attests_per_batch(transport):
    _, router, state = transport
    state['vectors'] = {'alpha':[.1,.9], 'beta':[.8,.2]}
    state['calls'].clear()
    result = router.embed_batch(['beta','alpha','beta'], backend='local')
    assert result == [[.8,.2],[.1,.9],[.8,.2]]
    bodies = [body for endpoint,body in state['calls'] if endpoint=='embed']
    assert len(bodies)==1 and bodies[0]['input']==['beta','alpha']
    assert sum(endpoint in ('tags','show') for endpoint,_ in state['calls'])==4
    result[0][0]=99
    assert result[2][0]==.8


def test_warm_cache_batch_has_constant_metadata_and_no_inference(transport):
    _, router, state = transport
    texts = [f'input {i}' for i in range(12)]
    router.embed_batch(texts, backend='local')
    state['calls'].clear()
    assert len(router.embed_batch(texts, backend='local'))==12
    assert [endpoint for endpoint,_ in state['calls']]==['tags','show','tags','show']


def test_replacement_between_batches_rejects_cached_outputs(transport):
    _, router, state = transport
    router.embed_batch(['alpha','beta'], backend='local')
    state['digest']='e'*64
    with pytest.raises(ValueError, match='digest'):
        router.embed_batch(['alpha','beta'], backend='local')


def test_mid_batch_replacement_rejects_and_does_not_publish_cache(transport):
    provider, router, state = transport
    opened=provider._open
    def replace_during_inference(request, **kwargs):
        response=opened(request, **kwargs)
        if request.full_url.endswith('/api/embed'):
            state['digest']='e'*64
        return response
    provider._open=replace_during_inference
    with pytest.raises(ValueError, match='digest|identity'):
        router.embed_batch(['alpha','beta'], backend='local')
    assert not list(router.cache.cache_dir.rglob('*.json'))


@pytest.mark.parametrize('rows', [[], [[.2,.8]], [[.2,.8],[0.,0.]], [[.2,.8],[1.]]])
def test_malformed_batch_is_atomic_before_cache_writes(transport, rows):
    _, router, state = transport
    state['response_rows']=rows
    with pytest.raises(ValueError):
        router.embed_batch(['alpha','beta'], backend='local')
    assert not list(router.cache.cache_dir.rglob('*.json'))


def test_only_cache_misses_reach_the_provider(transport):
    _, router, state = transport
    router.embed('alpha', backend='local')
    state['calls'].clear()
    router.embed_batch(['alpha','beta','alpha'], backend='local')
    bodies=[body for endpoint,body in state['calls'] if endpoint=='embed']
    assert len(bodies)==1 and bodies[0]['input']=='beta'


def test_batch_requests_are_bounded(transport):
    _, router, state = transport
    assert len(router.embed_batch([f'input {i}' for i in range(35)],backend='local'))==35
    bodies=[body for endpoint,body in state['calls'] if endpoint=='embed']
    assert len(bodies)==2 and all(len(body['input'])<=32 for body in bodies)


def test_frozen_identity_is_checked_before_cached_values_or_inference(transport):
    provider, router, state = transport
    identity=provider.embedding_identity('embed:latest')
    identity['dimension']=3
    with pytest.raises(ValueError, match='identity'):
        router.embed_batch(['alpha','beta'],backend='local',expected_identity=identity)
    assert not any(endpoint=='embed' for endpoint,_ in state['calls'])


def test_empty_batch_requires_no_transport(transport):
    _, router, state = transport
    state['calls'].clear()
    assert router.embed_batch([],backend='local')==[]
    assert state['calls']==[]


def test_scope_encoder_cannot_be_reused_after_verification(transport):
    provider, _, _ = transport
    with provider.verified_embedding_batch('embed:latest') as (_, encode):
        assert encode(['alpha'])==[[.2,.8]]
    with pytest.raises(ValueError, match='closed'):
        encode(['beta'])


def test_later_chunk_failure_cannot_publish_the_first_chunk(transport):
    provider, router, state = transport
    opened=provider._open
    calls=0
    def fail_second_chunk(request, **kwargs):
        nonlocal calls
        if request.full_url.endswith('/api/embed'):
            calls+=1
            if calls==2:
                raise TimeoutError('bounded transport failure')
        return opened(request, **kwargs)
    provider._open=fail_second_chunk
    with pytest.raises(TimeoutError):
        router.embed_batch([f'input {i}' for i in range(35)],backend='local')
    assert not list(router.cache.cache_dir.rglob('*.json'))


def test_uncached_batch_preserves_repeated_input_observations(transport):
    _, router, state = transport
    state['response_rows']=[[.1,.9],[.8,.2]]
    result=router.embed_batch(['alpha','alpha'],backend='local',use_cache=False)
    assert result==[[.1,.9],[.8,.2]]
    body=next(body for endpoint,body in state['calls'] if endpoint=='embed')
    assert body['input']==['alpha','alpha']
    assert not list(router.cache.cache_dir.rglob('*.json'))
