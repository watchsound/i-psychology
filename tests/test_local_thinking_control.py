"""Disclosed software transport tests; no model-quality claims or real calls."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from io import BytesIO
import json

import pytest

from psychology_base.applications.measurement.local_query_binding import LocalChatMembershipAssessor
from psychology_base.eval.benchmarks.local_membership_audit import audit_membership_capture
from psychology_base.eval.benchmarks.local_membership_diagnostic import run_local_membership_diagnostic
from psychology_base.llm.cache import chat_cache_key
from psychology_base.llm.config import default_config
from psychology_base.llm.providers.ollama import OllamaProvider
from psychology_base.llm.router import LLMRouter
from psychology_base.llm.types import ChatMessage, TaskTier
from tests.test_local_chat_membership import router as software_router, inputs
from tests.test_local_membership_diagnostic import setup, MODEL, DIGEST


def provider(think=None):
    return OllamaProvider(base_url='http://127.0.0.1:11434', simple_model=MODEL, advanced_model=MODEL,
        local_only=True, expected_digest=DIGEST, context_length=8192, seed=0, json_mode=True, think=think)


def software_http(monkeypatch, controls=None, trace=None):
    """Authored metadata/completion fixtures exercise the real provider."""
    calls = []
    def opened(self, request, *, timeout):
        path = request.full_url.rsplit('/', 1)[-1]
        payload = json.loads(request.data) if request.data else None
        calls.append((path, payload))
        if path == 'tags': value = dict(models=[dict(name=MODEL, size=10, digest=DIGEST)])
        elif path == 'show':
            value = dict(capabilities=['completion', 'embedding'], details=dict(family='test'),
                model_info={'test.context_length':8192, 'test.embedding_length':32})
            if controls is not None: value['thinking'] = deepcopy(controls)
        else:
            assert path == 'chat'
            message = dict(content='{"labels":["match","nonmatch"]}')
            if trace is not None: message['thinking'] = trace
            value = dict(model=MODEL, message=message, done=True, done_reason='stop',
                prompt_eval_count=100, eval_count=20)
        return BytesIO(json.dumps(value).encode())
    monkeypatch.setattr(OllamaProvider, '_open', opened)
    return calls


@pytest.mark.parametrize('think', [None, False, True])
def test_native_control_and_attestation_are_explicit_only_when_requested(monkeypatch, think):
    calls = software_http(monkeypatch, dict(values=[False, True]), 'Reasoning fixture' if think else None)
    p = provider(think); messages = [ChatMessage(role='user', content='Authored control')]
    result = p.chat(messages, tier=TaskTier.ADVANCED, max_tokens=64, temperature=0)
    expected = dict(model=MODEL, messages=[dict(role='user', content='Authored control')], stream=False,
        options=dict(temperature=0, num_predict=64, num_ctx=8192, seed=0), format='json')
    if think is not None: expected['think'] = think
    assert calls == [('tags', None), ('show', {'model':MODEL}), ('chat', expected)]
    assert result.completion_tokens == 20  # Native count is never reduced for reasoning.
    if think is None:
        assert 'think' not in result.metadata and 'thinking_control' not in result.metadata['model_attestation']
    else:
        assert result.metadata['think'] is think
        assert result.metadata['model_attestation']['thinking_control'] == dict(requested=think, supported=[False, True])


@pytest.mark.parametrize('controls', [None, {}, {'values':[]}, {'values':[True]},
    {'values':['low','high']}, {'values':[0,1]}, {'values':[False,'low']}, {'values':False}])
def test_unknown_or_unsupported_controls_stop_before_chat_without_fallback(monkeypatch, controls):
    calls = software_http(monkeypatch, controls)
    p = provider(False)
    with pytest.raises(ValueError, match='does not attest'): p.chat([ChatMessage(role='user', content='Test')])
    assert [path for path, _ in calls] == ['tags', 'show'] and p._model_metadata_cache is None


@pytest.mark.parametrize('think', [0, 1, 'false', 'low', [], {}])
def test_invalid_control_types_fail_without_metadata_or_native_requests(monkeypatch, think):
    calls = software_http(monkeypatch, dict(values=[False, True]))
    with pytest.raises(ValueError): provider(think).chat([ChatMessage(role='user', content='Test')])
    assert calls == []


def test_explicit_control_requires_pinned_transport_and_embedding_identity_is_unchanged(monkeypatch):
    calls = software_http(monkeypatch)
    with pytest.raises(ValueError): OllamaProvider(think=False).chat([ChatMessage(role='user', content='Test')])
    assert calls == []
    p = provider(False)
    attestation = p.attest_model(MODEL, capability='embedding')
    assert attestation == dict(model=MODEL, digest=DIGEST, family='test', size=10, dimension=32, context_length=8192)


@pytest.mark.parametrize('trace', ['unexpected reasoning', False, 0, [], {}])
def test_non_thinking_response_cannot_hide_or_accept_contradictory_trace(monkeypatch, trace):
    calls = software_http(monkeypatch, dict(values=[False, True]), trace)
    with pytest.raises(ValueError, match='contradicts'): provider(False).chat([ChatMessage(role='user', content='Test')])
    assert sum(path == 'chat' for path, _ in calls) == 1


def test_cache_identity_preserves_default_and_separates_each_explicit_mode():
    args = dict(provider='ollama', model_id=MODEL, tier=TaskTier.ADVANCED,
        messages=[ChatMessage(role='user', content='Test')], temperature=0, max_tokens=64)
    original = dict(purpose='chat', provider='ollama', model_id=MODEL, tier='advanced',
        messages=[dict(role='user', content='Test')], temperature=0, max_tokens=64)
    historical = sha256(json.dumps(original, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    assert chat_cache_key(**args) == historical == chat_cache_key(**args, think=None)
    assert len({historical, chat_cache_key(**args, think=False), chat_cache_key(**args, think=True)}) == 3
    with pytest.raises(ValueError): chat_cache_key(**args, think=0)


def test_actual_router_cache_reuses_each_mode_without_cross_mode_hits(tmp_path, monkeypatch):
    calls = software_http(monkeypatch, dict(values=[False, True]))
    config = default_config()
    config = replace(config, cache=replace(config.cache, dir=tmp_path/'cache'),
        execution=replace(config.execution, mode='research'), local_ollama=replace(config.local_ollama, enabled=False))
    p = provider(); r = LLMRouter(config, providers={'ollama':p})
    for mode in (None, False, True, False, None, True):
        p.think = mode
        completion = r.chat([ChatMessage(role='user', content='Test')], tier=TaskTier.ADVANCED, max_tokens=64, temperature=0)
        assert completion.metadata.get('think') is mode
    native = [payload for path, payload in calls if path == 'chat']
    assert len(native) == 3 and [payload.get('think') for payload in native] == [None, False, True]


@pytest.mark.parametrize('damage', [None, 'unsupported', 'trace', 'truncated', 'token_overflow', 'mutation', 'audit_mode', 'audit_supported', 'audit_invented_support'])
def test_existing_executor_and_native_auditor_enforce_control_without_retry(tmp_path, monkeypatch, damage):
    state = setup(tmp_path, monkeypatch)
    plan = state.read(state.plan_path); plan['think'] = False; state.save(state.plan_path, plan)
    original = OllamaProvider._open
    def opened(p, request, *, timeout):
        response = original(p, request, timeout=timeout)
        value = json.load(response)
        if request.full_url.endswith('/show') and damage != 'unsupported': value['thinking'] = dict(values=[False, True])
        if request.full_url.endswith('/chat'):
            if damage == 'trace': value['message']['thinking'] = 'Unexpected reasoning'
            if damage == 'truncated': value['done_reason'] = 'length'
            if damage == 'token_overflow': value['eval_count'] = 65
            if damage == 'mutation': p.think = True
        return BytesIO(json.dumps(value).encode())
    monkeypatch.setattr(OllamaProvider, '_open', opened)
    run_local_membership_diagnostic(**state.args)
    result = state.read(state.result_path)
    obs, events = state.read(state.out/'observations.json'), state.read(state.out/'transport.json')
    failed = damage in ('unsupported', 'trace', 'truncated', 'token_overflow', 'mutation')
    assert result['completed'] == (not failed)
    assert result['native_attempts'] == (0 if damage == 'unsupported' else 1 if failed else 3)
    assert result['paid_api_usd'] == '0' and not result['qualified']
    if damage in ('audit_mode', 'audit_supported', 'audit_invented_support'):
        meta = obs[0]['native_calls'][0]['completion']['metadata']
        if damage == 'audit_mode': meta['think'] = 0
        elif damage == 'audit_supported': meta['model_attestation']['thinking_control']['supported'] = [0]
        else: meta['model_attestation']['thinking_control']['supported'] = [False]
    args = dict(rows_for=lambda e:state.args['rows_for'](state.cases[e['case']], e),
        snapshot=lambda name:state.read(state.out/name), model=MODEL, digest=DIGEST, max_native=3, max_metadata=64, think=False)
    if damage in ('audit_mode', 'audit_supported', 'audit_invented_support'):
        with pytest.raises(ValueError, match='thinking control'):
            audit_membership_capture(events, obs, state.schedule, **args)
    else:
        replay = audit_membership_capture(events, obs, state.schedule, **args)
        assert len(replay['predictions']) == (0 if failed else 2)
    if not failed:
        with pytest.raises(ValueError): audit_membership_capture(events, obs, state.schedule, **dict(args, think=None))
    with pytest.raises(FileExistsError): run_local_membership_diagnostic(**state.args)


@pytest.mark.parametrize('value', [None, True, 0])
def test_assessor_identity_and_pre_dispatch_mutation_checks(value):
    r, seen = software_router(); p = r.providers['ollama']; p.think = False
    candidate = LocalChatMembershipAssessor(r)
    p.think = value
    with pytest.raises(ValueError, match='configuration changed'): candidate(inputs())
    assert seen == []


def test_malformed_frozen_control_fails_before_any_scope_is_started(tmp_path, monkeypatch):
    state = setup(tmp_path, monkeypatch); plan = state.read(state.plan_path)
    plan['think'] = 0; state.save(state.plan_path, plan)
    with pytest.raises(ValueError, match='Frozen thinking'): run_local_membership_diagnostic(**state.args)
    assert state.wire == [] and not (state.out/'started.json').exists()
