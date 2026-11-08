"""Portable contract tests; authored HTTP replies are not model-quality evidence."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from urllib.error import HTTPError

import pytest

from psychology_base.applications.measurement.local_query_binding import (
    LocalChatMembershipAssessor, REFERENCE_IDS_PROMPT, reference_messages,
)
from psychology_base.applications.measurement.reference_readings import (
    reference_projection, reference_response_schema, decode_reference_readings,
)
from psychology_base.eval.benchmarks.local_membership_audit import membership_wire_payload, audit_membership_capture
from psychology_base.eval.benchmarks.local_membership_diagnostic import run_local_membership_diagnostic
from psychology_base.eval.benchmarks.reference_extraction import rows_for, authored_schema_example, grade
from psychology_base.llm.accounting import request_input_bound
from psychology_base.llm.cache import chat_cache_key
from psychology_base.llm.config import default_config
from psychology_base.llm.execution import ExecutionDeferred
from psychology_base.llm.providers.ollama import OllamaProvider
from psychology_base.llm.router import LLMRouter
from psychology_base.llm.types import ChatMessage, TaskTier
from tests.test_reference_extraction import packet, extraction_router
from tests.test_reference_extraction_execution import setup as old_setup, MODEL, DIGEST

ROOT = Path(__file__).parents[1]
KIND = 'reference-extraction-v2'


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def sample(key='reference-alias-en'):
    inputs, refs = packet()
    entry = next(e for e in inputs['schedule'] if e['case'] == key)
    rows = rows_for(inputs['cases'][key], entry)
    focus = json.loads(rows[0]['pair'][0])
    raw, analysis = authored_schema_example(refs[key], focus)
    return rows, focus, raw, analysis


@pytest.mark.parametrize('key', ['reference-alias-en', 'reference-alias-zh', 'reference-quotation-en',
    'reference-correlated-zh', 'reference-implicit-zh'])
def test_projection_preserves_exact_source_ids_roles_unicode_and_focus(key):
    rows, focus, _, _ = sample(key)
    before = deepcopy(focus)
    projection = reference_projection(focus)
    turns = focus['prior_utterances'] + [focus['focus_utterance']]
    assert projection == dict(schema='reference-turn-ids-v1', context_scope=focus['context_scope'],
        turns=[dict(id=i, **t) for i, t in enumerate(turns)], focus_turn=len(turns)-1)
    messages = reference_messages(focus, explicit_ids=True)
    assert messages[0].content == REFERENCE_IDS_PROMPT and json.loads(messages[1].content) == projection
    wire = membership_wire_payload(rows, MODEL, assessment_kind=KIND, max_tokens=1024)
    assert wire['messages'] == [dict(role=m.role, content=m.content) for m in messages]
    assert wire['format'] == reference_response_schema()
    assert 'descriptions' not in projection and focus == before
    projection['turns'][0]['text'] = 'mutation'
    assert focus == before


@pytest.mark.parametrize('prior_count', [0, 1, 2])
def test_focus_id_uses_only_the_delivered_window_not_global_record_positions(prior_count):
    _, focus, _, _ = sample()
    focus['prior_utterances'] = focus['prior_utterances'][:prior_count]
    projection = reference_projection(focus)
    assert projection['focus_turn'] == prior_count
    assert [t['id'] for t in projection['turns']] == list(range(prior_count+1))


@pytest.mark.parametrize('damage', ['future', 'extra', 'scope', 'role', 'text'])
def test_invalid_or_extended_source_windows_fail_before_projection(damage):
    _, focus, _, _ = sample()
    if damage == 'future': focus['prior_utterances'] *= 2
    elif damage == 'extra': focus['answer'] = 'not permitted'
    elif damage == 'scope': focus['context_scope'] = ''
    elif damage == 'role': focus['focus_utterance']['speaker'] = 'invented'
    else: focus['focus_utterance']['text'] = ''
    with pytest.raises(ValueError): reference_messages(focus, explicit_ids=True)


def test_schema_is_bounded_answer_free_and_returns_independent_trees():
    schema = reference_response_schema()
    assert schema['required'] == ['mentions', 'entities', 'focus_references', 'readings', 'coverage']
    assert schema['additionalProperties'] is False
    props = schema['properties']
    assert props['mentions']['maxItems'] == 16
    assert props['entities']['maxItems'] == props['focus_references']['maxItems'] == props['readings']['maxItems'] == 8
    assert props['readings']['minItems'] == 1
    assert props['mentions']['items']['properties']['turn'] == dict(type='integer', minimum=0, maximum=2)
    assert set(props['mentions']['items']['properties']['kind']['enum']) == {'named', 'deictic', 'anaphoric', 'implicit'}
    assert 'Mira' not in json.dumps(schema) and 'Jo' not in json.dumps(schema)
    schema['properties']['coverage']['enum'].append('invented')
    assert 'invented' not in reference_response_schema()['properties']['coverage']['enum']


def assessor(raw):
    router, seen = extraction_router(raw)
    provider = router.providers['ollama']
    provider.response_schema = reference_response_schema()
    original = router.chat
    def chat(messages, **kwargs):
        completion = original(messages, **kwargs)
        return replace(completion, metadata=dict(completion.metadata, response_schema_sha256=digest(provider.response_schema)))
    router.chat = chat
    return LocalChatMembershipAssessor(router, format_version=4, max_input=7168, max_tokens=1024,
        assessment_kind=KIND), seen


def test_new_mode_retains_canonical_analysis_and_emits_no_expected_answers():
    rows, focus, raw, expected = sample()
    candidate, seen = assessor(raw)
    assert candidate(rows) == expected and len(seen) == 1
    payload = json.loads(seen[0][0][1].content)
    assert payload == reference_projection(focus)
    assert 'descriptions' not in payload and 'reference' not in payload


@pytest.mark.parametrize('damage', ['missing', 'changed', 'mode', 'schema_attestation', 'protocol'])
def test_schema_mutation_and_protocol_changes_cannot_be_accepted(damage):
    rows, _, raw, _ = sample()
    candidate, seen = assessor(raw)
    if damage == 'missing': candidate.provider.response_schema = None
    elif damage == 'changed': candidate.provider.response_schema['properties']['mentions']['maxItems'] = 99
    elif damage == 'mode': candidate.assessment_kind = 'reference-extraction-v1'
    elif damage == 'protocol': candidate.prompt_profile = 'reference-possibility-v1'
    else:
        original = candidate.router.chat
        def no_hash(messages, **kwargs):
            result = original(messages, **kwargs)
            return replace(result, metadata={k:v for k,v in result.metadata.items() if k != 'response_schema_sha256'})
        candidate.router.chat = no_hash
    with pytest.raises(ValueError): candidate(rows)
    assert len(seen) == (1 if damage == 'schema_attestation' else 0)


def test_schema_overhead_is_admitted_before_dispatch():
    rows, focus, raw, _ = sample()
    candidate, seen = assessor(raw)
    messages = reference_messages(focus, explicit_ids=True)
    candidate.max_input = request_input_bound(messages)
    candidate._configuration = candidate._config()
    with pytest.raises(ValueError, match='input bound'): candidate(rows)
    assert seen == []


@pytest.mark.parametrize('json_mode,schema', [(False, None), (True, None), (False, {'type':'object'}),
    (True, {'type':'object'})])
def test_native_provider_preserves_default_wire_and_uses_schema_only_when_requested(monkeypatch, json_mode, schema):
    provider = OllamaProvider(json_mode=json_mode, response_schema=deepcopy(schema))
    requests = []
    def software_http(self, request, *, timeout):
        requests.append(json.loads(request.data))
        return BytesIO(json.dumps(dict(message=dict(content='{}'), done=True, done_reason='stop',
            prompt_eval_count=100, eval_count=2)).encode())
    monkeypatch.setattr(OllamaProvider, '_open', software_http)
    completion = provider.chat([ChatMessage(role='user', content='test')], max_tokens=64, temperature=0)
    expected = dict(model='qwen2.5:7b', messages=[dict(role='user', content='test')], stream=False,
        options=dict(temperature=0, num_predict=64))
    if schema is not None: expected['format'] = schema
    elif json_mode: expected['format'] = 'json'
    assert requests == [expected]
    if schema is None: assert 'response_schema_sha256' not in completion.metadata
    else: assert completion.metadata['response_schema_sha256'] == digest(schema)


@pytest.mark.parametrize('schema', [[], {}, 'json', False, {'invalid':float('nan')}])
def test_malformed_schema_fails_before_any_metadata_or_native_request(monkeypatch, schema):
    provider = OllamaProvider(local_only=True, response_schema=schema)
    def forbidden(*args, **kwargs): raise AssertionError('No network expected')
    monkeypatch.setattr(OllamaProvider, '_open', forbidden)
    with pytest.raises(ValueError): provider.chat([ChatMessage(role='user', content='test')])


def test_router_cache_separates_schemas_and_reuses_identical_schema(tmp_path, monkeypatch):
    config = default_config()
    config = replace(config, cache=replace(config.cache, dir=tmp_path/'cache'),
        local_ollama=replace(config.local_ollama,enabled=False),
        execution=replace(config.execution, mode='research'))
    provider = OllamaProvider()
    calls = []
    def software_http(self, request, *, timeout):
        payload = json.loads(request.data); calls.append(payload)
        return BytesIO(json.dumps(dict(message=dict(content=json.dumps(payload.get('format'))),
            done_reason='stop', prompt_eval_count=100, eval_count=2)).encode())
    monkeypatch.setattr(OllamaProvider, '_open', software_http)
    router = LLMRouter(config, providers={'ollama':provider})
    messages = [ChatMessage(role='user', content='same input')]
    outputs = []
    for schema in [None, {'type':'object'}, {'type':'object'}, {'type':'array'}, None]:
        provider.response_schema = schema
        outputs.append(router.chat(messages, tier=TaskTier.ADVANCED, max_tokens=64, temperature=0).content)
    assert len(calls) == 3 and outputs[0] == outputs[4] and outputs[1] == outputs[2] != outputs[3]
    args = dict(provider='ollama', model_id='test', tier=TaskTier.SIMPLE, messages=messages, temperature=0, max_tokens=64)
    historical = dict(purpose='chat', provider='ollama', model_id='test', tier='simple', temperature=0, max_tokens=64,
        messages=[dict(role='user', content='same input')])
    assert chat_cache_key(**args) == digest(historical) == chat_cache_key(**args, response_schema=None)
    assert chat_cache_key(**args, response_schema={'type':'object'}) != chat_cache_key(**args)


def setup_execution(tmp_path, monkeypatch):
    state = old_setup(tmp_path, monkeypatch)
    plan = state.read(state.plan_path); plan['assessment_kind'] = KIND
    state.save(state.plan_path, plan)
    return state


@pytest.mark.parametrize('damage', [None, 'api_schema', 'source', 'wire_schema', 'schema_hash'])
def test_actual_executor_router_and_auditor_handle_schema_errors_without_weaker_retry(tmp_path, monkeypatch, damage):
    state = setup_execution(tmp_path, monkeypatch)
    original = OllamaProvider._open
    def software_http(provider, request, *, timeout):
        if request.full_url.endswith('/chat') and damage == 'api_schema':
            # Explicit simulated server rejection, never an accuracy observation.
            raise HTTPError(request.full_url,400,'Bad Request',{},BytesIO(b'{"error":"unsupported schema grammar"}'))
        response = original(provider, request, timeout=timeout)
        if request.full_url.endswith('/chat') and damage == 'source':
            value = json.load(response); graph = json.loads(value['message']['content'])
            graph['mentions'][0]['turn'] = 2  # Schema-compatible, absent from this one-turn source.
            value['message']['content'] = json.dumps(graph)
            return BytesIO(json.dumps(value).encode())
        return response
    monkeypatch.setattr(OllamaProvider, '_open', software_http)
    run_local_membership_diagnostic(**state.args)
    result = state.read(state.result_path)
    obs, events = state.read(state.out/'observations.json'), state.read(state.out/'transport.json')
    if damage == 'api_schema':
        assert result['native_attempts'] == 1 and not result['completed']
        failure = next(e for e in events if e['path']=='/api/chat')['http_error']
        import base64
        assert failure['status'] == 400
        assert base64.b64decode(failure['body_base64']) == b'{"error":"unsupported schema grammar"}'
    else: assert result['native_attempts'] == 2
    if damage == 'schema_hash': obs[0]['native_calls'][0]['completion']['metadata']['response_schema_sha256'] = 'wrong'
    if damage == 'wire_schema':
        event = next(e for e in events if e['path'] == '/api/chat')
        event['request']['format'] = 'json'
    args = dict(rows_for=lambda e:rows_for(state.cases[e['case']],e), snapshot=lambda p:state.read(state.out/p),
        model=MODEL, digest=DIGEST, max_native=2, max_metadata=64, max_tokens=1024, max_input=7168, assessment_kind=KIND)
    if damage in ('wire_schema', 'schema_hash'):
        with pytest.raises(ValueError): audit_membership_capture(events, obs, state.schedule, **args)
    else:
        replay = audit_membership_capture(events, obs, state.schedule, **args)
        assert len(replay['predictions']) == (2 if damage is None else 0)
        assert result['graded']['summary']['all']['failed'] == (0 if damage is None else (1 if damage == 'api_schema' else 2))
    assert not result['qualified'] and result['paid_api_usd'] == '0'


def test_recorded_failed_completions_keep_original_source_validation_failures():
    fixture = json.loads((ROOT/'benchmarks/fixtures/application_context/reference-interface-invalid-v1.json').read_text(encoding='utf-8'))
    assert len(fixture['cases']) == 7 and fixture['kind'] == 'observed-invalid-model-output'
    for case in fixture['cases']:
        assert sha256(case['content'].encode()).hexdigest() == case['content_sha256']
        with pytest.raises(ValueError, match=case['error']):
            decode_reference_readings(case['content'], case['focus'])


def test_small_prospective_screen_keeps_missing_failed_and_repeat_denominators():
    inputs, refs = packet()
    schedule = [deepcopy(e) for e in inputs['schedule'] if not e['repeat']][:7]
    schedule.append(dict(schedule[0], id=schedule[0]['id']+'/repeat', repeat=True))
    cases = {e['case']:inputs['cases'][e['case']] for e in schedule}
    refs = {key:refs[key] for key in cases}
    observations = []
    for entry in schedule:
        focus = json.loads(rows_for(cases[entry['case']],entry)[0]['pair'][0])
        _, analysis = authored_schema_example(refs[entry['case']], focus)
        observations.append(dict(id=entry['id'],analysis=analysis))
    def screen(obs): return grade(cases,refs,schedule,obs,expected_conditions=8,expected_repeats=1)
    assert screen(observations)['candidate_screen_passed']
    assert not grade(cases,refs,schedule,observations)['candidate_screen_passed']
    assert not screen(observations[:-1])['candidate_screen_passed']
    assert screen(observations[:-1])['summary']['all']['missing'] == 1
    observations[-1] = dict(id=observations[-1]['id'],error='declared software failure')
    assert screen(observations)['summary']['all']['failed'] == 1
    assert not screen(observations)['candidate_screen_passed']


@pytest.mark.parametrize('kind,schema', [(KIND,None), (KIND,{'type':'object'}),
    ('membership',reference_response_schema()), ('reference-extraction-v1',reference_response_schema())])
def test_protocol_schema_mismatch_is_rejected_before_dispatch(kind,schema):
    _,_,raw,_ = sample()
    router,seen = extraction_router(raw);router.providers['ollama'].response_schema = schema
    with pytest.raises(ValueError,match='response schema'):
        LocalChatMembershipAssessor(router,format_version=4,assessment_kind=kind)
    assert seen == []


def test_schema_mutated_during_dispatch_cannot_publish_analysis():
    rows,_,raw,_ = sample();candidate,seen = assessor(raw)
    original = candidate.router.chat
    def mutated(messages,**kwargs):
        result = original(messages,**kwargs)
        candidate.provider.response_schema['properties']['coverage']['enum'].append('invented')
        return result
    candidate.router.chat = mutated
    with pytest.raises(ValueError,match='configuration changed'):candidate(rows)
    assert len(seen) == 1


@pytest.mark.parametrize('read_failure',[False,True])
def test_http_error_body_hash_or_read_failure_remains_auditable(tmp_path,monkeypatch,read_failure):
    state = setup_execution(tmp_path,monkeypatch)
    original = OllamaProvider._open
    class Unreadable(BytesIO):
        def read(self,*args): raise TimeoutError('Declared error-body timeout')
    def software_http(provider,request,*,timeout):
        if request.full_url.endswith('/chat'):
            body = Unreadable() if read_failure else BytesIO(b'{"error":"grammar"}')
            raise HTTPError(request.full_url,400,'Bad Request',{},body)
        return original(provider,request,timeout=timeout)
    monkeypatch.setattr(OllamaProvider,'_open',software_http)
    run_local_membership_diagnostic(**state.args)
    obs,events = state.read(state.out/'observations.json'),state.read(state.out/'transport.json')
    event = next(e for e in events if e['path']=='/api/chat')
    def snapshot(name):
        if name.endswith('-finished.json'):
            return next(e for e in events if f"native-{e['sequence']:03d}-finished.json"==name)
        return state.read(state.out/name)
    args = dict(rows_for=lambda e:rows_for(state.cases[e['case']],e),snapshot=snapshot,
        model=MODEL,digest=DIGEST,max_native=2,max_metadata=64,max_tokens=1024,max_input=7168,assessment_kind=KIND)
    replay = audit_membership_capture(events,obs,state.schedule,**args)
    assert replay['native_attempts']==1 and replay['native_responses']==0 and replay['predictions']=={}
    if read_failure:
        assert 'Declared error-body timeout' in event['http_error']['body_read_error']
    else:
        event['http_error']['body_sha256']='0'*64
        with pytest.raises(ValueError,match='error body changed'):
            audit_membership_capture(events,obs,state.schedule,**args)
