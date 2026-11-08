"""Software HTTP fixtures verify integration, never model accuracy."""
from copy import deepcopy
from dataclasses import replace
from io import BytesIO
import json

import pytest

from psychology_base.applications.measurement.local_query_binding import (
    LocalChatMembershipAssessor, REFERENCE_SELECTIONS_PROMPT, reference_messages,
)
from psychology_base.applications.measurement.reference_readings import (
    reference_candidate_view, reference_selection_schema,
)
from psychology_base.eval.benchmarks.reference_extraction import authored_selection_example, rows_for
from psychology_base.eval.benchmarks.local_membership_audit import membership_wire_payload, audit_membership_capture
from psychology_base.eval.benchmarks.local_membership_diagnostic import run_local_membership_diagnostic
from psychology_base.llm.accounting import request_input_bound
from psychology_base.llm.providers.ollama import OllamaProvider
from tests.test_reference_extraction import packet, extraction_router
from tests.test_reference_interface import sample as old_sample, digest
from tests.test_reference_extraction_execution import setup as old_setup, MODEL, DIGEST

KIND = 'reference-selections-v1'


def sample(key='reference-alias-en'):
    rows, focus, _, _ = old_sample(key)
    _, refs = packet()
    raw, analysis = authored_selection_example(refs[key], focus)
    return rows, focus, raw, analysis


def assessor(focus, raw):
    router, seen = extraction_router(raw)
    provider = router.providers['ollama']
    provider.response_schema = reference_selection_schema(focus)
    original = router.chat
    def chat(messages, **kwargs):
        completion = original(messages, **kwargs)
        return replace(completion, metadata=dict(completion.metadata,
            response_schema_sha256=digest(provider.response_schema)))
    router.chat = chat
    return LocalChatMembershipAssessor(router, format_version=4, max_input=7168,
        max_tokens=1024, assessment_kind=KIND, reference_scope=focus), seen


@pytest.mark.parametrize('key', ['reference-alias-en', 'reference-alias-zh', 'reference-quotation-en',
    'reference-correlated-zh', 'reference-implicit-zh', 'referent-zh-ambiguous', 'referent-en-unique'])
def test_model_free_wire_is_source_only_and_returns_canonical_analysis(key):
    rows, focus, raw, expected = sample(key)
    candidate, seen = assessor(focus, raw)
    assert candidate(rows) == expected and len(seen) == 1
    messages = reference_messages(focus, source_selections=True)
    assert messages[0].content == REFERENCE_SELECTIONS_PROMPT
    assert json.loads(messages[1].content) == reference_candidate_view(focus)
    wire = membership_wire_payload(rows, MODEL, assessment_kind=KIND, max_tokens=1024)
    assert wire['messages'] == [dict(role=m.role, content=m.content) for m in messages]
    assert wire['format'] == reference_selection_schema(focus)
    assert 'descriptions' not in json.loads(messages[1].content)


def test_scope_is_copied_identity_binds_source_and_other_scope_is_rejected():
    rows, focus, raw, _ = sample()
    original = deepcopy(focus)
    candidate, seen = assessor(focus, raw)
    focus['focus_utterance']['text'] += ' Extra.'
    other, _ = assessor(focus, raw)
    assert candidate.identity != other.identity
    assert candidate(rows)
    changed = deepcopy(rows)
    for row in changed:
        row['pair'] = (json.dumps(focus), row['pair'][1])
    with pytest.raises(ValueError, match='pinned focus'): candidate(changed)
    assert len(seen) == 1 and original != focus


@pytest.mark.parametrize('when', ['before', 'during', 'missing_hash'])
def test_schema_mutation_or_wrong_attestation_cannot_publish_analysis(when):
    rows, focus, raw, _ = sample()
    candidate, seen = assessor(focus, raw)
    original = candidate.router.chat
    def chat(messages, **kwargs):
        result = original(messages, **kwargs)
        if when == 'during': candidate.provider.response_schema['additionalProperties'] = True
        if when == 'missing_hash': result = replace(result, metadata={k:v for k,v in result.metadata.items()
            if k != 'response_schema_sha256'})
        return result
    candidate.router.chat = chat
    if when == 'before': candidate.provider.response_schema = None
    with pytest.raises(ValueError): candidate(rows)
    assert len(seen) == (0 if when == 'before' else 1)


def test_schema_input_overhead_is_counted_before_dispatch():
    rows, focus, raw, _ = sample()
    candidate, seen = assessor(focus, raw)
    candidate.max_input = request_input_bound(reference_messages(focus, source_selections=True))
    candidate._configuration = candidate._config()
    with pytest.raises(ValueError, match='input bound'): candidate(rows)
    assert seen == []


@pytest.mark.parametrize('damage', [None, 'source', 'coverage', 'transport', 'audit_schema', 'audit_analysis'])
def test_real_executor_router_capture_and_auditor_with_disclosed_software_http(tmp_path, monkeypatch, damage):
    state = old_setup(tmp_path, monkeypatch)
    plan = state.read(state.plan_path); plan['assessment_kind'] = KIND
    state.save(state.plan_path, plan)
    entry = state.schedule[0]
    focus = json.loads(state.args['rows_for'](state.cases[entry['case']], entry)[0]['pair'][0])
    units = reference_candidate_view(focus)['turns'][-1]['units']
    first_person = next(i for i, text in units if text == '我')
    wire = dict(coverage='enumerated', readings=[[dict(span=[first_person, first_person],
        kind='deictic', target=dict(speaker='user'))]])
    original = OllamaProvider._open
    def software_http(provider, request, *, timeout):
        if request.full_url.endswith('/chat') and damage == 'transport':
            raise TimeoutError('Declared software transport failure')
        response = original(provider, request, timeout=timeout)
        if request.full_url.endswith('/chat'):
            native = json.load(response); value = deepcopy(wire)
            if damage == 'source': value['readings'][0][0]['span'] = [999, 999]
            if damage == 'coverage':
                value['coverage'] = 'open'  # No null: reject, never invent one.
            native['message']['content'] = json.dumps(value, ensure_ascii=False)
            return BytesIO(json.dumps(native).encode())
        return response
    monkeypatch.setattr(OllamaProvider, '_open', software_http)
    run_local_membership_diagnostic(**state.args)
    result = state.read(state.result_path)
    obs, events = state.read(state.out/'observations.json'), state.read(state.out/'transport.json')
    assert result['native_attempts'] == (1 if damage == 'transport' else 2), result.get('error')
    assert result['completed'] == (damage != 'transport')
    assert result['paid_api_usd'] == '0' and not result['qualified']
    assert all('labels' not in o for o in obs)
    if damage == 'audit_schema': obs[0]['native_calls'][0]['completion']['metadata']['response_schema_sha256'] = 'wrong'
    if damage == 'audit_analysis': obs[0]['analysis']['coverage'] = 'unsupported'
    args = dict(rows_for=lambda e:state.args['rows_for'](state.cases[e['case']], e),
        snapshot=lambda p:state.read(state.out/p), model=MODEL, digest=DIGEST,
        max_native=2, max_metadata=64, max_tokens=1024, max_input=7168, assessment_kind=KIND)
    if damage in ('audit_schema', 'audit_analysis'):
        with pytest.raises(ValueError): audit_membership_capture(events, obs, state.schedule, **args)
    else:
        replay = audit_membership_capture(events, obs, state.schedule, **args)
        assert len(replay['predictions']) == (2 if damage is None else 0)
    with pytest.raises(FileExistsError): run_local_membership_diagnostic(**state.args)


def test_protocol_requires_specific_scope_and_schema_and_preserves_other_modes():
    _, focus, raw, _ = sample()
    router, seen = extraction_router(raw)
    with pytest.raises(ValueError):
        LocalChatMembershipAssessor(router, format_version=4, assessment_kind=KIND)
    with pytest.raises(ValueError, match='response schema'):
        LocalChatMembershipAssessor(router, format_version=4, assessment_kind=KIND, reference_scope=focus)
    with pytest.raises(ValueError, match='scope requires'):
        LocalChatMembershipAssessor(router, format_version=4, assessment_kind='reference-extraction-v1', reference_scope=focus)
    with pytest.raises(ValueError, match='Conflicting'):
        reference_messages(focus, source_selections=True, explicit_ids=True)
    assert seen == []
