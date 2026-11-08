"""Runtime assessor transport/schema checks, never evidence of semantic accuracy."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from psychology_base.applications.measurement import memory_grounding as grounding
from psychology_base.eval.benchmarks import application_context as study


def criteria(**changes):
    return dict(addresses_request=True, grounded=True, attribution_correct=True,
                abstains_appropriately=True, critical_error=False) | changes


class Capture:
    controls = False

    def __init__(self, verdict):
        self.verdict = verdict
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return SimpleNamespace(content=json.dumps(self.verdict), finish_reason='stop')


def test_runtime_composition_is_explicit_and_preserves_historical_default():
    protocol = grounding.runtime_memory_protocol()
    assert protocol['grading_contract_version'] == 2
    assert protocol['runtime_memory_contract_version'] == 1
    prompt = protocol['memory_prompt']
    assert 'WITHHELD_REFERENCE' not in prompt and 'SOURCE_ABSENT' not in prompt
    assert grounding.CONTEXT_RECORD_RULE in prompt
    assert grounding.SPEAKER_OWNERSHIP_RULE in prompt
    assert grounding.MEMORY_REPORT_BOUNDARIES_V1 in prompt
    assert 'inconclusive' in prompt and 'complete' in prompt
    protocol['memory_prompt'] = 'changed'
    assert grounding.runtime_memory_protocol()['memory_prompt'] == prompt
    assert 'runtime_memory_contract_version' not in grounding.revised_grading_protocol({})


@pytest.mark.parametrize('app', ['companion', 'tutor'])
def test_existing_judge_boundary_dispatches_only_actual_runtime_evidence(app):
    router = Capture(dict(criteria(), decision='accept'))
    capture = dict(query='Did you ask?', context='exact\ncontext\u2028with data')
    before = copy.deepcopy(capture)
    result = study.judge_memory(router, {'app': app}, 'NEVER-SEND-ABSENCE-LABEL', capture,
                                'complete candidate', 'NEVER-SEND-REFERENCE',
                                protocol=grounding.runtime_memory_protocol())
    assert result['decision'] == 'accept' and study.judgment_resolved(result)
    assert len(router.calls) == 1
    messages, options = router.calls[0]
    assert json.loads(messages[1].content) == dict(APPLICATION=app, QUERY=capture['query'],
        VISIBLE_CONTEXT=capture['context'], REPLY='complete candidate')
    assert 'NEVER-SEND' not in ''.join(message.content for message in messages)
    assert options['max_tokens'] == 300 and options['temperature'] == 0
    assert capture == before


@pytest.mark.parametrize('decision', [None, '', True, 1, 'ACCEPT', [], {}])
def test_missing_or_malformed_decision_cannot_resolve(decision):
    verdict = dict(criteria(), decision=decision)
    result = grounding.parse_runtime_memory(json.dumps(verdict), 'stop', False)
    assert result['parse_failed'] and not study.judgment_resolved(result)


def test_inconclusive_is_distinct_from_fabrication_and_never_resolved():
    result = grounding.parse_runtime_memory(json.dumps(dict(criteria(), decision='inconclusive')), 'stop', False)
    assert not result['parse_failed'] and result['criteria']['grounded']
    assert not result['criteria']['critical_error']
    assert result['inconclusive'] and not study.judgment_resolved(result)


@pytest.mark.parametrize('decision,flags', [
    ('accept', criteria(grounded=False, abstains_appropriately=False, critical_error=True)),
    ('reject', criteria()),
])
def test_decision_cannot_override_criteria(decision, flags):
    result = grounding.parse_runtime_memory(json.dumps(dict(flags, decision=decision)), 'stop', False)
    assert result['consistency_errors'] and not study.judgment_resolved(result)
    # A forged/stale error list cannot make the conflicting decision resolved.
    result['consistency_errors'] = []
    assert not study.judgment_resolved(result)


def test_resolved_rejection_and_safe_low_relevance_remain_distinct():
    wrong = criteria(grounded=False, abstains_appropriately=False, critical_error=True)
    rejected = grounding.parse_runtime_memory(json.dumps(dict(wrong, decision='reject')), 'stop', False)
    assert study.judgment_resolved(rejected) and rejected['decision'] == 'reject'
    safe = grounding.parse_runtime_memory(json.dumps(dict(criteria(addresses_request=False), decision='accept')), 'stop', False)
    assert study.judgment_resolved(safe) and not safe['criteria']['addresses_request']


@pytest.mark.parametrize('raw,finish', [
    ('{}', 'stop'), ('{"decision":"accept","decision":"reject"}', 'stop'),
    (json.dumps(dict(criteria(), decision='accept')), 'length'),
    (json.dumps(dict(criteria(), decision='accept', grounded='true')), 'stop'),
])
def test_strict_shared_parser_failures_remain_unresolved(raw, finish):
    result = grounding.parse_runtime_memory(raw, finish, False)
    assert result['parse_failed'] and not study.judgment_resolved(result)


@pytest.mark.parametrize('mutation', ['boolean-version', 'unknown-version', 'legacy-parser', 'changed-prompt'])
def test_runtime_contract_drift_rejects_before_dispatch(mutation):
    protocol = grounding.runtime_memory_protocol()
    if mutation == 'boolean-version':
        protocol['runtime_memory_contract_version'] = True
    elif mutation == 'unknown-version':
        protocol['runtime_memory_contract_version'] = 2
    elif mutation == 'legacy-parser':
        protocol['grading_contract_version'] = 1
    else:
        protocol['memory_prompt'] += ' changed'
    router = Capture({})
    with pytest.raises(ValueError, match='runtime|Runtime'):
        study.judge_memory(router, {'app': 'companion'}, None, {'query': '', 'context': ''}, '', None, protocol=protocol)
    assert not router.calls


@pytest.mark.parametrize('app,query,context,reply', [
    ('assistant', '', '', ''), ('companion', None, '', ''),
    ('tutor', '', [], ''), ('tutor', '', '', {'text': 'candidate'}),
])
def test_payload_rejects_unknown_application_or_nontext(app, query, context, reply):
    with pytest.raises(ValueError):
        grounding.runtime_memory_payload(app, query, context, reply)


@pytest.mark.parametrize('field,value', [
    ('runtime_memory_contract_version', True), ('runtime_memory_contract_version', 2),
    ('contract_version', 2.0), ('inconclusive', True), ('inconclusive', 0),
    ('decision', None), ('parse_failed', 0),
])
def test_stale_or_forged_resolution_metadata_cannot_authorize(field, value):
    result = grounding.parse_runtime_memory(json.dumps(dict(criteria(), decision='accept')), 'stop', False)
    result[field] = value
    assert not grounding.runtime_memory_resolved(result)


ROOT = Path(__file__).resolve().parents[1]
BANK = json.loads((ROOT / 'benchmarks/fixtures/application_context/runtime-report-boundaries-v1.json').read_text(encoding='utf-8'))


@pytest.mark.parametrize('case', BANK['cases'], ids=lambda case: case['id'])
def test_authored_contrasts_have_valid_contracts_and_no_provider_label_leakage(case):
    """Checks request construction and authored label schema, not LLM judgments."""
    class StopBeforeDispatch(Exception):
        pass
    class RequestOnly:
        controls = False
        def chat(self, messages, **kwargs):
            self.payload = json.loads(messages[1].content)
            self.prompt = messages[0].content
            raise StopBeforeDispatch
    capture = RequestOnly()
    context = '\n'.join(json.dumps(record, ensure_ascii=False) for record in case['records'])
    with pytest.raises(StopBeforeDispatch):
        study.judge_memory(capture, {'app': case['app']}, None,
                           {'query': case['query'], 'context': context}, case['response'], None,
                           protocol=grounding.runtime_memory_protocol())
    assert capture.payload == dict(APPLICATION=case['app'], QUERY=case['query'],
                                  VISIBLE_CONTEXT=context, REPLY=case['response'])
    assert case['rationale'] not in capture.prompt
    if case['expected_decision'] == 'inconclusive':
        assert case['expected_criteria'] is None
    else:
        assert not grounding.memory_consistency_errors(case['expected_criteria'])
        verdict = grounding.parse_runtime_memory(json.dumps(dict(case['expected_criteria'], decision=case['expected_decision'])), 'stop', False)
        assert grounding.runtime_memory_resolved(verdict)
    assert BANK['qualification_eligible'] is False and BANK['empirical_model_results'] is False


def test_frozen_development_packet_retains_actual_replies_and_historical_results():
    import hashlib
    path = ROOT / '.data/application_context/runtime-verifier-preparation-v1/development-requests-v1.json'
    if not path.is_file():
        pytest.skip('Optional provisioned exact real-provider replies are unavailable')
    spec = json.loads((ROOT / 'docs/experiments/phase-9-application-context/runtime-verifier-qualification-spec-v1.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == spec['development']['sha256']
    packet = json.loads(path.read_text(encoding='utf-8'))
    old = ROOT / '.data/application_context/speaker-evidence-boundaries-v1'
    actual = {}
    for suffix in ('hosted-run', 'hosted-run-continuation-v1'):
        for line in (old / suffix / 'responses.jsonl').read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            actual[row['index']] = row
    history = json.loads((ROOT / 'docs/experiments/phase-9-application-context/speaker-evidence-boundaries-hosted-review-v1.json').read_text(encoding='utf-8'))
    assert len(packet['rows']) == 56
    for row in packet['rows']:
        assert row['qualification_eligible'] is False
        assert row['messages'][0]['content'] == grounding.runtime_memory_protocol()['memory_prompt']
        payload = json.loads(row['messages'][1]['content'])
        assert set(payload) == {'APPLICATION', 'QUERY', 'VISIBLE_CONTEXT', 'REPLY'}
        assert row['bounded_input_bytes'] <= 12000
        if row['origin'] == 'exact_saved_real_reply':
            assert payload['REPLY'] == actual[row['index']]['response']
            assert row['historical_outcome'] == history['rows'][row['index']]['outcome']
    assert history['counts'] == {'pass': 31, 'fail': 4, 'inconclusive': 5}
    assert packet['provider_calls'] == packet['model_completions'] == 0
    assert spec['status'] == 'frozen-requirements-not-executable'
    assert not spec['verifier_qualified'] and not spec['runtime_enabled']
