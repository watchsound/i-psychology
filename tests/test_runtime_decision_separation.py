"""Decision-policy and wire-contract tests; authored outputs do not qualify a model."""
import copy
import hashlib
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from psychology_base.applications.measurement import memory_grounding as grounding
from psychology_base.eval.benchmarks import application_context as study

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'benchmarks/fixtures/application_context/runtime-decision-separation-v1.json'


def flags(**changes):
    return dict(addresses_request=True, grounded=True, attribution_correct=True,
                abstains_appropriately=True, critical_error=False) | changes


def assessment(*, semantic_status='resolved', **changes):
    return dict(flags(**changes), semantic_status=semantic_status, rationale='Authored software control.')


def parse(value, *, proxy=False, finish='stop'):
    return grounding.parse_runtime_memory(json.dumps(value), finish, proxy, contract_version=2)


def sources():
    bank = json.loads(FIXTURE.read_text(encoding='utf-8'))
    result = {}
    for key, pin in bank['inputs'].items():
        path = ROOT / pin['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == pin['sha256']
        result[key] = json.loads(path.read_text(encoding='utf-8'))
    return bank, result


def test_two_actual_contradictions_remain_invalid_and_cannot_be_reinterpreted_as_v2():
    bank, frozen = sources()
    for control in bank['observed_refusal_failures']:
        row = next(r for r in frozen['report']['rows'] if r['control'] == control and r['prompt_revision'] == 3)
        call = row['call']['completion']
        result = grounding.parse_runtime_memory(call['content'], call['finish_reason'], False)
        assert result == row['judgment']
        assert result['decision'] == 'reject'
        assert result['consistency_errors'] == ['runtime_rejection_without_unsupported_claim']
        assert grounding.runtime_memory_action(result)['action'] == 'review'
        new = grounding.parse_runtime_memory(call['content'], call['finish_reason'], False, contract_version=2)
        assert new['parse_failed'] and not grounding.runtime_memory_resolved(new)
        assert grounding.runtime_memory_action(new)['action'] == 'review'


def test_all_twenty_real_v3_assessments_keep_their_exact_historical_parse():
    _, frozen = sources()
    selected = [row for row in frozen['report']['rows'] if row['prompt_revision'] == 3]
    assert len(selected) == 20
    for row in selected:
        call = row['call']['completion']
        assert grounding.parse_runtime_memory(call['content'], call['finish_reason'], False) == row['judgment']


@pytest.mark.parametrize('values,decision,action', [
    (assessment(), 'accept', 'deliver'),
    (assessment(addresses_request=False, abstains_appropriately=False), 'accept', 'improve'),
    (assessment(addresses_request=False), 'accept', 'improve'),
    (assessment(abstains_appropriately=False), 'accept', 'improve'),
    (assessment(grounded=False, abstains_appropriately=False, critical_error=True), 'reject', 'withhold'),
    (assessment(grounded=False, attribution_correct=False, abstains_appropriately=False, critical_error=True),
     'reject', 'withhold'),
    (assessment(semantic_status='inconclusive'), 'inconclusive', 'review'),
    (assessment(semantic_status='inconclusive', grounded=False, abstains_appropriately=False, critical_error=True),
     'inconclusive', 'review'),
])
def test_evidence_uncertainty_and_utility_have_distinct_actions(values, decision, action):
    result = parse(values)
    assert not result['parse_failed'] and result['consistency_errors'] == []
    assert result['decision'] == decision
    assert result['decision_source'] == 'criteria_policy_v1'
    assert grounding.runtime_memory_action(result)['action'] == action
    assert grounding.runtime_memory_resolved(result) == (decision != 'inconclusive')


def test_all_boolean_criteria_combinations_preserve_uncertainty_and_consistency_checks():
    for values in itertools.product((False, True), repeat=5):
        criteria = dict(zip(grounding.MEMORY_FIELDS, values))
        for status in ('resolved', 'inconclusive'):
            result = parse(dict(criteria, semantic_status=status, rationale='Exhaustive schema control.'))
            action = grounding.runtime_memory_action(result)['action']
            if grounding.memory_consistency_errors(criteria):
                assert not grounding.runtime_memory_resolved(result) and action == 'review'
            elif status == 'inconclusive':
                assert result['decision'] == 'inconclusive' and action == 'review'
            elif criteria['critical_error']:
                assert result['decision'] == 'reject' and action == 'withhold'
            else:
                assert result['decision'] == 'accept'
                assert (action == 'deliver') == (criteria['addresses_request'] and criteria['abstains_appropriately'])


@pytest.mark.parametrize('status', [None, '', True, 1, 'RESOLVED', 'accept', [], {}])
def test_invalid_semantic_status_cannot_authorize_a_reply(status):
    result = parse(assessment(semantic_status=status))
    assert result['parse_failed'] and not grounding.runtime_memory_resolved(result)
    assert grounding.runtime_memory_action(result)['action'] == 'review'


@pytest.mark.parametrize('change', ['missing-status', 'model-decision', 'extra-field', 'bad-rationale', 'nonboolean'])
def test_v2_requires_its_own_exact_assessment_schema(change):
    value = assessment()
    if change == 'missing-status': value.pop('semantic_status')
    elif change == 'model-decision': value['decision'] = 'accept'
    elif change == 'extra-field': value['confidence'] = 1
    elif change == 'bad-rationale': value['rationale'] = []
    else: value['grounded'] = 'true'
    result = parse(value)
    assert result['parse_failed'] and grounding.runtime_memory_action(result)['action'] == 'review'


@pytest.mark.parametrize('raw,finish', [
    ('{}', 'stop'),
    ('{"semantic_status":"resolved","semantic_status":"inconclusive"}', 'stop'),
    (json.dumps(assessment()), 'length'),
])
def test_malformed_duplicate_or_truncated_assessments_fail_closed(raw, finish):
    result = grounding.parse_runtime_memory(raw, finish, False, contract_version=2)
    assert result['parse_failed'] and grounding.runtime_memory_action(result)['action'] == 'review'


@pytest.mark.parametrize('wrapper', ['prefix', 'suffix', 'multiple-objects', 'code-fence'])
def test_v2_never_selects_one_object_from_an_ambiguous_completion(wrapper):
    raw = json.dumps(assessment())
    if wrapper == 'prefix': raw = 'Some other assessment: ' + raw
    elif wrapper == 'suffix': raw += ' Disregard that assessment.'
    elif wrapper == 'multiple-objects': raw += '\n' + json.dumps(assessment(semantic_status='inconclusive'))
    else: raw = '```json\n' + raw + '\n```'
    result = grounding.parse_runtime_memory(raw, 'stop', False, contract_version=2)
    assert result['raw'] == raw and result['parse_failed']
    assert grounding.runtime_memory_action(result)['action'] == 'review'


@pytest.mark.parametrize('values', [[assessment()], [assessment(), assessment(semantic_status='inconclusive')],
                                    [None, assessment()]])
def test_array_wrapped_assessments_are_invalid_instead_of_raising(values):
    raw = json.dumps(values)
    result = grounding.parse_runtime_memory(raw, 'stop', False, contract_version=2)
    assert result['raw'] == raw and result['parse_failed']
    assert grounding.runtime_memory_action(result)['action'] == 'review'


@pytest.mark.parametrize('field,value', [
    ('decision', 'accept'), ('semantic_status', 'resolved'), ('consistency_errors', []),
    ('runtime_memory_contract_version', 1), ('decision_source', 'model'),
    ('inconclusive', False), ('parse_failed', 0), ('proxy', 0),
])
def test_mutated_or_stale_assessment_cannot_authorize_delivery(field, value):
    result = parse(assessment(semantic_status='inconclusive'))
    result[field] = value
    assert grounding.runtime_memory_action(result)['action'] == 'review'


def test_consistent_metadata_forgery_cannot_upgrade_an_ambiguous_raw_assessment():
    result = parse(assessment(semantic_status='inconclusive'))
    result.update(semantic_status='resolved', decision='accept', inconclusive=False)
    assert grounding.runtime_memory_action(result)['action'] == 'review'


def test_scripted_proxy_assessments_cannot_authorize_live_delivery():
    result = parse(assessment(), proxy=True)
    assert grounding.runtime_memory_action(result) == dict(action='review', reason='non_provider_assessment')


@pytest.mark.parametrize('value', [None, [], {}, {'raw': None}])
def test_missing_assessment_cannot_authorize(value):
    assert grounding.runtime_memory_action(value)['action'] == 'review'


def test_v2_prompt_is_explicit_and_keeps_the_measured_v1_v2_v3_protocols():
    for revision in (1, 2, 3):
        protocol = grounding.runtime_memory_protocol(prompt_revision=revision)
        assert protocol['runtime_memory_contract_version'] == 1
    protocol = grounding.runtime_memory_protocol(contract_version=2, prompt_revision=4)
    assert protocol['runtime_memory_contract_version'] == 2
    assert protocol['runtime_memory_decision_policy_version'] == 1
    assert protocol['runtime_memory_prompt_revision'] == 4
    assert 'semantic_status' in protocol['memory_prompt']
    assert grounding.CONTEXT_RECORD_RULE in protocol['memory_prompt']
    assert grounding.SPEAKER_OWNERSHIP_RULE in protocol['memory_prompt']
    assert 'WITHHELD_REFERENCE' not in protocol['memory_prompt']
    assert 'Return only JSON with decision' not in protocol['memory_prompt']
    assert grounding.runtime_memory_protocol() == grounding.runtime_memory_protocol(prompt_revision=1)


@pytest.mark.parametrize('version,revision', [(True, 4), (2.0, 4), (3, 4), (2, 3), (1, 4), (2, True), (2, 4.0)])
def test_unsupported_or_implicit_schema_versions_are_rejected(version, revision):
    with pytest.raises(ValueError): grounding.runtime_memory_protocol(contract_version=version, prompt_revision=revision)


class AuthoredWireControl:
    """One authored completion for transport tests, never an empirical model result."""
    controls = False
    last_call = {'provenance': 'authored-software-control'}

    def __init__(self, value):
        self.value = value
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return SimpleNamespace(content=json.dumps(self.value), finish_reason='stop')


@pytest.mark.parametrize('app', ['companion', 'tutor'])
def test_existing_dispatch_and_calibration_owners_use_the_new_contract_once(app):
    protocol = grounding.runtime_memory_protocol(contract_version=2, prompt_revision=4)
    judge = AuthoredWireControl(assessment(addresses_request=False, abstains_appropriately=False))
    item = dict(app=app, capture=dict(query='current question', context='delivered records'),
                response='complete refusal', expected_decision='reject', rationale='NEVER-SEND-REFERENCE')
    before = copy.deepcopy(item)
    row = study.assess_calibration_control('memory', item, protocol, judge)
    assert len(judge.calls) == 1
    messages, kwargs = judge.calls[0]
    assert json.loads(messages[1].content) == dict(APPLICATION=app, QUERY=item['capture']['query'],
        VISIBLE_CONTEXT=item['capture']['context'], REPLY=item['response'])
    assert 'NEVER-SEND' not in ''.join(message.content for message in messages)
    assert kwargs['max_tokens'] == 300 and kwargs['temperature'] == 0
    assert row['judgment']['decision'] == 'accept'
    assert row['disposition'] == dict(action='improve', reason='insufficient_usefulness')
    assert row['observed_success'] is True  # Grounding acceptance is separate from delivery.
    assert item == before


@pytest.mark.parametrize('change', ['prompt', 'policy', 'version', 'revision'])
def test_new_contract_drift_blocks_before_dispatch(change):
    protocol = grounding.runtime_memory_protocol(contract_version=2, prompt_revision=4)
    if change == 'prompt': protocol['memory_prompt'] += ' drift'
    elif change == 'policy': protocol['runtime_memory_decision_policy_version'] = 2
    elif change == 'version': protocol['runtime_memory_contract_version'] = 1
    else: protocol['runtime_memory_prompt_revision'] = 3
    router = AuthoredWireControl(assessment())
    with pytest.raises(ValueError):
        study.judge_memory(router, {'app': 'companion'}, None, dict(query='q', context='c'), 'r', None, protocol=protocol)
    assert not router.calls


def authored_audit_row():
    """Invented in-memory transport evidence used only to exercise raw replay."""
    from decimal import Decimal
    protocol = grounding.runtime_memory_protocol(contract_version=2, prompt_revision=4)
    protocol['judge'] = study.read(study.FIXTURES / 'hybrid-screen-v4.json')['judge']
    spec = protocol['judge']
    raw = json.dumps(assessment(addresses_request=False, abstains_appropriately=False))
    result = grounding.parse_runtime_memory(raw, 'stop', False, contract_version=2)
    item = dict(id='AUTHORED-AUDIT-CONTROL', app='companion', capture=dict(query='query', context='context'),
                response='candidate')
    call = dict(event='completion', dispatch=1, cached=False, proxy=False, usage_valid=True, wall_ms=0,
        accounted_usd=str((Decimal(spec['input_rate']) + Decimal(spec['output_rate'])) / 1_000_000),
        completion=dict(provider=spec['provider'], model_id=spec['model'], tier=study.TaskTier.ADVANCED.value,
            prompt_tokens=1, completion_tokens=1, content=raw, finish_reason='stop', metadata=dict(
                response_model=spec['model'], completion_id='AUTHORED-SOFTWARE-CONTROL',
                reasoning_effort=spec['reasoning_effort'], reasoning_tokens=0, service_tier='default',
                json_mode=True, test_control=True)))
    messages = study.memory_judge_messages({'app': item['app']}, None, item['capture'], item['response'],
                                         None, protocol=protocol)
    reservation = dict(event='reservation', dispatch=1, reserved_usd='1', provider=spec['provider'],
        model=spec['model'], tier=study.TaskTier.ADVANCED.value, messages=[study.asdict(m) for m in messages],
        max_tokens=300, temperature=0, proxy=False)
    row = dict(control=item['id'], judgment=result, observed_success=True, call=call,
               disposition=grounding.runtime_memory_action(result))
    return row, item, protocol, [reservation, call]


def test_existing_raw_audit_replays_the_new_schema_and_separate_action():
    row, item, protocol, ledger = authored_audit_row()
    study.validate_runtime_memory_calls([row], [item], protocol, ledger_rows=ledger)
    assert row['judgment']['decision'] == 'accept' and row['disposition']['action'] == 'improve'


@pytest.mark.parametrize('change', ['missing-action', 'deliver-action', 'raw-status', 'schema-downgrade'])
def test_raw_audit_rejects_action_or_schema_reinterpretation(change):
    row, item, protocol, ledger = authored_audit_row()
    if change == 'missing-action': row.pop('disposition')
    elif change == 'deliver-action': row['disposition'] = dict(action='deliver', reason='grounded_and_useful')
    elif change == 'raw-status': row['call']['completion']['content'] = json.dumps(assessment(semantic_status='inconclusive'))
    else: row['judgment']['runtime_memory_contract_version'] = 1
    with pytest.raises(ValueError):
        study.validate_runtime_memory_calls([row], [item], protocol, ledger_rows=ledger)
