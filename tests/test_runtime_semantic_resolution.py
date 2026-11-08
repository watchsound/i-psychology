"""Actual failure preservation and prospective prompt plumbing, never LLM accuracy."""
import hashlib
import json

import pytest

from psychology_base.applications.measurement import memory_grounding as grounding
from psychology_base.eval.benchmarks import application_context as study
from psychology_base.eval.benchmarks import runtime_report_diagnostic as diagnostic
from tests.test_runtime_decision_separation import AuthoredWireControl, assessment, authored_audit_row


@pytest.fixture(scope='module')
def frozen():
    bank = study.read(study.FIXTURES / 'runtime-semantic-resolution-v1.json')
    sources = {}
    for key, pin in bank['inputs'].items():
        path = study.ROOT / pin['path']
        assert study.shared.digest(path) == pin['sha256']
        sources[key] = study.read(path)
    assert diagnostic.items() == sources['reused_development']
    cases = {row['id']: row for row in diagnostic.items(revision=3)}
    original = {row['id']: row for row in sources['controls']['controls']}
    for identifier in bank['exact_observed_control_ids']:
        assert all(cases[identifier][key] == original[identifier][key]
                   for key in ('capture', 'response', 'expected_decision', 'expected_criteria'))
    return bank, sources, cases


def test_exact_real_failure_remains_a_failure_and_not_a_repaired_model_result(frozen):
    bank, sources, _ = frozen
    row = next(row for row in sources['report']['rows'] if row['control'] == bank['failed_control'])
    actual = row['call']['completion']
    result = grounding.parse_runtime_memory(actual['content'], actual['finish_reason'], False, contract_version=2)
    assert result == row['judgment']
    assert result['semantic_status'] == 'resolved' and result['decision'] == 'accept'
    assert grounding.runtime_memory_action(result) == row['disposition'] == dict(
        action='deliver', reason='grounded_and_useful')
    reference = next(row for row in sources['reference_review']['judgments']
                     if row['control'] == bank['failed_control'])
    assert reference['decision'] == 'inconclusive' and reference['criteria'] is None
    assert not sources['report']['passed'] and not sources['report']['runtime_enabled']


def test_all_three_real_assessments_keep_their_raw_parse_action_and_request_audit(frozen):
    _, sources, _ = frozen
    report, protocol = sources['report'], sources['plan']['protocol']
    assert len(report['rows']) == 3
    for row in report['rows']:
        completion = row['call']['completion']
        assert grounding.parse_runtime_memory(completion['content'], completion['finish_reason'],
            False, contract_version=2) == row['judgment']
        assert grounding.runtime_memory_action(row['judgment']) == row['disposition']
    study.validate_runtime_memory_calls(report['rows'], sources['controls']['controls'], protocol)


def test_exposed_regression_packet_reuses_sources_without_duplicate_requests(frozen):
    bank, _, cases = frozen
    assert bank['qualification_eligible'] is False and bank['provider_calls'] == 0
    assert len(cases) == 49 and len(bank['additional_cases']) == 27
    assert bank['equivalent_variants'] == {'en-tutor-competing-affirmation': bank['failed_control']}
    assert all(item['qualification_eligible'] is False for item in cases.values())
    keys = {study.shared.stable_key(grounding.runtime_memory_payload(item['app'],
        item['capture']['query'], item['capture']['context'], item['response'])) for item in cases.values()}
    assert len(keys) == len(cases)
    matrix = {(row['language'], row['app'], row['contrast']) for row in bank['additional_cases']}
    matrix.add(('en', 'tutor', 'competing-affirmation'))
    assert len(matrix) == 28
    assert {contrast for _, _, contrast in matrix} == {
        'competing-affirmation', 'scoped-other-role', 'explicit-retraction',
        'independent-own-support', 'definite-error-after-ambiguity', 'unsupported-never', 'quoted-affirmation'}
    assert all(sum(lang == language and app == role for lang, app, _ in matrix) == 7
               for language in ('en', 'zh') for role in ('companion', 'tutor'))


def test_original_protocol_bytes_and_default_selection_are_preserved(frozen):
    bank, _, _ = frozen
    for revision in (1, 2, 3, 4):
        protocol = grounding.runtime_memory_protocol(prompt_revision=revision,
            contract_version=2 if revision == 4 else 1)
        actual = hashlib.sha256(json.dumps(protocol, sort_keys=True, ensure_ascii=False,
            separators=(',', ':')).encode()).hexdigest()
        assert actual == bank['historical_protocol_sha256'][str(revision)]
    assert grounding.runtime_memory_protocol() == grounding.runtime_memory_protocol(prompt_revision=1)


def test_claim_evidence_precedes_resolution_without_a_second_schema_or_policy():
    old = grounding.runtime_memory_protocol(contract_version=2, prompt_revision=4)
    new = grounding.runtime_memory_protocol(contract_version=2, prompt_revision=5)
    prompt = new['memory_prompt']
    assert new == dict(old, memory_prompt=prompt, runtime_memory_prompt_revision=5)
    assert prompt.index('2. Claim evidence:') < prompt.index('3. Resolve the complete reply:')
    assert prompt.index('3. Resolve the complete reply:') < prompt.index('4. Assess the five existing criteria')
    for section in ('4. Assess the five existing criteria separately.',):
        assert old['memory_prompt'].partition(section)[2] == prompt.partition(section)[2]
    assert grounding.CONTEXT_RECORD_RULE in prompt and grounding.SPEAKER_OWNERSHIP_RULE in prompt
    assert 'its you names APPLICATION' in prompt
    assert "unquoted own speech, I names APPLICATION" in prompt
    assert 'Keep quoted and reported speakers in their own scope' in prompt
    assert 'role is not itself a retraction' in prompt
    assert 'every plausible reading' in prompt
    assert 'Return only one JSON object with semantic_status' in prompt


def test_all_exposed_requests_keep_one_four_field_boundary_without_gold_leakage(frozen):
    _, _, cases = frozen
    protocol = grounding.runtime_memory_protocol(contract_version=2, prompt_revision=5)
    for item in cases.values():
        messages = study.memory_judge_messages(item, False, item['capture'], item['response'],
            'NEVER-SEND-REFERENCE', protocol=protocol)
        assert messages[0].content == protocol['memory_prompt']
        assert json.loads(messages[1].content) == grounding.runtime_memory_payload(item['app'],
            item['capture']['query'], item['capture']['context'], item['response'])
        assert 'NEVER-SEND' not in ''.join(message.content for message in messages)
        assert study.request_input_bound(messages) <= 12000


@pytest.mark.parametrize('app', ['companion', 'tutor'])
@pytest.mark.parametrize('status,action', [('inconclusive', 'review'), ('resolved', 'deliver')])
def test_existing_dispatch_uses_the_candidate_once_with_authored_wire_controls(app, status, action):
    protocol = grounding.runtime_memory_protocol(contract_version=2, prompt_revision=5)
    judge = AuthoredWireControl(assessment(semantic_status=status))
    result = study.judge_memory(judge, {'app': app}, False, dict(query='question', context='evidence'),
        'candidate', 'NEVER-SEND-REFERENCE', protocol=protocol)
    assert len(judge.calls) == 1
    assert judge.calls[0][0][0].content == protocol['memory_prompt']
    assert grounding.runtime_memory_action(result)['action'] == action


def test_candidate_reuses_raw_call_and_action_auditing_with_authored_wire_evidence():
    row, item, protocol, ledger = authored_audit_row()
    protocol.update(grounding.runtime_memory_protocol(contract_version=2, prompt_revision=5))
    ledger[0]['messages'] = [study.asdict(message) for message in study.memory_judge_messages(
        item, False, item['capture'], item['response'], None, protocol=protocol)]
    study.validate_runtime_memory_calls([row], [item], protocol, ledger_rows=ledger)
    row['disposition'] = dict(action='deliver', reason='grounded_and_useful')
    with pytest.raises(ValueError):
        study.validate_runtime_memory_calls([row], [item], protocol, ledger_rows=ledger)


@pytest.mark.parametrize('field,value', [('memory_prompt', 'drift'),
    ('runtime_memory_prompt_revision', 4), ('runtime_memory_decision_policy_version', 2)])
def test_candidate_prompt_or_policy_drift_stops_before_any_wire_call(field, value):
    protocol = grounding.runtime_memory_protocol(contract_version=2, prompt_revision=5)
    protocol[field] = value
    judge = AuthoredWireControl(assessment())
    with pytest.raises(ValueError):
        study.judge_memory(judge, {'app': 'tutor'}, False, dict(query='q', context='c'), 'r', None,
            protocol=protocol)
    assert not judge.calls


@pytest.mark.parametrize('contract,revision', [(1, 5), (2, 6), (2, True), (2, 5.0)])
def test_candidate_needs_an_explicit_supported_version(contract, revision):
    with pytest.raises(ValueError):
        grounding.runtime_memory_protocol(contract_version=contract, prompt_revision=revision)
