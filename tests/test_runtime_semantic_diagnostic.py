"""Paired development plumbing and raw audit; authored outputs are not model evidence."""
import copy
import importlib.util
import json
from collections import Counter

import pytest

from psychology_base.applications.measurement import memory_grounding as grounding
from psychology_base.eval.benchmarks import application_context as study
from psychology_base.eval.benchmarks import runtime_report_diagnostic as diagnostic
from tests.test_runtime_semantic_resolution import frozen as frozen_regressions
from tests.test_runtime_decision_separation import authored_audit_row


def runner():
    path = study.ROOT / 'benchmarks/scripts/phase9_runtime_report_diagnostic.py'
    spec = importlib.util.spec_from_file_location('semantic_diagnostic_cli_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def cases():
    return diagnostic.items(revision=3)


@pytest.fixture(scope='module')
def historical_ledger(tmp_path_factory):
    """Keep offline plan tests valid after the approved live budget is consumed."""
    entry = runner()
    plan = study.read(entry.profile(3).plan)
    prefix = b''.join(line for line in entry.LEDGER.read_bytes().splitlines(keepends=True)
                      if json.loads(line)['dispatch'] <= plan['baseline']['provider_attempts'])
    path = tmp_path_factory.mktemp('paired-historical-ledger') / 'ledger.jsonl'
    path.write_bytes(prefix)
    assert study.shared.digest(path) == plan['baseline_ledger_sha256']
    return path


def scripted_rows(cases):
    """Invented outputs strictly for metric tests, never persisted as real results."""
    by_id = {item['id']: item for item in cases}
    rows = []
    for entry in diagnostic.schedule(cases, revision=3):
        item = by_id[entry['control']]
        criteria = item['expected_criteria'] or dict(addresses_request=True, grounded=True,
            attribution_correct=True, abstains_appropriately=True, critical_error=False)
        value = dict(criteria, semantic_status='inconclusive' if item['expected_decision'] == 'inconclusive'
                     else 'resolved', rationale='AUTHORED METRIC CONTROL; NOT MODEL EVIDENCE')
        result = grounding.parse_runtime_memory(json.dumps(value), 'stop', True, contract_version=2)
        rows.append(dict(entry, judgment=result))
    return rows


def replace_output(row, **changes):
    value = json.loads(row['judgment']['raw'])
    value.update(changes)
    row['judgment'] = grounding.parse_runtime_memory(json.dumps(value), 'stop', True, contract_version=2)


def test_existing_case_owner_reuses_all_regressions_and_only_exposed_cases(cases, frozen_regressions):
    bank, sources, _ = frozen_regressions
    # Assert against pinned source rows, not another call to the new catalog owner.
    expected = {row['id']: row for row in sources['reused_development']}
    observed = set(bank['exact_observed_control_ids'])
    expected.update({row['id']: row for row in sources['controls']['controls'] if row['id'] in observed})
    expected.update({row['id']: row for row in bank['additional_cases']})
    assert len(cases) == 49 and {item['id'] for item in cases} == set(expected)
    assert all(item['qualification_eligible'] is False for item in cases)
    actual = {row['id']: row for row in cases}
    for identifier, item in expected.items():
        assert all(actual[identifier][key] == item[key] for key in
                   ('capture', 'response', 'expected_decision', 'expected_criteria'))
    assert bank['failed_control'] in actual
    assert {row['control'] for row in sources['report']['rows']} <= set(actual)
    payloads = [study.shared.stable_key(grounding.runtime_memory_payload(item['app'],
        item['capture']['query'], item['capture']['context'], item['response'])) for item in cases]
    assert len(set(payloads)) == 49


def test_full_paired_schedule_is_deterministic_and_counterbalances_arm_order(cases):
    schedule = diagnostic.schedule(cases, revision=3)
    assert len(schedule) == 98 and schedule == diagnostic.schedule(cases, revision=3)
    assert [row['index'] for row in schedule] == list(range(98))
    assert Counter(row['prompt_revision'] for row in schedule) == {4: 49, 5: 49}
    for index in range(49):
        pair = schedule[index * 2:index * 2 + 2]
        assert {row['control'] for row in pair} == {cases[index]['id']}
        assert [row['prompt_revision'] for row in pair] == ([4, 5] if index % 2 == 0 else [5, 4])


def test_arm_selection_cannot_leak_new_schema_metadata_into_a_legacy_arm():
    original = dict(grounding.runtime_memory_protocol(), screening=True)
    candidate = diagnostic.arm_protocol(original, 5)
    assert candidate['runtime_memory_contract_version'] == 2
    assert candidate['runtime_memory_prompt_revision'] == 5
    assert diagnostic.arm_protocol(candidate, 1) == original
    assert diagnostic.arm_protocol(candidate, 4) == dict(original,
        **grounding.runtime_memory_protocol(contract_version=2, prompt_revision=4))


def test_metrics_parse_the_actual_v2_schema_and_retain_all_denominators(cases):
    metrics = diagnostic.metrics(cases, scripted_rows(cases), revision=3)
    assert metrics['complete'] and metrics['total'] == metrics['completed'] == 98
    assert metrics['remaining'] == 0 and metrics['qualification_eligible'] is False
    assert metrics['candidate_reference_screen_passed'] and metrics['no_observed_regressions']
    assert metrics['paired'] == dict(total=49, completed=49, improved=0, regressed=0,
        both_match=49, neither_matches=0)
    for arm in metrics['arms'].values():
        assert arm['total'] == arm['completed'] == 49
        assert arm['settled_full_matches'] == arm['settled_total'] == 43
        assert arm['ambiguous_inconclusive'] == arm['ambiguous_total'] == 6
        assert arm['malformed_or_inconsistent'] == 0


def test_a_repaired_candidate_can_pass_its_screen_while_unchanged_baseline_fails(cases):
    rows = scripted_rows(cases)
    row = next(row for row in rows if row['control'] == 'rtq-3a394b90dfcbb264'
               and row['prompt_revision'] == 4)
    replace_output(row, semantic_status='resolved')
    metrics = diagnostic.metrics(cases, rows, revision=3)
    assert not metrics['all_reference_matches']
    assert metrics['candidate_reference_screen_passed'] and metrics['no_observed_regressions']
    assert metrics['paired']['improved'] == 1 and metrics['paired']['regressed'] == 0


def test_candidate_regression_and_partial_pairs_cannot_receive_passing_credit(cases):
    rows = scripted_rows(cases)
    candidate = next(row for row in rows if row['prompt_revision'] == 5)
    replace_output(candidate, semantic_status='inconclusive')
    metrics = diagnostic.metrics(cases, rows, revision=3)
    assert metrics['paired']['regressed'] == 1
    assert not metrics['candidate_reference_screen_passed'] and not metrics['no_observed_regressions']
    empty = diagnostic.metrics(cases, [], revision=3)
    partial = diagnostic.metrics(cases, rows[:1], revision=3)
    assert empty['remaining'] == 98 and partial['remaining'] == 97
    assert empty['paired']['completed'] == partial['paired']['completed'] == 0
    assert not empty['no_observed_regressions'] and not partial['candidate_reference_screen_passed']


def test_inconclusive_flags_have_no_boolean_gold_and_malformed_metadata_gets_no_match(cases):
    rows = scripted_rows(cases)
    original = next(row for row in rows if row['judgment']['decision'] == 'inconclusive')
    replace_output(original, addresses_request=False, abstains_appropriately=False)
    assert diagnostic.metrics(cases, rows, revision=3)['all_reference_matches']
    original['judgment']['parse_failed'] = True
    metrics = diagnostic.metrics(cases, rows, revision=3)
    assert not metrics['all_reference_matches']
    assert sum(arm['malformed_or_inconsistent'] for arm in metrics['arms'].values()) == 1


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'reorder'])
def test_pair_schedule_drift_rejects(cases, change):
    rows = scripted_rows(cases)
    if change == 'missing': rows.pop(1)
    elif change == 'duplicate': rows[1] = copy.deepcopy(rows[0])
    else: rows[0], rows[1] = rows[1], rows[0]
    with pytest.raises(ValueError, match='schedule'):
        diagnostic.metrics(cases, rows, revision=3)


def test_new_budget_and_exact_outbound_payloads_reuse_the_existing_runner(cases, historical_ledger):
    entry = runner()
    entry.LEDGER = historical_ledger
    summary = entry.baseline(revision=3)
    protocol = entry.protocol_for(summary, revision=3)
    assert summary['provider_attempts'] == 898 and summary['accounted_usd'] == '0.107133775'
    assert entry.profile(3).calls == 98 and entry.profile(3).cap == entry.Decimal('0.1617')
    assert protocol['budget'] == dict(max_calls=996, max_usd='0.268833775', local_only=False)
    requests = entry.requests(cases, protocol)
    assert len(requests) == 98
    for index in range(0, 98, 2):
        first, second = requests[index:index + 2]
        assert first['messages'][1] == second['messages'][1]
        assert first['messages'][0] != second['messages'][0]
        assert set(json.loads(first['messages'][1]['content'])) == {
            'APPLICATION', 'QUERY', 'VISIBLE_CONTEXT', 'REPLY'}
        assert max(first['bounded_input_bytes'], second['bounded_input_bytes']) <= 12000


def test_full_development_loop_and_raw_audit_cover_both_v2_arms(cases, monkeypatch, historical_ledger):
    """Invented transport rows exercise execution and audit, not semantic ability."""
    entry = runner()
    entry.LEDGER = historical_ledger
    protocol = entry.protocol_for(entry.baseline(revision=3), revision=3)
    judge = object.__new__(study.RecordingRouter)
    judge.controls = False
    judge.provider = protocol['judge']['provider']
    judge.model = protocol['judge']['model']
    judge.remote_spec = protocol['judge']
    judge.max_input = protocol['judge']['max_input']
    ledger = []
    seen = []
    template, _, _, _ = authored_audit_row()
    def assess(kind, item, current, router):
        number = len(seen) + 1
        seen.append(item['id'])
        criteria = item['expected_criteria'] or dict(addresses_request=True, grounded=True,
            attribution_correct=True, abstains_appropriately=True, critical_error=False)
        raw = json.dumps(dict(criteria, rationale='AUTHORED WIRE CONTROL; NOT MODEL EVIDENCE',
            semantic_status='inconclusive' if item['expected_decision'] == 'inconclusive' else 'resolved'))
        result = grounding.parse_runtime_memory(raw, 'stop', False, contract_version=2)
        call = copy.deepcopy(template['call'])
        call['dispatch'] = number
        call['completion']['content'] = raw
        call['completion']['metadata']['completion_id'] = f'AUTHORED-PAIRED-{number}'
        messages = study.memory_judge_messages(item, None, item['capture'], item['response'], None, protocol=current)
        ledger.extend([dict(event='reservation', dispatch=number, reserved_usd='1', provider=judge.provider,
            model=judge.model, tier=study.TaskTier.ADVANCED.value, max_tokens=300, temperature=0,
            proxy=False, messages=[study.asdict(message) for message in messages]), call])
        return dict(judgment=result, call=call,
            observed_success=study.runtime_memory_resolved(result) and result['decision'] == 'accept',
            disposition=grounding.runtime_memory_action(result))
    monkeypatch.setattr(study, 'assess_calibration_control', assess)
    validate = diagnostic.validate
    monkeypatch.setattr(diagnostic, 'validate', lambda report, controls, proto:
        validate(report, controls, proto, ledger_rows=ledger))
    progress = []
    report = diagnostic.run(cases, protocol, judge, on_assessment=lambda row: progress.append(row))
    assert len(seen) == len(progress) == len(report['rows']) == 98 and len(ledger) == 196
    assert report['metrics']['candidate_reference_screen_passed']
    assert report['passed'] is report['runtime_enabled'] is report['qualification_eligible'] is False
    progress[0]['judgment']['raw'] = 'callback mutation'
    assert report['rows'][0]['judgment']['raw'] != 'callback mutation'
    validate(report, cases, protocol, ledger_rows=ledger)
    report['rows'][0]['disposition'] = dict(action='deliver', reason='grounded_and_useful')
    with pytest.raises(ValueError): validate(report, cases, protocol, ledger_rows=ledger)


@pytest.mark.parametrize('revision', [True, 3.0, '3', 0, 11])
def test_unknown_study_revision_never_selects_a_candidate(revision):
    with pytest.raises(ValueError): runner().profile(revision)
    with pytest.raises(ValueError): diagnostic.items(revision=revision)


def test_stopped_previous_scope_cannot_authorize_new_requests(tmp_path, monkeypatch):
    entry = runner()
    plan = tmp_path / 'new-plan.json'
    plan.write_text('{}', encoding='utf-8')
    old = study.ROOT / 'docs/experiments/phase-9-application-context/runtime-decision-separation-qualification-hosted-approval-v1.json'
    def forbidden(*args, **kwargs): pytest.fail('Old approval must stop before preflight/provider access')
    monkeypatch.setattr(entry, 'validate', forbidden)
    with pytest.raises(ValueError, match='approval'): entry.run(old, plan, revision=3)


@pytest.mark.parametrize('change', ['source-hash', 'unobserved-case', 'duplicate-id',
    'qualification-claim', 'duplicate-payload'])
def test_changed_semantic_sources_and_exposure_claims_stop_offline(tmp_path, monkeypatch, change):
    bank = study.read(diagnostic.SEMANTIC)
    if change == 'source-hash': bank['inputs']['report']['sha256'] = 'wrong'
    elif change == 'unobserved-case': bank['exact_observed_control_ids'][-1] = 'UNSENT-CASE'
    elif change == 'duplicate-id': bank['additional_cases'].append(copy.deepcopy(bank['additional_cases'][0]))
    elif change == 'qualification-claim': bank['additional_cases'][0]['qualification_eligible'] = True
    else:
        bank['additional_cases'][1]['capture'] = copy.deepcopy(bank['additional_cases'][0]['capture'])
        bank['additional_cases'][1]['response'] = bank['additional_cases'][0]['response']
    path = tmp_path / 'changed-source.json'
    path.write_text(json.dumps(bank), encoding='utf-8')
    monkeypatch.setattr(diagnostic, 'SEMANTIC', path)
    with pytest.raises(ValueError): diagnostic.items(revision=3)


@pytest.fixture(scope='module')
def prepared_candidate(tmp_path_factory, historical_ledger):
    entry = runner()
    entry.LEDGER = historical_ledger
    selected = entry.profile(3)
    if not selected.plan.exists(): pytest.skip('Frozen packet is prepared after pre-freeze tests pass')
    controls = diagnostic.items(revision=3)
    summary = entry.baseline(revision=3)
    proposal = dict(entry.fields(controls, entry.protocol_for(summary, revision=3),
        study.load_config(), summary, revision=3), timestamp_utc='LOCAL-ONLY-PROSPECTIVE-PLAN')
    path = tmp_path_factory.mktemp('paired-local-plan') / 'plan.json'
    path.write_text(json.dumps(proposal), encoding='utf-8')
    assert entry.validate(path, revision=3)[0] == proposal
    return entry, proposal


@pytest.mark.parametrize('field,value', [('additional_attempts', 99), ('prompt_revisions', [5]),
    ('fixture_sha256', 'wrong'), ('requests_sha256', 'wrong'), ('endpoint', 'https://example.com/v1'),
    ('shared_attempt_cap', 997), ('additional_conservative_usd_cap', '1'), ('qualification_eligible', True)])
def test_exact_frozen_source_prompt_and_budget_drift_stop_offline(prepared_candidate, tmp_path, field, value):
    entry, original = prepared_candidate
    plan = copy.deepcopy(original)
    plan[field] = value
    path = tmp_path / 'drift.json'
    path.write_text(json.dumps(plan), encoding='utf-8')
    with pytest.raises(ValueError): entry.validate(path, revision=3)


def test_consumed_approved_scope_cannot_make_another_provider_request(monkeypatch):
    entry = runner()
    approval = study.ROOT / 'docs/experiments/phase-9-application-context/runtime-semantic-resolution-diagnostic-hosted-approval-v1.json'
    def forbidden(*args, **kwargs): pytest.fail('Consumed scope must stop before provider construction')
    monkeypatch.setattr(study, 'make_routers', forbidden)
    with pytest.raises(ValueError, match='budget|baseline|drift'):
        entry.run(approval, revision=3)
