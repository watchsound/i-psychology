"""Prompt selection, exact evidence and diagnostic accounting; no model accuracy claims."""
import copy
import importlib.util
import json

import pytest

from psychology_base.applications.measurement import memory_grounding as grounding
from psychology_base.eval.benchmarks import application_context as study


def diagnostic():
    from psychology_base.eval.benchmarks import runtime_report_diagnostic
    return runtime_report_diagnostic


def entrypoint(tmp_path=None, monkeypatch=None):
    path = study.ROOT / 'benchmarks/scripts/phase9_runtime_report_diagnostic.py'
    spec = importlib.util.spec_from_file_location('report_diagnostic_entrypoint_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if tmp_path is not None:
        plan = study.read(module.PLAN)
        prefix = b''.join(line for line in module.LEDGER.read_bytes().splitlines(keepends=True)
                          if json.loads(line)['dispatch'] <= 832)
        ledger = tmp_path / 'diagnostic-historical-ledger.jsonl'
        ledger.write_bytes(prefix)
        assert study.shared.digest(ledger) == plan['baseline_ledger_sha256']
        monkeypatch.setattr(module, 'LEDGER', ledger)
    return module


def test_compact_revision_is_opt_in_and_keeps_the_existing_semantic_schema():
    before = grounding.runtime_memory_protocol()
    revised = grounding.runtime_memory_protocol(prompt_revision=2)
    assert grounding.runtime_memory_protocol() == before
    assert revised['runtime_memory_contract_version'] == before['runtime_memory_contract_version'] == 1
    assert revised['runtime_memory_prompt_revision'] == 2
    assert len(revised['memory_prompt'].encode()) < len(before['memory_prompt'].encode()) * .65
    assert grounding.CONTEXT_RECORD_RULE in revised['memory_prompt']
    assert grounding.SPEAKER_OWNERSHIP_RULE in revised['memory_prompt']


@pytest.mark.parametrize('revision', [True, 2.0, None, 0, 4, '2'])
def test_unknown_or_coerced_prompt_revisions_reject(revision):
    with pytest.raises(ValueError):
        grounding.runtime_memory_protocol(prompt_revision=revision)


def test_both_prompts_use_the_same_complete_four_field_boundary():
    capture = {'query': 'Who said it?', 'context': 'exact\ncontext'}
    before = copy.deepcopy(capture)
    messages = [study.memory_judge_messages({'app': 'tutor'}, 'HIDDEN ABSENCE', capture,
        'whole reply; including an unsupported tail', 'HIDDEN GOLD',
        protocol=grounding.runtime_memory_protocol(prompt_revision=revision)) for revision in (1, 2)]
    assert messages[0][1] == messages[1][1]
    assert set(json.loads(messages[0][1].content)) == {'APPLICATION', 'QUERY', 'VISIBLE_CONTEXT', 'REPLY'}
    assert all('HIDDEN' not in ''.join(message.content for message in pair) for pair in messages)
    assert capture == before
    revised = grounding.runtime_memory_protocol(prompt_revision=2)
    revised['memory_prompt'] += ' altered'
    with pytest.raises(ValueError):
        study.memory_judge_messages({'app': 'tutor'}, None, capture, '', None, protocol=revised)


def test_diagnostic_preserves_the_observed_failure_and_uses_only_exposed_controls():
    owner = diagnostic()
    items = owner.items()
    assert len(items) == 20 and len({item['id'] for item in items}) == 20
    source = study.read(study.ROOT / '.data/application_context/runtime-qualification-v1/controls-v1.json')['controls']
    failed = next(item for item in source if item['id'] == 'rtq-036c4b9aab201fa9')
    actual = next(item for item in items if item['id'] == failed['id'])
    assert actual['capture'] == failed['capture'] and actual['response'] == failed['response']
    assert actual['expected_criteria'] == failed['expected_criteria']
    scoped = next(item for item in items if item['id'] == 'zh-travel-scoped-report')
    direct = next(item for item in items if item['id'] == 'zh-travel-direct-support')
    assert scoped['capture'] == failed['capture'] and scoped['expected_decision'] == 'accept'
    assert direct['response'] == failed['response'] and direct['expected_decision'] == 'accept'
    assert all(item['qualification_eligible'] is False for item in items)
    assert {item['source'] for item in items} == {'exposed-authored-development', 'msc_456'}


def test_diagnostic_schedule_is_paired_and_checks_all_unhelpful_and_ambiguous_cases():
    owner = diagnostic()
    items = owner.items()
    schedule = owner.schedule(items)
    assert len(schedule) == 40
    for index in range(0, 40, 2):
        pair = schedule[index:index + 2]
        assert pair[0]['control'] == pair[1]['control']
        assert {row['prompt_revision'] for row in pair} == {1, 2}
    assert sum(item['expected_decision'] == 'inconclusive' for item in items) == 2
    assert sum(item['expected_criteria'] is not None and item['expected_criteria']['addresses_request'] is False for item in items) == 2


def test_provisional_flags_and_unrun_rows_never_receive_settled_accuracy_credit():
    owner = diagnostic()
    items = owner.items()
    schedule = owner.schedule(items)
    by_id = {item['id']: item for item in items}
    rows = []
    for entry in schedule:
        item = by_id[entry['control']]
        flags = item['expected_criteria'] or dict(addresses_request=True, grounded=True,
            attribution_correct=True, abstains_appropriately=True, critical_error=False)
        judgment = grounding.parse_runtime_memory(json.dumps(dict(flags, decision=item['expected_decision'])), 'stop', True)
        rows.append(dict(entry, judgment=judgment))
    metrics = owner.metrics(items, rows)
    assert not metrics['qualification_eligible']
    assert metrics['complete'] and metrics['all_reference_matches']
    for arm in metrics['arms'].values():
        assert arm['settled_total'] == arm['settled_completed'] == arm['settled_full_matches'] == 18
        assert arm['ambiguous_total'] == arm['ambiguous_inconclusive'] == 2
    row = next(row for row in rows if by_id[row['control']]['expected_criteria'] is not None)
    row['judgment'] = grounding.parse_runtime_memory(json.dumps(dict(row['judgment']['criteria'], decision='inconclusive')), 'stop', True)
    metrics = owner.metrics(items, rows)
    assert not metrics['all_reference_matches']
    assert metrics['arms'][str(row['prompt_revision'])]['settled_full_matches'] == 17
    prefix = owner.metrics(items, rows[:1])
    assert not prefix['complete'] and not prefix['all_reference_matches']


@pytest.mark.parametrize('change', ['skip', 'duplicate', 'reverse'])
def test_diagnostic_schedule_mutations_reject(change):
    owner = diagnostic()
    items = owner.items()
    rows = owner.schedule(items)
    if change == 'skip': rows.pop(0)
    elif change == 'duplicate': rows[1] = rows[0]
    else: rows.reverse()
    with pytest.raises(ValueError):
        owner.metrics(items, rows)


def test_prompt_arm_selection_does_not_retain_the_other_revision():
    owner = diagnostic()
    original = grounding.runtime_memory_protocol()
    revised = owner.arm_protocol(original, 2)
    assert owner.arm_protocol(revised, 1) == original
    assert original == grounding.runtime_memory_protocol()


def observed_failure():
    report = study.read(study.ROOT / '.data/application_context/runtime-qualification-hosted-v1/report.json')
    plan = study.read(study.ROOT / 'docs/experiments/phase-9-application-context/runtime-qualification-hosted-plan-v1.json')
    controls = study.read(study.ROOT / '.data/application_context/runtime-qualification-v1/controls-v1.json')['controls']
    item = next(item for item in controls if item['id'] == report['rows'][0]['control'])
    ledger = [json.loads(line) for line in (study.ROOT / '.data/application_context/provider-events.jsonl').read_text(encoding='utf-8').splitlines()]
    return report['rows'], [item], plan['protocol'], ledger


def test_shared_raw_audit_replays_the_exact_actual_failure_without_qualifying_it():
    rows, controls, protocol, ledger = observed_failure()
    study.validate_runtime_memory_calls(rows, controls, protocol, ledger_rows=ledger)
    assert rows[0]['judgment']['decision'] == 'accept' and controls[0]['expected_decision'] == 'reject'


@pytest.mark.parametrize('change', ['repeat', 'payload', 'reinterpret-as-revision-2', 'completion', 'raw'])
def test_shared_raw_audit_rejects_actual_call_reuse_or_relabelling(change):
    rows, controls, protocol, ledger = observed_failure()
    kw = {}
    if change == 'repeat': rows.append(copy.deepcopy(rows[0]))
    elif change == 'payload':
        next(row for row in ledger if row['event'] == 'reservation' and row['dispatch'] == 832)['messages'][1]['content'] = '{}'
    elif change == 'reinterpret-as-revision-2':
        kw['protocol_for_row'] = lambda row: diagnostic().arm_protocol(protocol, 2)
    elif change == 'completion': rows[0]['call']['completion']['metadata']['completion_id'] = 'changed'
    else: rows[0]['judgment']['raw'] = '{}'
    with pytest.raises(ValueError):
        study.validate_runtime_memory_calls(rows, controls, protocol, ledger_rows=ledger, **kw)


@pytest.mark.parametrize('revision', [1, 2])
def test_development_loop_completes_both_arms_even_when_semantic_checks_fail(monkeypatch, revision):
    """Scripted schema/control-flow test; never saved as model-quality evidence."""
    owner = diagnostic()
    controls = owner.items()
    protocol = dict(grounding.runtime_memory_protocol(), screening=True,
                    judge=study.read(study.FIXTURES / 'hybrid-screen-v4.json')['judge'])
    if revision != 1:
        protocol['runtime_report_diagnostic_revision'] = revision
    expected = 40 if revision == 1 else 23
    judge = object.__new__(study.RecordingRouter)
    judge.controls = False
    judge.provider = protocol['judge']['provider']
    judge.model = protocol['judge']['model']
    judge.remote_spec = protocol['judge']
    judge.max_input = protocol['judge']['max_input']
    seen = []
    def assess(kind, item, current, router):
        seen.append((item['id'], current['memory_prompt']))
        flags = dict(addresses_request=False, grounded=True, attribution_correct=True,
                     abstains_appropriately=False, critical_error=False)
        result = grounding.parse_runtime_memory(json.dumps(dict(flags, decision='accept')), 'stop', True)
        return dict(judgment=result, call={'scripted_test_only': True}, observed_success=True)
    monkeypatch.setattr(study, 'assess_calibration_control', assess)
    monkeypatch.setattr(owner, 'validate', lambda *args, **kwargs: None)
    snapshots = []
    def progress(row):
        snapshots.append(copy.deepcopy(row))
        row['judgment']['raw'] = 'callback mutation'
    report = owner.run(controls, protocol, judge, on_assessment=progress)
    assert len(seen) == len(snapshots) == len(report['rows']) == expected
    assert report['metrics']['complete'] and not report['metrics']['all_reference_matches']
    assert report['qualification_eligible'] is report['runtime_enabled'] is report['passed'] is False
    assert report['rows'][0]['judgment']['raw'] != 'callback mutation'
    with pytest.raises(TypeError): owner.run(controls, protocol, judge, on_assessment=3)
    judge.model = 'wrong'
    with pytest.raises(ValueError): owner.run(controls, protocol, judge)
    assert len(seen) == expected


def test_entrypoint_freezes_identical_payloads_and_the_actual_shared_budget(tmp_path, monkeypatch):
    cli = entrypoint(tmp_path, monkeypatch)
    baseline = cli.baseline()
    protocol = cli.protocol_for(baseline)
    assert protocol['budget'] == dict(max_calls=872, max_usd='0.159352275', local_only=False)
    assert cli.CAP == cli.Decimal('0.0660')
    requests = cli.requests(diagnostic().items(), protocol)
    assert len(requests) == 40
    for index in range(0, 40, 2):
        assert requests[index]['messages'][1] == requests[index + 1]['messages'][1]
        assert requests[index]['messages'][0] != requests[index + 1]['messages'][0]
        assert max(requests[index]['bounded_input_bytes'], requests[index + 1]['bounded_input_bytes']) <= 12000
    tiny = copy.deepcopy(protocol)
    tiny['judge']['max_input'] = 1
    with pytest.raises(ValueError): cli.requests(diagnostic().items(), tiny)


@pytest.mark.parametrize('approval', [{'approved': False}, {'approved': 'true'}, {'approved': True, 'plan_sha256': 'wrong'}])
def test_new_diagnostic_cannot_reuse_earlier_scope_or_enter_provider_code(tmp_path, monkeypatch, approval):
    cli = entrypoint()
    plan = tmp_path / 'plan.json'
    plan.write_text('{}', encoding='utf-8')
    receipt = tmp_path / 'approval.json'
    receipt.write_text(json.dumps(approval), encoding='utf-8')
    def forbidden(*args, **kwargs): pytest.fail('Approval must be checked before preflight/config/provider access')
    monkeypatch.setattr(cli, 'validate', forbidden)
    with pytest.raises(ValueError, match='approval'): cli.run(receipt, plan)


@pytest.mark.parametrize('field,value', [
    ('additional_attempts', 41), ('additional_conservative_usd_cap', '1'),
    ('requests_sha256', 'wrong'), ('diagnostic_owner_sha256', 'wrong'),
    ('prompt_revisions', [2]), ('endpoint', 'https://example.com/v1'),
    ('qualification_eligible', True),
])
def test_diagnostic_plan_drift_rejects_in_an_isolated_current_environment(tmp_path, monkeypatch, field, value):
    cli = entrypoint(tmp_path, monkeypatch)
    controls = diagnostic().items()
    summary = cli.baseline()
    plan = dict(cli.fields(controls, cli.protocol_for(summary), study.load_config(), summary), timestamp_utc='test-only')
    path = tmp_path / 'isolated-diagnostic-plan.json'
    path.write_text(json.dumps(plan), encoding='utf-8')
    assert cli.validate(path)[0] == plan
    plan[field] = value
    path.write_text(json.dumps(plan), encoding='utf-8')
    with pytest.raises(ValueError): cli.validate(path)


def test_live_diagnostic_plan_is_bound_to_its_environment_code_and_ledger():
    cli = entrypoint()
    plan = study.read(cli.PLAN)
    matching = (plan['runtime_identity'] == study.runtime_identity(study.load_config())
        and plan['code_sha256'] == study.fingerprints()
        and plan['baseline_ledger_sha256'] == study.shared.digest(cli.LEDGER))
    if matching:
        assert cli.validate()[0] == plan
    else:
        with pytest.raises(ValueError): cli.validate()
