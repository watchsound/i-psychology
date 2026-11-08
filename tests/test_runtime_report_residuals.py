"""Residual provenance, version compatibility and local experiment plumbing.

Scripted flags below check software accounting only, never semantic repair.
"""
import copy
import importlib.util
import json

import pytest

from psychology_base.applications.measurement import memory_grounding as grounding
from psychology_base.eval.benchmarks import application_context as study
from psychology_base.eval.benchmarks import runtime_report_diagnostic as diagnostic


def cli(tmp_path, monkeypatch):
    path = study.ROOT / 'benchmarks/scripts/phase9_runtime_report_diagnostic.py'
    spec = importlib.util.spec_from_file_location('residual_diagnostic_cli_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prior = study.read(study.ROOT / 'docs/experiments/phase-9-application-context/runtime-report-diagnostic-hosted-analysis-v1.json')
    ledger = tmp_path / 'historical-ledger-872.jsonl'
    ledger.write_bytes(b''.join(line for line in module.LEDGER.read_bytes().splitlines(keepends=True)
                                if json.loads(line)['dispatch'] <= 872))
    assert study.shared.digest(ledger) == prior['ledger_sha256']
    monkeypatch.setattr(module, 'LEDGER', ledger)
    return module


def test_revision_three_preserves_both_observed_prompt_revisions_and_schema():
    frozen = study.read(study.ROOT / 'docs/experiments/phase-9-application-context/runtime-report-diagnostic-hosted-plan-v1.json')
    for n in (1, 2):
        assert study.shared.stable_key(grounding.runtime_memory_protocol(prompt_revision=n)) == frozen['prompt_sha256'][str(n)]
    revised = grounding.runtime_memory_protocol(prompt_revision=3)
    assert revised['runtime_memory_contract_version'] == 1
    assert revised['runtime_memory_prompt_revision'] == 3
    assert grounding.runtime_memory_protocol() == grounding.runtime_memory_protocol(prompt_revision=1)
    prompt = revised['memory_prompt']
    assert prompt.index('Decision precedence') < prompt.index('Claim evidence')
    assert 'outer recorded speaker' in prompt
    assert grounding.CONTEXT_RECORD_RULE in prompt and grounding.SPEAKER_OWNERSHIP_RULE in prompt


@pytest.mark.parametrize('n', [True, None, 3.0, '3', 0, 4])
def test_prompt_revision_selection_remains_strict(n):
    with pytest.raises(ValueError): grounding.runtime_memory_protocol(prompt_revision=n)


def test_residuals_are_the_three_exact_observed_mismatches_with_unchanged_references():
    frozen = diagnostic.residuals()
    assert set(frozen) == {'en-unqualified-affirmation', 'zh-unqualified-affirmation', 'zh-flattened-nested-report'}
    controls = {item['id']: item for item in diagnostic.items()}
    bank = study.read(study.FIXTURES / 'runtime-report-residuals-v1.json')
    for row in bank['residuals']:
        assert controls[row['control']]['expected_decision'] == row['expected_decision']
        assert controls[row['control']]['expected_criteria'] == row['expected_criteria']
    for ids in bank['positive_contrasts'].values():
        assert all(controls[cid]['expected_decision'] == 'accept' for cid in ids)


def test_new_schedule_covers_all_twenty_cases_and_only_three_old_prompt_sentinels():
    controls = diagnostic.items()
    old = diagnostic.schedule(controls)
    new = diagnostic.schedule(controls, revision=2)
    assert len(old) == 40 and len(new) == 23
    assert {row['control'] for row in new if row['prompt_revision'] == 3} == {item['id'] for item in controls}
    assert {row['control'] for row in new if row['prompt_revision'] == 2} == set(diagnostic.residuals())
    assert [row['index'] for row in new] == list(range(23))


@pytest.mark.parametrize('revision', [True, None, 2.0, '2', 0, 11])
def test_schedule_revision_selection_remains_strict(revision):
    with pytest.raises(ValueError): diagnostic.schedule(diagnostic.items(), revision=revision)


def test_new_metrics_use_the_actual_unequal_denominators_and_no_provisional_gold():
    controls = diagnostic.items()
    by_id = {item['id']: item for item in controls}
    rows = []
    for entry in diagnostic.schedule(controls, revision=2):
        item = by_id[entry['control']]
        flags = item['expected_criteria'] or dict(addresses_request=True, grounded=True,
            attribution_correct=True, abstains_appropriately=True, critical_error=False)
        result = grounding.parse_runtime_memory(json.dumps(dict(flags, decision=item['expected_decision'])), 'stop', True)
        rows.append(dict(entry, judgment=result))
    measured = diagnostic.metrics(controls, rows, revision=2)
    assert measured['complete'] and measured['all_reference_matches']
    assert measured['arms']['2']['total'] == 3 and measured['arms']['2']['settled_total'] == 1
    assert measured['arms']['3']['total'] == 20 and measured['arms']['3']['settled_total'] == 18
    assert measured['arms']['2']['ambiguous_total'] == measured['arms']['3']['ambiguous_total'] == 2
    partial = diagnostic.metrics(controls, rows[:1], revision=2)
    assert not partial['complete'] and not partial['all_reference_matches']
    assert sum(arm['settled_full_matches'] for arm in partial['arms'].values()) <= 1
    broken = copy.deepcopy(rows)
    broken[0]['prompt_revision'] = 1
    with pytest.raises(ValueError): diagnostic.metrics(controls, broken, revision=2)


def test_original_wrong_verdicts_are_never_rewritten_to_the_expected_decision():
    bank = study.read(study.FIXTURES / 'runtime-report-residuals-v1.json')
    report = study.read(study.ROOT / bank['inputs']['report']['path'])
    for row in report['rows']:
        if row['prompt_revision'] == 2 and row['control'] in {r['control'] for r in bank['residuals']}:
            call = row['call']['completion']
            assert grounding.parse_runtime_memory(call['content'], call['finish_reason'], False) == row['judgment']


def test_twenty_candidate_payloads_and_sentinels_use_the_existing_message_owner(tmp_path, monkeypatch):
    entry = cli(tmp_path, monkeypatch)
    protocol = entry.protocol_for(entry.baseline(revision=2), revision=2)
    rows = entry.requests(diagnostic.items(), protocol)
    assert len(rows) == 23
    by_id = {item['id']: item for item in diagnostic.items()}
    for row in rows:
        item = by_id[row['control']]
        messages = study.memory_judge_messages({'app': item['app']}, None, item['capture'], item['response'], None,
            protocol=diagnostic.arm_protocol(protocol, row['prompt_revision']))
        assert row['messages'] == [study.asdict(message) for message in messages]
        assert set(json.loads(row['messages'][1]['content'])) == {'APPLICATION', 'QUERY', 'VISIBLE_CONTEXT', 'REPLY'}
        assert row['bounded_input_bytes'] <= 12000
    assert protocol['budget'] == dict(max_calls=895, max_usd='0.140376275', local_only=False)
    assert entry.profile(2).cap == entry.Decimal('0.03795')


@pytest.mark.parametrize('change', ['prompt', 'revision'])
def test_revision_three_drift_blocks_before_the_judge_boundary(change):
    protocol = grounding.runtime_memory_protocol(prompt_revision=3)
    if change == 'prompt': protocol['memory_prompt'] += ' drift'
    else: protocol['runtime_memory_prompt_revision'] = 4
    with pytest.raises(ValueError):
        study.memory_judge_messages({'app': 'companion'}, None, dict(query='q', context='c'), 'r', None, protocol=protocol)


@pytest.mark.parametrize('change', ['source', 'label', 'observed'])
def test_residual_source_or_label_drift_blocks_selection(tmp_path, monkeypatch, change):
    bank = study.read(diagnostic.RESIDUALS)
    if change == 'source': bank['inputs']['report']['sha256'] = 'wrong'
    elif change == 'label': bank['residuals'][0]['expected_decision'] = 'reject'
    else: bank['residuals'][0]['observed_decision'] = 'inconclusive'
    path = tmp_path / 'residual-drift.json'
    path.write_text(json.dumps(bank), encoding='utf-8')
    monkeypatch.setattr(diagnostic, 'RESIDUALS', path)
    with pytest.raises(ValueError): diagnostic.residuals()


def test_old_paid_approval_cannot_authorize_the_changed_prompt(tmp_path, monkeypatch):
    entry = cli(tmp_path, monkeypatch)
    plan = tmp_path / 'new-plan.json'
    plan.write_text('{}', encoding='utf-8')
    old = study.read(study.ROOT / 'docs/experiments/phase-9-application-context/runtime-report-diagnostic-hosted-approval-v1.json')
    receipt = tmp_path / 'old-approved-scope.json'
    receipt.write_text(json.dumps(old), encoding='utf-8')
    def forbidden(*args, **kwargs): pytest.fail('Old scope must block before config/provider validation')
    monkeypatch.setattr(entry, 'validate', forbidden)
    with pytest.raises(ValueError, match='approval'): entry.run(receipt, plan, revision=2)


@pytest.mark.parametrize('field,value', [
    ('additional_attempts', 24), ('prompt_revisions', [3]),
    ('residuals_sha256', 'wrong'), ('additional_conservative_usd_cap', '1'),
])
def test_new_plan_drift_rejects_after_a_positive_isolated_preflight(tmp_path, monkeypatch, field, value):
    entry = cli(tmp_path, monkeypatch)
    summary = entry.baseline(revision=2)
    controls = diagnostic.items()
    plan = dict(entry.fields(controls, entry.protocol_for(summary, revision=2), study.load_config(), summary, revision=2),
                timestamp_utc='test-only')
    path = tmp_path / 'new-isolated-plan.json'
    path.write_text(json.dumps(plan), encoding='utf-8')
    assert entry.validate(path, revision=2)[0] == plan
    plan[field] = value
    path.write_text(json.dumps(plan), encoding='utf-8')
    with pytest.raises(ValueError): entry.validate(path, revision=2)


def test_actual_new_plan_binds_the_live_environment_and_ledger():
    path = study.ROOT / 'benchmarks/scripts/phase9_runtime_report_diagnostic.py'
    spec = importlib.util.spec_from_file_location('actual_residual_plan_test', path)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    plan = study.read(entry.profile(2).plan)
    matching = (plan['runtime_identity'] == study.runtime_identity(study.load_config())
        and plan['code_sha256'] == study.fingerprints()
        and plan['baseline_ledger_sha256'] == study.shared.digest(entry.LEDGER))
    if matching:
        assert entry.validate(revision=2)[0] == plan
    else:
        with pytest.raises(ValueError): entry.validate(revision=2)
