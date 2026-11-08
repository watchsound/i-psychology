"""Fresh/exposed allocation and prospective execution plumbing, never model accuracy."""
import copy
import importlib.util
import json

import pytest

from psychology_base.eval.benchmarks import application_context as study
from psychology_base.applications.measurement import memory_grounding as grounding
from tests.test_runtime_qualification import controls, simulated_run


def runner():
    path = study.ROOT / 'benchmarks/scripts/phase9_runtime_qualification_run.py'
    spec = importlib.util.spec_from_file_location('decision_qualification_cli_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def historical_ledger(tmp_path_factory):
    """Use the real, pinned pre-run bytes; live accounting must keep advancing."""
    entry = runner()
    plan = entry.read(entry.profile(2).plan)
    prefix = b''.join(line for line in entry.LEDGER.read_bytes().splitlines(keepends=True)
                      if json.loads(line)['dispatch'] <= 895)
    path = tmp_path_factory.mktemp('qualification-baseline') / 'ledger.jsonl'
    path.write_bytes(prefix)
    assert entry.sha(path) == plan['baseline_ledger_sha256']
    return path


def test_fresh_schedule_excludes_the_entire_exposed_family_from_every_sentinel(controls):
    by_id = {row['id']: row for row in controls}
    old = study.runtime_qualification_schedule(controls)
    new = study.runtime_qualification_schedule(controls, qualification_version=2)
    assert len(old) == len(new) == 64
    assert old[:40] == new[:40]
    assert all(by_id[row['control']]['source'] != 'msc_456' for row in new[40:])
    assert len({row['control'] for row in new[40:]}) == 12
    assert {row['repeat'] for row in new[40:]} == {2, 3}
    assert len({(by_id[row['control']]['language'], by_id[row['control']]['app'],
                 by_id[row['control']]['expected_decision']) for row in new[40:]}) == 12


def test_all_reference_labels_and_the_original_independent_review_are_reused_unchanged(controls):
    entry = runner()
    packet, original, review = entry.checked_inputs()
    assert original == controls
    assert study.validate_runtime_qualification_review(controls, review)
    exposure = study.runtime_qualification_identity(qualification_version=2)
    assert 'exposure_manifest_sha256' in exposure
    assert study.runtime_qualification_identity().keys() < exposure.keys()


def test_freshness_is_bound_to_actual_prior_provider_payloads(controls):
    exposure = runner().checked_exposure(controls)
    assert exposure['fresh_primary_cases'] == 38 and exposure['exposed_development_cases'] == 2
    assert exposure['exact_prior_payload_exposures'] == {'rtq-036c4b9aab201fa9': [832, 833, 834, 873]}


def test_changed_prior_ledger_cannot_certify_fresh_cases(controls, tmp_path, monkeypatch):
    entry = runner()
    ledger = tmp_path / 'wrong-ledger.jsonl'
    ledger.write_text('', encoding='utf-8')
    monkeypatch.setattr(entry, 'LEDGER', ledger)
    with pytest.raises(ValueError, match='prefix'): entry.checked_exposure(controls)


def test_new_metrics_distinguish_fresh_accuracy_from_exposed_development(controls, monkeypatch):
    report, protocol, review, ledger, invoked, validate = simulated_run(monkeypatch, controls, qualification_version=2)
    assert len(invoked) == 64 and report['passed']
    assert report['runtime_qualification_version'] == 2
    assert report['schema'] == 'runtime-memory-qualification-v2'
    metrics = report['metrics']
    assert metrics['fresh_primary_total'] == metrics['fresh_primary_completed'] == metrics['fresh_full_matches'] == 38
    assert metrics['fresh_source_families'] == 15
    assert metrics['exposed_development_total'] == metrics['exposed_development_full_matches'] == 2
    assert metrics['fresh_sentinel_judgments'] == 24
    assert all(row['disposition'] == grounding.runtime_memory_action(row['judgment']) for row in report['rows'])
    assert validate(report, controls, protocol, review=review, ledger_rows=ledger)


def test_first_invalid_exposed_control_stops_without_consuming_the_fresh_packet(controls, monkeypatch):
    report, protocol, review, ledger, invoked, validate = simulated_run(monkeypatch, controls,
        qualification_version=2, malformed_first=True)
    assert len(invoked) == 1 and report['metrics']['remaining_judgments'] == 63
    assert report['metrics']['fresh_primary_completed'] == 0
    assert report['metrics']['exposed_development_completed'] == 1
    assert not report['passed'] and report['metrics']['irreversible_failure']
    assert not validate(report, controls, protocol, review=review, ledger_rows=ledger)


def test_fresh_noncritical_reference_error_cannot_hide_in_overall_agreement(controls):
    by_id = {row['id']: row for row in controls}
    rows = []
    changed = False
    for entry in study.runtime_qualification_schedule(controls, qualification_version=2):
        item = by_id[entry['control']]
        flags = copy.deepcopy(item['expected_criteria'] or dict(addresses_request=True, grounded=True,
            attribution_correct=True, abstains_appropriately=True, critical_error=False))
        if not changed and item['source'] != 'msc_456' and item['control_type'] == 'unsupported':
            flags['addresses_request'] = not flags['addresses_request']
            changed = True
        raw = json.dumps(dict(flags, semantic_status='inconclusive' if item['expected_decision'] == 'inconclusive'
                              else 'resolved', rationale='AUTHORED SOFTWARE CONTROL'))
        rows.append(dict(entry, judgment=grounding.parse_runtime_memory(raw, 'stop', False, contract_version=2)))
    metrics = study.runtime_qualification_metrics(controls, rows, qualification_version=2)
    assert metrics['fresh_full_matches'] == 37
    assert not metrics['passed'] and 'fresh_reference_mismatch' in metrics['failure_reasons']


@pytest.mark.parametrize('version', [True, 2.0, '2', 0, 3])
def test_unknown_qualification_versions_cannot_select_a_schedule(controls, version):
    with pytest.raises(ValueError): study.runtime_qualification_schedule(controls, qualification_version=version)


def test_new_runner_reuses_the_judge_only_contract_and_sets_an_exact_new_budget(historical_ledger):
    entry = runner()
    entry.LEDGER = historical_ledger
    profile = entry.profile(2)
    summary = entry.baseline(revision=2)
    protocol = entry.protocol_for(summary, revision=2)
    assert profile.calls == 64 and profile.cap == entry.Decimal('0.1056')
    assert summary['provider_attempts'] == 895 and summary['accounted_usd'] == '0.106581150'
    assert protocol['budget'] == dict(max_calls=959, max_usd='0.212181150', local_only=False)
    assert protocol['runtime_memory_contract_version'] == 2 and protocol['runtime_memory_prompt_revision'] == 4
    assert protocol['runtime_qualification_version'] == 2
    assert protocol['judge'] == entry.SPEC


def test_new_requests_use_only_existing_evidence_fields_and_no_reference_labels(controls, historical_ledger):
    entry = runner()
    entry.LEDGER = historical_ledger
    protocol = entry.protocol_for(entry.baseline(revision=2), revision=2)
    requests = entry.requests(controls, protocol, revision=2)
    assert len(requests) == 64
    by_id = {row['id']: row for row in controls}
    for row in requests:
        item = by_id[row['control']]
        assert row['messages'] == [study.asdict(message) for message in study.memory_judge_messages(
            {'app': item['app']}, None, item['capture'], item['response'], None, protocol=protocol)]
        assert set(json.loads(row['messages'][1]['content'])) == {'APPLICATION', 'QUERY', 'VISIBLE_CONTEXT', 'REPLY'}
        assert row['bounded_input_bytes'] <= 12000


def test_consumed_23_request_approval_cannot_authorize_the_new_candidate(tmp_path, monkeypatch):
    entry = runner()
    plan = tmp_path / 'new-plan.json'
    plan.write_text('{}', encoding='utf-8')
    old = study.ROOT / 'docs/experiments/phase-9-application-context/runtime-report-residuals-hosted-approval-v2.json'
    def forbidden(*args, **kwargs): pytest.fail('Old scope must stop before configuration or provider access')
    monkeypatch.setattr(entry, 'validate', forbidden)
    with pytest.raises(ValueError, match='approval'):
        entry.run(old, plan, revision=2)


@pytest.fixture(scope='module')
def prospective_candidate(historical_ledger, tmp_path_factory):
    """Rebuild a local proposal from current code, never rewrite the hosted freeze."""
    entry = runner()
    entry.LEDGER = historical_ledger
    packet, items, review = entry.checked_inputs(revision=2)
    plan = dict(entry.plan_fields(packet, study.load_config(), items, review,
        entry.baseline(revision=2), revision=2), timestamp_utc='LOCAL-OFFLINE-TEST-PROPOSAL')
    path = tmp_path_factory.mktemp('qualification-local-proposal') / 'plan.json'
    path.write_text(json.dumps(plan), encoding='utf-8')
    assert entry.validate(path, revision=2)[0] == plan
    return entry, plan


def test_local_candidate_rebuilds_the_exact_model_requests_and_new_cap(prospective_candidate):
    entry, plan = prospective_candidate
    assert plan['fresh_primary_cases'] == 38 and plan['exposed_development_primary_cases'] == 2
    assert plan['fresh_sentinel_judgments'] == 24
    assert plan['fresh_strata'] == {'en/companion': 10, 'en/tutor': 10, 'zh/companion': 10, 'zh/tutor': 8}
    assert plan['shared_attempt_cap'] == 959 and plan['additional_conservative_usd_cap'] == '0.1056'
    assert plan['runtime_enabled'] is False


@pytest.mark.parametrize('field,value', [
    ('additional_attempts', 65), ('fresh_primary_cases', 40), ('exposed_development_primary_cases', 0),
    ('exposure_manifest_sha256', 'wrong'), ('requests_sha256', 'wrong'),
    ('reference_contract_sha256', 'wrong'), ('endpoint', 'https://example.com/v1'),
    ('shared_attempt_cap', 960), ('additional_conservative_usd_cap', '1'),
])
def test_candidate_source_scope_and_budget_drift_stop_offline(prospective_candidate, tmp_path, field, value):
    entry, original = prospective_candidate
    plan = copy.deepcopy(original)
    plan[field] = value
    path = tmp_path / 'drift.json'
    path.write_text(json.dumps(plan), encoding='utf-8')
    with pytest.raises(ValueError): entry.validate(path, revision=2)


def test_executed_hosted_freeze_cannot_reuse_its_consumed_live_baseline():
    entry = runner()
    with pytest.raises(ValueError):
        entry.validate(revision=2)
