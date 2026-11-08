"""Read-only historical replay and configurable bounds, without another executor."""
from types import SimpleNamespace

import pytest

from tests.test_prompt_interference import run


def test_original_44_call_audit_is_preserved_except_reported_checkout_drift(run, monkeypatch):
    def forbidden(*args, **kwargs): pytest.fail('Read-only replay cannot construct a provider')
    monkeypatch.setattr(run.base.study, 'make_routers', forbidden)
    previous = run.base.read(run.parent.HERE / 'source-relations-verification-v1.json')['audit']
    current = run.audit.analyze()
    assert current['complete'] and current['generated_conditions'] == 44
    assert {k:v for k,v in current.items() if k != 'live_source_differences'} == {
        k:v for k,v in previous.items() if k != 'live_source_differences'}
    assert 'docs/experiments/phase-9-memory-navigation/verify-source-relations-v1.py' in current['live_source_differences']


def test_audit_cannot_reinterpret_consumed_44_calls_as_a_20_call_experiment(run):
    wrong = SimpleNamespace(**(dict(vars(run.parent)) | {'MAX_CALLS': 20}))
    with pytest.raises(ValueError, match='Executed freeze'):
        run.audit.analyze(wrong)


def test_original_report_counts_and_frozen_decisions_are_unchanged(run):
    report = run.load('prompt_interference_historical_report_test', 'summarize-source-relations-v1.py')
    previous = run.base.read(run.parent.HERE / 'source-relations-results-v1.json')
    current = report.summarize()
    assert {k:v for k,v in current.items() if k != 'audit'} == {k:v for k,v in previous.items() if k != 'audit'}


def test_prompt_effect_gate_keeps_incomplete_and_unstable_evidence_inconclusive(run):
    """Declared label controls test gate accounting, never model accuracy."""
    report = run.load('prompt_interference_gate_test', 'summarize-prompt-interference-v1.py')
    probes, refs, _ = run.build()
    labels = {p['id']:dict(binding='correct' if p['arm']=='minimal' else 'incorrect', resolution='answer') for p in probes}
    assert report.interpret(refs, labels, probes, complete=True)['evidence_for_instruction_bundle_interference']
    assert not report.interpret(refs, labels, probes, complete=False)['evidence_for_instruction_bundle_interference']
    repeat = next(p for p in probes if p['repeat'] and p['arm']=='minimal')
    labels[repeat['id']]['binding'] = 'incorrect'
    assert not report.interpret(refs, labels, probes, complete=True)['evidence_for_instruction_bundle_interference']
    labels[repeat['id']]['binding'] = 'correct'
    control = next(r for r in refs if r['family'] == 'groups')
    labels[f"{control['id']}/minimal/r0"]['binding'] = 'incorrect'
    assert not report.interpret(refs, labels, probes, complete=True)['evidence_for_instruction_bundle_interference']
