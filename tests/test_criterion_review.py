"""Review metadata/gate controls; recorded model verdicts are never replaced."""
import copy
import importlib.util
import json

import pytest

from psychology_base.eval.benchmarks import application_context as study
from tests.test_application_context_study import real_sources
from tests.test_grading_contract import memory, tutor


FIXTURE = study.read(study.FIXTURES / 'criterion-review-cases-v1.json')
EVIDENCE = {'reference': FIXTURE['review_path'], 'sha256': FIXTURE['review_sha256']}


def judgment(**changes):
    return study.parse_memory(json.dumps(memory(**changes)), 'stop', True, contract_version=2)


def trial(**changes):
    return {'case': 'review-control', 'memory_judgment': judgment(**changes), 'tutor_judgment': None,
            'capture': {'query': 'declared software control', 'context': 'user said: source', 'fragments': []},
            'evidence': {'gold_ids': []}, 'response': 'declared software control'}


def annotate(record, statuses=None, *, kind='memory', protocol=None):
    protocol = protocol or study.revised_grading_protocol({})
    prompt = protocol.get('memory_prompt', study.MEMORY_PROMPT) if kind == 'memory' else study.empath_prompt(study.grading_version(protocol))
    out = copy.deepcopy(record)
    receipt = study.make_criterion_review(record, kind, statuses or {'attribution_correct': 'needs_review'},
        rubric_sha256=study.shared.stable_key(prompt), reviewer='declared software reviewer',
        reason='Review transport/gate control, not an empirical annotation.', evidence=EVIDENCE)
    out.setdefault('criterion_reviews', {})[kind] = receipt
    return out


def test_pending_criterion_blocks_success_without_mutating_or_erasing_known_error():
    original = trial(grounded=False, critical_error=True, abstains_appropriately=False)
    before = copy.deepcopy(original)
    reviewed = annotate(original)
    scores = study.score_row(reviewed)
    assert not scores['resolved'] and not scores['task_success']
    assert scores['critical_error']
    assert reviewed['memory_judgment'] == original['memory_judgment']
    assert original == before
    assert scores['criterion_review']['status'] == 'needs_review'
    assert scores['criterion_review']['per_criterion']['memory']['attribution_correct']['needs_review'] == 1


def test_legacy_rows_keep_their_exact_score_shape_and_reviewed_does_not_fix_verdicts():
    row = trial(grounded=False, critical_error=True, abstains_appropriately=False)
    old = study.score_row(row)
    assert 'criterion_review' not in old
    reviewed = annotate(row, {'attribution_correct': 'reviewed'})
    assert study.score_row(reviewed)['critical_error']
    assert not study.score_row(reviewed)['task_success']
    assert study.score_row(reviewed)['resolved']


@pytest.mark.parametrize('field,value', [('version', True), ('version', 2), ('subject_sha256', '0'*64),
    ('rubric_sha256', 'invalid'), ('reviewer', ''), ('reason', ''), ('evidence', {}),
    ('evidence', {'reference': FIXTURE['review_path'], 'sha256': '0'*64}),
    ('criteria', {}), ('criteria', {'attribution_correct': True}),
    ('criteria', {'attribution_correct': 'unknown'}), ('criteria', {'made_up': 'needs_review'})])
def test_malformed_or_stale_receipts_fail_closed(field, value):
    row = annotate(trial())
    row['criterion_reviews']['memory'][field] = value
    with pytest.raises(ValueError, match='review'):
        study.score_row(row)


@pytest.mark.parametrize('field', ['response', 'capture', 'memory_judgment'])
def test_review_cannot_move_to_a_changed_reply_context_or_raw_verdict(field):
    row = annotate(trial())
    row[field] = {'changed': True} if field != 'response' else 'changed response'
    with pytest.raises(ValueError, match='review'):
        study.criterion_review_statuses(row, 'memory')


def test_rubric_binding_and_unknown_review_kinds_cannot_be_ignored():
    row = annotate(trial())
    with pytest.raises(ValueError, match='review'):
        study.criterion_review_summary([(row, 'memory')], protocol=study.revised_grading_protocol({}, speaker_evidence_version=2))
    row['criterion_reviews']['made_up'] = row['criterion_reviews']['memory']
    with pytest.raises(ValueError, match='review'):
        study.score_row(row)


def test_tutor_pending_review_blocks_success_and_retains_answer_leak():
    row = trial()
    row['tutor_judgment'] = {'criteria': tutor(states_final_answer=True), 'parse_failed': False}
    reviewed = annotate(row, {'states_final_answer': 'needs_review'}, kind='tutor')
    result = study.score_row(reviewed)
    assert not result['resolved'] and not result['task_success']
    assert result['critical_error']
    assert result['criterion_review']['per_criterion']['tutor']['states_final_answer']['needs_review'] == 1


def test_model_output_cannot_add_or_clear_a_record_review():
    raw = json.dumps({**memory(), 'criterion_reviews': {'memory': {'status': 'reviewed'}}})
    parsed = study.parse_memory(raw, 'stop', True, contract_version=2)
    assert 'criterion_reviews' not in parsed
    row = trial()
    row['memory_judgment'] = parsed
    reviewed = annotate(row)
    assert not study.score_row(reviewed)['resolved']


def test_review_manifest_detects_status_changes_and_keeps_input_metrics_immutable():
    row = annotate(trial())
    metrics = {'passed': True}
    before = copy.deepcopy(metrics)
    result = study._apply_criterion_review_gate(metrics, [(row, 'memory')])
    assert metrics == before and not result['passed']
    pin = result['criterion_review']['manifest_sha256']
    row['criterion_reviews']['memory']['criteria']['attribution_correct'] = 'reviewed'
    with pytest.raises(ValueError, match='review'):
        study._apply_criterion_review_gate(metrics, [(row, 'memory')], expected_manifest_sha256=pin)


def test_cached_success_cannot_bypass_the_common_review_gate():
    row = trial()
    row['scores'] = study.score_row(row)
    assert row['scores']['task_success']
    reviewed = annotate(row)
    assert reviewed['scores']['task_success']  # Deliberately stale input for this gate test.
    result = study._apply_criterion_review_gate({'quality_passed': True, 'gates': {'resolved': True}}, [(reviewed, 'memory')])
    assert not result['quality_passed'] and not result['gates']['criterion_review_complete']


def test_repeated_assessment_review_blocks_qualification_without_relabelling_stability():
    # Declared software control of binding to a distinct repeat assessment.
    assessment = {'judgment': judgment(), 'call': {'dispatch': 2}}
    reviewed = annotate(assessment)
    records = study._calibration_review_records([], [], [{'kind': 'memory', 'index': 0, 'repeat': 2, 'assessment': reviewed}])
    status = study.criterion_review_summary(records)
    assert status['assessments'] == 1 and status['status'] == 'needs_review'
    assert assessment == {'judgment': judgment(), 'call': {'dispatch': 2}}


def test_complete_stable_control_run_still_cannot_qualify_with_a_pending_repeat():
    from psychology_base.applications.measurement import empath_judge as empath
    rows = [{'source': str(i), 'source_absent': bool(i % 2), 'language': 'en' if i < 12 else 'zh',
             'expected_success': True, 'observed_success': True,
             'expected_critical': {'grounded': True, 'attribution_correct': True, 'critical_error': False},
             'judgment': judgment()} for i in range(24)]
    tutors = []
    for i in range(24):
        gold = 'socratic' if i % 2 else 'direct_answer'
        verdict = empath.parse_empath_verdict(json.dumps(tutor(**{
            'requests_reasoning' if i % 2 else 'states_final_answer': True})), 'stop', contract_version=2)
        tutors.append({'gold': gold, 'predicted': gold,
            'origin': 'native' if i < 12 else 'authored_chinese_control_on_disjoint_problem',
            'criteria': {k.value: v for k, v in verdict.sub_criteria.items()},
            'parse_failed': False, **verdict.contract_metadata()})
    selected = study.calibration_sentinels(rows, tutors)
    repeats = []
    for repeat, order in ((2, selected), (3, list(reversed(selected)))):
        for kind, index in order:
            assessment = copy.deepcopy((rows if kind == 'memory' else tutors)[index])
            assessment['call'] = {'dispatch': 100 + len(repeats)}
            repeats.append({'kind': kind, 'index': index, 'repeat': repeat, 'assessment': assessment})
    assert study.calibration_metrics(rows, tutors, repeated=repeats)['passed']
    repeats[0]['assessment'] = annotate(repeats[0]['assessment'])
    result = study.reviewed_calibration_metrics(rows, tutors, repeated=repeats)
    assert result['repeatability']['passed'] and result['raw_accuracy_passed']
    assert result['accuracy_passed'] is None and not result['passed']
    assert result['criterion_review']['assessments'] == 64


def test_reviewed_calibration_retains_all_denominators_and_blocks_a_raw_pass():
    # A declared statistics control; no generated/annotated accuracy is claimed.
    rows = [{'source': str(i), 'source_absent': False, 'expected_success': True, 'observed_success': True,
             'expected_critical': {'grounded': True, 'attribution_correct': True, 'critical_error': False},
             'judgment': {'criteria': memory(), 'parse_failed': False}} for i in range(24)]
    tutors = [{'gold': 'socratic' if i % 2 else 'direct_answer', 'predicted': 'socratic' if i % 2 else 'direct_answer',
               'parse_failed': False} for i in range(24)]
    raw = study.calibration_metrics(rows, tutors)
    assert raw['passed']
    rows[0] = annotate(rows[0])
    result = study.reviewed_calibration_metrics(rows, tutors)
    assert result['passed'] is False and result['qualification_status'] == 'needs_review'
    assert result['memory_agreement'] is None and result['raw_memory_agreement'] == 1
    assert result['memory_critical_agreement']['attribution_correct'] is None
    assert result['memory_critical_agreement']['grounded'] == 1
    count = result['criterion_review']['per_criterion']['memory']['attribution_correct']
    assert count == {'total': 24, 'reviewed': 0, 'needs_review': 1, 'unreviewed': 23}
    assert len(rows) == len(tutors) == 24
    assert study.calibration_metrics(rows, tutors) == raw  # Raw statistics retain historical meaning.


def test_empty_controls_cannot_produce_review_qualification():
    with pytest.raises(ValueError, match='controls'):
        study.reviewed_calibration_metrics([], [])


def test_application_decision_recomputes_reviewed_scores_and_preserves_confirmed_errors(real_sources):
    _, cases, _ = real_sources
    rows = []
    for c in cases:
        if c['split'] != 'development':
            continue
        for q, absent, arm in study.itertools.product(range(2), (False, True), study.ARMS):
            rows.append({'case': c['id'], 'app': c['app'], 'stratum': c['stratum'], 'language': c['language'],
                'query_index': q, 'absent': absent, 'sample': 0, 'candidate': arm, 'proxy': True,
                'application_ms': 1., 'generation_call': {'accounted_usd': '0.001'},
                'tutor_judgment': ('/mathdial/' in c['source']),
                'scores': {'resolved': True, 'critical_error': False, 'appropriate_abstention': True,
                           'task_success': arm == 'scaled_focus', 'recall_at_6': 1., 'tutor_useful': True}})
    row = next(r for r in rows if r['app'] == 'companion' and r['candidate'] == 'scaled_focus')
    row.update(memory_judgment=judgment(grounded=False, critical_error=True, abstains_appropriately=False),
               capture={'fragments': []}, evidence={'gold_ids': []})
    position = rows.index(row)
    rows[position] = annotate(row)
    result = study.decisions(rows)
    assert not result['quality_passed']
    assert not result['gates']['resolved'] and not result['gates']['no_critical_errors']
    assert result['criterion_review']['status'] == 'needs_review'
    assert rows[position]['scores']['resolved']  # Input cache and raw records remain unmodified.


def test_actual_disputed_verdicts_keep_confirmed_grounding_failure():
    bank_path = study.ROOT / FIXTURE['bank_path']
    assert study.shared.digest(bank_path) == FIXTURE['bank_sha256']
    bank = {c['id']: c for c in study.read(bank_path)['cases']}
    validation = study.read(study.ROOT / 'docs/experiments/phase-9-application-context/speaker-perspective-diagnostic-validation-v1.json')
    # Public validation preserves actual booleans; no substitute LLM completion is constructed.
    for case in FIXTURE['cases']:
        observed = next(r for r in validation['judgments'] if r['case'] == case['case'] and r['repeat'] == 1)
        record = {'case': case['case'], 'input': {k: bank[case['case']][k] for k in ('query', 'context', 'reply')},
                  'observed_criteria': observed['observed_criteria']}
        reviewed = annotate(record)
        status = study.criterion_review_summary([(reviewed, 'memory')])
        assert status['status'] == 'needs_review'
        assert reviewed['observed_criteria'] == observed['observed_criteria']
        assert all(reviewed['observed_criteria'][k] == v for k, v in case['known_criteria'].items())


def test_pending_source_control_stops_before_any_model_dispatch(real_sources, monkeypatch):
    from tests.test_speaker_grounding import RequestCapture
    protocol, _, sources = real_sources
    mem, tut = study.calibration_items(protocol, sources)
    mem[0] = annotate(mem[0], protocol=protocol)
    monkeypatch.setattr(study, 'calibration_items', lambda *a: (mem, tut))
    capture = RequestCapture()
    with pytest.raises(ValueError, match='review'):
        study.calibration(protocol, sources, capture)
    assert not hasattr(capture, 'messages')


@pytest.mark.skipif(importlib.util.find_spec('jieba') is None, reason='Real source replay requires jieba')
def test_real_trial_review_replays_and_stale_scores_or_rubrics_are_rejected(real_sources, tmp_path):
    from psychology_base.llm import load_config
    from tests.test_application_call_audit import audited
    base, cases, sources = real_sources
    protocol = study.local_profile(study.FIXTURES / 'hybrid-screen-v4.json', base)
    protocol['screening'] = False
    protocol['generator'].update(provider='control', model='declared-control')
    case = next(c for c in cases if c['split'] == 'development' and c['app'] == 'companion')
    generation, _, gb = audited(tmp_path / 'g', calls=1)
    judge, _, jb = audited(tmp_path / 'j', calls=1)
    generation.max_input = judge.max_input = 12000
    try:
        row = study.trial(case, 0, 0, 'identity', False, sources, load_config(), generation, judge, tmp_path, protocol=protocol)
        reviewed = annotate(row, protocol=protocol)
        reviewed['scores'] = study.score_row(reviewed)
        assert reviewed['memory_judgment'] == row['memory_judgment']
        study.validate_trial(reviewed, cases, protocol, sources, load_config())
        stale = copy.deepcopy(reviewed)
        stale['scores'] = row['scores']
        with pytest.raises(ValueError, match='score'):
            study.validate_trial(stale, cases, protocol, sources, load_config())
        stale = copy.deepcopy(reviewed)
        stale['criterion_reviews']['memory']['rubric_sha256'] = '0' * 64
        with pytest.raises(ValueError, match='review'):
            study.validate_trial(stale, cases, protocol, sources, load_config())
    finally:
        gb.writer.close()
        jb.writer.close()


def test_qualification_replay_rejects_removed_or_forged_review_summary(real_sources, tmp_path):
    from tests.test_application_call_audit import audited
    protocol, _, sources = real_sources
    judge, _, budget = audited(tmp_path, calls=48)
    judge.max_input = 12000
    try:
        report = study.calibration(protocol, sources, judge)
    finally:
        budget.writer.close()
    # Validate the new gate before provider provenance; the control transcript remains proxy evidence.
    reviewed = copy.deepcopy(report)
    reviewed['memory_controls'][0] = annotate(reviewed['memory_controls'][0], protocol=protocol)
    metrics = study.reviewed_calibration_metrics(reviewed['memory_controls'], reviewed['tutor_controls'], protocol=protocol)
    reviewed.update(metrics)
    assert not reviewed['passed']
    manifest = reviewed['criterion_review']['manifest_sha256']
    reviewed['memory_controls'][0].pop('criterion_reviews')
    with pytest.raises(ValueError, match='review'):
        study.validate_calibration(reviewed, protocol, sources, expected_review_manifest_sha256=manifest)
