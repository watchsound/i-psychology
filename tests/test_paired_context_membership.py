"""Authored software/data-contract controls; not model or annotation accuracy."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from psychology_base.applications.measurement.semantic_membership import membership_inputs
from psychology_base.eval.benchmarks.membership_preparation import (
    build_source_ledger, recover_authored_coordinates, review_projection,
)

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location('paired_context_test_owner',
    ROOT / 'docs/experiments/phase-9-memory-navigation/paired-context-membership-v1.py')
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def source_fixture():
    return study.read(ROOT / 'benchmarks/fixtures/application_context/membership-paired-context-sources-v1.json')


def positioned(tmp_path):
    cases, declaration = study.source_cases(source_fixture())
    path = tmp_path / 'declaration.json'
    path.write_text(json.dumps(declaration, ensure_ascii=False), encoding='utf-8')
    return [recover_authored_coordinates(c, declaration_path=path, declaration_sha256=study.sha(path)) for c in cases]


def test_source_construction_has_complete_language_application_context_cells_and_no_labels():
    source = source_fixture()
    before = deepcopy(source)
    cases, declaration = study.source_cases(source)
    assert source == before and len(cases) == 64 and len(declaration['sequences']) == 64
    assert len({c['id'] for c in cases}) == 64
    assert len({c['family'] for c in cases}) == 4
    assert {(c['application'], c['language'], c['context_condition'], c['focus_kind']) for c in cases} == {
        (a, l, m, f) for a in ('tutor', 'companion') for l in ('en', 'zh')
        for m in study.MODES for f in ('act', 'question-control')}
    assert all(not ({'expected', 'label', 'judgments'} & set(c)) for c in cases)


@pytest.mark.parametrize('change', ['duplicate_family', 'application', 'split', 'missing_language',
    'missing_context', 'missing_focus', 'empty_text', 'missing_target', 'author'])
def test_incomplete_or_invalid_source_declarations_fail_before_annotation(change):
    source = source_fixture()
    family = source['families'][0]
    if change == 'duplicate_family': source['families'].append(deepcopy(family))
    elif change == 'application': family['application'] = 'other'
    elif change == 'split': family['split'] = 'qualification'
    elif change == 'missing_language': family['languages'].pop('zh')
    elif change == 'missing_context': family['languages']['en']['contexts'].pop('missing')
    elif change == 'missing_focus': family['languages']['zh']['foci'].pop('act')
    elif change == 'empty_text': family['languages']['zh']['foci']['act'] = ' '
    elif change == 'missing_target': family['languages']['en']['descriptions'].pop()
    else: source['author'] = ''
    with pytest.raises(ValueError): study.source_cases(source)


def test_all_windows_and_actor_constraints_come_from_existing_coordinate_and_input_owners(tmp_path):
    cases = positioned(tmp_path)
    groups = set()
    for case in cases:
        rows = study.prior.review_projection(case)
        assert [r['source_order']['position'] for r in case['records']] == [0, 1]
        groups.add(case['records'][0]['source_order']['group'])
        assert all(r['context_record_ids'] == ([0] if r['record'] == 1 else []) for r in rows)
        native = membership_inputs(case, application=case['application'], format_version=4)
        assert [r['constraint'] for r in native] == ['nonmatch', None, 'nonmatch', None]
    assert len(groups) == 64


def test_review_packet_whitelist_hides_source_conditions_proposed_labels_and_split(tmp_path):
    case = positioned(tmp_path)[0]
    rows = review_projection(case)
    tasks = [dict(id=case['id'], rows=rows, source_case=case, proposed_label='match')]
    rows[0]['proposed_label'] = 'nonmatch'
    before = deepcopy(tasks)
    packet = study.blinded_packet(tasks)
    assert tasks == before
    item = packet['cases'][0]
    assert set(item) == {'id', 'rows'}
    assert all(set(r) == {'slot', 'record', 'input_sha256', 'context_record_ids', 'input'} for r in item['rows'])
    assert case['family'] not in json.dumps(packet) and 'proposed_label' not in json.dumps(packet['cases'])
    assert item['rows'][0]['input'] == rows[0]['input']


def test_metadata_does_not_change_the_actual_reviewed_input(tmp_path):
    case = positioned(tmp_path)[0]
    original = review_projection(case)
    case.update(family='not-a-model-feature', context_condition='not-a-model-feature', split='tuning',
                focus_kind='not-a-model-feature', expected='not-a-model-feature')
    assert review_projection(case) == original


def test_translations_context_variants_and_controls_share_one_partition_component(tmp_path):
    cases = positioned(tmp_path)
    ledger, overlaps = study.extend_ledger(build_source_ledger([]), cases, dict(normalized_string_sha256=[]))
    assert not overlaps and len(ledger['components']) == 4
    assert all(len(c['members']) == 16 for c in ledger['components'])
    components = {k: c for c in ledger['components'] for k in c['members']}
    for case in cases:
        assert 'qualification' not in components[case['source_entry']]['allowed_partitions']
    cases[0]['split'] = 'tuning' if cases[0]['split'] == 'train' else 'train'
    with pytest.raises(ValueError, match='admission'):
        study.extend_ledger(build_source_ledger([]), cases, dict(normalized_string_sha256=[]))


def test_unknown_prior_literal_use_blocks_the_whole_derived_family(tmp_path):
    cases = positioned(tmp_path)
    long_turn = next(r['text'] for c in cases for r in c['records'] if len(r['text']) >= 30)
    exposure = dict(normalized_string_sha256=[study.exposure_owner.text_key(long_turn).hex()])
    with pytest.raises(ValueError, match='Prior overlap'):
        study.extend_ledger(build_source_ledger([]), cases, exposure)


def software_references(cases):
    # Intended-pattern controls for report logic only, never actual annotations.
    patterns = {'unique-a': ('match', 'nonmatch'), 'unique-b': ('nonmatch', 'match'),
                'ambiguous': ('uncertain', 'uncertain'), 'missing': ('uncertain', 'uncertain')}
    return [dict(id=c['id'], judgments=[dict(slot=s, record=1,
        label=('nonmatch' if c['focus_kind'] == 'question-control' else patterns[c['context_condition']][s]))
        for s in (0, 1)]) for c in cases]


def test_contrast_check_reports_disagreement_without_rewriting_labels():
    cases, _ = study.source_cases(source_fixture())
    refs = software_references(cases)
    before = deepcopy(refs)
    report = study.contrast_report(cases, refs)
    assert refs == before and len(report) == 8
    assert all(r['matches_intended_contrast'] and r['controls_invariant'] for r in report)
    act = next(c for c in cases if c['focus_kind'] == 'act' and c['context_condition'] == 'unique-a')
    next(r for r in refs if r['id'] == act['id'])['judgments'][0]['label'] = 'uncertain'
    changed = deepcopy(refs)
    assert not all(r['matches_intended_contrast'] for r in study.contrast_report(cases, refs))
    assert refs == changed


@pytest.mark.parametrize('change', ['focus', 'target', 'application', 'split', 'omitted_case'])
def test_contrast_check_rejects_a_broken_matched_comparison(change):
    cases, _ = study.source_cases(source_fixture())
    refs = software_references(cases)
    if change == 'focus': cases[0]['records'][1]['text'] += ' Another act.'
    elif change == 'target': cases[0]['descriptions'][0] += ' Another target.'
    elif change == 'application': cases[0]['application'] = 'other'
    elif change == 'split': cases[0]['split'] = 'other'
    else: cases.pop()
    with pytest.raises(ValueError): study.contrast_report(cases, refs)
