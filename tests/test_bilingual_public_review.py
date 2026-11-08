"""Authored software-contract fixtures, not empirical model or reviewer accuracy."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from psychology_base.eval.benchmarks.membership_preparation import build_source_ledger
from tests.test_bilingual_membership_preparation import annotated_case, entry

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location('public_window_review_owner',
    ROOT / 'docs/experiments/phase-9-memory-navigation/review-bilingual-public-windows-v1.py')
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def partitions(split='train'):
    return {bank: {'one': split} for bank in study.BANKS}


@pytest.mark.parametrize('bank', study.BANKS)
@pytest.mark.parametrize('leaf', sorted(study.PREPARATION_LEAVES))
def test_prepared_artifact_role_joins_the_frozen_case_partition(bank, leaf):
    role, purposes = study.classify_hit('.data/application_context/' + bank + '/' + leaf,
                                      dict(cases=['one'], split=None), partitions())
    assert role == 'prepared_split' and purposes == ['train']


@pytest.mark.parametrize('path', [
    '.data/application_context/membership-training-source-draft-v1.json',
    '.data/application_context/membership-encoder-source-draft-v1.json',
    'docs/experiments/phase-9-memory-navigation/membership-encoder-context-review-v1.json',
    'docs/experiments/phase-9-memory-navigation/membership-encoder-data-freeze-v2.json',
])
def test_preliminary_split_and_description_review_are_not_model_exposure(path):
    _, exposure = study.classify_hit(path, dict(cases=['one'], split='tuning'), partitions())
    assert exposure == []


@pytest.mark.parametrize('path', [
    '.data/application_context/membership-head-pilot-v1/tuning-results.json',
    '.data/application_context/membership-encoder-finalization-v1/adapted/validation-results.json',
    '.data/application_context/membership-encoder-finalization-v1/frozen/validation-results.json',
    '.data/application_context/membership-encoder-audit-v1/errors.json',
    '.data/application_context/membership-head-pilot-audit-v1/errors.json',
])
def test_validation_or_error_review_never_turns_tuning_into_training(path):
    _, purposes = study.classify_hit(path, dict(cases=['one'], split=None), partitions('tuning'))
    assert purposes == ['tuning']
    with pytest.raises(ValueError, match='training seed'):
        study.classify_hit(path, dict(cases=['one'], split=None), partitions('train'))


@pytest.mark.parametrize('path', [
    '.data/application_context/unknown/train-pairs.json',
    '.data/application_context/membership-encoder-data-v1/unreviewed.json',
    '.data/application_context/membership-encoder-data-v1/extra/train-pairs.json',
])
def test_filename_similarity_cannot_establish_exposure_purpose(path):
    with pytest.raises(ValueError, match='Unreviewed'):
        study.classify_hit(path, dict(cases=['one'], split='train'), partitions())


def test_declared_partition_conflicts_and_unknown_case_fail():
    path = '.data/application_context/membership-training-v1/inputs.json'
    for hit in (dict(cases=['one'], split='tuning'), dict(cases=['absent'], split='train')):
        with pytest.raises(ValueError): study.classify_hit(path, hit, partitions())


def inspection():
    path = '.data/application_context/membership-training-v1/inputs.json'
    report = dict(checked_pins=1, source_cases={'one': dict(source='corpus.json', row=0)},
        matching_files=[dict(path=path, sha256='test-pin', hits=[dict(pointer='/0', cases=['one'], split='train')])])
    ledger = build_source_ledger([
        dict(entry('observed'), source='corpus.json', row=0, exposure=['unclassified_prior_use']),
        dict(entry('unrelated'), source='corpus.json', row=1, exposure=['unclassified_prior_use']),
        entry('regression', keys=['source/observed'], exposure=['regression'])])
    return ledger, report, {path: 'test-pin'}


def test_resolve_only_reviewed_observation_preserves_regression_and_parent_evidence():
    ledger, report, pins = inspection()
    before = deepcopy((ledger, report))
    resolved, evidence = study.resolve_exposure(ledger, report, partitions(), pins)
    assert (ledger, report) == before
    entries = {e['id']: e for e in resolved['entries']}
    assert entries['observed']['previous_exposure'] == ['unclassified_prior_use']
    assert entries['observed']['exposure'] == ['train']
    assert entries['unrelated']['exposure'] == ['unclassified_prior_use']
    component = next(c for c in resolved['components'] if 'observed' in c['members'])
    assert component['allowed_partitions'] == ['regression']
    assert evidence[0]['pointer'] == '/0' and evidence[0]['exposure'] == ['train']


@pytest.mark.parametrize('change', ['pin', 'count', 'missing_events', 'unknown_purpose'])
def test_resolution_refuses_incomplete_or_changed_evidence(change):
    ledger, report, pins = inspection()
    if change == 'pin': report['matching_files'][0]['sha256'] = 'changed'
    elif change == 'count': report['checked_pins'] = 0
    elif change == 'missing_events': report['matching_files'] = []
    elif change == 'unknown_purpose':
        path = '.data/application_context/unknown/inputs.json'
        report['matching_files'][0]['path'] = path
        pins = {path: 'test-pin'}
    with pytest.raises(ValueError): study.resolve_exposure(ledger, report, partitions(), pins)


def review_case(language='zh'):
    case, _, _ = annotated_case()
    case.update(origin='public-source', language=language,
                source='.data/kdconv/data/film/train.json' if language == 'zh' else '.data/mathdial/train.jsonl')
    notes = [['First explicit software-review rationale.', 'Second explicit software-review rationale.'],
             ['Third explicit software-review rationale.', 'Fourth explicit software-review rationale.']]
    manual = dict(reviewer='Disclosed software-fixture author', cases={case['id']: dict(vectors=['MN', 'NM'], notes=notes)})
    return case, manual


def test_compiler_requires_complete_manual_labels_and_exact_citations():
    case, manual = review_case()
    manual['cases'][case['id']]['context_citations'] = {'0/1': [0]}
    before = deepcopy((case, manual))
    cases, references, annotations = study.compile_review([case], manual)
    assert (case, manual) == before and cases == [case]
    assert len(references[0]['judgments']) == 4
    assert [j['label'] for j in references[0]['judgments']] == ['match', 'nonmatch', 'nonmatch', 'match']
    assert references[0]['judgments'][1]['context_evidence'][0]['record'] == 0
    assert annotations[0]['membership_loss'] and not annotations[0]['reference_loss']
    assert not annotations[0]['independent_review']


@pytest.mark.parametrize('change', ['missing_case', 'missing_cell', 'missing_note', 'unseen_context', 'bool_context', 'invalid_label', 'dash', 'mask_type'])
def test_manual_review_is_never_completed_with_inferred_or_dummy_cells(change):
    case, manual = review_case()
    record = manual['cases'][case['id']]
    if change == 'missing_case': manual['cases'] = {}
    elif change == 'missing_cell': record['vectors'][0] = 'M'
    elif change == 'missing_note': record['notes'][0][0] = ''
    elif change == 'unseen_context': record['context_citations'] = {'0/0': [1]}
    elif change == 'bool_context': record['context_citations'] = {'0/1': [True]}
    elif change == 'invalid_label': record['vectors'][0] = 'MX'
    elif change == 'dash': record['notes'][0][0] = '-'
    elif change == 'mask_type': record['membership_loss'] = 'yes'
    with pytest.raises(ValueError): study.compile_review([case], manual)


@pytest.mark.parametrize('reason', ['unresolved', 'manual_mask', 'mathdial'])
def test_review_dispute_attribution_and_loss_masks_remain_separate(reason):
    case, manual = review_case('en' if reason == 'mathdial' else 'zh')
    record = manual['cases'][case['id']]
    if reason == 'unresolved': record['state'] = 'unresolved'
    elif reason == 'manual_mask': record['membership_loss'] = False
    _, references, annotations = study.compile_review([case], manual)
    assert not annotations[0]['membership_loss']
    assert 'uncertain' not in {j['label'] for j in references[0]['judgments']}
    if reason == 'mathdial': assert annotations[0]['source_basis'] == 'filtered_mathdial_import_shim'


def test_language_is_not_a_substitute_for_corpus_provenance():
    case, manual = review_case()
    case['source'] = '.data/some-other-chinese-corpus/train.json'
    with pytest.raises(ValueError, match='corpus provenance'):
        study.compile_review([case], manual)
    case['source'] = '.data/mathdial/train.jsonl'
    with pytest.raises(ValueError, match='language mapping'):
        study.compile_review([case], manual)


def test_explicit_verified_snapshot_admission_keeps_manual_review_masks():
    case, manual = review_case('en')
    _, _, metadata = study.compile_review([case], manual, mathdial_attribution_verified=True)
    assert metadata[0]['membership_loss'] and metadata[0]['attribution_status'] == 'cached_revision_card_pinned'
    manual['cases'][case['id']]['state'] = 'unresolved'
    _, _, metadata = study.compile_review([case], manual, mathdial_attribution_verified=True)
    assert not metadata[0]['membership_loss']
    with pytest.raises(ValueError, match='explicit boolean'):
        study.compile_review([case], manual, mathdial_attribution_verified='yes')


def test_serialized_batch_preserves_each_cases_own_partition():
    ledger = build_source_ledger([
        dict(entry('observed-one'), source='corpus', row=0, exposure=['unclassified_prior_use']),
        dict(entry('observed-two'), source='corpus', row=1, exposure=['unclassified_prior_use'])])
    path = '.data/application_context/membership-training-v1/inputs.json'
    inspection = dict(checked_pins=1, source_cases={'one': dict(source='corpus', row=0),
        'two': dict(source='corpus', row=1)}, matching_files=[dict(path=path, sha256='test-pin',
        hits=[dict(pointer='/batch', cases=['one', 'two'], split=None)])])
    splits = {bank: {'one': 'train', 'two': 'tuning'} for bank in study.BANKS}
    resolved, _ = study.resolve_exposure(ledger, inspection, splits, {path: 'test-pin'})
    entries = {e['id']: e for e in resolved['entries']}
    assert entries['observed-one']['exposure'] == ['train']
    assert entries['observed-two']['exposure'] == ['tuning']


def proof_fixture(tmp_path, monkeypatch):
    project = tmp_path / 'project'
    folder = project / 'proof'
    folder.mkdir(parents=True)
    monkeypatch.setattr(study, 'ROOT', project)
    revision = 'a' * 40
    files = {'README.md': '---\nlicense: cc-by-4.0\n---\n', 'train.jsonl': '{}\n', 'test.jsonl': '{}\n'}
    info = {'download_checksums': {f'hf://datasets/eth-nlped/mathdial@{revision}/{s}.jsonl':
        {'num_bytes': len(files[s + '.jsonl'].encode())} for s in ('train', 'test')}}
    files['dataset_info.json'] = json.dumps(info)
    for name, content in files.items(): (folder / name).write_text(content, encoding='utf-8', newline='')
    source_paths = ['benchmarks/scripts/provision_mathdial.py', '.cache/audit-mathdial-cached-snapshot-v1.py']
    for source in source_paths:
        path = project / source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# Disclosed source-pin fixture, not an importer.\n', encoding='utf-8')
    shims = {}
    for split in ('train', 'test'):
        name = '.data/mathdial/' + split + '.jsonl'
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}\n', encoding='utf-8', newline='')
        shims[name] = {'sha256': study.sha(path), 'original_sha256': study.sha(folder / (split + '.jsonl'))}
    report = dict(schema='mathdial-cached-snapshot-provenance-v1', publisher_card_license='cc-by-4.0',
        revision=revision, complete_shims_reconstructed=True, exact_serialized_bytes_match=True,
        files={name: study.sha(folder / name) for name in files}, shims=shims,
        importer_sha256=study.sha(project / source_paths[0]), audit_source_sha256=study.sha(project / source_paths[1]))
    (folder / 'report.json').write_text(json.dumps(report), encoding='utf-8')
    return project, folder, report


def test_cached_provenance_checks_snapshot_and_derivative_pins(tmp_path, monkeypatch):
    _, folder, report = proof_fixture(tmp_path, monkeypatch)
    assert study.validate_mathdial_provenance(folder) == report


@pytest.mark.parametrize('change', ['card', 'original', 'shim', 'revision', 'empty_revision', 'incomplete', 'importer', 'audit', 'replay'])
def test_attribution_is_not_accepted_from_a_license_string_alone(tmp_path, monkeypatch, change):
    project, folder, report = proof_fixture(tmp_path, monkeypatch)
    if change == 'card':
        (folder / 'README.md').write_text('No recorded license', encoding='utf-8')
        report['files']['README.md'] = study.sha(folder / 'README.md')
    elif change == 'original': (folder / 'train.jsonl').write_text('changed', encoding='utf-8')
    elif change == 'shim': (project / '.data/mathdial/train.jsonl').write_text('changed', encoding='utf-8')
    elif change == 'revision': report['revision'] = 'b' * 40
    elif change == 'empty_revision': report['revision'] = ''
    elif change == 'incomplete': report['files'].pop('test.jsonl')
    elif change == 'importer': (project / 'benchmarks/scripts/provision_mathdial.py').write_text('changed', encoding='utf-8')
    elif change == 'audit': report['audit_source_sha256'] = 'changed'
    elif change == 'replay': report['complete_shims_reconstructed'] = False
    (folder / 'report.json').write_text(json.dumps(report), encoding='utf-8')
    with pytest.raises(ValueError): study.validate_mathdial_provenance(folder)


def impact_fixture():
    return dict(schema='mathdial-public-coordinate-impact-v1', snapshot_provenance_sha256='snapshot-pin',
        audit_source_sha256='audit-pin', all_selected_first_six_positions_unchanged=True,
        cases=[dict(id='one', unchanged=True, first_six_native_positions=list(range(6)),
                    first_six_import_positions=list(range(6)))])


def test_exact_selected_coordinate_coverage_is_required():
    report = impact_fixture()
    study.validate_coordinate_impact(report, {'one'}, 'snapshot-pin', 'audit-pin')
    for cases in ([], report['cases'] * 2):
        altered = dict(report, cases=cases)
        with pytest.raises(ValueError):
            study.validate_coordinate_impact(altered, {'one'}, 'snapshot-pin', 'audit-pin')


@pytest.mark.parametrize('change', ['id', 'native_gap', 'import_gap', 'false_flag', 'audit', 'snapshot'])
def test_coordinate_summary_cannot_hide_a_changed_or_unchecked_window(change):
    report = impact_fixture()
    if change == 'id': report['cases'][0]['id'] = 'other'
    elif change == 'native_gap': report['cases'][0]['first_six_native_positions'][-1] = 6
    elif change == 'import_gap': report['cases'][0]['first_six_import_positions'][-1] = 6
    elif change == 'false_flag': report['cases'][0]['unchanged'] = False
    elif change == 'audit': report['audit_source_sha256'] = 'changed'
    elif change == 'snapshot': report['snapshot_provenance_sha256'] = 'changed'
    with pytest.raises(ValueError):
        study.validate_coordinate_impact(report, {'one'}, 'snapshot-pin', 'audit-pin')
