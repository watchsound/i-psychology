"""Disclosed software fixtures; these do not measure model/reviewer accuracy."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from psychology_base.eval.benchmarks.membership_preparation import (
    build_source_ledger, compile_membership_review, prepare_membership_bank, review_projection,
)
from tests.test_bilingual_membership_preparation import annotated_case, entry
from tests.test_bilingual_public_review import review_case, study as historical_public

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location('authored_review_test_owner',
    ROOT / 'docs/experiments/phase-9-memory-navigation/review-bilingual-authored-contrasts-v1.py')
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def fixture():
    case, _, _ = annotated_case()
    manual = dict(reviewer='Disclosed software-fixture author', cases={case['id']: dict(
        vectors=['MN', 'NM'], notes=[['direct_act|Explicit question.', 'different_act|Statement is not a question.'],
            ['different_act|Question is not a statement.', 'direct_act|Explicit statement.']],
        state='implementer_reviewed', membership_loss=True,
        pairs=[{k: r[k] for k in ('slot', 'record', 'input_sha256')} for r in review_projection(case)])})
    metadata = {case['id']: dict(admitted=True, attribution_status='disclosed_project_authored', source_basis='software_fixture')}
    return case, manual, metadata


def test_current_compiler_uses_canonical_windows_and_does_not_mutate_inputs():
    case, manual, metadata = fixture()
    manual['cases'][case['id']]['context_citations'] = {'0/1': [0]}
    before = deepcopy((case, manual, metadata))
    references, annotations = compile_membership_review([case], manual, metadata)
    assert (case, manual, metadata) == before
    assert references[0]['judgments'][1]['context_evidence'][0]['quote'] == case['records'][0]['text']
    semantic, complete = prepare_membership_bank([case], references, annotations, build_source_ledger([entry(case['id'])]))
    assert len(semantic) == len(complete) == 4
    assert annotations[0]['membership_loss'] and not annotations[0]['reference_loss']
    assert not annotations[0]['independent_review']


@pytest.mark.parametrize('change', ['focus', 'context', 'target', 'actor', 'application', 'missing_pin'])
def test_first_pass_cannot_be_reused_after_payload_or_window_changes(change):
    case, manual, metadata = fixture()
    if change == 'focus': case['records'][1]['text'] += ' Later.'
    elif change == 'context': case['records'][0]['source_order']['group'] = 'b' * 32
    elif change == 'target': case['descriptions'][0] = 'my denial'
    elif change == 'actor': case['records'][0]['speaker'] = 'tutor'
    elif change == 'application':
        case['application'] = 'companion'
        case['records'][0]['speaker'] = 'tutor'
    else: manual['cases'][case['id']].pop('pairs')
    with pytest.raises(ValueError, match='exact supplied'):
        compile_membership_review([case], manual, metadata)


@pytest.mark.parametrize('change', ['missing_case', 'duplicate_case', 'missing_cell', 'bad_label',
    'missing_note', 'blank_rationale', 'wrong_reason', 'actor_reason', 'unreviewed', 'independent',
    'bad_loss', 'bad_admission', 'missing_source', 'missing_reviewer', 'bad_record', 'bad_source', 'bool_pin'])
def test_no_inferred_labels_dummy_notes_or_unsupported_review_claims(change):
    case, manual, metadata = fixture()
    record = manual['cases'][case['id']]
    cases = [case]
    if change == 'missing_case': manual['cases'] = {}
    elif change == 'duplicate_case': cases.append(deepcopy(case))
    elif change == 'missing_cell': record['vectors'][0] = 'M'
    elif change == 'bad_label': record['vectors'][0] = 'MX'
    elif change == 'missing_note': record['notes'][0][0] = 'No canonical reason'
    elif change == 'blank_rationale': record['notes'][0][0] = 'direct_act|  '
    elif change == 'wrong_reason': record['notes'][0][0] = 'different_act|Wrong label/reason pair'
    elif change == 'actor_reason': record['notes'][0][0] = 'wrong_outer_actor|Invented constraint'
    elif change == 'unreviewed': record['state'] = 'projected_only'
    elif change == 'independent': record['state'] = 'independently_reviewed'
    elif change == 'bad_loss': record['membership_loss'] = 'yes'
    elif change == 'bad_admission': metadata[case['id']]['admitted'] = 'yes'
    elif change == 'missing_source': metadata = {}
    elif change == 'missing_reviewer': manual['reviewer'] = ''
    elif change == 'bad_record': manual['cases'][case['id']] = None
    elif change == 'bad_source': metadata[case['id']] = None
    elif change == 'bool_pin': record['pairs'][0]['slot'] = False
    with pytest.raises(ValueError): compile_membership_review(cases, manual, metadata)


@pytest.mark.parametrize('citations', [{'0/1': [True]}, {'0/0': [1]}, {'0/1': [0, 0]},
    {'2/0': []}, {'0/1': '0'}])
def test_context_citations_are_exact_unique_predecessors(citations):
    case, manual, metadata = fixture()
    manual['cases'][case['id']]['context_citations'] = citations
    with pytest.raises(ValueError): compile_membership_review([case], manual, metadata)


@pytest.mark.parametrize('state,loss,admitted', [('unresolved', True, True), ('invalid', True, True),
    ('implementer_reviewed', False, True), ('implementer_reviewed', True, False)])
def test_review_and_attribution_masks_do_not_change_semantic_labels(state, loss, admitted):
    case, manual, metadata = fixture()
    manual['cases'][case['id']].update(state=state, membership_loss=loss)
    metadata[case['id']]['admitted'] = admitted
    refs, annotations = compile_membership_review([case], manual, metadata)
    semantic, complete = prepare_membership_bank([case], refs, annotations, build_source_ledger([entry(case['id'])]))
    assert semantic == [] and len(complete) == 4
    assert not any(r['label'] == 'uncertain' for r in complete)


@pytest.mark.parametrize('speaker,label', [('tutor', 'N'), (None, 'U')])
def test_actor_uncertainty_never_counts_as_semantic_supervision(speaker, label):
    case, manual, metadata = fixture()
    case['records'][0]['speaker'] = speaker
    record = manual['cases'][case['id']]
    record['vectors'] = [label + v[1:] for v in record['vectors']]
    for notes in record['notes']: notes[0] = '-'
    record['pairs'] = [{k: r[k] for k in ('slot', 'record', 'input_sha256')} for r in review_projection(case)]
    refs, annotations = compile_membership_review([case], manual, metadata)
    semantic, complete = prepare_membership_bank([case], refs, annotations, build_source_ledger([entry(case['id'])]))
    assert len(semantic) == 2 and len(complete) == 4
    record['vectors'][0] = 'M' + record['vectors'][0][1:]
    with pytest.raises(ValueError, match='code-owned'): compile_membership_review([case], manual, metadata)


def test_noncontiguous_record_ids_are_not_array_indices():
    case, manual, metadata = fixture()
    case['records'][0]['id'], case['records'][1]['id'] = 8, 12
    manual['cases'][case['id']]['pairs'] = [
        {k: r[k] for k in ('slot', 'record', 'input_sha256')} for r in review_projection(case)]
    refs, _ = compile_membership_review([case], manual, metadata)
    assert [j['record'] for j in refs[0]['judgments']] == [8, 12, 8, 12]


@pytest.mark.parametrize('language', ['en', 'zh'])
def test_current_compiler_preserves_historical_public_reference_contract(language):
    case, old = review_case(language)
    _, expected_refs, expected_annotations = historical_public.compile_review([case], old, mathdial_attribution_verified=True)
    manual = deepcopy(old)
    record = manual['cases'][case['id']]
    record.update(state='implementer_reviewed', membership_loss=True,
        pairs=[{k: r[k] for k in ('slot', 'record', 'input_sha256')} for r in review_projection(case)])
    for slot, notes in enumerate(record['notes']):
        record['notes'][slot] = [('direct_act' if record['vectors'][slot][i] == 'M' else 'different_act') + '|' + n
                                 for i, n in enumerate(notes)]
    previous = expected_annotations[0]
    metadata = {case['id']: dict(admitted=True, **{k: previous[k] for k in ('attribution_status', 'source_basis')})}
    assert compile_membership_review([case], manual, metadata) == (expected_refs, expected_annotations)


def authored_fixture():
    case, manual, _ = fixture()
    for r in case['records']: r.pop('source_order')
    manual['cases'][case['id']]['pairs'] = [
        {k: r[k] for k in ('slot', 'record', 'input_sha256')} for r in review_projection(case)]
    manual['input_sha256'] = 'test-bank-pin'
    tasks = [dict(id=case['id'], rows=review_projection(case))]
    ledger = build_source_ledger([dict(entry(case['id']), origin='authored', language='en')])
    return case, manual, tasks, ledger


@pytest.mark.parametrize('change', [None, 'order', 'source', 'task', 'bank_pin', 'duplicate_task'])
def test_authored_admission_reuses_empty_unlabeled_windows_and_exact_source(change):
    case, manual, tasks, ledger = authored_fixture()
    if change == 'order': case['records'][0]['source_order'] = dict(group='c' * 32, position=0)
    elif change == 'source': ledger['entries'][0]['origin'] = 'public-source'
    elif change == 'task': tasks[0]['rows'][0]['input_sha256'] = 'changed'
    elif change == 'bank_pin': manual['input_sha256'] = 'wrong-bank'
    elif change == 'duplicate_task': tasks += deepcopy(tasks)
    if change is not None:
        with pytest.raises(ValueError): study.compile_authored([case], tasks, manual, ledger, 'test-bank-pin')
    else:
        selected, refs, annotations = study.compile_authored([case], tasks, manual, ledger, 'test-bank-pin')
        assert selected == [case] and len(refs[0]['judgments']) == 4
        assert annotations[0]['source_basis'] == 'authored_contrast_alternatives_without_declared_chronology'


def test_coverage_uses_eligible_rows_keeps_zero_cells_and_discloses_missing_application():
    case, manual, metadata = fixture()
    manual['cases'][case['id']]['vectors'][0] = 'MU'
    manual['cases'][case['id']]['notes'][0][1] = 'missing_reference|Unspecified object in this software fixture.'
    refs, annotations = compile_membership_review([case], manual, metadata)
    semantic, complete = prepare_membership_bank([case], refs, annotations, build_source_ledger([entry(case['id'])]))
    report = study.coverage([case], annotations, semantic, complete)
    assert report['semantic_labels'] == dict(match=2, nonmatch=1, uncertain=1)
    assert len(report['cells']) == 24
    assert all(c['rows'] == 0 for c in report['cells'] if c['application'] == 'companion')
    assert not report['training_ready'] and not report['all_required_label_marginals_present']
    assert len(report['missing_application_language_labels']) == 9
    assert report['application_language_component_targets'][0]['train_components'] == 1
    assert not any(c['meets_component_targets'] for c in report['application_language_component_targets'])


def test_parent_verifier_uses_preserved_archives_not_current_source_and_rejects_tampering(tmp_path):
    folder = tmp_path / 'frozen'
    folder.mkdir()
    (folder / 'executed-source').mkdir()
    for name in ('inputs.json', 'references.json', 'artifact.json'):
        (folder / name).write_text('[]', encoding='utf-8')
    source = folder / 'executed-source/old.py'
    source.write_text('historic = True', encoding='utf-8')
    plan = dict(source={'old.py': study.sha(source)}, input_sha256=study.sha(folder / 'inputs.json'),
        references_sha256=study.sha(folder / 'references.json'), artifacts_canonical_sha256={'artifact.json': 'canonical-pin'})
    study.verify_parent(folder, plan, lambda value: 'canonical-pin')
    source.write_text('historic = False', encoding='utf-8')
    with pytest.raises(ValueError, match='archive changed'):
        study.verify_parent(folder, plan, lambda value: 'canonical-pin')
