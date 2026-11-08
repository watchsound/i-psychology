"""Disclosed software controls for review admission; no accuracy measurement."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from psychology_base.eval.benchmarks.membership_preparation import (
    build_source_ledger, compile_membership_review, review_projection,
)
from tests.test_bilingual_authored_review import fixture
from tests.test_bilingual_membership_preparation import entry

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location('independent_review_test_owner',
    ROOT/'docs/experiments/phase-9-memory-navigation/compare-paired-context-reviews-v1.py')
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def inputs(two=False):
    case, manual, metadata = fixture()
    case['records'][0]['speaker'] = 'tutor'
    key = case['id']
    record = manual['cases'][key]
    record['vectors'] = ['NN', 'NM']
    for notes in record['notes']: notes[0] = '-'
    record['pairs'] = [{k:r[k] for k in ('slot','record','input_sha256')} for r in review_projection(case)]
    cases = [case]
    if two:
        other = deepcopy(case)
        other.update(id='second', source_entry='second')
        other['records'][1]['text'] = 'It worked again.'
        other_record = deepcopy(record)
        other_record['pairs'] = [{k:r[k] for k in ('slot','record','input_sha256')} for r in review_projection(other)]
        manual['cases']['second'] = other_record
        metadata['second'] = deepcopy(metadata[key])
        cases.append(other)
    refs, annotations = compile_membership_review(cases, manual, metadata)
    ledger = build_source_ledger([entry(c['id'], keys=['reviewed-family/shared']) for c in cases])
    second = deepcopy(manual)
    second.update(reviewer='Separate software-fixture reviewer', reviewer_kind='llm',
        independent_of_preparer=True, blinded_to_proposed_labels=True, packet_sha256='test-packet', disputes=[])
    for record in second['cases'].values(): record['membership_loss'] = False
    return cases, refs, annotations, second, ledger, 'test-packet'


def test_agreement_keeps_independent_labels_and_admits_only_semantic_rows():
    args = inputs()
    before = deepcopy(args)
    refs, annotations, semantic, complete, report = study.compare_reviews(*args)
    assert args == before
    assert report['semantic_label_matches'] == report['semantic_label_total'] == 2
    assert report['complete_label_matches'] == report['complete_label_total'] == 4
    assert len(semantic) == 2 and len(complete) == 4
    assert annotations[0]['state'] == 'independently_reviewed' and annotations[0]['independent_review']
    assert len(annotations[0]['reviewers']) == 2 and not annotations[0]['reference_loss']
    assert refs[0]['reviewer'] == args[3]['reviewer']
    assert not report['empirical_model_accuracy'] and not report['clinical_gold'] and not report['training_ready']


def test_array_vector_normalization_preserves_every_cell_and_the_raw_review():
    args = inputs()
    expected = study.compare_reviews(*args)
    for record in args[3]['cases'].values():
        record['vectors'] = [list(vector) for vector in record['vectors']]
    before = deepcopy(args)
    assert study.compare_reviews(*args) == expected and args == before
    next(iter(args[3]['cases'].values()))['vectors'][0][0] = 'nonmatch'
    with pytest.raises(ValueError, match='single-character'):
        study.compare_reviews(*args)


@pytest.mark.parametrize('change', ['same_reviewer', 'wrong_packet', 'not_independent', 'not_blinded',
    'bad_kind', 'missing_case', 'stale_pair', 'actor_label', 'duplicate_metadata', 'unknown_dispute', 'bad_dispute_flag'])
def test_unestablished_or_incomplete_independent_review_cannot_promote_a_case(change):
    cases, refs, annotations, second, ledger, pin = inputs()
    key = cases[0]['id']
    if change == 'same_reviewer': second['reviewer'] = refs[0]['reviewer']
    elif change == 'wrong_packet': second['packet_sha256'] = 'other'
    elif change == 'not_independent': second['independent_of_preparer'] = 'yes'
    elif change == 'not_blinded': second['blinded_to_proposed_labels'] = False
    elif change == 'bad_kind': second['reviewer_kind'] = 'unknown'
    elif change == 'missing_case': second['cases'] = {}
    elif change == 'stale_pair': second['cases'][key]['pairs'][0]['input_sha256'] = 'changed'
    elif change == 'actor_label': second['cases'][key]['vectors'][0] = 'MN'
    elif change == 'duplicate_metadata': annotations += deepcopy(annotations)
    elif change == 'unknown_dispute': second['disputes'] = [dict(case='absent', rationale='Unknown scope.', blocking=True)]
    else: second['disputes'] = [dict(case=key, rationale='Incorrect flag type.', blocking='yes')]
    with pytest.raises(ValueError): study.compare_reviews(cases, refs, annotations, second, ledger, pin)


def test_disagreement_holds_all_translations_variants_in_component_without_changing_either_first_pass():
    args = inputs(two=True)
    key = args[0][0]['id']
    record = args[3]['cases'][key]
    record['vectors'][0] = 'NM'
    record['notes'][0][1] = 'direct_act|Injected disagreement for a software boundary control.'
    before = deepcopy(args)
    _, annotations, semantic, complete, report = study.compare_reviews(*args)
    assert args == before and not semantic and len(complete) == 8
    assert len(report['quarantined_components']) == 1
    assert report['directly_blocked_cases'] == [key]
    assert all(a['state'] == 'unresolved' and not a['membership_loss'] for a in annotations)
    assert report['semantic_label_total'] == 4 and report['semantic_label_matches'] == 3
    assert len(report['label_disagreements']) == 1
    assert sum(s['total'] for s in report['label_strata']) == 4


@pytest.mark.parametrize('blocking', [False, True])
def test_global_taxonomy_caveat_and_blocking_label_uncertainty_remain_distinct(blocking):
    args = inputs()
    args[3]['disputes'] = [dict(case=None, rationale='A declared review issue.', blocking=blocking)]
    _, annotations, semantic, _, report = study.compare_reviews(*args)
    assert bool(semantic) is not blocking
    assert bool(report['quarantined_components']) is blocking
    assert annotations[0]['membership_loss'] is not blocking
    assert report['semantic_label_matches'] == report['semantic_label_total'] == 2
    assert report['reviewer_disputes'] == args[3]['disputes']


def test_existing_mask_or_independent_unresolved_state_cannot_be_auto_admitted():
    for existing in (True, False):
        args = inputs()
        if existing: args[2][0].update(state='unresolved', membership_loss=False)
        else: args[3]['cases'][args[0][0]['id']]['state'] = 'unresolved'
        _, annotations, semantic, _, _ = study.compare_reviews(*args)
        assert not semantic and annotations[0]['state'] == 'unresolved'
