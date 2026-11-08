"""Portable reviewed-control checks; public source artifacts are not required."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from psychology_base.applications.measurement.semantic_membership import membership_inputs, reviewed_membership_rows

ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location('nonliteral_review_tests',ROOT/'docs/experiments/phase-9-memory-navigation/prepare-nonliteral-review-v1.py')
review=importlib.util.module_from_spec(spec);spec.loader.exec_module(review)


def authored():
    return review.build_authored(json.loads(review.FIXTURE.read_text(encoding='utf-8')))


def selected(family,language='en'):
    cases,refs2,refs3=authored()
    i=next(i for i,c in enumerate(cases) if c['family']=='new-authored-recovery-'+family and c['language']==language)
    return cases[i],refs2[i],refs3[i]


def label(ref,slot,record):
    return next(j['label'] for j in ref['judgments'] if j['slot']==slot and j['record']==record)


def test_all_bilingual_controls_are_complete_and_excluded_from_training():
    cases,r2,r3=authored()
    assert len(cases)==24 and len({c['family'] for c in cases})==12
    assert all(c['split']=='regression' and c['origin']=='authored' for c in cases)
    for refs in (r2,r3):
        assert sum(len(reviewed_membership_rows(c,r)) for c,r in zip(cases,refs))==288
    with pytest.raises(ValueError,match='train/tuning'):
        review.preparation.validate_bank(cases,r2)


@pytest.mark.parametrize('language',['en','zh'])
@pytest.mark.parametrize('family',['person-occupation','person-nationality'])
def test_pronouns_become_positive_only_with_the_actual_named_context(family,language):
    case,r2,r3=selected(family,language)
    assert label(r2,1,2)=='uncertain' and label(r3,1,2)=='match'
    j=next(j for j in r3['judgments'] if j['slot']==1 and j['record']==2)
    assert [e['record'] for e in j['context_evidence']]==[1]
    assert next(r for r in membership_inputs(case,format_version=3) if r['slot']==1 and r['record']==2)['context_record_ids']==(0,1)


@pytest.mark.parametrize('language',['en','zh'])
def test_genuine_ambiguity_and_wrong_source_do_not_get_resolved_by_context(language):
    for family in ('ambiguous-reference','cross-group'):
        case,_,r3=selected(family,language)
        assert label(r3,0,1)==label(r3,1,1)=='uncertain'
        if family=='cross-group':
            assert next(r for r in membership_inputs(case,format_version=3) if r['record']==1)['context_record_ids']==()


@pytest.mark.parametrize('language',['en','zh'])
def test_balanced_descriptions_discriminate_polarity_completion_and_direction(language):
    for family,positive,negative in [('polarity',0,1),('completion',1,0),('preference-direction',0,1)]:
        _,r2,r3=selected(family,language)
        for ref in (r2,r3):
            assert label(ref,0,positive)=='match' and label(ref,0,negative)=='nonmatch'
            assert label(ref,1,negative)=='match' and label(ref,1,positive)=='nonmatch'


@pytest.mark.parametrize('language',['en','zh'])
def test_later_attempts_remain_positive_and_unknown_actors_remain_uncertain(language):
    _,r2,_=selected('encouragement',language)
    assert label(r2,1,1)==label(r2,1,3)=='match'
    _,r2,_=selected('invitation',language)
    assert label(r2,1,1)==label(r2,1,5)=='match'
    case,_,r3=selected('outer-actor',language)
    for slot in (0,1):
        row=next(r for r in reviewed_membership_rows(case,r3) if r['slot']==slot and r['record']==4)
        assert row['label']=='uncertain' and row['constraint']=='uncertain' and not row['semantic_training']


def test_all_input_pairs_ignore_annotation_changes_and_recall_order():
    cases,refs2,refs3=authored()
    for case,ref in zip(cases,refs3):
        before={(r['slot'],r['record']):r for r in membership_inputs(case,format_version=3)}
        changed=deepcopy(case);changed['records'].reverse();changed['expected_label']='private-label'
        assert {(r['slot'],r['record']):r for r in membership_inputs(changed,format_version=3)}==before
        assert not any('private-label' in str(r['pair']) for r in before.values())
        assert len(reviewed_membership_rows(case,ref))==12


@pytest.mark.parametrize('mutation',[
    lambda u:u[0].update(record=99),
    lambda u:u.append(deepcopy(u[0])),
    lambda u:u[0].update(support=[5]),
    lambda u:u[0].update(support=[]),
    lambda u:u[0].update(support=[1,1]),
])
def test_invalid_context_updates_cannot_become_reviewed_references(mutation):
    data=json.loads(review.FIXTURE.read_text(encoding='utf-8'))
    mutation(data['families'][0]['context_updates'])
    with pytest.raises((ValueError,KeyError)):
        review.build_authored(data)


def test_original_review_records_all_four_challenges_without_source_text_or_relabeling_history():
    data=json.loads(review.REVIEW.read_text(encoding='utf-8'))
    assert len(data['cases'])==4 and len({c['id'] for c in data['cases']})==4
    for case in data['cases']:
        assert len(case['labels'])==2 and all(len(v)==6 and set(v)<=set('MNU') for v in case['labels'])
        assert len(case['rationales'])==2 and all(len(v)==6 and all(v) for v in case['rationales'])
        assert case['limitations']
    assert sum(len(c['context_updates']) for c in data['cases'])==2
    assert 'not independent' in data['reviewer']
