"""Authored packet and grading controls, not measured model predictions."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

from tests.test_independent_membership import packet as prior_packet

ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location('pointwise_packet_test',ROOT/'docs/experiments/phase-9-memory-navigation/run-pointwise-membership-v1.py')
study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)

def packet():
    parent,refs=prior_packet();bank=json.loads(study.FOCUS_BANK.read_text(encoding='utf-8'))
    denial=json.loads(study.DENIAL_BANK.read_text(encoding='utf-8'))
    return study.build_packet(parent,refs,bank,denial)

def oracle(inputs,refs):
    result=[]
    for e in inputs['schedule']:
        c=inputs['cases'][e['case']]
        truth={j['slot']:j['label'] for j in refs[c['id']]['judgments'] if j['record']==c['focus_id']}
        result.append(dict(id=e['id'],labels=[truth[s] for s in e['description_order']]))
    return result

def test_scope_preserves_existing_cases_annotations_and_complete_bilingual_contrasts():
    parent,annotations=prior_packet();before=deepcopy((parent,annotations))
    inputs,refs=study.build_packet(parent,annotations,json.loads(study.FOCUS_BANK.read_text(encoding='utf-8')),
        json.loads(study.DENIAL_BANK.read_text(encoding='utf-8')))
    assert (parent,annotations)==before and len(inputs['cases'])==10 and len(inputs['schedule'])==20
    assert len(inputs['contrast_pairs'])==14
    assert sum(2//e['descriptions_per_call'] for e in inputs['schedule'])==30
    for key in inputs['cases'].keys() & parent['cases'].keys():
        assert inputs['cases'][key]==parent['cases'][key] and refs[key]==annotations[key]
    for i in range(10):
        a,b=inputs['schedule'][2*i:2*i+2]
        assert a['case']==b['case'] and a['prompt_profile']==b['prompt_profile']==study.PROFILE
        assert study.rows_for(inputs['cases'][a['case']],a)==study.rows_for(inputs['cases'][b['case']],b)
        assert [a['grouping'],b['grouping']]==(['joint','pointwise'] if i%2==0 else ['pointwise','joint'])
    for lang in ('en','zh'):
        assert {f'referent-{lang}-{v}' for v in ('unique','ambiguous')} <= inputs['cases'].keys()
    observed=oracle(inputs,refs)
    denial=next(o for o in observed if o['id'].startswith('denial-zh-act/'))
    assert denial['labels']==['match','nonmatch']

def test_canonical_grade_counts_families_missing_failures_and_degenerate_predictions():
    inputs,refs=packet();cases,schedule=inputs['cases'],inputs['schedule'];observed=oracle(inputs,refs)
    good=study.score(cases,refs,schedule,observed)
    assert good['candidate_screen_passed'] and not good['qualified']
    assert good['by_family']['act']['pointwise']['planned_labels']==12
    assert good['by_family']['reference']['pointwise']['planned_labels']==8
    assert good['by_grouping']['pointwise']['positives_retained']==7
    assert good['by_grouping']['pointwise']['uncertainty_retained']==4
    assert not study.score(cases,refs,schedule,observed[:-1])['candidate_screen_passed']
    for label in ('match','nonmatch','uncertain'):
        constant=[dict(id=e['id'],labels=[label,label]) for e in schedule]
        assert not study.score(cases,refs,schedule,constant)['candidate_screen_passed']

def test_incomplete_scope_preserves_planned_positive_and_uncertainty_denominators():
    inputs,refs=packet()
    result=study.score(inputs['cases'],refs,inputs['schedule'],[])
    for grouping in study.GROUPS:
        summary=result['by_grouping'][grouping]
        assert summary['completed']==summary['measured_labels']==0
        assert summary['planned_labels']==20 and summary['positives_expected']==7
        assert summary['uncertainty_expected']==4 and summary['uncertainty_retained']==0
