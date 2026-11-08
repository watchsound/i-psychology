"""Software/packet controls; synthetic responses are not model accuracy evidence."""
from copy import deepcopy
from dataclasses import replace
import importlib.util
from itertools import product
import json
from pathlib import Path

import pytest

from psychology_base.applications.measurement.local_query_binding import (
    LocalChatMembershipAssessor, REFERENCE_POSSIBILITY_RULES, membership_prompt,
)
from psychology_base.applications.measurement.semantic_membership import reviewed_membership_rows
from psychology_base.eval.benchmarks.local_membership_audit import membership_wire_payload
from tests.test_local_chat_membership import router
from tests.test_pointwise_membership_packet import packet as parent_packet, oracle

ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location('reference_packet_test',ROOT/'docs/experiments/phase-9-memory-navigation/run-reference-possibility-v1.py')
study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)

def packet():
    parent,refs=parent_packet()
    return study.build_packet(parent,refs,json.loads(study.BANK.read_text(encoding='utf-8')))

def test_new_profile_is_only_an_opt_in_appendix_and_has_a_distinct_identity():
    assert membership_prompt(study.PROFILES[1])==membership_prompt(study.PROFILES[0])+'\n'+REFERENCE_POSSIBILITY_RULES
    value,_=router()
    old=LocalChatMembershipAssessor(value,format_version=4,prompt_profile=study.PROFILES[0])
    new=LocalChatMembershipAssessor(value,format_version=4,prompt_profile=study.PROFILES[1])
    assert old.identity!=new.identity
    assert LocalChatMembershipAssessor(value).identity==LocalChatMembershipAssessor(value,prompt_profile='baseline-v1').identity

@pytest.mark.parametrize('labels',list(product(('match','nonmatch','uncertain'),repeat=2)))
def test_candidate_keeps_all_vectors_and_exact_joint_wire_payload(labels):
    inputs,_=packet();entry=next(e for e in inputs['schedule'] if e['prompt_profile']==study.PROFILES[1])
    rows=study.rows_for(inputs['cases'][entry['case']],entry)
    value,seen=router(lambda c:replace(c,content=json.dumps(dict(labels=list(labels)))))
    assessor=LocalChatMembershipAssessor(value,format_version=4,prompt_profile=study.PROFILES[1])
    assert assessor(rows)==list(labels) and len(seen)==1
    expected=membership_wire_payload(rows,'authored-test',prompt_profile=study.PROFILES[1])
    assert [dict(role=m.role,content=m.content) for m in seen[0][0]]==expected['messages']
    assert assessor.last_calls[0]['row_indices']==[0,1]

def test_profile_mutation_is_rejected_before_dispatch():
    inputs,_=packet();entry=inputs['schedule'][0]
    value,seen=router();assessor=LocalChatMembershipAssessor(value,format_version=4,prompt_profile=study.PROFILES[1])
    assessor.prompt_profile=study.PROFILES[0]
    with pytest.raises(ValueError):assessor(study.rows_for(inputs['cases'][entry['case']],entry))
    assert not seen

def test_packet_reuses_all_eight_existing_cases_without_mutation():
    parent,annotations=parent_packet();bank=json.loads(study.BANK.read_text(encoding='utf-8'))
    before=deepcopy((parent,annotations,bank))
    inputs,refs=study.build_packet(parent,annotations,bank)
    assert (parent,annotations,bank)==before
    assert len(inputs['cases'])==16 and len(inputs['schedule'])==32 and len(inputs['contrast_pairs'])==20
    reused=inputs['cases'].keys() & parent['cases'].keys()
    assert len(reused)==8
    for key in reused:assert inputs['cases'][key]==parent['cases'][key] and refs[key]==annotations[key]
    for index in range(16):
        a,b=inputs['schedule'][2*index:2*index+2]
        assert a['case']==b['case'] and a['descriptions_per_call']==b['descriptions_per_call']==2
        assert [a['prompt_profile'],b['prompt_profile']]==list(study.PROFILES if index%2==0 else study.PROFILES[::-1])
        ca=inputs['cases'][a['case']]
        assert study.rows_for(ca,a)==study.rows_for(ca,b)
        left,right=[membership_wire_payload(study.rows_for(ca,e),study.MODEL,prompt_profile=e['prompt_profile']) for e in (a,b)]
        assert left['messages'][1]==right['messages'][1]
        payload=json.loads(left['messages'][1]['content'])
        assert set(payload)=={'focus_input','descriptions'}
        assert all(r['constraint'] is None for r in study.rows_for(ca,a))
        assert refs[ca['id']]['contract_version']==4
        reviewed_membership_rows(ca,refs[ca['id']],application=ca['application'])

@pytest.mark.parametrize('lang',['en','zh'])
def test_authored_reference_boundaries_have_explicit_reviewed_labels_and_evidence(lang):
    inputs,refs=packet()
    expected={'unique-other':['nonmatch','match'],'direct-name':['nonmatch','match'],
        'missing-antecedent':['uncertain','uncertain'],'incompatible-act':['nonmatch','nonmatch']}
    reasons={'unique-other':['different_participant','direct_act'],'direct-name':['different_participant','direct_act'],
        'missing-antecedent':['missing_reference','missing_reference'],'incompatible-act':['different_act','different_act']}
    for variant,labels in expected.items():
        key=f'reference-{variant}-{lang}';ref=refs[key]
        focus=[r for r in ref['judgments'] if r['record']==2]
        assert [r['label'] for r in focus]==labels and [r['reason'] for r in focus]==reasons[variant]
        assert all(r['rationale'] and r['evidence'] for r in ref['judgments'])
        assert [r['record'] for r in focus[0]['context_evidence']]==([0,1] if variant=='unique-other' else [])
    assert len(inputs['cases'][f'reference-missing-antecedent-{lang}']['records'])==1

@pytest.mark.parametrize('damage',['profile','group','version','case'])
def test_single_case_executor_hook_rejects_unfrozen_configuration(damage):
    inputs,_=packet();entry=deepcopy(inputs['schedule'][0]);case=inputs['cases'][entry['case']]
    if damage=='profile':entry['prompt_profile']='unknown'
    elif damage=='group':entry['descriptions_per_call']=1
    elif damage=='version':entry['version']=3
    else:entry['case']='wrong'
    with pytest.raises(ValueError):study.rows_for(case,entry)

def test_canonical_grading_gate_keeps_existing_error_but_rejects_new_label_regressions():
    inputs,refs=packet();cases,schedule=inputs['cases'],inputs['schedule'];observed=oracle(inputs,refs)
    for o,e in zip(observed,schedule):
        if e['case']=='original-reflection/with-prior':o['labels']=['match','nonmatch']
    result=study.score(cases,refs,schedule,observed)
    assert result['candidate_reference_screen_passed'] and not result['qualified']
    assert result['by_family']['reference'][study.PROFILES[1]]['correct_conditions']==12
    assert result['by_family']['act'][study.PROFILES[1]]['correct_conditions']==3
    broken=deepcopy(observed)
    target=next(o for o,e in zip(broken,schedule) if e['case']=='both-en' and e['prompt_profile']==study.PROFILES[1])
    target['labels'][1]='uncertain'
    result=study.score(cases,refs,schedule,broken)
    assert not result['candidate_reference_screen_passed'] and len(result['new_act_label_errors'])==1

@pytest.mark.parametrize('label',['match','nonmatch','uncertain'])
def test_constant_answers_cannot_pass_reference_screen(label):
    inputs,refs=packet();schedule=inputs['schedule']
    result=study.score(inputs['cases'],refs,schedule,[dict(id=e['id'],labels=[label,label]) for e in schedule])
    assert not result['candidate_reference_screen_passed']

@pytest.mark.parametrize('failure',['missing','failed'])
def test_incomplete_comparator_cannot_pass_and_keeps_planned_denominators(failure):
    inputs,refs=packet();schedule=inputs['schedule'];observed=oracle(inputs,refs)
    if failure=='missing':observed.pop(0)
    else:observed[0]=dict(id=observed[0]['id'],error='authored software failure')
    result=study.score(inputs['cases'],refs,schedule,observed)
    assert not result['candidate_reference_screen_passed'] and result['summary']['all']['completed']==31
    empty=study.score(inputs['cases'],refs,schedule,[])
    for profile in study.PROFILES:
        summary=empty['by_profile'][profile]
        assert summary['planned']==16 and summary['planned_pairs']==32 and summary['completed']==0
        assert summary['planned_positive_labels']==10 and summary['planned_uncertainty_labels']==8

def test_duplicate_or_unexpected_observations_are_rejected():
    inputs,refs=packet();observed=oracle(inputs,refs)
    with pytest.raises(ValueError):study.score(inputs['cases'],refs,inputs['schedule'],observed+[observed[0]])
    observed[0]['id']='not scheduled'
    with pytest.raises(ValueError):study.score(inputs['cases'],refs,inputs['schedule'],observed)
