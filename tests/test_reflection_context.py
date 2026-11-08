"""Authored software checks; fixture labels never stand in for model accuracy."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from psychology_base.applications.measurement.semantic_membership import membership_inputs
from psychology_base.eval.benchmarks.membership_focus import focus_rows,grade
from psychology_base.llm.providers.ollama import OllamaProvider
from tests.test_membership_focus import packet as legacy_packet

ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location('reflection_context_tests',ROOT/'docs/experiments/phase-9-memory-navigation/run-reflection-context-v1.py')
study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)

def software_source():
    texts=['Almost finished; one step left.','12 divided by 3 is 4.','Check the previous arithmetic.',
           '8 divided by 2 is 4.','Can you explain the earlier error?','I understand my earlier mistake now.']
    case=dict(id='authored-software-source',language='en',application='companion',origin='authored',split='regression',
        descriptions=['the encouraging near-completion remark',"the learner's first calculation"],
        records=[dict(id=i,speaker='user' if i%2 else 'tutor',text=t,source_order=dict(group='a'*32,position=i))
                 for i,t in enumerate(texts)])
    labels=[['match','nonmatch','nonmatch','nonmatch','nonmatch','nonmatch'],
            ['nonmatch','match','nonmatch','match','nonmatch','nonmatch']]
    judgments=[]
    for s,vector in enumerate(labels):
        for i,label in enumerate(vector):
            reason='wrong_outer_actor' if s==1 and i%2==0 else ('direct_act' if label=='match' else 'different_act')
            judgments.append(dict(slot=s,record=i,label=label,reason=reason,
                rationale='Explicit authored software annotation; not model evidence.',
                evidence=[dict(start=0,end=len(texts[i]),quote=texts[i])],context_evidence=[]))
    return case,dict(id=case['id'],contract_version=4,reviewed=True,reviewer='Authored software fixture',judgments=judgments)

def packet():
    c,r=software_source()
    return study.build_packet(c,r,json.loads(study.FIXTURE.read_text(encoding='utf-8')))

def oracle(inputs,refs):
    result=[]
    for e in inputs['schedule']:
        c=inputs['case']['cases'][e['case']]
        truth={j['slot']:j['label'] for j in refs[e['case']]['judgments'] if j['record']==c['focus_id']}
        result.append(dict(id=e['id'],labels=[truth[s] for s in e['description_order']]))
    return result

def test_packet_scope_same_speaker_positive_and_no_source_mutation():
    c,r=software_source();original=deepcopy((c,r));bank=json.loads(study.FIXTURE.read_text(encoding='utf-8'))
    inputs,refs=study.build_packet(c,r,bank)
    assert (c,r)==original and len(inputs['case']['cases'])==len(refs)==len(inputs['schedule'])==8
    assert [e['variant'] for e in inputs['schedule']]==['with-prior','focus-only','focus-only','with-prior']*2
    for e in inputs['schedule']:
        case=inputs['case']['cases'][e['case']]
        rows=study.rows_for(inputs['case'],e)
        assert len(rows)==2 and all(row['constraint'] is None and row['input_format_version']==4 for row in rows)
        assert json.loads(rows[0]['pair'][0])['focus_utterance']['speaker']=='user'
    true_user=next(e for e in inputs['schedule'] if e['act']=='same-speaker-encouragement')
    assert oracle(inputs,refs)[inputs['schedule'].index(true_user)]['labels']==['match','nonmatch']

def test_context_ablation_changes_only_prior_and_preserves_original_focus_wire():
    inputs,_=packet();bundle=inputs['case'];source,_=software_source()
    for contrast in bundle['contrast_pairs']:
        a,b=[next(e for e in inputs['schedule'] if e['id']==contrast[k]) for k in ('left','right')]
        x,y=study.rows_for(bundle,a),study.rows_for(bundle,b)
        assert [r['pair'][1] for r in x]==[r['pair'][1] for r in y]
        full,isolated=json.loads(x[0]['pair'][0]),json.loads(y[0]['pair'][0])
        assert full['focus_utterance']==isolated['focus_utterance'] and isolated['prior_utterances']==[]
        assert full['context_scope']==isolated['context_scope'] and full['prior_utterances']
        if a['act'].startswith('original-'):
            focus=bundle['cases'][a['case']]['focus_id']
            prior=[r for r in membership_inputs(source,application='companion',format_version=4) if r['record']==focus]
            assert study.wire_payload(x,'authored-test-model')==study.wire_payload(prior,'authored-test-model')
            expected=[3,4] if focus==5 else [0]
            assert [r['text'] for r in full['prior_utterances']]==[source['records'][i]['text'] for i in expected]

def test_context_support_cannot_be_silently_removed():
    c,r=software_source();text=c['records'][4]['text']
    next(j for j in r['judgments'] if j['slot']==0 and j['record']==5)['context_evidence']=[dict(record=4,start=0,end=len(text),quote=text)]
    with pytest.raises(ValueError,match='context-independent'):
        study.build_packet(c,r,json.loads(study.FIXTURE.read_text(encoding='utf-8')))

def test_grader_separates_correctness_missing_failure_and_stability():
    inputs,refs=packet();args=inputs['case'],refs,inputs['schedule']
    empty=study.score(*args,[])
    assert empty['summary']['all']['planned_pairs']==16 and empty['summary']['all']['missing']==8
    assert not any(p['complete'] for p in empty['contrasts'])
    observations=oracle(inputs,refs);full=study.score(*args,observations)
    assert full['summary']['all']['correct_pairs']==16 and full['summary']['all']['correct_conditions']==8
    assert all(p['complete'] and p['same'] for p in full['contrasts']) and not full['qualified']
    observations[0]=dict(id=observations[0]['id'],error='Authored software failure');observations.pop()
    partial=study.score(*args,observations)
    assert partial['summary']['all']['failed']==partial['summary']['all']['missing']==1
    assert sum(p['complete'] for p in partial['contrasts'])==2

@pytest.mark.parametrize('labels',[['nonmatch','nonmatch'],['match','nonmatch'],['nonmatch','match']])
def test_stable_constant_answers_cannot_pass_semantics(labels):
    inputs,refs=packet()
    result=study.score(inputs['case'],refs,inputs['schedule'],[dict(id=e['id'],labels=labels) for e in inputs['schedule']])
    assert all(p['complete'] and p['same'] for p in result['contrasts'])
    assert result['summary']['all']['correct_pairs']<16 and result['summary']['all']['correct_conditions']<8

@pytest.mark.parametrize('mutation',[
    lambda p:p[0].update(relation='different'),lambda p:p[0].update(left='unknown'),
    lambda p:p[0].update(right=p[0]['left']),lambda p:p.append(deepcopy(p[0])),
    lambda p:p[0].update(kind=''),lambda p:p[0].update(extra=True),
])
def test_explicit_contrasts_reject_bad_identity_or_reference_claims(mutation):
    inputs,refs=packet();mutation(inputs['case']['contrast_pairs'])
    with pytest.raises(ValueError):study.score(inputs['case'],refs,inputs['schedule'],[])

def test_explicit_legacy_contrasts_and_default_input_remain_identical():
    cases,refs,schedule=legacy_packet();old=grade(cases,refs,schedule,[])
    pairs=[dict(kind=c['kind'],left=c['left'],right=c['right'],relation=c['expected_relation']) for c in old['contrasts']]
    assert grade(cases,refs,schedule,[],contrast_pairs=pairs)==old
    for e in schedule:
        assert focus_rows(cases[e['case']],e)==focus_rows(cases[e['case']],e,format_version=3)
    for bad in (1,2,5,True,'4'):
        with pytest.raises(ValueError):focus_rows(cases[schedule[0]['case']],schedule[0],format_version=bad)

def test_adapter_tightens_real_capture_caps_without_provider_replacement():
    owner=study.execution_owner()
    provider=OllamaProvider(base_url='http://127.0.0.1:11434',local_only=True)
    kwargs=dict(max_attempts=12,max_metadata=128,deadline=10,observer=lambda *_:None,validate_chat=lambda *_:None)
    cap=owner.LocalNativeCapture(provider,**kwargs)
    assert cap.provider is provider and cap.max_attempts==8 and cap.max_metadata==96
    assert owner.LIMITS['max_native_attempts']==8 and owner.LIMITS['max_wall_s']==1800
    with pytest.raises(ValueError):owner.LocalNativeCapture(provider,**dict(kwargs,max_attempts=11))
