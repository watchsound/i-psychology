"""Authored software controls; simulated responses are not model evidence."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import importlib.util
from itertools import product
import json
from pathlib import Path
import time

import pytest

from psychology_base.applications.measurement.local_query_binding import (
    MEMBERSHIP_PROMPT,LocalChatMembershipAssessor,membership_prompt)
from psychology_base.eval.benchmarks.application_calls import LocalNativeCapture
from psychology_base.eval.benchmarks.local_membership_audit import membership_wire_payload
from psychology_base.llm.providers.ollama import OllamaProvider
from tests.test_local_chat_membership import router
from tests.test_reflection_context import packet as reflection_packet

ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location('independent_membership_tests',
    ROOT/'docs/experiments/phase-9-memory-navigation/run-independent-membership-v1.py')
study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)

PARAGRAPH=('Evaluate each description independently. The supplied descriptions are not exhaustive or '
    'mutually exclusive choices: none, one, or several may match. Apply the label definitions '
    'below to each description; do not select a description merely because it is the closest '
    'among those supplied.\n')

def packet():
    inputs,refs=reflection_packet()
    bank=json.loads(study.FIXTURE.read_text(encoding='utf-8'))
    return study.build_packet(inputs,refs,bank)

def oracle(inputs,refs):
    """Explicit software oracle for grading tests only; never inference."""
    result=[]
    for e in inputs['schedule']:
        case=inputs['cases'][e['case']]
        truth={j['slot']:j['label'] for j in refs[e['case']]['judgments'] if j['record']==case['focus_id']}
        result.append(dict(id=e['id'],labels=[truth[s] for s in e['description_order']]))
    return result

def test_baseline_and_identity_are_exactly_preserved():
    assert hashlib.sha256(MEMBERSHIP_PROMPT.encode()).hexdigest()=='1d2eff4226b9d063d42e9374afa287bdcec75d005f6439fefc72c1008674a985'
    value,_=router()
    for version,pin in [(3,'8ed227360079a0f75218f1c340d18bb1d77ed179f279c295621b56e862472643'),
                        (4,'e7fed5ccad8c8dece3b7c88351368b20debe41692b23afdfefb17df86a0b875a')]:
        assert LocalChatMembershipAssessor(value,format_version=version).identity==pin
    candidate=membership_prompt(study.PROFILES[1])
    boundary='Allowed labels: match, nonmatch, uncertain. No additional keys or explanation.\n'
    assert candidate==MEMBERSHIP_PROMPT.replace(boundary,boundary+PARAGRAPH)
    assert candidate.count(PARAGRAPH)==1 and candidate.replace(PARAGRAPH,'')==membership_prompt()

def test_packet_retains_all_references_and_counterbalances_only_prompt():
    parent,annotations=reflection_packet();before=deepcopy((parent,annotations))
    inputs,refs=study.build_packet(parent,annotations,json.loads(study.FIXTURE.read_text(encoding='utf-8')))
    assert (parent,annotations)==before
    assert len(inputs['cases'])==10 and len(inputs['schedule'])==20 and len(inputs['contrast_pairs'])==10
    for key,c in parent['case']['cases'].items():
        assert inputs['cases'][key]==c and refs[key]==annotations[key]
    for i in range(10):
        a,b=inputs['schedule'][i*2:i*2+2]
        assert a['case']==b['case'] and a['description_order']==b['description_order']
        assert (a['prompt_profile'],b['prompt_profile'])==(study.PROFILES if i%2==0 else study.PROFILES[::-1])
        rows=study.rows_for(inputs['cases'],a)
        assert rows==study.rows_for(inputs['cases'],b)
        assert len(rows)==2 and all(r['constraint'] is None for r in rows)
        x,y=[membership_wire_payload(rows,'authored',prompt_profile=e['prompt_profile']) for e in (a,b)]
        assert x['messages'][1:]==y['messages'][1:] and x['options']==y['options']
        assert 'reviewer' not in x['messages'][1]['content'] and 'rationale' not in x['messages'][1]['content']
    assert {e['case'] for e in inputs['schedule'][:10]}=={
        'original-reflection/with-prior','original-calculation/with-prior',
        'same-speaker-encouragement/with-prior','both-en','referent-zh-ambiguous'}

@pytest.mark.parametrize('vector',list(product(('match','nonmatch','uncertain'),repeat=2)))
def test_candidate_accepts_all_vectors_in_one_call_and_rejects_profile_mutation(vector):
    inputs,_=packet();entry=inputs['schedule'][1];rows=study.rows_for(inputs['cases'],entry)
    value,seen=router(lambda c:replace(c,content=json.dumps(dict(labels=list(vector)))))
    baseline=LocalChatMembershipAssessor(value,format_version=4)
    candidate=LocalChatMembershipAssessor(value,format_version=4,prompt_profile=study.PROFILES[1])
    assert baseline.identity!=candidate.identity
    assert candidate(rows)==list(vector) and len(seen)==1
    assert seen[0][0][0].content==membership_prompt(study.PROFILES[1])
    assert seen[0][1]['use_cache'] is False
    candidate.prompt_profile=study.PROFILES[0]
    with pytest.raises(ValueError):candidate(rows)
    assert len(seen)==1

def test_screen_counts_missing_failures_and_constant_nonmatch_honestly():
    inputs,refs=packet();cases,schedule=inputs['cases'],inputs['schedule']
    observed=oracle(inputs,refs)
    good=study.score(cases,refs,schedule,observed)
    assert good['candidate_screen_passed'] and not good['qualified']
    assert all(v==dict(planned=10,completed=10,correct=10) for v in good['by_profile'].values())
    assert not study.score(cases,refs,schedule,[])['candidate_screen_passed']
    assert not study.score(cases,refs,schedule,observed[:-1])['candidate_screen_passed']
    candidate_id=next(e['id'] for e in schedule if e['prompt_profile']==study.PROFILES[1])
    assert not study.score(cases,refs,schedule,[o for o in observed if o['id']!=candidate_id])['candidate_screen_passed']
    failed=[dict(id=o['id'],error='Authored transport failure') if o['id']==candidate_id else deepcopy(o) for o in observed]
    report=study.score(cases,refs,schedule,failed)
    assert report['summary']['all']['failed']==1 and not report['candidate_screen_passed']
    constant=[dict(id=e['id'],labels=['nonmatch','nonmatch']) for e in schedule]
    report=study.score(cases,refs,schedule,constant)
    assert not report['candidate_screen_passed'] and report['summary']['all']['unsafe_exclusions']>0

def test_executor_keeps_real_owners_and_tightens_scope_without_network():
    owner=study.execution_owner()
    provider=owner.OllamaProvider(base_url='http://127.0.0.1:11434',simple_model=owner.MODEL,
        advanced_model=owner.MODEL,request_timeout_s=180,model_metadata_timeout_s=60,model_metadata_ttl_s=60,
        local_only=True,expected_digest=owner.DIGEST,context_length=8192,seed=0,json_mode=True)
    assert isinstance(provider,OllamaProvider)
    assert provider.model_metadata_timeout_s==180 and provider.model_metadata_ttl_s==60
    assert provider.request_timeout_s==180 and provider.local_only and provider.json_mode
    capture=owner.LocalNativeCapture(provider,max_attempts=42,max_metadata=256,deadline=time.monotonic()+100,
        observer=lambda *args:None,validate_chat=lambda *args:None)
    assert isinstance(capture,LocalNativeCapture)
    assert capture.max_attempts==20 and capture.max_metadata==128
    assert owner.LIMITS['max_wall_s']==2400 and owner.LIMITS['paid_api_usd']=='0'
    value,_=router()
    candidate=owner.LocalChatMembershipAssessor(value,prompt_profile=study.PROFILES[1])
    assert isinstance(candidate,LocalChatMembershipAssessor) and candidate.format_version==4
    with pytest.raises(ValueError):owner.LocalNativeCapture(provider,max_attempts=21,max_metadata=256,deadline=1)
