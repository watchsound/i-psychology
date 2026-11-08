"""Disclosed fake traces test tamper rejection; not empirical model evidence."""
from copy import deepcopy
import hashlib
import json

import pytest

from psychology_base.applications.measurement.semantic_membership import membership_inputs
from psychology_base.eval.benchmarks.local_membership_audit import audit_membership_capture,membership_wire_payload

MODEL='authored-test-model';DIGEST='a'*64

def fixture():
    case=dict(language='en',descriptions=['the encouraging near-completion remark',"the learner's first calculation"],
        records=[dict(id=0,speaker='user',text='I see my earlier mistake now.',source_order=dict(group='b'*32,position=0))])
    rows=membership_inputs(case,application='companion',format_version=4)
    content=json.dumps(dict(labels=['nonmatch','nonmatch']))
    raw=dict(model=MODEL,done=True,done_reason='stop',message=dict(content=content),prompt_eval_count=30,eval_count=7)
    completion=dict(content=content,provider='ollama',model_id=MODEL,prompt_tokens=30,completion_tokens=7,finish_reason='stop',
        metadata=dict(model_attestation=dict(digest=DIGEST),local_only=True,done=True,response_model=MODEL,
            context_length=8192,seed=0,json_mode=True))
    events=[]
    for path,body,response in [('/api/tags',None,dict(models=[])),('/api/chat',membership_wire_payload(rows,MODEL),raw)]:
        text=json.dumps(response)
        events.append(dict(sequence=len(events)+1,path=path,condition='c1',request=body,
            response=response,response_text=text,response_sha256=hashlib.sha256(text.encode()).hexdigest(),elapsed_ms=1))
    obs=[dict(id='c1',labels=['nonmatch','nonmatch'],native_calls=[dict(messages=deepcopy(events[-1]['request']['messages']),
        row_indices=[0,1],completion=completion)])]
    return rows,events,obs,[dict(id='c1',prompt_profile='baseline-v1')]

def replay(rows,events,obs,schedule,**overrides):
    snapshots={}
    for e in events:
        snapshots[f"native-{e['sequence']:03d}-finished.json"]=deepcopy(e)
        snapshots[f"native-{e['sequence']:03d}-started.json"]={k:deepcopy(e[k]) for k in ('sequence','path','condition','request')}
    kwargs=dict(rows_for=lambda _:rows,snapshot=lambda p:snapshots[p],model=MODEL,digest=DIGEST,max_native=1,max_metadata=1)
    kwargs.update(overrides)
    return audit_membership_capture(events,obs,schedule,**kwargs)

def rewrite_raw(events,obs):
    e=events[-1];text=json.dumps(e['response']);e.update(response_text=text,response_sha256=hashlib.sha256(text.encode()).hexdigest())

def test_exact_native_replay_and_separate_request_denominators():
    rows,events,obs,schedule=fixture();r=replay(rows,events,obs,schedule)
    assert r==dict(native_attempts=1,native_responses=1,metadata_attempts=1,predictions={'c1':['nonmatch','nonmatch']})

@pytest.mark.parametrize('mutation',[
    lambda e,o:o[0]['labels'].__setitem__(0,'match'),
    lambda e,o:e[-1].update(response_sha256='0'*64),
    lambda e,o:e[-1]['request']['options'].update(num_ctx=4096),
    lambda e,o:e[-1]['response'].update(model='wrong-model'),
    lambda e,o:e[-1]['response'].update(done_reason='length'),
    lambda e,o:o[0]['native_calls'][0]['completion'].update(prompt_tokens=31),
    lambda e,o:o[0]['native_calls'][0].update(row_indices=[1,0]),
    lambda e,o:o[0]['native_calls'][0]['completion']['metadata']['model_attestation'].update(digest='b'*64),
    lambda e,o:e[-1].update(condition='unknown'),
    lambda e,o:o.append(deepcopy(o[0])),
    lambda e,o:e[0].update(path='/api/show',request={'model':'wrong-model'}),
    lambda e,o:o[0]['native_calls'][0]['completion']['metadata'].update(local_only=False),
    lambda e,o:o[0]['native_calls'][0]['completion']['metadata'].update(seed=False),
    lambda e,o:o[0]['native_calls'][0]['completion']['metadata'].update(json_mode=False),
])
def test_changed_trace_or_observation_cannot_pass(mutation):
    rows,events,obs,schedule=fixture();mutation(events,obs)
    # Keep JSON/hash consistent for semantic mutations so identity/token checks
    # also run rather than relying exclusively on byte-integrity rejection.
    if events[-1]['response_sha256']!='0'*64:rewrite_raw(events,obs)
    with pytest.raises(ValueError):replay(rows,events,obs,schedule)

def test_no_completion_or_missing_observation_never_becomes_a_prediction():
    rows,events,obs,schedule=fixture()
    with pytest.raises(ValueError):replay(rows,events,[],schedule)
    obs[0].pop('labels');obs[0]['error']='Authored postcondition failure';obs[0]['native_calls'][0]['completion']=None
    result=replay(rows,events,obs,schedule)
    assert result['native_attempts']==1 and result['predictions']=={}

def test_metadata_failure_before_dispatch_stays_unmeasured_and_caps_remain_strict():
    rows,events,obs,schedule=fixture()
    with pytest.raises(ValueError):replay(rows,events,obs,schedule,max_native=0)
    with pytest.raises(ValueError):replay(rows,events,obs,schedule,max_metadata=0)
    events=events[:1];events[0]={k:v for k,v in events[0].items() if k not in ('response','response_text','response_sha256')}
    events[0]['error']='Authored transport timeout'
    result=replay(rows,events,[dict(id='c1',error='Authored failure',native_calls=[])],schedule)
    assert result['native_attempts']==0 and result['metadata_attempts']==1 and result['predictions']=={}

def test_malformed_native_labels_cannot_be_accepted():
    rows,events,obs,schedule=fixture()
    malformed=json.dumps(dict(labels=['nonmatch','nonmatch','match']))
    events[-1]['response']['message']['content']=malformed
    obs[0]['native_calls'][0]['completion']['content']=malformed
    obs[0]['labels']=['nonmatch','nonmatch','match'];rewrite_raw(events,obs)
    with pytest.raises(ValueError,match='malformed'):replay(rows,events,obs,schedule)
