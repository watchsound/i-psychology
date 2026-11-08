"""End-to-end software transport simulation, never model-quality evidence."""
from copy import deepcopy
from dataclasses import replace
import hashlib
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from psychology_base.eval.benchmarks import local_membership_diagnostic as executor
from psychology_base.eval.benchmarks.local_membership_audit import audit_membership_capture
from psychology_base.eval.benchmarks.membership_focus import condition,focus_rows,grade
from psychology_base.llm.execution import current_operation
from psychology_base.llm.providers.ollama import OllamaProvider

MODEL='declared-software-test';DIGEST='a'*64
ROOT=Path(__file__).parents[1]

def setup(tmp_path,monkeypatch,damage=None):
    read_log=[];checks=[];wire=[];operations=[];chat_count=0
    def read(p):read_log.append(p);return json.loads(p.read_text(encoding='utf-8'))
    def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    fixture=read(ROOT/'benchmarks/fixtures/application_context/pointwise-denial-control-v1.json')
    case,ref=fixture['case'],fixture['reference'];cases={case['id']:case};refs={case['id']:ref}
    schedule=[dict(condition(case,profile='independent-membership-v1'),descriptions_per_call=size) for size in (2,1)]
    for e,name in zip(schedule,('joint','pointwise')):e['id']+='/'+name
    out=tmp_path/'out';out.mkdir();plan_path=tmp_path/'plan.json';result_path=tmp_path/'result.json'
    save(out/'inputs.json',dict(cases=cases,schedule=schedule));save(out/'references.json',refs)
    (out/'declared-source.txt').write_text('Explicit software test source',encoding='utf-8')
    limits=dict(max_native_attempts=3,max_metadata_attempts=64,max_wall_s=3600,finalization_reserve_s=180,
        request_timeout_s=180,operation_timeout_s=300,metadata_timeout_s=180,metadata_ttl_s=60,
        context_length=8192,max_input=8000,max_tokens=64,paid_api_usd='0')
    plan=dict(limits=limits,paid_calls=False,date='2026-10-28',model=MODEL,digest=DIGEST,seed=0,
        contract_version=4,input_sha256=sha(out/'inputs.json'),references_sha256=sha(out/'references.json'),
        declared_source_sha256=sha(out/'declared-source.txt'))
    save(plan_path,plan)
    def validate(p,o):
        assert sha(o/'declared-source.txt')==p['declared_source_sha256'];checks.append(True)
    archive=SimpleNamespace(base=SimpleNamespace(read=read,save=save,sha=sha),validate_frozen_sources=validate)
    def fake_open(provider,request,*,timeout):
        nonlocal chat_count
        assert out/'references.json' not in read_log
        path=request.full_url.rsplit('/',1)[-1]
        payload=json.loads(request.data) if request.data else None
        wire.append((path,payload,timeout))
        if path=='tags':data=dict(models=[dict(name=MODEL,size=10,digest=DIGEST)])
        elif path=='show':data=dict(capabilities=['completion'],details=dict(family='test'),model_info={'test.context_length':8192})
        elif path=='ps':data=dict(models=[])
        else:
            assert path=='chat';chat_count+=1
            op=current_operation.get();operations.append((op.id,op.deadline))
            if damage=='transport' and chat_count==3:raise TimeoutError('Declared second pointwise request failed')
            labels=[['match','nonmatch'],['match'],['nonmatch']][chat_count-1]
            if damage=='labels':labels=['nonmatch','nonmatch']
            if damage=='format' and chat_count==3:labels=['match','nonmatch']
            data=dict(model=MODEL,done=True,done_reason='stop',message=dict(content=json.dumps(dict(labels=labels))),
                prompt_eval_count=100,eval_count=12)
        return BytesIO(json.dumps(data).encode())
    monkeypatch.setattr(OllamaProvider,'_open',fake_open)
    visited=[]
    def rows_for(single_case,entry):
        assert single_case['id']==entry['case'];visited.append(entry['id'])
        return focus_rows(single_case,entry,format_version=4)
    contrast=[dict(kind='grouping',left=schedule[0]['id'],right=schedule[1]['id'],relation='same')]
    grading=lambda c,r,s,o:grade(c,r,s,o,contrast_pairs=contrast)
    args=dict(out=out,plan_path=plan_path,result_path=result_path,archive=archive,rows_for=rows_for,grade=grading)
    return SimpleNamespace(args=args,read=read,save=save,wire=wire,ops=operations,checks=checks,visited=visited,
        schedule=schedule,cases=cases,out=out,result_path=result_path,plan_path=plan_path)

@pytest.mark.parametrize('damage',[None,'format','transport'])
def test_actual_executor_and_real_router_capture_every_call_and_fail_closed(tmp_path,monkeypatch,damage):
    state=setup(tmp_path,monkeypatch,damage)
    original_connect=executor.socket.socket.connect
    executor.run_local_membership_diagnostic(**state.args)
    assert executor.socket.socket.connect is original_connect
    result=state.read(state.result_path);obs=state.read(state.out/'observations.json');events=state.read(state.out/'transport.json')
    assert state.visited==[e['id'] for e in state.schedule] and len(state.checks)==2
    assert result['native_attempts']==3 and result['paid_api_usd']=='0' and not result['qualified']
    assert [len(json.loads(p['messages'][1]['content'])['descriptions']) for path,p,t in state.wire if path=='chat']==[2,1,1]
    assert state.ops[1]==state.ops[2] and state.ops[0][0]!=state.ops[1][0]
    assert all(0<t<=180 for path,p,t in state.wire)
    replay=audit_membership_capture(events,obs,state.schedule,
        rows_for=lambda e:focus_rows(state.cases[e['case']],e,format_version=4),snapshot=lambda p:state.read(state.out/p),
        model=MODEL,digest=DIGEST,max_native=3,max_metadata=64)
    if damage is None:
        assert result['completed'] and len(replay['predictions'])==2
        assert obs[0]['labels']==obs[1]['labels']==['match','nonmatch']
    else:
        assert 'labels' not in obs[1] and len(replay['predictions'])==1
        assert obs[1]['format_failure']==(damage=='format')
        assert result['completed']==(damage=='format')
        assert result['graded']['summary']['all']['failed']==1
    with pytest.raises(FileExistsError):executor.run_local_membership_diagnostic(**state.args)
    assert len([x for x in state.wire if x[0]=='chat'])==3

def test_bad_single_case_boundary_fails_before_attestation(tmp_path,monkeypatch):
    state=setup(tmp_path,monkeypatch)
    state.args['rows_for']=lambda case,e:case['not_a_case_mapping'][e['case']]
    with pytest.raises(KeyError):executor.run_local_membership_diagnostic(**state.args)
    assert not state.wire and not (state.out/'started.json').exists()

def test_invalid_shared_execution_policy_cannot_consume_or_dispatch_a_scope(tmp_path,monkeypatch):
    state=setup(tmp_path,monkeypatch)
    plan=state.read(state.plan_path);plan['limits']['operation_timeout_s']=600;state.save(state.plan_path,plan)
    with pytest.raises(ValueError):executor.run_local_membership_diagnostic(**state.args)
    assert not state.wire and not (state.out/'started.json').exists()


def test_membership_limits_are_checked_before_any_metadata_or_consumed_scope(tmp_path,monkeypatch):
    state=setup(tmp_path,monkeypatch)
    plan=state.read(state.plan_path);plan['limits']['max_tokens']=1024;state.save(state.plan_path,plan)
    with pytest.raises(ValueError):executor.run_local_membership_diagnostic(**state.args)
    assert not state.wire and not (state.out/'started.json').exists()


@pytest.mark.parametrize('change',[
    {'limits':{'metadata_ttl_s':120}}, {'limits':{'metadata_timeout_s':True}},
    {'stop_on_label_error':'true'},
    {'stop_on_label_error':True,'assessment_kind':'reference-extraction-v2'},
])
def test_lazy_metadata_settings_and_invalid_stop_protocol_are_rejected_offline(tmp_path,monkeypatch,change):
    state=setup(tmp_path,monkeypatch)
    plan=state.read(state.plan_path)
    plan['limits'].update(change.get('limits',{}))
    plan.update({k:v for k,v in change.items() if k!='limits'})
    state.save(state.plan_path,plan)
    with pytest.raises(ValueError):executor.run_local_membership_diagnostic(**state.args)
    assert not state.wire and not (state.out/'started.json').exists()


def test_explicit_semantic_stop_keeps_unattempted_conditions_in_denominator(tmp_path,monkeypatch):
    state=setup(tmp_path,monkeypatch,damage='labels')
    plan=state.read(state.plan_path);plan['stop_on_label_error']=True;state.save(state.plan_path,plan)
    executor.run_local_membership_diagnostic(**state.args)
    result=state.read(state.result_path)
    assert not result['completed'] and result['stop_reason']=='semantic_gate'
    assert result['native_attempts']==1 and result['graded']['summary']['all']['missing']==1
    assert result['graded']['summary']['all']['measured_pairs']==2
    assert result['graded']['summary']['all']['planned_pairs']==4
    assert len([x for x in state.wire if x[0]=='chat'])==1

def test_condition_admission_reserves_the_whole_two_call_operation(tmp_path,monkeypatch):
    state=setup(tmp_path,monkeypatch)
    original_time=executor.time
    clock=[original_time.monotonic()]
    monkeypatch.setattr(executor,'time',SimpleNamespace(monotonic=lambda:clock[0],time_ns=original_time.time_ns))
    original_open=OllamaProvider._open
    def advance_after_preflight(provider,request,*,timeout):
        result=original_open(provider,request,timeout=timeout)
        if request.full_url.endswith('/ps'):clock[0]+=3121
        return result
    monkeypatch.setattr(OllamaProvider,'_open',advance_after_preflight)
    executor.run_local_membership_diagnostic(**state.args)
    result=state.read(state.result_path)
    assert not result['completed'] and result['native_attempts']==0 and result['metadata_attempts']==3
    assert result['graded']['summary']['all']['missing']==2
