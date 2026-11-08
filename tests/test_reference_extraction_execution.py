"""Actual executor/router/native audit with disclosed software HTTP responses."""
from copy import deepcopy
from io import BytesIO
import json

import pytest

from psychology_base.eval.benchmarks.local_membership_audit import audit_membership_capture
from psychology_base.eval.benchmarks.reference_extraction import reviewed_reference, authored_schema_example, grade
from psychology_base.eval.benchmarks.membership_focus import condition,focus_rows
from psychology_base.eval.benchmarks import local_membership_diagnostic as executor
from psychology_base.llm.providers.ollama import OllamaProvider
from tests.test_local_membership_diagnostic import setup as prior_setup, MODEL, DIGEST


def setup(tmp_path,monkeypatch,damage=None):
    state=prior_setup(tmp_path,monkeypatch)
    case=next(iter(state.cases.values()))
    schedule=[dict(condition(case,repeat=repeat),version=4,descriptions_per_call=2) for repeat in (False,True)]
    rows=focus_rows(case,schedule[0],format_version=4);focus=json.loads(rows[0]['pair'][0])
    declared=dict(names={},focus=[['我',0,'deictic']],readings=[['speaker:user']],coverage='enumerated',
        rationale='Explicit authored software role-reference control, not measured accuracy.')
    ref=reviewed_reference(case,schedule[0],declared)
    raw,_=authored_schema_example(ref,focus)
    state.save(state.out/'inputs.json',dict(cases=state.cases,schedule=schedule))
    state.save(state.out/'references.json',{case['id']:ref})
    plan=state.read(state.plan_path)
    plan.update(assessment_kind='reference-extraction-v1',input_sha256=state.args['archive'].base.sha(state.out/'inputs.json'),
        references_sha256=state.args['archive'].base.sha(state.out/'references.json'))
    plan['limits'].update(max_native_attempts=2,max_input=7168,max_tokens=1024)
    state.save(state.plan_path,plan)
    state.args.update(rows_for=lambda c,e:focus_rows(c,e,format_version=4),grade=grade)
    original=OllamaProvider._open
    count=0
    def fake_open(provider,request,*,timeout):
        nonlocal count
        if request.full_url.endswith('/chat'):
            count+=1
            if damage=='transport' and count==2:raise TimeoutError('Declared extraction transport failure')
            response=json.load(original(provider,request,timeout=timeout))
            response['message']['content']=json.dumps(raw,ensure_ascii=False)
            if damage=='format' and count==2:response['message']['content']='{"coverage":"open"}'
            if damage=='attestation' and count==2:response['model']='different-model'
            return BytesIO(json.dumps(response).encode())
        return original(provider,request,timeout=timeout)
    monkeypatch.setattr(OllamaProvider,'_open',fake_open)
    state.schedule=schedule
    return state


@pytest.mark.parametrize('damage',[None,'format','transport','attestation'])
def test_extraction_uses_shared_executor_and_raw_audit_without_label_or_reference_leakage(tmp_path,monkeypatch,damage):
    state=setup(tmp_path,monkeypatch,damage)
    executor.run_local_membership_diagnostic(**state.args)
    result=state.read(state.result_path);obs=state.read(state.out/'observations.json');events=state.read(state.out/'transport.json')
    assert result['assessment_kind']=='reference-extraction-v1' and result['native_attempts']==2
    assert not result['qualified'] and result['paid_api_usd']=='0'
    assert all('labels' not in o for o in obs)
    for path,payload,timeout in state.wire:
        if path=='chat':assert set(json.loads(payload['messages'][1]['content']))=={'focus_input'}
        assert 0<timeout<=180
    replay=audit_membership_capture(events,obs,state.schedule,
        rows_for=lambda e:focus_rows(state.cases[e['case']],e,format_version=4),snapshot=lambda p:state.read(state.out/p),
        model=MODEL,digest=DIGEST,max_native=2,max_metadata=64,max_tokens=1024,max_input=7168,
        assessment_kind='reference-extraction-v1')
    assert len(replay['predictions'])==(2 if damage is None else 1)
    if damage is None:
        assert result['completed'] and obs[0]['analysis']==obs[1]['analysis']
    else:
        assert 'analysis' not in obs[1] and 'error' in obs[1]
        assert result['completed']==(damage=='format')
        assert result['graded']['summary']['all']['failed']==1
    with pytest.raises(FileExistsError):executor.run_local_membership_diagnostic(**state.args)


@pytest.mark.parametrize('damage',['kind','profile','group','output','input'])
def test_invalid_extraction_configuration_fails_before_attestation(tmp_path,monkeypatch,damage):
    state=setup(tmp_path,monkeypatch)
    plan=state.read(state.plan_path)
    if damage=='kind':plan['assessment_kind']='unknown'
    elif damage=='output':plan['limits']['max_tokens']=1025
    elif damage=='input':plan['limits']['max_input']=20
    else:
        inputs=state.read(state.out/'inputs.json')
        if damage=='profile':inputs['schedule'][0]['prompt_profile']='independent-membership-v1'
        else:inputs['schedule'][0]['descriptions_per_call']=1;plan['limits']['max_native_attempts']=3
        state.save(state.out/'inputs.json',inputs)
        plan['input_sha256']=state.args['archive'].base.sha(state.out/'inputs.json')
    state.save(state.plan_path,plan)
    with pytest.raises(ValueError):executor.run_local_membership_diagnostic(**state.args)
    assert state.wire==[] and not (state.out/'started.json').exists()


def test_changed_extracted_analysis_cannot_pass_native_replay(tmp_path,monkeypatch):
    state=setup(tmp_path,monkeypatch);executor.run_local_membership_diagnostic(**state.args)
    obs=state.read(state.out/'observations.json');events=state.read(state.out/'transport.json')
    obs[0]['analysis']['readings']=[[None]]
    with pytest.raises(ValueError,match='differs'):
        audit_membership_capture(events,obs,state.schedule,
            rows_for=lambda e:focus_rows(state.cases[e['case']],e,format_version=4),snapshot=lambda p:state.read(state.out/p),
            model=MODEL,digest=DIGEST,max_native=2,max_metadata=64,max_tokens=1024,max_input=7168,
            assessment_kind='reference-extraction-v1')


@pytest.mark.parametrize('deadline',[False,float('nan'),float('inf'),0])
def test_invalid_or_expired_shared_clock_cannot_dispatch(tmp_path,monkeypatch,deadline):
    state=setup(tmp_path,monkeypatch);plan=state.read(state.plan_path)
    plan['shared_deadline_monotonic']=deadline;state.save(state.plan_path,plan)
    with pytest.raises(ValueError,match='Shared diagnostic deadline'):
        executor.run_local_membership_diagnostic(**state.args)
    assert state.wire==[] and not (state.out/'started.json').exists()


def test_continuation_cannot_reset_the_original_wall_allowance(tmp_path,monkeypatch):
    state=setup(tmp_path,monkeypatch)
    original_time=executor.time;clock=[original_time.monotonic()]
    monkeypatch.setattr(executor,'time',type('Clock',(),dict(monotonic=staticmethod(lambda:clock[0]),time_ns=staticmethod(original_time.time_ns))))
    plan=state.read(state.plan_path);plan['shared_deadline_monotonic']=clock[0]+500;state.save(state.plan_path,plan)
    original_open=OllamaProvider._open
    def after_preflight(provider,request,*,timeout):
        result=original_open(provider,request,timeout=timeout)
        if request.full_url.endswith('/ps'):clock[0]+=21
        return result
    monkeypatch.setattr(OllamaProvider,'_open',after_preflight)
    executor.run_local_membership_diagnostic(**state.args)
    result=state.read(state.result_path)
    assert not result['completed'] and result['native_attempts']==0 and result['metadata_attempts']==3
    assert result['shared_deadline_monotonic']==plan['shared_deadline_monotonic']
    assert result['graded']['summary']['all']['missing']==2
