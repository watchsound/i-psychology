"""Resource/transport simulations test guards, never model capability."""
from copy import deepcopy
from io import BytesIO
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from psychology_base.eval.benchmarks.local_membership_diagnostic import (
    diagnostic_router,local_memory_admission,run_local_membership_diagnostic,
)
from psychology_base.llm.providers.ollama import OllamaProvider
from tests.test_local_membership_diagnostic import setup,MODEL,DIGEST

ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location('resident_memory_sentinel_test',
    ROOT/'docs/experiments/phase-9-memory-navigation/run-membership-local-sentinel-v2.py')
adapter=importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def plan():
    return dict(model=MODEL,digest=DIGEST,limits=dict(context_length=8192,
        cold_memory_minimum_bytes=7624568755,resident_memory_minimum_bytes=1024**3))


def resident():
    return dict(models=[dict(name=MODEL,digest=DIGEST,context_length=8192,size=6550825550)])


def test_exact_residency_avoids_counting_model_allocation_twice():
    value=local_memory_admission(plan(),resident(),4*1024**3)
    assert value['admitted'] and value['exact_model_context_resident']
    assert value['required_bytes']==1024**3
    cold=local_memory_admission(plan(),dict(models=[]),4*1024**3)
    assert not cold['admitted'] and cold['required_bytes']==7624568755
    assert not local_memory_admission(plan(),resident(),1024**3-1)['admitted']
    assert local_memory_admission(plan(),dict(models=[]),7624568755)['admitted']


@pytest.mark.parametrize('damage',['name','digest','context','missing-context','boolean-context','size','duplicate'])
def test_changed_or_ambiguous_residency_preserves_the_cold_reserve(damage):
    state=resident()
    item=state['models'][0]
    if damage=='name':item['name']='other-model'
    elif damage=='digest':item['digest']='b'*64
    elif damage=='context':item['context_length']=4096
    elif damage=='missing-context':del item['context_length']
    elif damage=='boolean-context':item['context_length']=True
    elif damage=='size':item['size']=0
    else:state['models'].append(deepcopy(item))
    value=local_memory_admission(plan(),state,4*1024**3)
    assert not value['admitted'] and not value['exact_model_context_resident']
    assert value['required_bytes']==7624568755


@pytest.mark.parametrize('damage',['negative','boolean','bad-residency','reversed-bounds'])
def test_invalid_memory_inputs_fail_closed(damage):
    p,state,available=plan(),resident(),4*1024**3
    if damage=='negative':available=-1
    elif damage=='boolean':available=True
    elif damage=='bad-residency':state=dict(models=None)
    else:p['limits']['resident_memory_minimum_bytes']=10**20
    with pytest.raises(ValueError):local_memory_admission(p,state,available)


def test_corrected_actual_stack_constructs_offline_and_reuses_unchanged_packet(monkeypatch):
    import socket
    def no_network(*args,**kwargs):raise AssertionError('Offline construction only')
    monkeypatch.setattr(socket.socket,'connect',no_network)
    _,inputs,_=adapter.sentinel.verified_packet()
    config=adapter.configuration()
    assert config['resident_memory_admission'] is True
    assert len(adapter.study.preflight(inputs,config))==1
    actual=diagnostic_router(config)
    assert actual.execution.request_timeout_s==actual.execution.operation_timeout_s==1200


@pytest.mark.parametrize('loaded',[False,True])
def test_shared_executor_captures_memory_evidence_before_generation(tmp_path,monkeypatch,loaded):
    import psutil
    state=setup(tmp_path,monkeypatch)
    config=state.read(state.plan_path)
    config['resident_memory_admission']=True
    config['limits'].update(plan()['limits'])
    state.save(state.plan_path,config)
    monkeypatch.setattr(psutil,'virtual_memory',lambda:SimpleNamespace(available=4*1024**3))
    original=OllamaProvider._open
    def memory_transport(self,request,*,timeout):
        response=original(self,request,timeout=timeout)
        if request.full_url.endswith('/api/ps'):
            response.close()
            return BytesIO(json.dumps(resident() if loaded else dict(models=[])).encode())
        return response
    monkeypatch.setattr(OllamaProvider,'_open',memory_transport)
    run_local_membership_diagnostic(**state.args)
    result=state.read(state.result_path)
    assert result['memory_admission']['admitted'] is loaded
    sample=state.read(state.out/'memory-admission.json')
    assert sample['decision']==result['memory_admission']
    events=state.read(state.out/'transport.json')
    assert all(e['start_monotonic']>=sample['monotonic'] for e in events if e['path']=='/api/chat')
    assert result['native_attempts']==(3 if loaded else 0)
    assert result['residency_preflight']==(resident() if loaded else dict(models=[]))
    if not loaded:
        assert not result['completed'] and result['graded']['summary']['all']['missing']==2
