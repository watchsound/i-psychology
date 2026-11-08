"""Declared protocol fixtures exercise real owners; no model-quality claims.

The tiny GGUF/DLL files below are software test inputs and are never loaded by
an inference engine. Real child-process tests use Python, not a language model.
"""
from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
from types import SimpleNamespace

import psutil
import pytest

from psychology_base.llm.local_backend import (
    NativeBackendSpec, OwnedLlamaServer, inspect_gguf, installed_ollama_model,
    local_memory_admission, native_spec_from_record, verify_file,
)
from psychology_base.llm.providers.llama_cpp import LlamaCppProvider, decode_native_response, validate_native_props
from psychology_base.llm.types import ChatMessage, TaskTier
from psychology_base.llm.config import default_config
from psychology_base.llm.execution import ExecutionPolicy, ExecutionDeferred
from psychology_base.llm.router import LLMRouter
from psychology_base.applications.measurement.local_query_binding import LocalChatMembershipAssessor
from psychology_base.eval.benchmarks.application_calls import LocalNativeCapture
from psychology_base.eval.benchmarks.local_membership_audit import audit_membership_capture, membership_wire_payload
from psychology_base.eval.benchmarks.membership_focus import condition, focus_rows


TEMPLATE = "{{ messages }}{% if enable_thinking %}<|im_start|>assistant\n<think>\n{% endif %}</think>"
TOKENS = ['<think>', '</think>', '<|im_start|>', '<|im_end|>', 'Analysis.', '{"labels":["match","nonmatch"]}']


def inputs():
    fixture=Path(__file__).parents[1]/'benchmarks/fixtures/application_context/pointwise-denial-control-v1.json'
    case=json.loads(fixture.read_text(encoding='utf-8'))['case']
    return focus_rows(case,dict(condition(case,profile='baseline-v1'),descriptions_per_call=2),format_version=4)


def text(value):
    value = value.encode()
    return struct.pack('<Q',len(value))+value


def gguf_bytes(*, template=TEMPLATE, extra=()):
    fields = [('general.architecture',8,text('qwen35')),
        ('qwen35.context_length',4,struct.pack('<I',32768)),
        ('tokenizer.chat_template',8,text(template)),
        ('tokenizer.ggml.tokens',9,struct.pack('<IQ',8,len(TOKENS))+b''.join(map(text,TOKENS))),
        ('tokenizer.ggml.eos_token_id',4,struct.pack('<I',3)), *extra]
    return struct.pack('<4sIQQ',b'GGUF',3,1,len(fields))+b''.join(text(k)+struct.pack('<I',t)+v for k,t,v in fields)


@pytest.fixture
def native_spec(tmp_path):
    root = tmp_path/'models'
    blobs = root/'blobs';blobs.mkdir(parents=True)
    weight = gguf_bytes();weight_hash = sha256(weight).hexdigest()
    (blobs/('sha256-'+weight_hash)).write_bytes(weight)
    manifest = root/'manifests'/'registry.ollama.ai'/'library'/'fixture'/'latest'
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(dict(layers=[dict(mediaType='application/vnd.ollama.image.model',
        digest='sha256:'+weight_hash,size=len(weight))])),encoding='utf-8')
    source_hash = sha256(manifest.read_bytes()).hexdigest()
    model = installed_ollama_model(root,manifest,model_id='fixture:latest',source_digest=source_hash)
    runtime = tmp_path/'runtime';runtime.mkdir()
    exe = runtime/'llama-server.exe';exe.write_bytes(b'Declared non-executable test fixture')
    libs = []
    for name in ('libllama-server-impl.dll','libllama-common.dll','libllama.dll','ggml-base.dll'):
        path=runtime/name;path.write_bytes(name.encode());libs.append(verify_file(path))
    return NativeBackendSpec(verify_file(exe),tuple(libs),model,max_tokens=64,thinking_tokens=16)


def props(spec):
    return dict(model_alias=spec.model.model_id,model_path=str(spec.model.weight.path),
        default_generation_settings=dict(n_ctx=spec.context_length),total_slots=1,
        chat_template=spec.model.template,is_sleeping=False,ui=False,cors_proxy_enabled=False,
        build_info='fixture 39febd2ad')


def response(spec, rendered, *, content=TOKENS[5], finish='stop'):
    ids=[4,1,5,3] if finish=='stop' else [4]*spec.max_tokens
    reasoning='Analysis.'
    return dict(model=spec.model.model_id,object='chat.completion',system_fingerprint='fixture 39febd2ad',
        choices=[dict(index=0,message=dict(role='assistant',content=content,reasoning_content=reasoning),finish_reason=finish)],
        usage=dict(prompt_tokens=100,completion_tokens=len(ids),total_tokens=100+len(ids)),
        __verbose=dict(tokens=ids,prompt=rendered,content=reasoning+'</think>\n'+content,
            tokens_predicted=len(ids),tokens_evaluated=100,stop=True,truncated=False,
            stop_type='eos' if finish=='stop' else 'limit',generation_settings=dict(seed=0,temperature=0,
                n_predict=spec.max_tokens,stream=False,reasoning_format='deepseek',grammar='root ::= object',
                generation_prompt='<think>\n')))


def router_for(provider, *, mode='local', cache=False):
    cfg=default_config()
    policy=ExecutionPolicy(mode=mode,request_timeout_s=180,operation_timeout_s=180)
    cfg=replace(cfg,cache=replace(cfg.cache,enabled=cache),execution=policy)
    return LLMRouter(cfg,providers={},local_chat_provider=provider)


def software_backend(native_spec,tmp_path,monkeypatch,*,damage=None):
    """Explicit native HTTP/process fixtures, shared real provider/router/capture."""
    owner=OwnedLlamaServer(native_spec,log_path=tmp_path/'unused.log')
    def identity(self,*,listener=True):
        if self.closed:raise ValueError('Fixture process is closed')
        return dict(pid=12345,created=1.25,spec_sha256=sha256(json.dumps(self.spec.record(),sort_keys=True,separators=(',',':')).encode()).hexdigest())
    monkeypatch.setattr(OwnedLlamaServer,'assert_owned',identity)
    provider=LlamaCppProvider(owner,request_timeout_s=180)
    calls=[]; rendered=['']
    def opened(self,request,*,timeout):
        path=request.full_url[len(self.base_url):]
        payload=json.loads(request.data) if request.data else None
        calls.append((path,payload,timeout))
        if path=='/props':value=props(native_spec)
        elif path=='/apply-template':
            rendered[0]=json.dumps(payload['messages'],ensure_ascii=False)+'<|im_start|>assistant\n<think>\n'
            value=dict(prompt=rendered[0])
            if damage=='render':value['prompt']='x'*native_spec.context_length
        elif path=='/health':value=dict(status='ok')
        elif path=='/v1/chat/completions':
            if damage=='transport':raise TimeoutError('Declared native transport failure')
            value=response(native_spec,rendered[0],finish='length' if damage=='length' else 'stop')
            if damage=='format':
                value['choices'][0]['message']['content']='{"labels":["unsupported","nonmatch"]}'
                value['__verbose']['content']='Analysis.</think>\n'+value['choices'][0]['message']['content']
        else:raise AssertionError(path)
        return BytesIO(json.dumps(value).encode())
    monkeypatch.setattr(LlamaCppProvider,'_open',opened)
    return provider, calls, identity(owner)


def test_real_asset_resolution_is_bounded_and_reconstructs_the_exact_record(native_spec):
    record=native_spec.record()
    assert native_spec_from_record(record).record()==record
    assert native_spec.model.thinking_start==0 and native_spec.model.thinking_end==1
    data=inspect_gguf(native_spec.model.weight.path)
    assert data['template_sha256']==sha256(TEMPLATE.encode()).hexdigest()
    assert data['header_bytes']==native_spec.model.weight.signature[0]
    changed=deepcopy(record);changed['model']['weight']['bytes']+=1
    with pytest.raises(ValueError):native_spec_from_record(changed)


@pytest.mark.parametrize('payload',[
    b'GGUF'+b'\x03\x00\x00\x00',
    gguf_bytes(template='no thinking control'),
    gguf_bytes(extra=[('general.architecture',8,text('duplicate'))]),
    gguf_bytes(extra=[('oversized',8,struct.pack('<Q',2**63))]),
    gguf_bytes(extra=[('nested',9,struct.pack('<IQ',9,1))]),
    gguf_bytes(extra=[('unknown',90,b'')]),
    gguf_bytes(extra=[('array',9,struct.pack('<IQ',0,1_000_001))]),
    gguf_bytes()[:-3],
])
def test_malformed_gguf_never_falls_back_to_a_guessed_template(tmp_path,payload):
    path=tmp_path/'bad.gguf';path.write_bytes(payload)
    with pytest.raises((ValueError,UnicodeError)):inspect_gguf(path)


@pytest.mark.parametrize('change',[
    dict(thinking_tokens=True),dict(thinking_tokens=0),dict(thinking_tokens=64),
    dict(max_tokens=True),dict(max_tokens=4097),dict(port=11434),dict(port=True),
    dict(port=65536),dict(seed=True),dict(context_length=65536),dict(cold_memory_minimum_bytes=1),
])
def test_invalid_native_controls_fail_offline(native_spec,change):
    with pytest.raises(ValueError):replace(native_spec,**change)


def test_changed_or_unpinned_artifacts_are_rejected_before_start(native_spec,tmp_path):
    native_spec.check()
    (native_spec.executable.path.parent/'unexpected.dll').write_bytes(b'extra fixture')
    with pytest.raises(ValueError,match='library set'):native_spec.check()
    native_spec.model.weight.path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='artifact changed'):native_spec.model.check()


def test_separate_process_cannot_claim_shared_resident_memory(native_spec):
    plan=dict(model='fixture',digest='a'*64,limits=dict(cold_memory_minimum_bytes=100,resident_memory_minimum_bytes=10,context_length=8192))
    residency=dict(models=[dict(name='fixture',digest='a'*64,context_length=8192,size=90)])
    assert local_memory_admission(plan,residency,20)['admitted']
    decision=local_memory_admission(plan,residency,20,separate_process=True)
    assert decision==dict(available_bytes=20,required_bytes=100,exact_model_context_resident=False,admitted=False)


def test_complete_native_evidence_is_distinct_from_verified_forcing(native_spec):
    decoded=decode_native_response(response(native_spec,'rendered'),native_spec.record(),rendered_prompt='rendered')
    assert decoded['content']==TOKENS[5]
    assert decoded['closure']==dict(observed=True,generated_tokens_before_close=1,
        requested_thinking_tokens=16,budget_forcing_verified=False)


@pytest.mark.parametrize('mutate',[
    lambda r:r['usage'].update(completion_tokens=True),
    lambda r:r['usage'].update(prompt_tokens=True),
    lambda r:r['usage'].update(total_tokens=0),
    lambda r:r['__verbose'].update(tokens=[4,1,5]),
    lambda r:r['__verbose'].update(tokens=[4,1,True,3]),
    lambda r:r['__verbose'].update(tokens=[0,1,5,3]),
    lambda r:r['__verbose'].update(tokens=[1,1,5,3]),
    lambda r:r['__verbose'].update(tokens=[4,4,5,3]),
    lambda r:r['__verbose'].update(tokens=[4,1,3,5]),
    lambda r:r['__verbose'].update(truncated=True),
    lambda r:r['__verbose'].update(stop_type='limit'),
    lambda r:r['__verbose'].update(prompt='different input'),
    lambda r:r['__verbose']['generation_settings'].update(seed=True),
    lambda r:r['__verbose']['generation_settings'].update(temperature=False),
    lambda r:r['__verbose']['generation_settings'].update(n_predict=1000),
    lambda r:r['__verbose']['generation_settings'].update(generation_prompt=''),
    lambda r:r['__verbose']['generation_settings'].update(grammar=''),
    lambda r:r['choices'][0]['message'].update(content=''),
    lambda r:r['choices'][0]['message'].update(tool_calls=[{}]),
    lambda r:r['choices'][0].update(finish_reason='tool_calls'),
    lambda r:r.update(model='different model'),
    lambda r:r.pop('__verbose'),
])
def test_native_response_cannot_hide_truncation_missing_tags_or_forged_settings(native_spec,mutate):
    raw=response(native_spec,'rendered');mutate(raw)
    with pytest.raises(ValueError):decode_native_response(raw,native_spec.record(),rendered_prompt='rendered')


@pytest.mark.parametrize('damage',['length','format','transport','render'])
def test_real_assessor_rejects_failed_native_protocol_without_retry(native_spec,tmp_path,monkeypatch,damage):
    provider,calls,_=software_backend(native_spec,tmp_path,monkeypatch,damage=damage)
    assessor=LocalChatMembershipAssessor(router_for(provider),format_version=4,
        max_tokens=64,reasoning_budget=True,max_input=8000)
    with pytest.raises((ValueError,ExecutionDeferred)):assessor(inputs())
    assert sum(path=='/v1/chat/completions' for path,_,_ in calls)==(0 if damage=='render' else 1)
    if damage=='length':assert assessor.last_calls[0]['completion'].finish_reason=='length'


def test_router_capture_assessor_and_independent_audit_share_the_native_path(native_spec,tmp_path,monkeypatch):
    provider,wire,identity=software_backend(native_spec,tmp_path,monkeypatch)
    router=router_for(provider)
    assessor=LocalChatMembershipAssessor(router,format_version=4,max_tokens=64,reasoning_budget=True)
    rows=inputs();entry=dict(id='fixture-condition',prompt_profile='baseline-v1',descriptions_per_call=2)
    expected=membership_wire_payload(rows,provider.advanced_model,think=True,native_spec=native_spec.record())
    snapshots={}
    def observer(event,phase):snapshots[f"native-{event['sequence']:03d}-{phase}.json"]=deepcopy(event)
    def validate(payload):assert payload==expected
    with LocalNativeCapture(provider,max_attempts=1,max_metadata=3,deadline=time.monotonic()+120,
            observer=observer,validate_chat=validate) as capture:
        capture.condition=entry['id'];labels=assessor(rows)
    assert labels==['match','nonmatch'] and capture.attempts==1 and capture.metadata_attempts==2
    assert [path for path,_,_ in wire]==['/props','/apply-template','/v1/chat/completions']
    assert all(0<t<=180 for _,_,t in wire)
    assert all(e['endpoint']==provider.base_url for e in capture.events)
    observation=dict(id=entry['id'],labels=labels,native_calls=[dict(row_indices=c['row_indices'],
        messages=[asdict(m) for m in c['messages']],completion=asdict(c['completion'])) for c in assessor.last_calls])
    snapshots['native-process-started.json']=dict(spec=native_spec.record(),identity=identity,argv=native_spec.argv())
    snapshots['native-process.json']=dict(snapshots['native-process-started.json'],cleanup=dict(verified_exit=True,job_verified_empty=True))
    args=dict(rows_for=lambda e:rows,snapshot=lambda name:snapshots[name],model=provider.advanced_model,
        digest=provider.expected_digest,max_native=1,max_metadata=3,think=True,native_spec=native_spec.record())
    result=audit_membership_capture(capture.events,[observation],[entry],**args)
    assert result['predictions']=={entry['id']:labels}
    changed=deepcopy(observation);changed['native_calls'][0]['completion']['metadata']['thinking_closure']['budget_forcing_verified']=True
    with pytest.raises(ValueError):audit_membership_capture(capture.events,[changed],[entry],**args)
    wrong=deepcopy(capture.events);wrong[-1]['endpoint']='http://127.0.0.1:11434'
    with pytest.raises(ValueError):audit_membership_capture(wrong,[observation],[entry],**args)


def test_native_injection_cannot_enable_paid_research_cache_or_serving_routes(native_spec,tmp_path):
    provider=LlamaCppProvider(OwnedLlamaServer(native_spec,log_path=tmp_path/'unused.log'),request_timeout_s=180)
    for mode,cache in [('paid',False),('research',False),('local',True)]:
        with pytest.raises(ValueError):router_for(provider,mode=mode,cache=cache)
    cfg=replace(default_config(),cache=replace(default_config().cache,enabled=False))
    with pytest.raises(ValueError):LLMRouter(cfg,providers={'llama_cpp':provider})
    router=router_for(provider)
    with pytest.raises(ValueError,match='cache'):router.chat([ChatMessage('user','fixture')],max_tokens=64,temperature=0)
    with pytest.raises(ExecutionDeferred):router.embed('fixture')


def test_native_owner_refuses_low_ram_before_any_spawn(native_spec,tmp_path,monkeypatch):
    monkeypatch.setattr(psutil,'virtual_memory',lambda:SimpleNamespace(available=1))
    monkeypatch.setattr(subprocess,'Popen',lambda *a,**k:pytest.fail('No process may start'))
    owner=OwnedLlamaServer(native_spec,log_path=tmp_path/'unused.log')
    with pytest.raises(MemoryError):owner.start()
    assert owner.memory_admission['admitted'] is False and not owner.started


@pytest.mark.parametrize('damage',[None,'length','format','transport','render','journal'])
def test_shared_executor_native_lifecycle_and_failed_denominators(native_spec,tmp_path,monkeypatch,damage):
    from tests.test_local_membership_diagnostic import setup
    from psychology_base.eval.benchmarks import local_membership_diagnostic as executor
    from psychology_base.eval.benchmarks.membership_focus import grade
    state=setup(tmp_path,monkeypatch)
    packet=state.read(state.out/'inputs.json');packet['schedule']=packet['schedule'][:1]
    state.save(state.out/'inputs.json',packet)
    plan=state.read(state.plan_path)
    plan.update(transport='llama_cpp',native_backend=native_spec.record(),think=True,reasoning_budget=True,
        model=native_spec.model.model_id,digest=native_spec.model.source_digest,
        input_sha256=state.args['archive'].base.sha(state.out/'inputs.json'))
    plan['limits'].update(max_native_attempts=1,max_metadata_attempts=12,metadata_timeout_s=60)
    state.save(state.plan_path,plan)
    state.args['grade']=lambda c,r,s,o:grade(c,r,s,o,contrast_pairs=[])
    if damage=='journal':
        original_save=state.args['archive'].base.save
        def save(path,value):
            if path.name=='native-process-started.json':raise OSError('Declared journal write failure')
            return original_save(path,value)
        state.args['archive'].base.save=save
    provider,wire,_=software_backend(native_spec,tmp_path,monkeypatch,damage=damage)
    lifecycle=[]
    def start(self):
        lifecycle.append('start');self.started=True
        self.memory_admission=dict(available_bytes=10**10,required_bytes=native_spec.cold_memory_minimum_bytes,
            exact_model_context_resident=False,admitted=True)
        return self
    def close(self):
        lifecycle.append('close');self.closed=True;self.cleanup=dict(verified_exit=True,job_verified_empty=True)
    monkeypatch.setattr(OwnedLlamaServer,'start',start)
    monkeypatch.setattr(OwnedLlamaServer,'close',close)
    before=executor.socket.socket.connect
    executor.run_local_membership_diagnostic(**state.args)
    assert executor.socket.socket.connect is before and lifecycle==['start','close']
    result=state.read(state.result_path)
    assert result['native_attempts']==(0 if damage in ('render','journal') else 1)
    assert result['qualified'] is False and result['paid_api_usd']=='0'
    assert result['memory_admission']['exact_model_context_resident'] is False
    if damage=='transport':assert any('Declared native transport failure' in v for v in result['error_chain'])
    assert state.read(state.out/'native-process.json')['cleanup']['verified_exit'] is True
    observations=state.read(state.out/'observations.json');events=state.read(state.out/'transport.json')
    replay=audit_membership_capture(events,observations,packet['schedule'],
        rows_for=lambda e:focus_rows(packet['cases'][e['case']],e,format_version=4),
        snapshot=lambda p:state.read(state.out/p),model=plan['model'],digest=plan['digest'],
        max_native=1,max_metadata=12,max_tokens=64,max_input=8000,think=True,native_spec=native_spec.record())
    assert len(replay['predictions'])==(1 if damage is None else 0)
    assert result['graded']['summary']['all']['planned_pairs']==2
    with pytest.raises(FileExistsError):executor.run_local_membership_diagnostic(**state.args)
    assert lifecycle==['start','close']


@pytest.mark.parametrize('change',[
    dict(ui=True),dict(cors_proxy_enabled=True),dict(total_slots=True),dict(total_slots=2),
    dict(chat_template='changed'),dict(model_path='different.gguf'),dict(model_alias='different'),dict(build_info='different'),
])
def test_bad_native_metadata_never_attests_a_model(native_spec,change):
    value=props(native_spec);value.update(change)
    with pytest.raises(ValueError):validate_native_props(value,native_spec.record())


def test_assessor_identity_detects_native_control_changes_before_dispatch(native_spec,tmp_path,monkeypatch):
    provider,calls,_=software_backend(native_spec,tmp_path,monkeypatch)
    assessor=LocalChatMembershipAssessor(router_for(provider),format_version=4,max_tokens=64,reasoning_budget=True)
    provider.server.spec=replace(native_spec,thinking_tokens=8)
    with pytest.raises(ValueError,match='configuration changed'):assessor(inputs())
    assert calls==[]


def test_owned_native_identity_checks_pid_command_and_listener(native_spec,tmp_path,monkeypatch):
    owner=OwnedLlamaServer(native_spec,log_path=tmp_path/'unused.log')
    owner._handle=SimpleNamespace(pid=123,poll=lambda:None);owner.started=True;owner._created=1.25
    state=dict(created=1.25,exe=str(native_spec.executable.path),argv=native_spec.argv(),connections=[])
    process=SimpleNamespace(create_time=lambda:state['created'],exe=lambda:state['exe'],
        cmdline=lambda:state['argv'],net_connections=lambda **kw:state['connections'])
    monkeypatch.setattr(psutil,'Process',lambda pid:process)
    assert owner.assert_owned(listener=False)['pid']==123
    with pytest.raises(ValueError,match='listening'):owner.assert_owned()
    state['connections']=[SimpleNamespace(status=psutil.CONN_LISTEN,laddr=SimpleNamespace(ip='127.0.0.1',port=native_spec.port))]
    assert owner.assert_owned()['pid']==123
    for key,wrong in [('created',2),('exe',str(tmp_path/'foreign.exe')),('argv',['foreign'])]:
        saved=state[key];state[key]=wrong
        with pytest.raises(ValueError,match='identity'):owner.assert_owned()
        state[key]=saved


def test_capture_refuses_foreign_endpoints_and_extra_native_attempts(native_spec,tmp_path,monkeypatch):
    from urllib.request import Request
    provider,_,_=software_backend(native_spec,tmp_path,monkeypatch)
    with LocalNativeCapture(provider,max_attempts=1,max_metadata=1,deadline=time.monotonic()+60,
            observer=lambda *args:None,validate_chat=lambda payload:None) as capture:
        for url in ('http://127.0.0.1:11434/props','http://example.com/props',provider.base_url+'/props?x=1',provider.base_url+'/completion'):
            with pytest.raises(ValueError):provider._open(Request(url),timeout=1)
        assert capture.attempts==capture.metadata_attempts==0
        provider.metadata('/props')
        with pytest.raises(ValueError,match='metadata attempt'):provider.metadata('/props')


def test_native_packet_preserves_evidence_and_references_without_exposing_labels():
    import importlib.util
    path=Path(__file__).parents[1]/'docs/experiments/phase-9-memory-navigation/run-membership-native-sentinel-v1.py'
    spec=importlib.util.spec_from_file_location('native_sentinel_packet_test',path)
    scope=importlib.util.module_from_spec(spec);spec.loader.exec_module(scope)
    base=SimpleNamespace(read=lambda p:json.loads(p.read_text(encoding='utf-8')),
        sha=lambda p:sha256(p.read_bytes()).hexdigest())
    assert scope.verify_prior(base)['reserved_rows_sha256']==base.sha(scope.QUALIFICATION)
    previous=json.loads((scope.PRIOR/'inputs.json').read_text(encoding='utf-8'))
    refs=json.loads((scope.PRIOR/'references.json').read_text(encoding='utf-8'))
    packet,new_refs=scope.build_packet(previous,refs)
    assert packet['cases']==previous['cases'] and new_refs==refs
    assert previous['schedule'][0]['id']+'/native-thinking-768-v1'==packet['schedule'][0]['id']
    entry=packet['schedule'][0];rows=scope.rows_for(packet['cases'][entry['case']],entry)
    wire=membership_wire_payload(rows,'qwen3.5:9b',prompt_profile=entry['prompt_profile'],max_tokens=1024,
        think=True,native_spec=dict(thinking_tokens=768))
    old=[e for e in json.loads((scope.PRIOR/'transport.json').read_text(encoding='utf-8')) if e['path']=='/api/chat'][0]
    assert wire['messages']==old['request']['messages']
    assert set(json.loads(wire['messages'][1]['content']))=={'focus_input','descriptions'}
    empty=scope.score(packet['cases'],new_refs,packet['schedule'],[])
    assert empty['summary']['all']['planned_pairs']==2 and empty['summary']['all']['missing']==1
    wrong=deepcopy(previous);wrong['schedule'][0]['repeat']=True
    with pytest.raises(ValueError):scope.build_packet(wrong,refs)
    prior_plan=json.loads(scope.PRIOR_PLAN.read_text(encoding='utf-8'))
    # Scope-control validation needs no installed model, DLL or local asset cache.
    assets=dict(max_tokens=1024,thinking_tokens=768,context_length=8192,seed=0,
        model=dict(model_id=old['request']['model'],source_digest=prior_plan['digest'],
            weight=dict(sha256='02d45dc1cf451ba2475ac33b301c2dd8f985abe4c182ce04a1f2f5bf0260278d')))
    plan=scope.configuration(assets)
    assert plan['limits']['max_native_attempts']==1 and plan['limits']['max_tokens']==1024
    for change in (dict(thinking_tokens=1024),dict(thinking_tokens=True),dict(max_tokens=2048)):
        with pytest.raises(ValueError):scope.configuration(dict(assets,**change))


def test_real_windows_job_cleans_a_child_even_after_its_parent_exits(tmp_path):
    if os.name!='nt':pytest.skip('Windows job ownership is Windows-specific')
    from psychology_base.llm.owned_process import WindowsProcessJob
    pid_path=tmp_path/'child.pid'
    release=tmp_path/'release'
    # Parent waits until assigned to our job, then creates a real descendant.
    script="import subprocess,sys,time,pathlib; p=pathlib.Path(sys.argv[1]); r=pathlib.Path(sys.argv[2]);\nwhile not r.exists():time.sleep(.02)\nc=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)']);p.write_text(str(c.pid))"
    job=WindowsProcessJob()
    parent=subprocess.Popen([sys.executable,'-B','-c',script,str(pid_path),str(release)],
        creationflags=subprocess.CREATE_NO_WINDOW,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    child=None
    try:
        job.assign(parent);release.write_text('assigned')
        until=time.monotonic()+10
        while not pid_path.exists() and time.monotonic()<until:time.sleep(.02)
        assert pid_path.exists()
        child=psutil.Process(int(pid_path.read_text()))
        parent.wait(timeout=10)
        assert child.is_running()
        result=job.stop()
        assert result['verified_exit'] and result['job_verified_empty'] and result['tracked_processes']>=2
        assert not child.is_running()
    finally:
        if job.handle is not None:job.stop()
        if parent.poll() is None:parent.kill();parent.wait(timeout=5)
        if child is not None and child.is_running():child.kill();child.wait(timeout=5)
