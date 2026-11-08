"""Local transport/accounting controls are not model-quality observations."""
import copy
import io
import json
from decimal import Decimal
from urllib.request import Request

import pytest

from psychology_base.eval.benchmarks import application_context as study
from psychology_base.eval.benchmarks.application_calls import Budget, RecordingRouter
from psychology_base.applications.pilot.session_runner import SessionLogWriter
from psychology_base.llm import load_config
from psychology_base.llm.providers.ollama import OllamaProvider, _RejectRedirects
from psychology_base.llm.types import ChatMessage, TaskTier
from tests.test_application_context_study import real_sources


PROFILE = study.FIXTURES/'local-screen-v1.json'


def test_overlay_keeps_source_freeze_and_exact_192_development_responses(real_sources):
    original,cases,_=real_sources
    frozen=copy.deepcopy(original)
    local=study.local_profile(PROFILE,original)
    assert original==frozen and original['samples']==3 and local['samples']==1
    for field in ('gates','source_pins','annotations','catalog_sha256','scoring_prompt_sha256'):
        assert original[field]==local[field]
    selected=[c for c in cases if c['split']=='development']
    rows=[dict(case=c['id'],app=c['app'],query_index=q,absent=a,candidate=arm,sample=0)
          for c in selected for q,a,arm in study.itertools.product(range(2),(False,True),study.ARMS)]
    assert study.matrix(rows,selected,local['samples'])=={'expected':192,'actual':192,'unique':192}
    with pytest.raises(ValueError): study.matrix(rows,selected)
    with pytest.raises(ValueError): study.matrix(rows+[rows[0]],selected,1)


@pytest.mark.parametrize('change',['parent','samples','split','cloud','same_family','zero_cost','extra','context'])
def test_local_profile_rejects_scope_or_provider_drift(real_sources,tmp_path,change):
    profile=study.read(PROFILE)
    if change=='parent': profile['parent_protocol_sha256']='changed'
    elif change=='samples': profile['samples']=3
    elif change=='split': profile['split']='evaluation'
    elif change=='cloud': profile['judge']['base_url']='https://example.com'
    elif change=='same_family': profile['judge']['family']=profile['generator']['family']
    elif change=='zero_cost': profile['budget']['max_usd']='1'
    elif change=='extra': profile['cloud_fallback']=True
    else: profile['generator']['context_length']=4096
    path=tmp_path/'profile.json'; path.write_text(json.dumps(profile),encoding='utf-8')
    with pytest.raises(ValueError): study.local_profile(path,real_sources[0])


@pytest.mark.parametrize('url',[
    'https://127.0.0.1:11434/api/chat','http://localhost:11434/api/chat',
    'http://example.com:11434/api/chat','http://127.0.0.1:11435/api/chat',
    'http://user@127.0.0.1:11434/api/chat','http://127.0.0.1:11434/api/chat?forward=remote'])
def test_transport_rejects_nonfrozen_addresses_before_network(url,monkeypatch):
    monkeypatch.setattr('urllib.request.build_opener',lambda *a:pytest.fail('Invalid endpoint reached transport'))
    with pytest.raises(ValueError): OllamaProvider(local_only=True)._open(Request(url),timeout=1)


def test_local_transport_bypasses_proxies_and_rejects_redirects(monkeypatch):
    from types import SimpleNamespace
    handlers=[]; requests=[]
    def build(*args):
        handlers.extend(args)
        return SimpleNamespace(open=lambda request,**kw:requests.append(request) or io.BytesIO(b'{}'))
    monkeypatch.setattr('urllib.request.build_opener',build)
    monkeypatch.setattr('urllib.request.urlopen',lambda *a,**kw:pytest.fail('Default/proxied transport used'))
    provider=OllamaProvider(local_only=True,base_url='http://127.0.0.1:11434')
    provider._open(Request(provider.base_url+'/api/tags'),timeout=1)
    assert len(requests)==1 and handlers[0].proxies=={} and isinstance(handlers[1],_RejectRedirects)
    with pytest.raises(ValueError):
        handlers[1].redirect_request(None,None,302,'redirect',{},'https://example.com')


def local_wrapper(tmp_path,monkeypatch,*,budget=None):
    base,_=study.load_inputs(); protocol=study.local_profile(PROFILE,base)
    budget=budget or Budget(SessionLogWriter(tmp_path/'events.jsonl'),max_calls=4,max_usd='0',local_only=True)
    generator,judge=study.make_routers(protocol,load_config(),budget)
    calls=[]
    def open_local(request,*,timeout):
        path=request.full_url.rsplit('/',1)[-1]
        if path=='tags':
            data={'models':[{'name':protocol['generator']['model'],'digest':protocol['generator']['digest'],'size':100}]}
        elif path=='show':
            data={'details':{'family':'qwen2'},'capabilities':['completion'],'model_info':{'qwen2.context_length':32768}}
        else:
            payload=json.loads(request.data); calls.append(payload)
            data={'model':payload['model'],'message':{'content':'Explicit unit transport control'},'done':True,
                  'done_reason':'stop','prompt_eval_count':10,'eval_count':2}
        return io.BytesIO(json.dumps(data).encode())
    monkeypatch.setattr(generator.router.providers['ollama'],'_open',open_local)
    monkeypatch.setattr('urllib.request.urlopen',lambda *a,**kw:pytest.fail('External/default transport called'))
    return generator,budget,calls,protocol


def test_actual_local_provider_records_usage_and_pinned_decoding(tmp_path,monkeypatch):
    wrapper,budget,calls,protocol=local_wrapper(tmp_path,monkeypatch)
    completion=wrapper.chat([ChatMessage('user','Declared unit input')],tier=TaskTier.ADVANCED,max_tokens=20)
    assert len(calls)==1 and calls[0]['options']=={'num_predict':20,'num_ctx':16384,'seed':0}
    assert completion.metadata['model_attestation']['digest']==protocol['generator']['digest']
    assert wrapper.last_call['local_only'] and wrapper.last_call['accounted_usd']=='0'
    assert budget.attempts==1 and budget.spent==0 and budget.summary()['hardware_cost_usd'] is None
    budget.writer.close()


def test_local_routers_reuse_existing_config_factory_and_have_no_other_providers(real_sources,tmp_path):
    from dataclasses import replace
    base=load_config(); selected=replace(base.providers['ollama'],request_timeout_s=321,embedding_model='declared-owner-setting')
    cfg=replace(base,providers={**base.providers,'ollama':selected})
    protocol=study.local_profile(PROFILE,real_sources[0])
    budget=Budget(SessionLogWriter(tmp_path/'events.jsonl'),max_calls=4,max_usd='0',local_only=True)
    for wrapper in study.make_routers(protocol,cfg,budget):
        assert set(wrapper.router.providers)=={'ollama'} and wrapper.router.cache is None
        provider=wrapper.router.providers['ollama']
        assert provider.request_timeout_s==321 and provider.embedding_model=='declared-owner-setting'
        assert provider.local_only and not wrapper.router.config.local_ollama.enabled
    budget.writer.close()


def test_local_ledger_preserves_historical_remote_spend(tmp_path,monkeypatch):
    from tests.test_application_call_audit import audited
    wrapper,_,remote=audited(tmp_path)
    wrapper.chat([ChatMessage('user','Historical unit request')],tier=TaskTier.ADVANCED,max_tokens=20)
    old_spend=remote.spent; remote.writer.close()
    path=tmp_path/'attempts.jsonl'
    local=Budget(SessionLogWriter(path),max_calls=3,max_usd='0',local_only=True)
    wrapper,local,calls,_=local_wrapper(tmp_path,monkeypatch,budget=local)
    wrapper.chat([ChatMessage('user','Local unit request')],tier=TaskTier.ADVANCED,max_tokens=20)
    assert local.spent==old_spend and local.attempts==2 and len(calls)==1
    with pytest.raises(ValueError): local.reserve(Decimal('0.01'),{'local_only':False})
    with pytest.raises(ValueError): local.reserve(Decimal(0),{'provider':'openai'})
    assert local.attempts==2
    local.writer.close()
    restored=Budget(SessionLogWriter(path),max_calls=3,max_usd='0',local_only=True)
    assert restored.spent==old_spend and restored.attempts==2 and not restored.uncertain
    restored.writer.close()


@pytest.mark.parametrize('mutation',['digest','family','cloud_forward','capacity','usage','returned_model'])
def test_local_identity_failures_never_qualify(tmp_path,monkeypatch,mutation):
    wrapper,budget,calls,_=local_wrapper(tmp_path,monkeypatch)
    provider=wrapper.router.providers['ollama']; original=provider._open
    def mutated(request,**kwargs):
        data=json.load(original(request,**kwargs))
        if mutation=='digest' and 'models' in data: data['models'][0]['digest']='bad'
        if mutation=='family' and 'details' in data: data['details']['family']='other'
        if mutation=='cloud_forward' and 'details' in data: data['remote_host']='https://example.com'
        if mutation=='capacity' and 'model_info' in data: data['model_info']['qwen2.context_length']=4096
        if mutation=='usage' and 'message' in data: data['prompt_eval_count']=0
        if mutation=='returned_model' and 'message' in data: data['model']='other'
        return io.BytesIO(json.dumps(data).encode())
    monkeypatch.setattr(provider,'_open',mutated)
    with pytest.raises(ValueError):
        wrapper.chat([ChatMessage('user','Unit probe')],tier=TaskTier.ADVANCED,max_tokens=20)
    assert len(calls)==(1 if mutation in ('usage','returned_model') else 0)
    budget.writer.close()


def test_calibration_early_stop_requires_impossible_fixed_denominator():
    def control(correct):
        return {'expected_success':True,'observed_success':correct,
                'expected_critical':{k:True for k in ('grounded','attribution_correct','critical_error')},
                'judgment':{'criteria':{k:True for k in ('grounded','attribution_correct','critical_error')},'parse_failed':False}}
    assert not study.calibration_failure_is_final([control(False)]*2,[])
    rows=[control(False)]*3
    assert study.calibration_failure_is_final(rows,[])
    metrics=study.calibration_metrics(rows,[])
    assert metrics['memory_agreement']==0 and metrics['tutor_kappa'] is None and not metrics['passed']
    rows=[control(True) for _ in range(3)]
    for row in rows: row['judgment']['criteria']['grounded']=False
    assert study.calibration_failure_is_final(rows,[])


def test_model_execution_without_local_profile_and_heldout_are_blocked(real_sources):
    from argparse import Namespace
    with pytest.raises(ValueError,match='External model execution is disabled'):
        study.run(Namespace(command='application-preflight',application_profile=None))
    with pytest.raises(ValueError,match='held-out'):
        study.run(Namespace(command='application-evaluation',application_profile=PROFILE))


def test_judge_preflight_matches_actual_english_rubric_and_unicode_payload_contract(real_sources):
    base=real_sources[0]; local=study.local_profile(PROFILE,base)
    requests=study.preflight_requests(local,'judge')
    assert len(requests)==2 and all([m.role for m in messages]==['system','user'] for messages in requests)
    assert requests[0][0]==requests[1][0]
    assert [json.loads(messages[1].content) for messages in requests]==[
        {'readiness_probe':'READY'},{'readiness_probe':'就绪'}]
    # Existing generator and archived cloud readiness contracts remain intact.
    assert study.preflight_requests(local,'generator')==study.preflight_requests(base,'generator')
    assert all(len(messages)==1 and messages[0].role=='user' for messages in study.preflight_requests(base,'judge'))


def test_failed_partial_calibration_is_valid_evidence_but_cannot_be_promoted(real_sources):
    """Declared parser/provenance control; no genuine model judgment is asserted."""
    from dataclasses import asdict
    from psychology_base.llm.types import Completion
    base,_,sources=real_sources; protocol=study.local_profile(PROFILE,base)
    memory,_=study.calibration_items(protocol,sources)
    item=memory[0]; spec=protocol['judge']
    judgment=study.parse_memory('{}','stop',False)
    completion=Completion('{}','ollama',spec['model'],TaskTier.ADVANCED,prompt_tokens=10,completion_tokens=2,
        finish_reason='stop',metadata={'local_only':True,'done':True,'response_model':spec['model'],
            'model_attestation':{'digest':spec['digest'],'family':spec['family']},
            'context_length':spec['context_length'],'seed':spec['seed'],'json_mode':True})
    call={'completion':{**asdict(completion),'tier':TaskTier.ADVANCED.value},'local_only':True,
          'proxy':False,'cached':False,'usage_valid':True,'accounted_usd':'0'}
    item.update(judgment=judgment,observed_success=False,call=call)
    report={'phase':'calibration','memory_controls':[item],'tutor_controls':[],
            'complete':False,'stopped_early':True,'remaining_controls':{'memory':23,'tutor':24},
            'generated_application_replies':0,'proxy':False,'code_sha256':study.fingerprints(),
            'protocol_sha256':study.protocol_sha(protocol),**study.calibration_metrics([item],[])}
    assert study.validate_calibration(report,protocol,sources) is False
    for mutation in ('passed','complete','counts','digest','raw'):
        bad=copy.deepcopy(report)
        if mutation in ('passed','complete'): bad[mutation]=True
        elif mutation=='counts': bad['remaining_controls']['memory']=0
        elif mutation=='digest': bad['memory_controls'][0]['call']['completion']['metadata']['model_attestation']['digest']='wrong'
        else: bad['memory_controls'][0]['call']['completion']['content']='different raw output'
        with pytest.raises(ValueError): study.validate_calibration(bad,protocol,sources)


@pytest.mark.skipif(study.optional_version('jieba') is None,reason='Real jieba backend required')
def test_versioned_resources_use_independent_real_recalls_and_validate_raw_batches(real_sources,monkeypatch):
    _,cases,sources=real_sources
    case=next(c for c in cases if c['split']=='development' and c['stratum']=='native_en_application')
    original=study.measure_recall; calls=[]
    def measured(system,cue):
        assert getattr(system,'audit_recall_count',0)==0
        result,metrics=original(system,cue)
        assert system.audit_recall_count==1
        calls.append(system.fragment_context_selection)
        return result,metrics
    monkeypatch.setattr(study,'measure_recall',measured)
    resources=study.capped_resources([case],sources,load_config(),procedure=study.RESOURCE_V2)
    assert len(calls)==404 and len(resources['measurements'])==4
    assert study.validate_resources(resources,[case],procedure=study.RESOURCE_V2)==resources['passed']
    for start in (0,202):
        assert calls[start:start+2]==list(study.ARMS)
        paired=calls[start+2:start+202]
        assert all(set(paired[i:i+2])==set(study.ARMS) for i in range(0,200,2))
    changed=copy.deepcopy(resources)
    changed['measurements'][0]['runs'][0]['samples'][0]['recall_ms']+=1
    with pytest.raises(ValueError): study.validate_resources(changed,[case],procedure=study.RESOURCE_V2)
    with pytest.raises(ValueError): study.validate_resources({'byte_cap':4096,'measurements':[], 'passed':True},[case],procedure=study.RESOURCE_V2)
