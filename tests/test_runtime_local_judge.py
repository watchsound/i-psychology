"""Local diagnostic controls use authored wire replies, never empirical labels."""
import copy
import io
import json
from decimal import Decimal

import pytest

from psychology_base.eval.benchmarks import application_context as study
from psychology_base.eval.benchmarks import runtime_report_diagnostic as diagnostic
from tests.test_runtime_semantic_diagnostic import runner


@pytest.fixture(scope='module')
def proposal():
    entry = runner()
    controls = diagnostic.items(revision=5)
    # Composition-only budget input; live preparation must read the real ledger.
    protocol = entry.protocol_for(dict(provider_attempts=999,accounted_usd='0.127250525'),revision=5)
    return entry, controls, protocol


def test_local_scope_keeps_all_49_inputs_and_fixes_v4(proposal):
    entry, controls, protocol = proposal
    assert controls == diagnostic.items(revision=3)
    requests = entry.requests(controls,protocol)
    assert len(requests) == 49 and [r['index'] for r in requests] == list(range(49))
    assert all(r['prompt_revision']==4 and r['max_tokens']==300 and r['temperature']==0 for r in requests)
    assert all('inference_arm' not in r for r in requests)
    assert protocol['judge']['provider']=='ollama' and protocol['judge']['model']=='qwen2.5:7b'
    assert protocol['judge']['json_mode'] and protocol['judge']['context_length']==16384
    assert protocol['budget']==dict(max_calls=1048,max_usd='0',local_only=True)
    assert protocol['judge']['chat_timeout_s']==300 and protocol['judge']['model_metadata_timeout_s']==60
    assert entry.profile(5).cap==0
    for request,item in zip(requests,controls):
        payload=json.loads(request['messages'][1]['content'])
        assert payload==study.runtime_memory_payload(item['app'],item['capture']['query'],item['capture']['context'],item['response'])


def test_local_router_factory_applies_frozen_timeouts_without_mutating_config(proposal,tmp_path):
    from psychology_base.applications.pilot.session_runner import SessionLogWriter
    _,_,protocol=proposal
    cfg=study.load_config();before=copy.deepcopy(cfg)
    with SessionLogWriter(tmp_path/'calls.jsonl') as writer:
        budget=study.Budget(writer,max_calls=49,max_usd='0',local_only=True)
        judge,=study.make_routers(protocol,cfg,budget,roles=('judge',))
        provider=judge.router.providers['ollama']
        assert provider.request_timeout_s==300 and provider.model_metadata_timeout_s==60
        assert cfg==before and judge.remote_spec is None and judge.local_spec==protocol['judge']
        assert set(judge.router.providers)=={'ollama'} and judge.router.cache is None


def test_full_local_wire_loop_and_audit_preserve_reference_and_runtime_boundaries(proposal,tmp_path,monkeypatch):
    """Real adapter/router/parser; only localhost HTTP responses are authored unit controls."""
    from psychology_base.applications.pilot.session_runner import SessionLogWriter
    _,controls,protocol=proposal
    with SessionLogWriter(tmp_path/'calls.jsonl') as writer:
        budget=study.Budget(writer,max_calls=49,max_usd='0',local_only=True)
        judge,=study.make_routers(protocol,study.load_config(),budget,roles=('judge',))
        spec=protocol['judge'];seen=[]
        def open_local(request,*,timeout):
            endpoint=request.full_url.rsplit('/',1)[-1]
            if endpoint=='tags':
                data=dict(models=[dict(name=spec['model'],digest=spec['digest'],size=4683087332)])
            elif endpoint=='show':
                data=dict(details=dict(family='qwen2'),capabilities=['completion'],model_info={'qwen2.context_length':32768})
            else:
                assert timeout==300
                payload=json.loads(request.data);item=controls[len(seen)];seen.append(payload)
                assert payload['options']==dict(temperature=0,num_predict=300,num_ctx=16384,seed=0)
                assert payload['format']=='json' and payload['stream'] is False
                raw=json.dumps(dict(item['expected_criteria'] or dict(addresses_request=True,grounded=True,
                    attribution_correct=True,abstains_appropriately=True,critical_error=False),
                    semantic_status='inconclusive' if item['expected_decision']=='inconclusive' else 'resolved',
                    rationale='AUTHORED WIRE CONTROL; NOT MODEL EVIDENCE'))
                data=dict(model=spec['model'],message=dict(content=raw),done=True,done_reason='stop',prompt_eval_count=100,eval_count=70)
            return io.BytesIO(json.dumps(data).encode())
        monkeypatch.setattr(judge.router.providers['ollama'],'_open',open_local)
        validate=diagnostic.validate
        def ledger():return [json.loads(line) for line in writer.path.read_text(encoding='utf-8').splitlines()]
        monkeypatch.setattr(diagnostic,'validate',lambda report,items,current:validate(report,items,current,ledger_rows=ledger()))
        progress=[]
        report=diagnostic.run(controls,protocol,judge,on_assessment=progress.append)
        assert len(seen)==len(progress)==budget.attempts==49 and budget.spent==0 and not budget.pending
        assert report['metrics']['candidate_reference_screen_passed']
        assert report['metrics']['arms']['4']['settled_full_matches']==43
        assert report['metrics']['arms']['4']['ambiguous_inconclusive']==6
        assert report['metrics']['arms']['4']['reasoning_tokens'] is None
        assert report['passed'] is report['runtime_enabled'] is report['qualification_eligible'] is False
        for mutation in ('reservation-local','digest','tokens','request','action'):
            changed=copy.deepcopy(report);events=ledger()
            if mutation=='reservation-local':events[0]['local_only']=False
            elif mutation=='digest':changed['rows'][0]['call']['completion']['metadata']['model_attestation']['digest']='changed'
            elif mutation=='tokens':changed['rows'][0]['call']['completion']['completion_tokens']=301
            elif mutation=='request':events[0]['messages'][1]['content']='changed'
            else:changed['rows'][0]['disposition']['action']='invented'
            with pytest.raises(ValueError):validate(changed,controls,protocol,ledger_rows=events)


def test_local_cli_uses_existing_executor_without_hosted_approval(proposal,monkeypatch):
    entry,_,_=proposal;seen=[]
    monkeypatch.setattr(study,'execute_runtime_study',lambda *args,**kwargs:seen.append((args,kwargs)))
    entry.run(None,revision=5)
    assert seen[0][0][0] is None and seen[0][1]['local_only'] is True
    assert seen[0][1]['assess'] is diagnostic.run


@pytest.mark.parametrize('field,value',[('screening',False),('local_only',False),('runtime_memory_prompt_revision',5)])
def test_local_protocol_drift_stops_before_generation(proposal,tmp_path,field,value):
    from psychology_base.applications.pilot.session_runner import SessionLogWriter
    _,controls,protocol=proposal;changed=copy.deepcopy(protocol);changed[field]=value
    with SessionLogWriter(tmp_path/'calls.jsonl') as writer:
        budget=study.Budget(writer,max_calls=49,max_usd='0',local_only=True)
        judge,=study.make_routers(protocol,study.load_config(),budget,roles=('judge',))
        with pytest.raises(ValueError):diagnostic.run(controls,changed,judge)
        assert budget.attempts==0


def test_local_metrics_do_not_credit_a_missing_or_failed_assessment(proposal):
    _,controls,_=proposal
    result=diagnostic.metrics(controls,[],revision=5)
    assert not result['complete'] and result['remaining']==49 and not result['candidate_reference_screen_passed']


@pytest.mark.parametrize('revision',[True,5.0,'5',0,11])
def test_invalid_revision_does_not_select_local_execution(revision):
    with pytest.raises(ValueError):runner().profile(revision)
    with pytest.raises(ValueError):diagnostic.items(revision=revision)


@pytest.mark.parametrize('mutation',['plan-local','protocol-local','remote-provider','endpoint','spend','budget-local'])
def test_local_executor_rejects_external_or_billable_scope_before_budget(proposal,tmp_path,monkeypatch,mutation):
    _,_,original=proposal;protocol=copy.deepcopy(original)
    plan=dict(local_only=True,endpoint='http://127.0.0.1:11434',additional_conservative_usd_cap='0')
    if mutation=='plan-local':plan['local_only']=False
    elif mutation=='protocol-local':protocol['local_only']=False
    elif mutation=='remote-provider':protocol['judge']['provider']='openai'
    elif mutation=='endpoint':plan['endpoint']='https://example.com'
    elif mutation=='spend':plan['additional_conservative_usd_cap']='.001'
    else:protocol['budget']['local_only']=False
    def forbidden(*args,**kwargs):pytest.fail('Invalid local scope must stop before budget/provider creation')
    monkeypatch.setattr(study,'study_budget',forbidden)
    with pytest.raises(ValueError,match='Local execution'):
        study.execute_runtime_study(None,tmp_path/'plan.json',validator=lambda path:(plan,None,None,None,protocol),
            run_directory=tmp_path/'run',assess=forbidden,ledger=tmp_path/'ledger',local_only=True)


def test_hosted_executor_still_requires_approval_before_validator(tmp_path):
    def forbidden(*args,**kwargs):pytest.fail('Hosted scope must require exact approval before validator/provider access')
    with pytest.raises(ValueError,match='approval'):
        study.execute_runtime_study(None,tmp_path/'plan.json',validator=forbidden,
            run_directory=tmp_path/'run',assess=forbidden,ledger=tmp_path/'ledger')
    with pytest.raises(ValueError,match='hosted approval'):
        study.execute_runtime_study(tmp_path/'old-approval.json',tmp_path/'plan.json',validator=forbidden,
            run_directory=tmp_path/'run',assess=forbidden,ledger=tmp_path/'ledger',local_only=True)


@pytest.fixture(scope='module')
def frozen_local_packet(tmp_path_factory):
    entry=runner();selected=entry.profile(5)
    if not selected.plan.exists():pytest.skip('Prepared after local pre-freeze tests pass')
    original=study.read(selected.plan)
    prefix=b''.join(line for line in entry.LEDGER.read_bytes().splitlines(keepends=True)
                    if json.loads(line)['dispatch']<=original['baseline']['provider_attempts'])
    directory=tmp_path_factory.mktemp('local-judge-frozen-ledger');ledger=directory/'ledger.jsonl';ledger.write_bytes(prefix)
    assert study.shared.digest(ledger)==original['baseline_ledger_sha256']
    entry.LEDGER=ledger
    controls=diagnostic.items(revision=5);summary=entry.baseline(revision=5)
    plan=dict(entry.fields(controls,entry.protocol_for(summary,revision=5),study.load_config(),summary,revision=5),
              timestamp_utc='LOCAL-ONLY-PROSPECTIVE-PLAN')
    path=directory/'plan.json';entry.save(path,plan)
    assert entry.validate(path,revision=5)[0]==plan
    return entry,plan


@pytest.mark.parametrize('field,value',[('local_only',False),('endpoint','https://example.com'),
    ('additional_attempts',50),('shared_attempt_cap',1049),('additional_conservative_usd_cap','1'),
    ('prompt_revisions',[5]),('health_result',{}),('requests_sha256','wrong')])
def test_local_frozen_scope_drift_rejects_offline(frozen_local_packet,tmp_path,field,value):
    entry,original=frozen_local_packet;changed=copy.deepcopy(original);changed[field]=value
    path=tmp_path/'changed-plan.json';entry.save(path,changed)
    with pytest.raises(ValueError):entry.validate(path,revision=5)
