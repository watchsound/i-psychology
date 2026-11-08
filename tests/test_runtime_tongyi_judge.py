"""Authored wire controls exercise real SDK/provider/audit owners, not judge accuracy."""
import copy
import json
from decimal import Decimal

import pytest

from psychology_base.eval.benchmarks import application_context as study
from psychology_base.eval.benchmarks import application_calls as calls
from psychology_base.eval.benchmarks import runtime_report_diagnostic as diagnostic
from psychology_base.applications.pilot.session_runner import SessionLogWriter
from psychology_base.llm.providers.openai import OpenAIProvider
from tests.test_runtime_semantic_diagnostic import runner


@pytest.fixture(scope='module')
def proposal():
    entry=runner()
    protocol=entry.protocol_for(dict(provider_attempts=1048,accounted_usd='0.127250525'),revision=6)
    return entry,diagnostic.items(revision=6),protocol


def wire_judge(tmp_path,protocol,*,response_change=None,status=200):
    openai=pytest.importorskip('openai');httpx=pytest.importorskip('httpx')
    budget=study.Budget(SessionLogWriter(tmp_path/'events.jsonl'),max_calls=1097,max_usd='1')
    judge,=study.make_routers(protocol,study.load_config(),budget,roles=('judge',))
    seen=[];controls=diagnostic.items(revision=6)
    def respond(request):
        payload=json.loads(request.content);seen.append(payload)
        if status!=200:
            return httpx.Response(status,json={'error':{'message':'AUTHORED TRANSPORT FAILURE'}},
                headers={'Location':'https://example.com/forbidden'})
        item=controls[len(seen)-1]
        raw=json.dumps(dict(item['expected_criteria'] or dict(addresses_request=True,grounded=True,
            attribution_correct=True,abstains_appropriately=True,critical_error=False),
            semantic_status='inconclusive' if item['expected_decision']=='inconclusive' else 'resolved',
            rationale='AUTHORED HTTP CONTROL; NOT REAL-MODEL EVIDENCE'))
        data=dict(id=f'chatcmpl-authored-{len(seen)}',object='chat.completion',created=1,
            model=calls.TONGYI_JUDGE_MODEL,choices=[dict(index=0,finish_reason='stop',
                message=dict(role='assistant',content=raw))],
            usage=dict(prompt_tokens=100,completion_tokens=70,total_tokens=170,
                completion_tokens_details=dict(reasoning_tokens=0)))
        if response_change:response_change(data)
        return httpx.Response(200,json=data)
    provider=judge.router.providers['tongyi']
    provider._client=openai.OpenAI(api_key='AUTHORED-UNIT-KEY',base_url=calls.TONGYI_JUDGE_ENDPOINT,
        max_retries=0,timeout=60,http_client=httpx.Client(transport=httpx.MockTransport(respond),follow_redirects=False))
    return judge,budget,seen


def assess_one(judge,controls,protocol):
    return study.assess_calibration_control('memory',controls[0],protocol,judge)


def ledger(budget):
    return [json.loads(line) for line in budget.writer.path.read_text(encoding='utf-8').splitlines()]


def test_scope_request_budget_and_existing_owner_reuse(proposal):
    entry,controls,protocol=proposal
    assert controls==diagnostic.items(revision=5)
    requests=entry.requests(controls,protocol)
    assert len(requests)==49 and all(row['prompt_revision']==4 for row in requests)
    assert all(row['max_tokens']==2048 and row['temperature']==0 for row in requests)
    assert all(row['inference_options']==calls.hosted_request_options(protocol['judge'])['inference_options'] for row in requests)
    assert protocol['budget']==dict(max_calls=1097,max_usd='0.162346285',local_only=False)
    assert entry.profile(6).cap==Decimal('0.03509576')
    for request,item in zip(requests,controls):
        assert json.loads(request['messages'][1]['content'])==study.runtime_memory_payload(
            item['app'],item['capture']['query'],item['capture']['context'],item['response'])
    first=calls.tongyi_judge_spec();first['call_policy']['max_tokens']=1
    assert calls.tongyi_judge_spec()['call_policy']['max_tokens']==2048


def test_factory_isolates_tongyi_without_mutating_config_or_native_defaults(proposal,tmp_path):
    from psychology_base.llm.providers.alibaba_tongyi import AlibabaTongyiProvider,DASHSCOPE_URL
    _,_,protocol=proposal;cfg=study.load_config();before=copy.deepcopy(cfg)
    with SessionLogWriter(tmp_path/'events.jsonl') as writer:
        budget=study.Budget(writer,max_calls=49,max_usd='1')
        judge,=study.make_routers(protocol,cfg,budget,roles=('judge',))
        provider=judge.router.providers['tongyi']
        assert type(provider) is OpenAIProvider and provider.name=='tongyi'
        assert provider.enable_thinking is False and provider.follow_redirects is False
        assert provider.json_mode and provider.reasoning_effort is None and provider.max_retries==0
        assert cfg==before and judge.router.cache is None and set(judge.router.providers)=={'tongyi'}
        assert AlibabaTongyiProvider().base_url==DASHSCOPE_URL


def test_full_sdk_wire_assessor_parser_action_and_raw_ledger_audit(proposal,tmp_path,monkeypatch):
    _,controls,protocol=proposal;judge,budget,seen=wire_judge(tmp_path,protocol)
    original=diagnostic.validate
    monkeypatch.setattr(diagnostic,'validate',lambda report,items,current:
        original(report,items,current,ledger_rows=ledger(budget)))
    try:
        report=diagnostic.run(controls,protocol,judge)
        assert len(seen)==budget.attempts==49 and not budget.pending
        assert report['metrics']['candidate_reference_screen_passed']
        assert report['passed'] is report['runtime_enabled'] is report['qualification_eligible'] is False
        assert report['metrics']['arms']['4']['reasoning_tokens']==0
        for request in seen:
            assert request['model']==calls.TONGYI_JUDGE_MODEL and request['max_tokens']==2048
            assert request['enable_thinking'] is False and request['temperature']==0
            assert request['response_format']=={'type':'json_object'}
            assert not {'reasoning_effort','service_tier','max_completion_tokens'} & request.keys()
        charge=(Decimal('.0375')*100+Decimal('.130')*70)/1_000_000
        assert budget.spent==49*charge
        for change in ('request','endpoint','model','action','usage','cost','reasoning','duplicate'):
            changed=copy.deepcopy(report);events=ledger(budget)
            if change=='request':events[0]['messages'][1]['content']='CHANGED'
            elif change=='endpoint':events[0]['inference_options']['endpoint']='https://example.com'
            elif change=='model':changed['rows'][0]['call']['completion']['metadata']['response_model']='wrong'
            elif change=='action':changed['rows'][0]['disposition']['action']='wrong'
            elif change=='usage':changed['rows'][0]['call']['completion']['completion_tokens']=2049
            elif change=='cost':changed['rows'][0]['call']['accounted_usd']='0'
            elif change=='reasoning':changed['rows'][0]['call']['completion']['metadata']['reasoning_tokens']=1
            else:changed['rows'][1]['call']['completion']['metadata']['completion_id']=changed['rows'][0]['call']['completion']['metadata']['completion_id']
            with pytest.raises(ValueError):original(changed,controls,protocol,ledger_rows=events)
    finally:
        judge.router.providers['tongyi']._client.close();budget.writer.close()


@pytest.mark.parametrize('status',[401,403,429,503,307])
def test_auth_errors_retries_and_redirects_do_not_dispatch_again(proposal,tmp_path,status):
    _,controls,protocol=proposal;judge,budget,seen=wire_judge(tmp_path,protocol,status=status)
    try:
        with pytest.raises(Exception):assess_one(judge,controls,protocol)
        assert len(seen)==budget.attempts==1 and not budget.pending and budget.uncertain
        assert [row['event'] for row in ledger(budget)]==['reservation','failure']
        assert budget.spent==Decimal('0.00071624')
    finally:
        judge.router.providers['tongyi']._client.close();budget.writer.close()


@pytest.mark.parametrize('field,value',[('region','singapore'),('base_url','https://example.com'),
    ('model','qwen3.7-flash'),('enable_thinking',True),('enable_thinking',0),('json_mode',False),
    ('input_rate','0'),('output_rate','0'),('max_input',12001),('chat_timeout_s',61),
    ('transport','native'),('call_policy',None)])
def test_profile_drift_rejected_before_provider_dispatch(proposal,tmp_path,field,value):
    _,_,protocol=proposal;changed=copy.deepcopy(protocol);changed['judge'][field]=value
    with SessionLogWriter(tmp_path/'events.jsonl') as writer:
        budget=study.Budget(writer,max_calls=49,max_usd='1')
        with pytest.raises(ValueError):study.make_routers(changed,study.load_config(),budget,roles=('judge',))
        assert budget.attempts==0


@pytest.mark.parametrize('field,value',[('version',True),('version',1),('max_tokens',2049),('temperature',False),('temperature',None)])
def test_call_policy_cannot_silently_widen_or_change(proposal,field,value):
    _,_,protocol=proposal;spec=copy.deepcopy(protocol['judge']);spec['call_policy'][field]=value
    with pytest.raises(ValueError):calls.inference_call_options(spec)


@pytest.mark.parametrize('change',['model','reasoning','reasoning_content','output_usage','input_usage','zero_output_usage'])
def test_response_attestation_and_usage_errors_are_durable_and_abort(proposal,tmp_path,change):
    _,controls,protocol=proposal
    def mutate(data):
        if change=='model':data['model']='qwen-alias-or-wrong-model'
        elif change=='reasoning':data['usage']['completion_tokens_details']['reasoning_tokens']=1
        elif change=='reasoning_content':data['choices'][0]['message']['reasoning_content']='UNEXPECTED THINKING'
        elif change=='output_usage':data['usage']['completion_tokens']=2049
        elif change=='zero_output_usage':data['usage']['completion_tokens']=0
        else:data['usage']['prompt_tokens']=12001
    judge,budget,seen=wire_judge(tmp_path,protocol,response_change=mutate)
    try:
        with pytest.raises(ValueError):assess_one(judge,controls,protocol)
        assert len(seen)==budget.attempts==1 and not budget.pending
        assert ledger(budget)[-1]['event']=='completion'
        if change.endswith('usage'):assert budget.spent==Decimal('0.00071624')
    finally:
        judge.router.providers['tongyi']._client.close();budget.writer.close()


def test_unreported_reasoning_is_unknown_and_length_limit_is_not_semantic_success(proposal,tmp_path):
    _,controls,protocol=proposal
    def mutate(data):
        data['usage'].pop('completion_tokens_details')
        data['choices'][0].update(finish_reason='length',message=dict(role='assistant',content=''))
    judge,budget,_=wire_judge(tmp_path,protocol,response_change=mutate)
    try:
        assessment=assess_one(judge,controls,protocol)
        row=dict(index=0,control=controls[0]['id'],prompt_revision=4,**assessment)
        metrics=diagnostic.metrics(controls,[row],revision=6)
        assert metrics['arms']['4']['reasoning_tokens'] is None
        assert metrics['arms']['4']['budget_limited']==metrics['arms']['4']['empty_outputs']==1
        assert not metrics['candidate_reference_screen_passed'] and metrics['remaining']==48
    finally:
        judge.router.providers['tongyi']._client.close();budget.writer.close()


def test_sdk_compatibility_options_are_opt_in_and_client_controls_are_applied():
    pytest.importorskip('openai')
    legacy=OpenAIProvider(api_key='AUTHORED-UNIT-KEY')
    bounded=OpenAIProvider(api_key='AUTHORED-UNIT-KEY',enable_thinking=False,
        request_timeout_s=60,follow_redirects=False,max_retries=0)
    client=bounded._ensure_client()
    try:
        assert legacy.enable_thinking is legacy.follow_redirects is legacy.request_timeout_s is None
        assert client.max_retries==0 and client._client.follow_redirects is False
        assert client.timeout==60
        with pytest.raises(ValueError):OpenAIProvider(enable_thinking=False,reasoning_effort='none')
        with pytest.raises(ValueError):OpenAIProvider(enable_thinking=0)
        with pytest.raises(ValueError):OpenAIProvider(request_timeout_s=True)
    finally:client.close()


def test_cli_retains_shared_approval_and_execution_owner(proposal,monkeypatch):
    entry,_,_=proposal;seen=[]
    monkeypatch.setattr(study,'execute_runtime_study',lambda *args,**kwargs:seen.append((args,kwargs)))
    entry.run('exact-approval.json',revision=6)
    assert seen[0][0][0]=='exact-approval.json'
    assert seen[0][1]['assess'] is diagnostic.run and 'local_only' not in seen[0][1]


def test_legacy_tongyi_call_options_stay_unchanged_but_screening_requires_opt_in(proposal,tmp_path):
    assert calls.inference_call_options(dict(provider='tongyi',model='qwen-max')) is None
    assert study.memory_judge_options(dict(judge=dict(provider='tongyi',model='qwen-max')))==dict(max_tokens=300,temperature=0)
    _,_,protocol=proposal;changed=copy.deepcopy(protocol);changed['judge'].pop('call_policy')
    with SessionLogWriter(tmp_path/'events.jsonl') as writer:
        budget=study.Budget(writer,max_calls=49,max_usd='1')
        with pytest.raises(ValueError):study.make_routers(changed,study.load_config(),budget,roles=('judge',))
        assert budget.attempts==0


@pytest.mark.parametrize('field,value',[('enable_thinking',True),('json_mode',False),
    ('follow_redirects',True),('max_retries',1),('request_timeout_s',61),('reasoning_effort','low')])
def test_actual_provider_drift_stops_before_reservation(proposal,tmp_path,field,value):
    _,controls,protocol=proposal;judge,budget,seen=wire_judge(tmp_path,protocol)
    setattr(judge.router.providers['tongyi'],field,value)
    try:
        with pytest.raises(ValueError):assess_one(judge,controls,protocol)
        assert len(seen)==budget.attempts==0
    finally:
        judge.router.providers['tongyi']._client.close();budget.writer.close()


def test_actual_sdk_timeout_drift_stops_before_reservation(proposal,tmp_path):
    _,controls,protocol=proposal;judge,budget,seen=wire_judge(tmp_path,protocol)
    judge.router.providers['tongyi']._client.timeout=61
    try:
        with pytest.raises(ValueError):assess_one(judge,controls,protocol)
        assert len(seen)==budget.attempts==0
    finally:
        judge.router.providers['tongyi']._client.close();budget.writer.close()


def test_shared_executor_uses_tongyi_endpoint_and_retains_approval_boundary(proposal,tmp_path,monkeypatch):
    from contextlib import contextmanager
    _,controls,protocol=proposal;judge,budget,seen=wire_judge(tmp_path,protocol)
    budget.writer.path.touch()  # The shared writer opens lazily; freeze the empty unit ledger.
    plan=dict(endpoint=calls.TONGYI_JUDGE_ENDPOINT,baseline_ledger_sha256=study.shared.digest(budget.writer.path),
        baseline=dict(provider_attempts=0,dispatches=0,cache_hits=0,accounted_usd='0'),
        additional_conservative_usd_cap='0.03509576')
    plan_path=tmp_path/'plan.json';plan_path.write_text(json.dumps(plan),encoding='utf-8')
    approval=tmp_path/'approval.json';approval.write_text(json.dumps(dict(approved=True,
        plan_sha256=study.shared.digest(plan_path),basis='AUTHORED UNIT APPROVAL')),encoding='utf-8')
    @contextmanager
    def local_budget(current):
        assert current==protocol
        yield budget
    def factory(current,cfg,current_budget):
        assert set(cfg.providers)=={'tongyi'} and cfg.get_provider('tongyi').base_url==calls.TONGYI_JUDGE_ENDPOINT
        assert current==protocol and current_budget is budget
        return judge
    def single_assessment(items,current,router,*,review,on_assessment):
        row=dict(index=0,control=items[0]['id'],prompt_revision=4,**assess_one(router,items,current))
        on_assessment(row)
        return dict(rows=[row],passed=False)
    monkeypatch.setattr(study,'study_budget',local_budget)
    try:
        study.execute_runtime_study(approval,plan_path,
            validator=lambda path:(plan,study.load_config(),controls,None,protocol),
            run_directory=tmp_path/'run',assess=single_assessment,ledger=budget.writer.path,judge_factory=factory)
        result=study.read(tmp_path/'run/result.json')
        assert result['judgments']==1 and len(seen)==1 and result['runtime_enabled'] is False
        assert result['accounting']['accounted_usd']==str(budget.spent)
    finally:
        judge.router.providers['tongyi']._client.close();budget.writer.close()


@pytest.mark.parametrize('revision',[True,6.0,'6',0,11])
def test_invalid_revision_cannot_select_alibaba(revision):
    with pytest.raises(ValueError):runner().profile(revision)
    with pytest.raises(ValueError):diagnostic.items(revision=revision)
