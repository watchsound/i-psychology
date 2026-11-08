"""Authored HTTP controls test execution/accounting, never real judge accuracy."""
import copy
import json
from contextlib import contextmanager
from decimal import Decimal

import pytest

from psychology_base.eval.benchmarks import application_context as study
from psychology_base.eval.benchmarks import application_calls as calls
from psychology_base.eval.benchmarks import runtime_report_diagnostic as diagnostic
from psychology_base.applications.pilot.session_runner import SessionLogWriter
from tests.test_runtime_semantic_diagnostic import runner


@pytest.fixture(scope='module')
def proposal():
    entry=runner();controls=diagnostic.items(revision=7)
    protocol=entry.protocol_for(dict(provider_attempts=1097,accounted_usd='0.1303510625'),revision=7)
    return entry,controls,protocol


@contextmanager
def wire_judges(tmp_path,proposal,*,response_change=None,status=200):
    openai=pytest.importorskip('openai');httpx=pytest.importorskip('httpx')
    _,controls,protocol=proposal
    budget=study.Budget(SessionLogWriter(tmp_path/'events.jsonl'),max_calls=1195,max_usd='1')
    judges=diagnostic.make_reasoning_judges(protocol,study.load_config(),budget)
    by_payload={study.memory_judge_messages({'app':item['app']},None,item['capture'],
        item['response'],None,protocol=protocol)[1].content:item for item in controls}
    seen=[]
    def respond(request):
        payload=json.loads(request.content);seen.append(payload)
        if status!=200:
            return httpx.Response(status,json={'error':{'message':'AUTHORED TRANSPORT FAILURE'}},
                headers={'Location':'https://example.com/forbidden'})
        item=by_payload[payload['messages'][1]['content']]
        content=json.dumps(dict(item['expected_criteria'] or dict(addresses_request=True,grounded=True,
            attribution_correct=True,abstains_appropriately=True,critical_error=False),
            semantic_status='inconclusive' if item['expected_decision']=='inconclusive' else 'resolved',
            rationale='AUTHORED WIRE CONTROL; NOT REAL-MODEL EVIDENCE'))
        thinking=payload['enable_thinking']
        data=dict(id=f'chatcmpl-authored-reasoning-{len(seen)}',object='chat.completion',created=1,
            model=calls.TONGYI_JUDGE_MODEL,choices=[dict(index=0,finish_reason='stop',
                message=dict(role='assistant',content=content,
                    reasoning_content='AUTHORED REASONING CONTROL' if thinking else ''))],
            usage=dict(prompt_tokens=100,completion_tokens=70,total_tokens=170,
                completion_tokens_details=dict(reasoning_tokens=20 if thinking else 0)))
        if response_change:response_change(data,payload)
        return httpx.Response(200,json=data)
    try:
        for judge in judges.values():
            judge.router.providers['tongyi']._client=openai.OpenAI(api_key='AUTHORED-UNIT-KEY',
                base_url=calls.TONGYI_JUDGE_ENDPOINT,max_retries=0,timeout=60,
                http_client=httpx.Client(transport=httpx.MockTransport(respond),follow_redirects=False))
        yield judges,budget,seen
    finally:
        for judge in judges.values():
            client=judge.router.providers['tongyi']._client
            if client is not None:client.close()
        budget.writer.close()


def ledger(budget):
    return [json.loads(line) for line in budget.writer.path.read_text(encoding='utf-8').splitlines()]


def first_assessment(proposal,judges,arm):
    _,controls,protocol=proposal
    entry=dict(index=0 if arm=='off' else 1,control=controls[0]['id'],prompt_revision=4,inference_arm=arm)
    result=study.assess_calibration_control('memory',controls[0],diagnostic.condition_protocol(protocol,entry),judges[arm])
    return dict(entry,**result)


def test_frozen_pairs_hold_everything_except_thinking_fixed(proposal):
    entry,controls,protocol=proposal
    assert controls==diagnostic.items(revision=6)
    requests=entry.requests(controls,protocol)
    assert len(requests)==98 and protocol['budget']==dict(max_calls=1195,max_usd='0.2267615025',local_only=False)
    assert entry.profile(7).cap==Decimal('0.09641044')
    for index in range(49):
        pair=requests[index*2:index*2+2]
        assert [row['inference_arm'] for row in pair]==(['off','on'] if index%2==0 else ['on','off'])
        assert pair[0]['messages']==pair[1]['messages'] and all(row['prompt_revision']==4 for row in pair)
        for row in pair:
            assert row['max_tokens']==4096 and row['temperature']==0
            assert row['inference_options']['configured_max_completion_tokens']==4096
            assert row['inference_options']['billable_output_ceiling']==4106
            assert row['inference_options']['enable_thinking'] is (row['inference_arm']=='on')
            assert set(json.loads(row['messages'][1]['content']))=={'APPLICATION','QUERY','VISIBLE_CONTEXT','REPLY'}
    off,on=[diagnostic.condition_protocol(protocol,dict(prompt_revision=4,inference_arm=arm))['judge'] for arm in ('off','on')]
    assert dict(off,enable_thinking=True)==on
    assert calls.billable_output_limit(off,4096)==4106
    assert calls.billable_output_limit(calls.tongyi_judge_spec(),2048)==2048


def test_real_sdk_98_request_loop_and_raw_audit_reuse_existing_owners(proposal,tmp_path,monkeypatch):
    _,controls,protocol=proposal
    validate=diagnostic.validate
    with wire_judges(tmp_path,proposal) as (judges,budget,seen):
        monkeypatch.setattr(diagnostic,'validate',lambda report,items,current:
            validate(report,items,current,ledger_rows=ledger(budget)))
        report=diagnostic.run(controls,protocol,judges)
        assert len(seen)==budget.attempts==98 and not budget.pending
        metrics=report['metrics']
        assert metrics['paired']['both_match']==49 and metrics['candidate_reference_screen_passed']
        assert metrics['arms']['on']['reasoning_tokens']==49*20
        assert report['passed'] is report['qualification_eligible'] is report['runtime_enabled'] is False
        for index in range(49):
            first,second=seen[index*2:index*2+2]
            assert dict(first,enable_thinking=second['enable_thinking'])==second
        for request in seen:
            assert request['max_completion_tokens']==4096 and 'max_tokens' not in request
            assert request['response_format']=={'type':'json_object'} and request['temperature']==0
            assert not {'reasoning_effort','service_tier','thinking_budget'} & request.keys()
        expected=98*(Decimal('.0375')*100+Decimal('.130')*70)/1_000_000
        assert budget.spent==expected  # Reasoning is already included in the 70 total output tokens.
        events=ledger(budget)
        for change in ('wire_limit','mode','reported_limit','duplicate','cost','action'):
            changed=copy.deepcopy(report);saved=copy.deepcopy(events)
            if change=='wire_limit':saved[0]['inference_options']['configured_max_completion_tokens']=8192
            elif change=='mode':changed['rows'][1]['call']['completion']['metadata']['enable_thinking']=False
            elif change=='reported_limit':changed['rows'][1]['call']['completion']['metadata']['configured_max_completion_tokens']=8192
            elif change=='duplicate':changed['rows'][1]['call']['completion']['metadata']['completion_id']=changed['rows'][0]['call']['completion']['metadata']['completion_id']
            elif change=='cost':changed['rows'][1]['call']['accounted_usd']='0'
            else:changed['rows'][1]['disposition']['action']='changed'
            with pytest.raises(ValueError):validate(changed,controls,protocol,ledger_rows=saved)


@pytest.mark.parametrize('tokens',[4096,4097,4106,4107])
def test_documented_output_margin_is_reserved_and_audited(proposal,tmp_path,tokens):
    def mutate(data,payload):data['usage']['completion_tokens']=tokens
    with wire_judges(tmp_path,proposal,response_change=mutate) as (judges,budget,seen):
        if tokens>4106:
            with pytest.raises(ValueError):first_assessment(proposal,judges,'off')
            assert budget.spent==Decimal('0.00098378')
        else:
            row=first_assessment(proposal,judges,'off')
            result=diagnostic.metrics(proposal[1],[row],revision=7)
            assert result['arms']['off']['output_limit_overshoots']==int(tokens>4096)
            assert budget.spent==(Decimal('.0375')*100+Decimal('.130')*tokens)/1_000_000
            assert not result['candidate_reference_screen_passed']
        assert len(seen)==budget.attempts==1 and not budget.pending


@pytest.mark.parametrize('value',[None,0,True,-1,71,'20',20.0])
def test_on_arm_requires_reported_bounded_observed_reasoning(proposal,tmp_path,value):
    def mutate(data,payload):data['usage']['completion_tokens_details']['reasoning_tokens']=value
    with wire_judges(tmp_path,proposal,response_change=mutate) as (judges,budget,seen):
        with pytest.raises(ValueError):first_assessment(proposal,judges,'on')
        assert len(seen)==budget.attempts==1 and not budget.pending


@pytest.mark.parametrize('field',['prompt_tokens','completion_tokens'])
@pytest.mark.parametrize('value',[True,'70',70.0])
def test_raw_usage_types_cannot_be_silently_coerced_by_sdk(proposal,tmp_path,field,value):
    def mutate(data,payload):data['usage'][field]=value
    with wire_judges(tmp_path,proposal,response_change=mutate) as (judges,budget,seen):
        with pytest.raises(ValueError):first_assessment(proposal,judges,'on')
        assert len(seen)==budget.attempts==1 and budget.spent==Decimal('0.00098378')
        assert ledger(budget)[-1]['usage_valid'] is False
        assert type(ledger(budget)[-1]['completion'][field]) is type(value)


def test_missing_reasoning_content_is_not_attested_as_thinking(proposal,tmp_path):
    def mutate(data,payload):data['choices'][0]['message'].pop('reasoning_content')
    with wire_judges(tmp_path,proposal,response_change=mutate) as (judges,budget,seen):
        with pytest.raises(ValueError):first_assessment(proposal,judges,'on')
        assert len(seen)==1


@pytest.mark.parametrize('value',[None,True,1,'   '])
def test_reasoning_evidence_must_be_raw_nonempty_text(proposal,tmp_path,value):
    def mutate(data,payload):data['choices'][0]['message']['reasoning_content']=value
    with wire_judges(tmp_path,proposal,response_change=mutate) as (judges,budget,seen):
        with pytest.raises(ValueError):first_assessment(proposal,judges,'on')
        assert len(seen)==budget.attempts==1


def test_truncation_empty_output_and_unknown_off_reasoning_receive_no_free_credit(proposal,tmp_path):
    def mutate(data,payload):
        if payload['enable_thinking']:
            data['choices'][0]['message']['content']=''
            data['choices'][0]['finish_reason']='length'
            data['usage']['completion_tokens']=4096
            data['usage']['completion_tokens_details']['reasoning_tokens']=4096
        else:data['usage'].pop('completion_tokens_details')
    with wire_judges(tmp_path,proposal,response_change=mutate) as (judges,_,_):
        rows=[first_assessment(proposal,judges,arm) for arm in ('off','on')]
        result=diagnostic.metrics(proposal[1],rows,revision=7)
        assert result['arms']['off']['reasoning_tokens'] is None
        assert result['arms']['on']['budget_limited']==result['arms']['on']['empty_outputs']==1
        assert result['paired']['regressed']==1 and not result['candidate_reference_screen_passed']


@pytest.mark.parametrize('field,value',[('total_completion_limit',False),('total_completion_limit',1),
    ('billable_output_ceiling',4096),('billable_output_ceiling',4107),('enable_thinking',1),
    ('model','qwen3.7-flash'),('json_mode',False),('input_rate','0'),('base_url','https://example.com')])
def test_profile_drift_cannot_widen_or_disable_the_budget(proposal,field,value):
    spec=copy.deepcopy(proposal[2]['judge']);spec[field]=value
    with pytest.raises(ValueError):calls.inference_call_options(spec)


@pytest.mark.parametrize('field,value',[('version',True),('max_tokens',4097),('temperature',None)])
def test_call_policy_is_frozen(proposal,field,value):
    spec=copy.deepcopy(proposal[2]['judge']);spec['call_policy'][field]=value
    with pytest.raises(ValueError):calls.inference_call_options(spec)


@pytest.mark.parametrize('field,value',[('total_completion_limit',False),('enable_thinking',False)])
def test_actual_provider_drift_blocks_before_reservation(proposal,tmp_path,field,value):
    with wire_judges(tmp_path,proposal) as (judges,budget,seen):
        setattr(judges['on'].router.providers['tongyi'],field,value)
        with pytest.raises(ValueError):first_assessment(proposal,judges,'on')
        assert len(seen)==budget.attempts==0


@pytest.mark.parametrize('status',[401,429,503,307])
def test_transport_failure_does_not_retry_or_redirect(proposal,tmp_path,status):
    with wire_judges(tmp_path,proposal,status=status) as (judges,budget,seen):
        with pytest.raises(Exception):first_assessment(proposal,judges,'on')
        assert len(seen)==budget.attempts==1 and not budget.pending
        assert budget.spent==Decimal('0.00098378') and ledger(budget)[-1]['event']=='failure'


def test_cli_and_pair_factory_reuse_the_existing_executor(proposal,monkeypatch):
    entry,_,_=proposal;seen=[]
    monkeypatch.setattr(study,'execute_runtime_study',lambda *args,**kwargs:seen.append((args,kwargs)))
    entry.run('AUTHORED-EXACT-APPROVAL.json',revision=7)
    assert seen[0][1]['judge_factory'] is diagnostic.make_reasoning_judges
    assert seen[0][1]['assess'] is diagnostic.run


@pytest.mark.parametrize('entry',[dict(prompt_revision=5,inference_arm='on'),dict(prompt_revision=4,inference_arm='low')])
def test_pairing_cannot_change_prompt_or_arm(proposal,entry):
    with pytest.raises(ValueError):diagnostic.condition_protocol(proposal[2],entry)


@pytest.mark.parametrize('revision',[True,7.0,'7',0,11])
def test_invalid_revision_cannot_select_the_comparison(revision):
    with pytest.raises(ValueError):runner().profile(revision)
    with pytest.raises(ValueError):diagnostic.items(revision=revision)
