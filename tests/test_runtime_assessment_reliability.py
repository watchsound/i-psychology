"""Software/real-failure regression controls; authored wire output is not accuracy."""
import copy
import json
from decimal import Decimal

import pytest

from psychology_base.applications.measurement import memory_grounding as grounding
from psychology_base.eval.benchmarks import application_context as study
from psychology_base.eval.benchmarks import application_calls as calls
from psychology_base.eval.benchmarks import runtime_report_diagnostic as diagnostic
from psychology_base.llm.providers.openai import OpenAIProvider
from psychology_base.llm.types import ChatMessage, TaskTier
from tests.test_runtime_semantic_diagnostic import runner


def proposal():
    entry=runner()
    protocol=entry.protocol_for(dict(provider_attempts=1244,accounted_usd='0.1549251950'),revision=9)
    return entry,diagnostic.items(revision=9),protocol


def test_explicit_candidate_keeps_the_measured_contract_parser_and_default():
    old=grounding.runtime_memory_protocol(contract_version=3,prompt_revision=6)
    new=grounding.runtime_memory_protocol(contract_version=3,prompt_revision=7)
    assert {key:value for key,value in new.items() if key not in ('memory_prompt','runtime_memory_prompt_revision')} == {
        key:value for key,value in old.items() if key not in ('memory_prompt','runtime_memory_prompt_revision')}
    assert new['memory_prompt'] != old['memory_prompt']
    assert grounding.runtime_memory_protocol()['runtime_memory_contract_version']==1
    assert grounding.runtime_memory_payload('tutor','q','opaque','r')['VISIBLE_CONTEXT']=='opaque'


def test_schema_matches_existing_contract_bounds_and_has_no_shared_mutable_state():
    schema=grounding.runtime_evidence_schema()
    assert set(schema['properties'])=={'readings','addresses_request','abstains_appropriately','rationale'}
    readings=schema['properties']['readings'];segments=readings['items']
    claims=segments['items']['properties']['claims'];claim=claims['items']
    assert [(value['minItems'],value['maxItems']) for value in (readings,segments,claims)]==[(1,8),(1,64),(0,32)]
    assert claim['properties']['actors']['items']['enum']==['user','companion','tutor',None]
    for obj in (schema,segments['items'],claim,claim['properties']['evidence']['items']):
        assert set(obj['required'])==set(obj['properties']) and obj['additionalProperties'] is False
    schema['properties']['readings']['maxItems']=100
    assert grounding.runtime_evidence_schema()['properties']['readings']['maxItems']==8
    first=calls.tongyi_reliability_spec();first['response_schema']['additionalProperties']=True
    assert calls.tongyi_reliability_spec()['response_schema']['additionalProperties'] is False


def test_all_actual_failed_run_judgments_and_actions_remain_unchanged():
    plan=study.read(study.ROOT/'docs/experiments/phase-9-application-context/runtime-evidence-assessment-diagnostic-plan-v1.json')
    report=study.read(study.ROOT/'.data/application_context/runtime-evidence-assessment-diagnostic-v1/hosted-run/report.json')
    controls={item['id']:item for item in study.read(study.ROOT/plan['controls_path'])}
    for row in report['rows']:
        item=controls[row['control']]
        payload=grounding.runtime_memory_payload(item['app'],item['capture']['query'],item['capture']['context'],
            item['response'],contract_version=3)
        raw=row['call']['completion']
        parsed=grounding.parse_runtime_memory(raw['content'],raw['finish_reason'],False,contract_version=3,evidence=payload)
        assert parsed==row['judgment']
        assert grounding.runtime_memory_action(parsed,evidence=payload)==row['disposition']
    assert len(report['rows'])==49
    assert sum(row['judgment']['parse_failed'] for row in report['rows'])==15


def test_pair_uses_existing_exposed_references_and_excludes_reference_labels():
    entry,controls,protocol=proposal()
    full={item['id']:item for item in diagnostic.items(revision=8)}
    assert len(controls)==16 and all(item==full[item['id']] for item in controls)
    assert {item['app'] for item in controls}=={'companion','tutor'}
    assert {item['language'] for item in controls}=={'en','zh'}
    schedule=diagnostic.schedule(controls,revision=9)
    requests=entry.requests(controls,protocol)
    assert len(requests)==32
    assert protocol['judge']==calls.tongyi_reliability_spec()
    assert protocol['budget']['max_calls']==1276
    assert Decimal(protocol['budget']['max_usd'])==Decimal('0.4320677870')
    for index,item in enumerate(controls):
        assert [row['prompt_revision'] for row in schedule[index*2:index*2+2]]==([6,7] if index%2==0 else [7,6])
        pair=requests[index*2:index*2+2]
        assert pair[0]['messages'][1]==pair[1]['messages'][1]
        for request in pair:
            payload=json.loads(request['messages'][1]['content'])
            assert set(payload)=={'APPLICATION','QUERY','VISIBLE_CONTEXT','REPLY'}
            assert payload['REPLY']==item['response'] and payload['QUERY']==item['capture']['query']
            assert request['bounded_input_bytes']==calls.request_input_bound(
                [ChatMessage(**message) for message in request['messages']],response_schema=protocol['judge']['response_schema'])
            assert request['bounded_input_bytes']<=12000 and request['max_tokens']==4096
            assert request['inference_options']['response_schema']==grounding.runtime_evidence_schema()
    ceiling=(Decimal('.345')*12000+Decimal('1.101')*4106)/1_000_000
    assert entry.profile(9).cap==ceiling*32==Decimal('0.277142592')


@pytest.mark.parametrize('change', ['model','thinking','format','schema','output','temperature','rate'])
def test_mutated_candidate_specs_are_rejected_before_construction(change):
    spec=calls.tongyi_reliability_spec()
    if change=='model':spec['model']='qwen-max'
    elif change=='thinking':spec['enable_thinking']=True
    elif change=='format':spec['json_mode']=False
    elif change=='schema':spec['response_schema']['additionalProperties']=True
    elif change=='output':spec['call_policy']['max_tokens']=8192
    elif change=='temperature':spec['call_policy']['temperature']=True
    else:spec['input_rate']='0.00001'
    with pytest.raises(ValueError):calls.hosted_request_options(spec)


@pytest.mark.parametrize('schema,json_mode', [('object',True),([],True),({'type':'array'},True),({'type':'object'},False)])
def test_provider_rejects_invalid_schema_configuration(schema,json_mode):
    with pytest.raises(ValueError):OpenAIProvider(response_schema=schema,json_mode=json_mode)


def wire_candidate(tmp_path,*,malformed=False):
    from tests.test_runtime_tongyi_judge import wire_judge
    _,controls,protocol=proposal()
    by_id={item['id']:item for item in controls}
    schedule=diagnostic.schedule(controls,revision=9)
    index=0
    def change(data):
        nonlocal index
        item=by_id[schedule[index]['control']];index+=1
        # Intentionally inadequate authored output: it calls ALL prose nonhistorical.
        # The quality screen must reject it; this only verifies transport and audit.
        value=dict(readings=[[dict(text=item['response'],claims=[])]],addresses_request=True,
                   abstains_appropriately=True,rationale='AUTHORED WIRE CONTROL; NOT MODEL ACCURACY')
        if malformed:value['decision']='accept'
        data['model']=protocol['judge']['model']
        data['choices'][0]['message']['content']=json.dumps(value,ensure_ascii=False)
    judge,budget,seen=wire_judge(tmp_path,protocol,response_change=change)
    return controls,protocol,judge,budget,seen


@pytest.mark.parametrize('malformed',[False,True])
def test_real_sdk_sends_schema_shared_audit_rejects_poor_authored_judgments(tmp_path,monkeypatch,malformed):
    from tests.test_runtime_tongyi_judge import ledger
    controls,protocol,judge,budget,seen=wire_candidate(tmp_path,malformed=malformed)
    original=diagnostic.validate
    monkeypatch.setattr(diagnostic,'validate',lambda report,items,current:
        original(report,items,current,ledger_rows=ledger(budget)))
    try:
        report=diagnostic.run(controls,protocol,judge)
        assert len(seen)==budget.attempts==32 and not budget.pending
        assert report['runtime_enabled'] is report['passed'] is report['qualification_eligible'] is False
        assert report['metrics']['candidate_reference_screen_passed'] is False
        rejects=sum(item['expected_decision']=='reject' for item in controls)
        for arm in ('6','7'):
            assert report['metrics']['arms'][arm]['malformed_or_inconsistent']==(16 if malformed else 0)
            assert report['metrics']['arms'][arm]['unsupported_accepted']==(0 if malformed else rejects)
        for request in seen:
            assert request['model']=='qwen3.7-plus-2026-05-26'
            assert request['response_format']==dict(type='json_schema',json_schema=dict(
                name='conversation_assessment',strict=True,schema=grounding.runtime_evidence_schema()))
            assert request['max_completion_tokens']==4096 and 'max_tokens' not in request
            assert request['enable_thinking'] is False
        for mutation in ('wire-schema','schema-evidence','action'):
            altered=copy.deepcopy(report);events=ledger(budget)
            if mutation=='wire-schema':
                next(row for row in events if row['event']=='reservation')['inference_options']['response_schema']['additionalProperties']=True
            elif mutation=='schema-evidence':altered['rows'][0]['call']['completion']['metadata']['response_schema_sha256']='forged'
            else:altered['rows'][0]['disposition']['action']='forged'
            with pytest.raises(ValueError):original(altered,controls,protocol,ledger_rows=events)
    finally:
        judge.router.providers['tongyi']._client.close()
        budget.writer.close()


def test_schema_input_overhead_is_bounded_before_dispatch(tmp_path):
    _,protocol,judge,budget,seen=wire_candidate(tmp_path)
    framing=calls.request_input_bound([ChatMessage('user','')])
    messages=[ChatMessage('user','x'*(12000-framing))]
    assert calls.request_input_bound(messages)==12000
    assert calls.request_input_bound(messages,response_schema=protocol['judge']['response_schema'])>12000
    try:
        with pytest.raises(ValueError,match='input bound'):
            judge.chat(messages,tier=TaskTier.ADVANCED,max_tokens=4096,temperature=0)
        assert not seen and budget.attempts==0
    finally:
        judge.router.providers['tongyi']._client.close()
        budget.writer.close()


def test_provider_schema_drift_is_rejected_without_call(tmp_path):
    controls,protocol,judge,budget,seen=wire_candidate(tmp_path)
    judge.router.providers['tongyi'].response_schema['additionalProperties']=True
    try:
        with pytest.raises(ValueError):study.assess_calibration_control('memory',controls[0],protocol,judge)
        assert not seen and budget.attempts==0
    finally:
        judge.router.providers['tongyi']._client.close()
        budget.writer.close()


@pytest.mark.parametrize('entry',[dict(prompt_revision=4),dict(prompt_revision=True),
    dict(prompt_revision=7,inference_arm='on')])
def test_pair_cannot_silently_change_contract_or_decoding(entry):
    with pytest.raises(ValueError):diagnostic.condition_protocol(proposal()[2],entry)


@pytest.mark.parametrize('revision',[True,9.0,'9',0,11])
def test_invalid_profile_cannot_select_or_dispatch(revision):
    with pytest.raises(ValueError):runner().profile(revision)
    with pytest.raises(ValueError):diagnostic.items(revision=revision)
