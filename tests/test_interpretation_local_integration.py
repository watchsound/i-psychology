"""Disclosed software completions exercise the real local router and auditor."""
from copy import deepcopy
from dataclasses import asdict, replace
from io import BytesIO
import json

import pytest

from psychology_base.applications.measurement.local_query_binding import (
    LocalChatMembershipAssessor, reference_messages, INTERPRETATION_PROMPT,
)
from psychology_base.eval.benchmarks.local_membership_audit import membership_wire_payload, audit_membership_capture
from psychology_base.eval.benchmarks.local_membership_diagnostic import run_local_membership_diagnostic
from psychology_base.llm.providers.llama_cpp import LlamaCppProvider
from tests.test_interpretation_contract import software_example, decode
from tests.test_reference_extraction import extraction_router
from tests.test_native_local_backend import native_spec, software_backend, router_for
from tests.test_native_membership_batch import batch


KIND='interpretation-selections-v2'


def rows(focus):
    return [dict(slot=i,record=0,constraint=None,input_format_version=4,
        pair=(json.dumps(focus,ensure_ascii=False),f'Description: predicate {i}')) for i in range(2)]


def semantic_wire(raw):
    return dict(coverage=raw['coverage'],readings=raw['readings'])


def test_source_interpretation_is_description_independent_and_pins_the_actual_window():
    focus,raw=software_example();router,seen=extraction_router(semantic_wire(raw))
    candidate=LocalChatMembershipAssessor(router,assessment_kind=KIND,format_version=4,
        max_input=7168,max_tokens=1024,reference_scope=focus)
    assert candidate(rows(focus))==decode(focus,raw)
    changed=rows(focus)
    for row in changed:row['pair']=(row['pair'][0],'Completely different target description')
    assert candidate(changed)==raw and len(seen)==2
    assert seen[0][0]==seen[1][0]==reference_messages(focus,interpretation=True)
    payload=json.loads(seen[0][0][1].content)
    assert 'descriptions' not in payload and 'targets' not in payload
    assert seen[0][0][0].content==INTERPRETATION_PROMPT
    wire=membership_wire_payload(rows(focus),'authored-test',assessment_kind=KIND,max_tokens=1024)
    assert wire['messages']==[asdict(m) for m in seen[0][0]] and wire['format']=='json'
    changed_scope=deepcopy(focus);changed_scope['context_scope']+=' changed'
    with pytest.raises(ValueError,match='pinned focus'):candidate(rows(changed_scope))
    assert len(seen)==2


@pytest.mark.parametrize('damage',['mode','schema','grouping','version','unfrozen_scope','identity','source','during'])
def test_source_adapter_rejects_configuration_or_response_damage_without_retry(damage):
    focus,raw=software_example();wire=semantic_wire(raw);router,seen=extraction_router(wire)
    args=dict(assessment_kind=KIND,format_version=4,max_input=7168,max_tokens=1024,reference_scope=focus)
    if damage=='mode':router.execution=replace(router.execution,mode='research')
    if damage=='schema':router.providers['ollama'].response_schema={'type':'object'}
    if damage=='grouping':args['descriptions_per_call']=1
    if damage=='version':args['format_version']=3
    if damage=='unfrozen_scope':args.pop('reference_scope')
    if damage in ('mode','schema','grouping','version','unfrozen_scope'):
        with pytest.raises(ValueError):LocalChatMembershipAssessor(router,**args)
        assert seen==[]
        return
    candidate=LocalChatMembershipAssessor(router,**args)
    if damage=='identity':wire['source_identity']='0'*64
    if damage=='source':raw['readings'][0]['bindings'][0]['span']=[999,999]
    if damage=='during':
        original=router.chat
        def chat(messages,**kwargs):
            result=original(messages,**kwargs)
            candidate.max_tokens=1
            return result
        router.chat=chat
    with pytest.raises(ValueError):candidate(rows(focus))
    assert len(seen)==1


def native_analysis_backend(spec,tmp_path,monkeypatch,raw):
    provider,wire,_=software_backend(spec,tmp_path,monkeypatch)
    rewrite_native_analysis(monkeypatch,raw)
    return provider,wire


def rewrite_native_analysis(monkeypatch,raw):
    original=LlamaCppProvider._open
    def opened(self,request,*,timeout):
        result=original(self,request,timeout=timeout)
        if request.full_url.endswith('/v1/chat/completions'):
            value=json.load(result);content=json.dumps(semantic_wire(raw),ensure_ascii=False)
            value['choices'][0]['message']['content']=content
            value['__verbose']['content']='Analysis.</think>\n'+content
            return BytesIO(json.dumps(value).encode())
        return result
    monkeypatch.setattr(LlamaCppProvider,'_open',opened)


def test_native_source_mode_uses_existing_provider_and_preserves_legacy_extraction_rejection(native_spec,tmp_path,monkeypatch):
    focus,raw=software_example()
    provider,wire=native_analysis_backend(native_spec,tmp_path,monkeypatch,raw)
    router=router_for(provider)
    candidate=LocalChatMembershipAssessor(router,assessment_kind=KIND,format_version=4,
        max_input=7168,max_tokens=native_spec.max_tokens,reference_scope=focus,reasoning_budget=True)
    assert candidate(rows(focus))==raw
    chats=[v for path,v,_ in wire if path=='/v1/chat/completions']
    assert len(chats)==1
    expected=membership_wire_payload(rows(focus),provider.advanced_model,assessment_kind=KIND,
        max_tokens=native_spec.max_tokens,think=True,native_spec=native_spec.record())
    assert chats[0]==expected
    for old_kind in ('reference-extraction-v1','reference-extraction-v2','reference-selections-v1'):
        with pytest.raises(ValueError):LocalChatMembershipAssessor(router,assessment_kind=old_kind,
            format_version=4,max_input=7168,max_tokens=native_spec.max_tokens)


def test_interpretation_validation_revision_changes_identity_and_invalidates_stale_assessors(monkeypatch):
    from psychology_base.applications.measurement import reference_readings as protocol
    focus,raw=software_example();router,seen=extraction_router(semantic_wire(raw))
    args=dict(assessment_kind=KIND,format_version=4,max_input=7168,max_tokens=1024,reference_scope=focus)
    old=LocalChatMembershipAssessor(router,**args)
    monkeypatch.setattr(protocol,'INTERPRETATION_VALIDATION_REVISION','declared-other-revision')
    new=LocalChatMembershipAssessor(router,**args)
    assert old.identity!=new.identity
    with pytest.raises(ValueError):old(rows(focus))
    assert not seen


def test_annotation_prompt_change_invalidates_stale_assessor_and_is_sent_without_source_labels(monkeypatch):
    from psychology_base.applications.measurement import local_query_binding as owner
    focus,raw=software_example();router,seen=extraction_router(semantic_wire(raw))
    args=dict(assessment_kind=KIND,format_version=4,max_input=7168,max_tokens=1024,reference_scope=focus)
    old=LocalChatMembershipAssessor(router,**args)
    changed=owner.INTERPRETATION_PROMPT+'\nProspective annotation amendment.'
    monkeypatch.setattr(owner,'INTERPRETATION_PROMPT',changed)
    new=LocalChatMembershipAssessor(router,**args)
    assert old.identity!=new.identity
    with pytest.raises(ValueError):old(rows(focus))
    assert not seen
    assert new(rows(focus))==raw and len(seen)==1
    assert seen[0][0][0].content==changed
    payload=json.loads(seen[0][0][1].content)
    assert 'readings' not in payload and 'descriptions' not in payload


@pytest.mark.parametrize('cap',[3072,4096])
def test_expanded_native_budget_dispatches_through_existing_capture_and_requires_explicit_thinking(
        native_spec,tmp_path,monkeypatch,cap):
    focus,raw=software_example()
    spec=replace(native_spec,max_tokens=cap,thinking_tokens=2048)
    provider,wire=native_analysis_backend(spec,tmp_path,monkeypatch,raw)
    provider=replace(provider,request_timeout_s=3600)
    from psychology_base.llm.execution import ExecutionPolicy
    from psychology_base.llm.config import default_config
    from psychology_base.llm.router import LLMRouter
    config=default_config()
    config=replace(config,cache=replace(config.cache,enabled=False),execution=ExecutionPolicy(
        offline_local_diagnostic=True,request_timeout_s=3600,operation_timeout_s=3600))
    router=LLMRouter(config,providers={},local_chat_provider=provider)
    args=dict(assessment_kind=KIND,format_version=4,max_input=8192-cap,max_tokens=cap,reference_scope=focus)
    with pytest.raises(ValueError):LocalChatMembershipAssessor(router,**args)
    with pytest.raises(ValueError):LocalChatMembershipAssessor(router,**dict(args,max_input=8193-cap),reasoning_budget=True)
    candidate=LocalChatMembershipAssessor(router,**args,reasoning_budget=True)
    assert candidate(rows(focus))==raw
    chats=[(value,timeout) for path,value,timeout in wire if path=='/v1/chat/completions']
    assert len(chats)==1 and chats[0][0]['max_tokens']==cap
    assert chats[0][0]['reasoning_budget_tokens']==2048 and 300<chats[0][1]<=3600
    assert spec.record()['max_tokens']==cap and spec.record()['thinking_tokens']==2048
    with pytest.raises(ValueError):replace(provider,request_timeout_s=3600.001)
    with pytest.raises(ValueError):replace(provider,model_metadata_timeout_s=61)


@pytest.mark.parametrize('semantic_failure',[False,True])
def test_existing_native_runner_capture_and_replay_audit_handle_v2_analysis(native_spec,tmp_path,monkeypatch,semantic_failure):
    state,_,lifecycle=batch(tmp_path,monkeypatch,native_spec)
    packet=state.read(state.out/'inputs.json');entry=packet['schedule'][0]
    entry['prompt_profile']='baseline-v1'
    packet['schedule']=[entry,dict(entry,id=entry['id']+'/second')]
    state.save(state.out/'inputs.json',packet)
    original_rows=state.args['rows_for'](packet['cases'][entry['case']],entry)
    focus=json.loads(original_rows[0]['pair'][0])
    from psychology_base.applications.measurement.reference_readings import interpretation_candidate_view
    raw=dict(schema='reference-source-selections-v2',source_identity=interpretation_candidate_view(focus)['source_identity'],
        coverage='enumerated',readings=[dict(bindings=[],events=[])])
    # An empty source-valid proposal is deliberately not semantic success.
    plan=state.read(state.plan_path)
    plan.update(assessment_kind=KIND,transport='llama_cpp',native_backend=native_spec.record(),think=True,
        reasoning_budget=True,model=native_spec.model.model_id,digest=native_spec.model.source_digest,
        native_operation_scope='condition',stop_on_semantic_error=True,
        input_sha256=state.args['archive'].base.sha(state.out/'inputs.json'))
    plan['limits'].update(max_native_attempts=2,max_metadata_attempts=14,max_wall_s=3600,metadata_timeout_s=60)
    state.save(state.plan_path,plan)
    rewrite_native_analysis(monkeypatch,raw)
    def software_grade(cases,references,schedule,observations):
        return dict(records=[dict(id=e['id'],correct=not semantic_failure and any(
            o['id']==e['id'] and o.get('analysis')==raw for o in observations)) for e in schedule],summary={})
    state.args['grade']=software_grade
    run_local_membership_diagnostic(**state.args)
    result=state.read(state.result_path)
    observed=state.read(state.out/'observations.json');events=state.read(state.out/'transport.json')
    assert result['completed'] is (not semantic_failure),result.get('error')
    assert result['native_attempts']==(1 if semantic_failure else 2)
    assert lifecycle==['start','close']
    assert 'analysis' in observed[0] and 'labels' not in observed[0]
    audit=audit_membership_capture(events,observed,packet['schedule'],
        rows_for=lambda e:state.args['rows_for'](packet['cases'][e['case']],e),
        snapshot=lambda p:state.read(state.out/p),model=plan['model'],digest=plan['digest'],
        max_native=2,max_metadata=14,max_tokens=plan['limits']['max_tokens'],
        max_input=plan['limits']['max_input'],assessment_kind=KIND,think=True,native_spec=native_spec.record())
    assert audit['native_attempts']==result['native_attempts']
    if semantic_failure:
        assert result['stop_reason']=='semantic_gate' and len(observed)==1
    changed=deepcopy(observed);changed[0]['analysis']['coverage']='unsupported'
    with pytest.raises(ValueError):audit_membership_capture(events,changed,packet['schedule'],
        rows_for=lambda e:state.args['rows_for'](packet['cases'][e['case']],e),
        snapshot=lambda p:state.read(state.out/p),model=plan['model'],digest=plan['digest'],
        max_native=2,max_metadata=14,max_tokens=plan['limits']['max_tokens'],
        max_input=plan['limits']['max_input'],assessment_kind=KIND,think=True,native_spec=native_spec.record())


def conditional_router(*, failure=None):
    """Scripted source/conditional fixtures; the actual assessor owns dispatch."""
    router,seen=extraction_router({})
    original=router.chat
    def chat(messages,**kwargs):
        result=original(messages,**kwargs);payload=json.loads(messages[1].content)
        if messages[0].content==INTERPRETATION_PROMPT:
            value=dict(coverage='unsupported' if failure=='unsupported' else 'enumerated',
                readings=[dict(bindings=[],events=[])])
            if failure=='source':value['source_identity']='0'*64
        else:
            source=payload['source'];text=source['turns'][source['focus_turn']]['text']
            vector=['match','nonmatch'] if text=='A question?' else ['nonmatch','match']
            value=dict(labels=[vector],open_labels=None)
            if failure=='conditional':value['labels']=[]
            if failure=='stale':value['request_identity']='0'*64
        return replace(result,content=json.dumps(value,ensure_ascii=False))
    router.chat=chat
    return router,seen


@pytest.mark.parametrize('application',['tutor','companion'])
@pytest.mark.parametrize('version',[4,7])
def test_one_interpreted_pool_assessment_is_shared_by_actual_selection_and_binding(application,version):
    from psychology_base.applications.context import render_context_block
    from psychology_base.applications.llm_responders import request_context_selection
    from psychology_base.applications.reporting_order import ReportingMembershipSession,bind_reporting_order
    from psychology_base.companion.reconstruction import ordered_selection
    from tests.test_source_order_visibility import source
    from tests.test_reporting_membership_session import QUERY
    fs=[source(i,text=text) for i,text in enumerate(('A question?','An answer.'))]
    router,seen=conditional_router()
    candidate=LocalChatMembershipAssessor(router,assessment_kind='interpreted-membership-v2',format_version=4,
        max_input=7168,max_tokens=1024)
    session=ReportingMembershipSession(application,QUERY,candidate)
    policy=request_context_selection('Policy.',QUERY,application=application,format_version=version,
        max_input=8000,mode='occurrences',membership_session=session)
    selected,trace=ordered_selection(fs,{f.id:0 for f in fs},6,policy)
    selected=[f for f,_ in selected]
    context=render_context_block(selected,assistant_speaker=application,format_version=version)
    binding=bind_reporting_order(application,QUERY,context,selection_trace=trace,membership_session=session)
    assert binding.selected is not None and binding.selected.source_position==0
    assert len(seen)==4  # Two stages per focus, not one call per reading/description.
    assert [m[0][0].content==INTERPRETATION_PROMPT for m in seen]==[True,False,True,False]
    assert all('descriptions' not in json.loads(seen[i][0][1].content) for i in (0,2))
    assert set(session.required_occurrences(list(reversed(fs))))=={f.id for f in selected}
    assert len(seen)==4


@pytest.mark.parametrize('failure,attempts',[('source',1),('unsupported',1),('conditional',2),('stale',2)])
def test_interpreted_assessment_does_not_repair_failures_or_retry(failure,attempts):
    router,seen=conditional_router(failure=failure)
    candidate=LocalChatMembershipAssessor(router,assessment_kind='interpreted-membership-v2',format_version=4,
        max_input=7168,max_tokens=1024)
    focus=dict(context_scope='Software test',prior_utterances=[],focus_utterance=dict(speaker='user',text='A question?'))
    from psychology_base.llm.execution import ExecutionDeferred
    with pytest.raises((ValueError,ExecutionDeferred)):candidate(rows(focus))
    assert len(seen)==attempts


def test_conditional_payload_cannot_exceed_the_bound_after_source_interpretation():
    router,seen=conditional_router()
    candidate=LocalChatMembershipAssessor(router,assessment_kind='interpreted-membership-v2',format_version=4,
        max_input=7168,max_tokens=1024)
    focus=dict(context_scope='Software test',prior_utterances=[],focus_utterance=dict(speaker='user',text='A question?'))
    supplied=rows(focus)
    for row in supplied:row['pair']=(row['pair'][0],'Description: '+'x'*5000)
    with pytest.raises(ValueError,match='frozen input scope'):candidate(supplied)
    assert len(seen)==1
