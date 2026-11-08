"""Structural proposal contracts; injected proposals are not model accuracy."""
from contextlib import nullcontext
from dataclasses import replace
import itertools
import json
from types import SimpleNamespace

import pytest

from psychology_base.applications.context import render_context_block
from psychology_base.applications.measurement.local_query_binding import LocalQueryProposer, proposal_messages
from psychology_base.applications.measurement.memory_grounding import decode_runtime_memory_context
from psychology_base.applications.reporting_order import bind_reporting_order, respond_with_reporting_order
from psychology_base.llm.execution import ExecutionDeferred, ExecutionPolicy
from tests.test_source_order_visibility import source

QUERY="Which came earlier in the conversation: the teacher's opening question or the learner's first response?"


def utterance(position,text,actor):
    f=source(position,text=text)
    return replace(f,tags=tuple('speaker:'+actor if t=='speaker:user' else t for t in f.tags))


def block(fs):return render_context_block(fs,assistant_speaker='tutor',format_version=4)


class Proposal:
    identity='authored-structural-test-only'
    def __init__(self,value):self.value=value;self.calls=[]
    def __call__(self,*args):self.calls.append(args);return self.value


def test_optional_proposer_does_not_run_for_supported_fast_path_or_unrelated_request():
    def forbidden(*args):pytest.fail('No semantic request for this operation')
    forbidden.identity='must-not-run'
    fs=[source(0,text='red'),source(1,text='blue')]
    for query in ('Help me solve this problem.','Which phrase was mentioned earlier: red or blue?'):
        bind_reporting_order('tutor',query,block(fs),proposer=forbidden)


def test_recognized_nonliteral_without_qualified_proposer_stays_unresolved_locally():
    value=bind_reporting_order('tutor',QUERY,'')
    assert value.resolution.reason=='semantic_binding_unavailable' and value.selected is None
    response=respond_with_reporting_order(lambda *a:pytest.fail('No fallback generation'),'', '',QUERY,
        application='tutor',enabled=True)
    assert 'does not establish' in response


def test_proposal_supplies_matches_but_existing_relations_resolve_first_and_order():
    fs=[utterance(0,'What did you try?','tutor'),utterance(1,'I tried addition.','user'),
        utterance(2,'What came next?','tutor'),utterance(3,'I checked it.','user')]
    for selected in itertools.permutations(fs):
        records=decode_runtime_memory_context(block(selected))['records']
        groups=[[dict(record=r['id'],quote=r['text']) for r in records if r['speaker']==actor] for actor in ('tutor','user')]
        proposal=Proposal(dict(complete=True,targets=groups))
        result=bind_reporting_order('tutor',QUERY,block(selected),proposer=proposal)
        assert result.selected.source_position==0 and len(proposal.calls)==1
        assert result.alternatives[0][0].speaker=='tutor'
        assert bind_reporting_order('tutor',QUERY.replace('earlier','later'),block(selected),proposer=proposal).selected.source_position==1


@pytest.mark.parametrize('kind',['incomplete','bad_key','boolean_id','missing_quote','wrong_actor','unknown_actor',
    'duplicate','same_occurrence','missing_prefix','different_group'])
def test_invalid_ambiguous_and_unowned_semantic_citations_never_become_local_answers(kind):
    fs=[utterance(0,'What did you try?','tutor'),utterance(1,'I tried addition.','user')]
    value=dict(complete=True,targets=[[dict(record=0,quote=fs[0].content)],[dict(record=1,quote=fs[1].content)]])
    if kind=='incomplete':value['complete']=False
    elif kind=='bad_key':value['confidence']=1
    elif kind=='boolean_id':value['targets'][0][0]['record']=False
    elif kind=='missing_quote':value['targets'][0][0]['quote']='not in evidence'
    elif kind=='wrong_actor':value['targets'].reverse()
    elif kind=='unknown_actor':fs[0]=replace(fs[0],tags=tuple(t for t in fs[0].tags if not t.startswith('speaker:')))
    elif kind=='duplicate':value['targets'][0]*=2
    elif kind=='same_occurrence':value['targets'][1]=value['targets'][0]
    elif kind=='missing_prefix':fs=[utterance(3,fs[0].content,'tutor'),utterance(4,fs[1].content,'user')]
    elif kind=='different_group':fs[1]=replace(fs[1],tags=source(1,group='other').tags)
    result=bind_reporting_order('tutor',QUERY,block(fs),proposer=Proposal(value))
    assert result.selected is None and result.resolution.status=='unresolved'


def test_initial_cannot_allocate_a_set_proportional_to_an_untrusted_position():
    fs=[utterance(10**9,'What did you try?','tutor'),utterance(10**9+1,'I tried addition.','user')]
    value=dict(complete=True,targets=[[dict(record=0,quote=fs[0].content)],[dict(record=1,quote=fs[1].content)]])
    # Source positions are bounded upstream, but valid large values must remain cheap.
    assert bind_reporting_order('tutor',QUERY,block(fs),proposer=Proposal(value)).selected is None


def test_semantic_payload_excludes_source_positions_and_expected_labels():
    records=decode_runtime_memory_context(block([utterance(0,'A question?','tutor'),utterance(1,'A response.','user')]))['records']
    messages=proposal_messages('tutor',QUERY,('first question','first response'),records)
    payload=json.loads(messages[1].content)
    assert set(payload)=={'application','query','descriptions','records'}
    assert all(set(r)=={'id','speaker','text'} for r in payload['records'])


def test_proposer_transport_defers_instead_of_generating_a_fallback_answer():
    def unavailable(*args):raise ExecutionDeferred('local unavailable')
    unavailable.identity='unavailable-test'
    with pytest.raises(ExecutionDeferred):
        bind_reporting_order('tutor',QUERY,block([utterance(0,'A question?','tutor'),
            utterance(1,'A response.','user')]),proposer=unavailable)


def test_local_adapter_uses_shared_router_once_and_rejects_nonlocal_execution():
    seen=[]
    router=SimpleNamespace(execution=ExecutionPolicy(mode='local'),operation=lambda *a:nullcontext(),
        chat=lambda *a,**kw:seen.append((a,kw)) or SimpleNamespace(content='{"complete":false,"targets":[[],[]]}',finish_reason='stop'))
    proposer=LocalQueryProposer(router,model_identity='pinned-test-model')
    assert proposer('tutor',QUERY,('a','b'),[])['complete'] is False and len(seen)==1
    assert seen[0][1]['use_cache'] is False
    router.execution=ExecutionPolicy(mode='research')
    with pytest.raises(ValueError,match='strict local'):LocalQueryProposer(router,model_identity='model')


def test_compound_nonliteral_request_keeps_existing_response_path():
    assert bind_reporting_order('tutor',QUERY+' Also solve the exercise.','') is None


@pytest.mark.parametrize('ids',[[0,0],[True],[3],['0']])
def test_compact_proposer_rejects_invalid_ids_before_making_citations(ids):
    router=SimpleNamespace(execution=ExecutionPolicy(mode='local'),operation=lambda *a:nullcontext(),
        chat=lambda *a,**kw:SimpleNamespace(content=json.dumps(dict(complete=True,targets=[ids,[1]])),finish_reason='stop'))
    proposer=LocalQueryProposer(router,model_identity='test-only')
    records=[dict(id=0,speaker='tutor',text='A question?'),dict(id=1,speaker='user',text='A reply.')]
    with pytest.raises(ValueError):proposer('tutor',QUERY,('a','b'),records)


def test_compact_ids_expand_to_actual_utterances_without_model_repetition():
    router=SimpleNamespace(execution=ExecutionPolicy(mode='local'),operation=lambda *a:nullcontext(),
        chat=lambda *a,**kw:SimpleNamespace(content='{"complete":true,"targets":[[0],[1]]}',finish_reason='stop'))
    records=[dict(id=0,speaker='tutor',text='A question?'),dict(id=1,speaker='user',text='原来的回答。')]
    proposer=LocalQueryProposer(router,model_identity='test-only')
    value=proposer('tutor',QUERY,('a','b'),records)
    assert value['targets']==[[dict(record=0,quote='A question?')],[dict(record=1,quote='原来的回答。')]]
