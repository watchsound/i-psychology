"""Real request composition and memory selection; captures do not grade models."""
import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from psychology_base.applications.context import render_context_block
from psychology_base.applications.llm_responders import request_context_selection, responder_messages
from psychology_base.applications.reporting_order import bind_reporting_order, respond_with_reporting_order
from psychology_base.companion.reconstruction import ResearchSelection, ordered_selection
from psychology_base.companion.system import CompanionSystem
from psychology_base.llm.accounting import request_input_bound
from psychology_base.llm.execution import ExecutionDeferred
from tests.test_source_order_visibility import source, STAMP
from tests.test_turn_provenance import build_agent

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / 'benchmarks/fixtures/application_context/request-headroom-oracles-v1.zip'
with ZipFile(PACKET) as archive:
    INPUT_BYTES, ORACLE_BYTES = archive.read('inputs.json'), archive.read('references.json')
INPUTS = {r['id']:r for r in json.loads(INPUT_BYTES)}
ORACLES = json.loads(ORACLE_BYTES)


def test_portable_oracles_preserve_the_prospective_frozen_bytes():
    plan = json.loads((ROOT / 'docs/experiments/phase-9-memory-navigation/request-headroom-development-freeze-v1.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(INPUT_BYTES).hexdigest() == plan['input_sha256']
    assert hashlib.sha256(ORACLE_BYTES).hexdigest() == plan['references_sha256']


@pytest.mark.parametrize('oracle', ORACLES, ids=lambda r:r['id'])
def test_prospective_full_message_oracles_remain_identical(oracle):
    item = INPUTS[oracle['id']]
    assert [asdict(m) for m in responder_messages(oracle['prompt'],item['context'],item['query'])] == oracle['messages']


@pytest.mark.parametrize('version', [1,2,3,4,5,6,7])
@pytest.mark.parametrize('app', ['companion','tutor'])
def test_headroom_counts_full_unicode_framing_and_preserves_original_fragments(app,version):
    fs = [source(i,text=('中文 quote"\n\u2028 ' * 3) + str(i)) for i in range(8)]
    prompt, query = 'actual system policy 中文', '我之前提到过什么？'
    render = lambda selected:render_context_block(selected,assistant_speaker=app,format_version=version)
    bound = request_input_bound(responder_messages(prompt,render(fs[:2]),query))
    policy = request_context_selection(prompt,query,application=app,format_version=version,max_input=bound)
    selected, trace = ordered_selection(fs,{f.id:8-i for i,f in enumerate(fs)},6,policy)
    selected = [f for f,_ in selected]
    assert selected == fs[:2] and all(any(f is original for original in fs) for f in selected)
    assert request_input_bound(responder_messages(prompt,render(selected),query)) <= bound
    assert trace is not None and policy.max_bytes == len(render(selected).encode('utf-8'))


@pytest.mark.parametrize('spare', [0,1,19])
def test_empty_only_fit_does_not_charge_nonempty_framing_or_force_a_fragment(spare):
    prompt, query = 'policy','question'
    bound = request_input_bound(responder_messages(prompt,'',query)) + spare
    policy = request_context_selection(prompt,query,application='companion',format_version=7,max_input=bound)
    f = source(0)
    selected,_ = ordered_selection([f],{f.id:1},6,policy)
    assert selected == []
    assert request_input_bound(responder_messages(prompt,'',query)) <= bound


def test_no_memory_can_fit_when_prompt_itself_is_too_large():
    with pytest.raises(ExecutionDeferred,match='without memory'):
        request_context_selection('oversized policy','query',application='tutor',format_version=7,max_input=1)


@pytest.mark.parametrize('app',['companion','tutor'])
@pytest.mark.parametrize('version',[4,7])
def test_real_application_selects_once_before_reconstruction_and_delivery(app,version,monkeypatch):
    from psychology_base.applications.context import with_runtime_ambiguity_policy
    from psychology_base.i18n.prompts import response_language_directive
    system = CompanionSystem(byte_cap=None,top_k=9)
    for i in range(9): system._ingest_fragment(source(i,text='notebook ' + ('long evidence ' * 15) + str(i)))
    captured, recalls = [], []
    original = system.recall
    def track(*args,**kwargs):
        result=original(*args,**kwargs);recalls.append(result);return result
    monkeypatch.setattr(system,'recall',track)
    agent = build_agent(app,system,lambda *args:captured.append(args) or 'What would you try next?')
    agent.context_format_version=version
    query='notebook'
    prompt=with_runtime_ambiguity_policy(agent.system_prompt,assistant_speaker=app,context_format_version=version)+response_language_directive(query)
    # Leave room for a strict subset without referring to preferred answers.
    first=system.companion.store.all()[0]
    one=render_context_block([first],assistant_speaker=app,format_version=version)
    agent.max_request_input=request_input_bound(responder_messages(prompt,one,query))+40
    receipt=agent.respond(query,timestamp=STAMP)
    assert len(recalls)==len(captured)==1
    assert len(recalls[0].fragments_used)==1
    assert captured[0][1]==render_context_block(recalls[0].fragments_used,assistant_speaker=app,format_version=version)
    assert request_input_bound(responder_messages(*captured[0])) <= agent.max_request_input
    assert system.activation.recall_count==system.witness_state.activation.recall_count==1
    assert receipt.response_text=='What would you try next?'
    assert len(system.companion.store.all())==11


@pytest.mark.parametrize('app',['companion','tutor'])
def test_nonbinding_budget_matches_existing_observations_witness_and_response(app,monkeypatch):
    from psychology_base.applications.emotional import companion_agent
    from psychology_base.applications.learning import tutor_agent
    exchange='aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa'
    monkeypatch.setattr(companion_agent,'new_exchange_id',lambda:exchange)
    monkeypatch.setattr(tutor_agent,'new_exchange_id',lambda:exchange)
    original=CompanionSystem(byte_cap=None)
    for i in range(3): original._ingest_fragment(source(i,text='notebook '+str(i)))
    states,receipts,requests=[],[],[]
    for bound in (None,8000):
        system=copy.deepcopy(original)
        seen=[]
        agent=build_agent(app,system,lambda *args:seen.append(args) or 'What would you try next?')
        agent.max_request_input=bound
        receipts.append(agent.respond('notebook',timestamp=STAMP))
        requests.append(seen)
        states.append(copy.deepcopy((system.companion.store.all(),system.witness.store.all(),system.activation,
            system.witness_state.activation,system.edges,system.witness_edges)))
    assert states[0]==states[1] and requests[0]==requests[1] and receipts[0]==receipts[1]


@pytest.mark.parametrize('app',['companion','tutor'])
def test_unfit_base_prompt_defers_before_memory_or_provider_work(app):
    system=CompanionSystem(byte_cap=None)
    agent=build_agent(app,system,lambda *args:pytest.fail('No model dispatch'))
    agent.max_request_input=1
    with pytest.raises(ExecutionDeferred):agent.respond('ordinary question',timestamp=STAMP)
    assert system.companion.store.all()==[] and system.activation.recall_count==0 and agent._turn_index==0


@pytest.mark.parametrize('configuration',['hierarchical','research','profile'])
def test_unsupported_budget_configuration_fails_closed_before_state_changes(configuration):
    system=CompanionSystem(byte_cap=None,fragment_context_selection='scaled_focus' if configuration=='profile' else 'identity')
    policy=request_context_selection('policy','query',application='companion',format_version=4,max_input=8000,
        mode='occurrences' if configuration=='profile' else 'legacy')
    if configuration=='hierarchical': system.consolidation_enabled=True
    elif configuration=='research':system._research_selection=ResearchSelection('legacy',lambda fs:'')
    with pytest.raises(ExecutionDeferred):system.recall('query',selection_budget=policy,max_fragments=6)
    assert system.activation.recall_count==system.witness_state.activation.recall_count==0


@pytest.mark.parametrize('invalid',[True,0,-1,1.5,'8000'])
def test_invalid_limits_are_rejected_before_agent_work(invalid):
    agent=build_agent('companion',CompanionSystem(byte_cap=None),lambda *args:'')
    with pytest.raises(ValueError):replace(agent,max_request_input=invalid)
    with pytest.raises(ValueError):request_context_selection('p','q',application='companion',format_version=4,max_input=invalid)


def test_budget_pruning_cannot_manufacture_unique_reporting_targets():
    before=json.loads(Path('benchmarks/fixtures/application_context/request-budget-ambiguity-v1.json').read_text(encoding='utf-8'))
    assert before['original_binding']['resolution']['status']=='unresolved'
    assert before['pruned_binding']['resolution']['status']=='resolved'
    actual=bind_reporting_order('companion',before['query'],before['selected_context'],evidence_pruned=True)
    assert actual.selected is None and actual.resolution.reason=='selection_omitted_evidence'
    assert actual.request_identity!=bind_reporting_order('companion',before['query'],before['selected_context']).request_identity
    reply=respond_with_reporting_order(lambda *a:pytest.fail('No model can repair known omission here'),
        '',before['selected_context'],before['query'],application='companion',enabled=True,evidence_pruned=True)
    assert 'incomplete' in reply and 'red' not in reply and 'blue' not in reply


@pytest.mark.parametrize('app',['companion','tutor'])
def test_actual_budgeted_local_shortcut_preserves_uncertainty_after_omission(app):
    system=CompanionSystem(byte_cap=None)
    for i in range(4):system._ingest_fragment(source(i,text=('notebook red ' if i%2 else 'notebook blue ')+str(i)))
    query='Which phrase was mentioned earlier: red or blue?'
    agent=build_agent(app,system,lambda *a:pytest.fail('Recognized query stays local'))
    agent.local_reporting_order=True
    from psychology_base.applications.context import with_runtime_ambiguity_policy
    prompt=with_runtime_ambiguity_policy(agent.system_prompt,assistant_speaker=app,context_format_version=4)
    one=render_context_block([system.companion.store.all()[0]],assistant_speaker=app,format_version=4)
    agent.max_request_input=request_input_bound(responder_messages(prompt,one,query))+10
    receipt=agent.respond(query,timestamp=STAMP)
    assert 'incomplete' in receipt.response_text and system.activation.recall_count==1


def test_pruning_flag_preserves_nonordering_delegation_without_calling_a_proposer():
    seen=[]
    reply=respond_with_reporting_order(lambda *a:seen.append(a) or 'captured response','policy','context',
        'Help me understand this exercise.',application='tutor',enabled=True,evidence_pruned=True)
    assert reply=='captured response' and seen==[('policy','context','Help me understand this exercise.')]


def test_omission_blocks_an_expensive_semantic_request():
    def forbidden(*a):pytest.fail('Do not run a model when omission already prevents a reliable answer')
    forbidden.identity='omission-test'
    query="Which came earlier in the conversation: the teacher's opening question or the learner's first response?"
    result=bind_reporting_order('tutor',query,'',proposer=forbidden,evidence_pruned=True)
    assert result.resolution.reason=='selection_omitted_evidence'


@pytest.mark.parametrize('app',['companion','tutor'])
@pytest.mark.parametrize('bound',[None,8000])
def test_reporting_request_preserves_repeated_original_occurrences(app,bound):
    system=CompanionSystem(byte_cap=None)
    for position,text in enumerate(('red notebook','blue notebook','red notebook')):
        system._ingest_fragment(source(position,text=text))
    recalled=[]
    original=system.recall
    def capture(*args,**kwargs):
        value=original(*args,**kwargs);recalled.append(value);return value
    system.recall=capture
    agent=build_agent(app,system,lambda *args:pytest.fail('Ambiguous reporting stays local'))
    agent.local_reporting_order=True;agent.max_request_input=bound
    receipt=agent.respond('Which phrase was reported earlier: red notebook or blue notebook?',timestamp=STAMP)
    assert len(recalled)==1 and len(recalled[0].fragments_used)==3
    assert recalled[0].metadata['ordered_selection']['mode']=='occurrences'
    assert 'does not establish' in receipt.response_text


def test_request_budget_does_not_install_an_unqualified_research_selection_mode():
    with pytest.raises(ValueError,match='legacy or occurrence'):
        request_context_selection('p','q',application='companion',format_version=4,max_input=8000,mode='utility_spans')
