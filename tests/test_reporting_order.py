"""Declared query grammar, real provenance and real application state; no inference."""
import copy
from dataclasses import replace
import itertools
import json
from pathlib import Path

import pytest

from psychology_base.applications.context import render_context_block
from psychology_base.applications.reporting_order import resolve_reporting_order, respond_with_reporting_order
from psychology_base.applications.safety.crisis_escalation import CrisisEscalation
from psychology_base.companion import turn_provenance as provenance
from psychology_base.companion.system import CompanionSystem
from tests.test_source_order_visibility import source, STAMP
from tests.test_turn_provenance import build_agent


CONTROLS = json.loads(Path('benchmarks/fixtures/application_context/reporting-order-controls-v1.json').read_text(encoding='utf-8'))
QUERY = CONTROLS['recognized'][0]['query']


def block(fs, version=7):
    return render_context_block(fs, assistant_speaker='companion', format_version=version)


def forbidden(*args):
    pytest.fail('A bounded local reporting decision cannot call a model')


@pytest.mark.parametrize('case', CONTROLS['recognized'])
@pytest.mark.parametrize('version', [3, 4, 5, 6, 7])
def test_frozen_questions_bind_original_targets_in_both_languages(case, version):
    first, second = case['quotes']
    fs = [source(8, text=second), source(2, text=first)]
    result = resolve_reporting_order('companion', case['query'], block(fs, version))
    assert result.language == case['language'] and result.polarity == case['polarity']
    if version == 3:
        assert result.status == 'unresolved' and result.reason == 'unresolved_order'
    else:
        assert result.status == 'resolved'
        assert result.query_target == (1 if case['polarity'] == 'earlier' else 2)


@pytest.mark.parametrize('query', CONTROLS['unsupported'])
def test_unrecognized_queries_keep_the_existing_responder_and_exact_arguments(query):
    captured = []
    def responder(*args):
        captured.append(args)
        return 'existing application response'
    assert resolve_reporting_order('tutor', query, '') is None
    assert respond_with_reporting_order(responder, 'custom policy', 'original context', query,
        application='tutor', enabled=True) == 'existing application response'
    assert captured == [('custom policy', 'original context', query)]


@pytest.mark.parametrize('version', [4, 5, 6, 7])
def test_all_mixed_source_permutations_keep_the_query_specific_answer(version):
    fs = [source(8, text='The blue notebook.'), source(2, text='The red notebook.'),
          source(0, group='other', text='The unrelated event happened in 1991.')]
    for selected in itertools.permutations(fs):
        result = resolve_reporting_order('companion', QUERY, block(selected, version))
        assert result.status == 'resolved' and result.query_target == 1
        reply = respond_with_reporting_order(forbidden, '', block(selected, version), QUERY,
            application='companion', enabled=True)
        assert 'first quoted statement was reported earlier' in reply
        assert 'notebook' not in reply and '1991' not in reply


@pytest.mark.parametrize('kind', ['missing', 'substring', 'duplicate_text', 'same_target', 'different_group',
    'missing_coordinate', 'conflict', 'conflict_in_third_turn', 'owner', 'perspective', 'invalid_context'])
def test_ambiguous_or_incomparable_evidence_never_acquires_order(kind):
    a, b = source(2, text='The red notebook.'), source(8, text='The blue notebook.')
    fs, query = [a, b], QUERY
    if kind == 'missing': fs = [b]
    elif kind == 'substring': a = source(2, text='I mentioned The red notebook. yesterday.'); fs = [a,b]
    elif kind == 'duplicate_text': fs += [source(0, group='other', text=a.content)]
    elif kind == 'same_target': query = QUERY.replace('The blue notebook.', 'The red notebook.')
    elif kind == 'different_group': fs[1] = source(8, group='other', text=b.content)
    elif kind == 'missing_coordinate': fs[1] = replace(b, parent_ids=('derived',))
    elif kind == 'conflict': fs[1] = source(2, text=b.content)
    elif kind == 'conflict_in_third_turn': fs += [source(2, text='Conflicting position for a third statement.')]
    elif kind == 'owner': fs[1] = source(8, user='bob', text=b.content)
    elif kind == 'perspective': fs[1] = source(8, extra=('subject:bob',), text=b.content)
    wire = 'invalid context' if kind == 'invalid_context' else block(fs)
    result = resolve_reporting_order('companion', query, wire)
    assert result.status == 'unresolved' and result.query_target is None
    reply = respond_with_reporting_order(forbidden, '', wire, query, application='companion', enabled=True)
    assert 'does not establish' in reply and '?' in reply
    assert 'never happened' not in reply and 'reported earlier' not in reply


def test_duplicate_positions_in_unrelated_group_do_not_invalidate_target_group():
    fs = [source(8, text='The blue notebook.'), source(2, text='The red notebook.'),
          source(0, group='other'), source(0, group='other', text='conflict elsewhere')]
    assert resolve_reporting_order('tutor', QUERY, block(fs)).query_target == 1


def test_reported_sequence_is_not_event_chronology_or_speaker_ownership():
    query = 'Which was reported earlier in the source dialogue: “The event was in 2025.” or “The event was in 2019.”?'
    fs = [source(8, text='The event was in 2019.'), source(2, text='The event was in 2025.')]
    reply = respond_with_reporting_order(forbidden, '', block(fs), query, application='tutor', enabled=True)
    assert 'first quoted statement was reported earlier' in reply
    assert '2025' not in reply and '2019' not in reply and 'you said' not in reply
    assert resolve_reporting_order('tutor', query.replace('was reported', 'happened'), block(fs)) is None


@pytest.mark.parametrize('app', ['companion', 'tutor'])
def test_default_preserves_existing_responder_and_opt_in_is_strict(app):
    args = ('custom prompt', block([source(2)]), QUERY)
    seen = []
    def responder(*actual): seen.append(actual); return 'unchanged'
    assert respond_with_reporting_order(responder, *args, application=app) == 'unchanged'
    assert seen == [args]
    for invalid in (1, None, 'yes'):
        with pytest.raises(ValueError, match='Boolean'):
            respond_with_reporting_order(forbidden, *args, application=app, enabled=invalid)


@pytest.mark.parametrize('app', ['companion', 'tutor'])
def test_actual_app_memory_and_telemetry_are_preserved_with_identical_reply(app, monkeypatch):
    from psychology_base.applications.emotional import companion_agent
    from psychology_base.applications.learning import tutor_agent
    exchange = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa'
    monkeypatch.setattr(companion_agent, 'new_exchange_id', lambda: exchange)
    monkeypatch.setattr(tutor_agent, 'new_exchange_id', lambda: exchange)
    initial = CompanionSystem(byte_cap=None)
    for f in [source(2, text='The red notebook.'), source(8, text='The blue notebook.')]:
        initial._ingest_fragment(f)
    expected = 'In the records available here, the first quoted statement was reported earlier.'
    states, receipts = [], []
    for enabled, version in ((False, 4), (True, 4), (True, 7)):
        system = copy.deepcopy(initial)
        agent = build_agent(app, system, forbidden if enabled else lambda *args: expected)
        assert agent.local_reporting_order is False and agent.context_format_version == 4
        agent.local_reporting_order, agent.context_format_version = enabled, version
        receipt = agent.respond(QUERY, timestamp=STAMP)
        assert receipt.response_text == expected and receipt.exchange_id == exchange
        observed = [f for f in system.companion.store.all() if provenance.parse_turn(f.tags).descriptor.kind == 'exchange']
        assert len(observed) == 2
        assert {f.content for f in observed} == {QUERY, expected}
        states.append(copy.deepcopy((system.companion.store.all(), system.witness.store.all(),
            system.activation, system.witness_state.activation, system.edges, system.witness_edges)))
        receipts.append(receipt)
    assert states[0] == states[1] == states[2]
    assert receipts[0] == receipts[1] == receipts[2]


@pytest.mark.parametrize('app', ['companion', 'tutor'])
def test_constructor_rejects_non_boolean_opt_in_before_state_changes(app):
    system = CompanionSystem(byte_cap=None)
    agent = build_agent(app, system, forbidden)
    for invalid in (1, None, 'true'):
        with pytest.raises(ValueError, match='Boolean'):
            replace(agent, local_reporting_order=invalid)
    assert system.companion.store.all() == []


@pytest.mark.parametrize('app', ['companion', 'tutor'])
def test_unrecognized_request_retains_existing_deferral_semantics(app):
    from psychology_base.llm.execution import ExecutionDeferred
    def deferred(*args): raise ExecutionDeferred('local model unavailable')
    system = CompanionSystem(byte_cap=None)
    agent = build_agent(app, system, deferred)
    agent.local_reporting_order = True
    with pytest.raises(ExecutionDeferred): agent.respond('Help me with my plan.', timestamp=STAMP)
    stored = system.companion.store.all()
    assert len(stored) == 1 and stored[0].content == 'Help me with my plan.'
    assert agent._turn_index == 1


def test_null_actor_does_not_become_first_person_history():
    fs = [replace(source(p, text=text), tags=tuple(t for t in source(p).tags if t != 'speaker:user'))
          for p, text in [(2, 'The red notebook.'), (8, 'The blue notebook.')]]
    resolution = resolve_reporting_order('companion', QUERY, block(fs))
    # Reliable source positions can order turns without identifying their actors.
    assert resolution.status == 'resolved' and resolution.query_target == 1
    from psychology_base.applications.measurement.memory_grounding import runtime_memory_payload
    records = runtime_memory_payload('companion', QUERY, block(fs), '', contract_version=3)['VISIBLE_CONTEXT']['records']
    assert all(r['speaker'] is None for r in records)
    reply = respond_with_reporting_order(forbidden, '', block(fs), QUERY, application='companion', enabled=True)
    assert 'first quoted statement' in reply and 'you said' not in reply and 'I said' not in reply


def test_crisis_gate_precedes_local_order_resolution():
    system = CompanionSystem(byte_cap=None)
    agent = build_agent('companion', system, forbidden)
    agent.local_reporting_order = True
    incidents = []
    agent.crisis_escalation = CrisisEscalation(handoff_sink=incidents.append)
    query = 'Which was said earlier in the source dialogue: “I want to kill myself” or “help”?'
    receipt = agent.respond(query, timestamp=STAMP)
    assert receipt.handoff_triggered and receipt.exchange_id is None and len(incidents) == 1
    assert system.companion.store.all() == []


def test_minor_gate_precedes_local_order_resolution():
    from psychology_base.applications.safety import AgeAttestation, ApplicationSurface, MinorExclusionGate
    system = CompanionSystem(byte_cap=None)
    agent = build_agent('companion', system, forbidden)
    agent.local_reporting_order = True
    agent.minor_exclusion_gate = MinorExclusionGate(surface=ApplicationSurface.EMOTIONAL_COMPANION)
    attestation = AgeAttestation(age_years=12, attested_at=STAMP.timestamp(), attestation_source='self_declared')
    with pytest.raises(PermissionError, match='under 14'):
        agent.respond(QUERY, timestamp=STAMP, attestation=attestation)
    assert system.companion.store.all() == []


@pytest.mark.parametrize('app', ['companion', 'tutor'])
def test_local_response_still_advances_identity_reminder(app):
    from datetime import timedelta
    from psychology_base.applications.safety import AIIdentityReminder
    agent = build_agent(app, CompanionSystem(byte_cap=None), forbidden)
    agent.local_reporting_order = True
    agent.ai_identity_reminder = AIIdentityReminder(reminder_interval_seconds=1, idle_threshold_seconds=10)
    first = agent.respond(QUERY, timestamp=STAMP)
    second = agent.respond(QUERY, timestamp=STAMP + timedelta(seconds=2))
    assert not first.reminder_fired and second.reminder_fired
    assert '?' in first.response_text and '?' in second.response_text


def test_tutor_order_answer_never_repeats_a_quoted_math_solution():
    query = 'Which was said earlier in the source dialogue: “The answer is 12345.” or “The answer is 12346.”?'
    fs = [source(8, text='The answer is 12346.'), source(2, text='The answer is 12345.')]
    reply = respond_with_reporting_order(forbidden, '', block(fs), query, application='tutor', enabled=True)
    assert 'first quoted statement' in reply and '12345' not in reply and '12346' not in reply


@pytest.mark.parametrize('keyword', ['earlİer', 'earlıer', 'earlſer'])
def test_unicode_keyword_variants_cannot_choose_the_opposite_direction(keyword):
    query = QUERY.replace('earlier', keyword)
    assert resolve_reporting_order('companion', query,
        block([source(2, text='The red notebook.'), source(8, text='The blue notebook.')])) is None


def test_ascii_case_matching_preserves_unicode_quotation_contents_and_partial_scope():
    query = 'WHICH was said EARLIER in the source dialogue: “红色笔记本。” or “蓝色笔记本。”?'
    fs = [source(2, text='红色笔记本。'), source(8, text='蓝色笔记本。')]
    result = resolve_reporting_order('companion', query, block(fs))
    assert result.status == 'resolved' and result.query_target == 1
    reply = respond_with_reporting_order(forbidden, '', block(fs), query, application='companion', enabled=True)
    assert reply.startswith('In the records available here,') and 'first quoted statement was reported earlier' in reply
    assert 'first ever' not in reply


def test_saved_historical_queries_retain_existing_response_path():
    saved = Path('.data/application_context/episode-context-v2/inputs.json')
    if not saved.exists():
        pytest.skip('.data/ corpus is gitignored; regenerate locally to exercise this guard')
    inputs = json.loads(saved.read_text(encoding='utf-8'))
    assert len(inputs) == 32
    for item in inputs:
        assert resolve_reporting_order(item['app'], item['query'], '') is None

