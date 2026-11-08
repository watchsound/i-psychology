"""Authored offline completions qualify reuse/accounting, never model semantics."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
import hashlib
import json

import pytest

from psychology_base.applications.llm_responders import make_llm_responder
from psychology_base.applications.measurement.local_query_binding import (
    LocalChatMembershipAssessor, INTERPRETATION_PROMPT, MembershipOutputError,
)
from psychology_base.applications.reporting_order import ReportingMembershipSession
from psychology_base.companion.system import CompanionSystem
from psychology_base.llm.cache import LLMCache
from psychology_base.llm.config import CacheConfig, LLMConfig, ProviderConfig
from psychology_base.llm.execution import ConversationLimits, ExecutionDeferred, ExecutionPolicy, InferenceLimit
from psychology_base.llm.providers.ollama import OllamaProvider
from psychology_base.llm.router import LLMRouter
from psychology_base.llm.types import Completion, TaskTier
from tests.test_source_order_visibility import source
from tests.test_turn_provenance import build_agent
from tests.test_conversation_budget import no_network
from tests.test_interpretation_contract import CASES


QUERY = 'Which came earlier in the conversation: the first question or the first answer?'
SWAPPED = 'Which came earlier in the conversation: the first answer or the first question?'
LIMITS = ConversationLimits(auxiliary=InferenceLimit(8, 65536))


def completions():
    return Completion('software fixture', 'ollama', 'authored-test', TaskTier.ADVANCED,
                      prompt_tokens=10, completion_tokens=20, finish_reason='stop', metadata={})


def response_text(response):
    return response.text if hasattr(response,'text') else response.response_text


def setup(tmp_path, monkeypatch, *, application='tutor', user='alice', cache=None):
    """Real router and admission; only model transport/metadata are prescribed."""
    provider = OllamaProvider(simple_model='authored-test', advanced_model='authored-test',
        local_only=True, expected_digest='a'*64, context_length=8192, seed=0, json_mode=True)
    cfg = LLMConfig('ollama', providers={'ollama': ProviderConfig('ollama', 'authored-test', 'authored-test')},
                    cache=CacheConfig(False), execution=ExecutionPolicy())
    router = LLMRouter(cfg, providers={'ollama': provider}, user_id=user)
    state = dict(calls=[], metadata=[], failure=None, analysis_factory=None, conditional_factory=None)
    monkeypatch.setattr(provider, 'require_local_model', lambda *a, **kw: state['metadata'].append((a, kw)))

    def chat(messages, *, tier, **kwargs):
        stage = ('source' if messages[0].content == INTERPRETATION_PROMPT else
                 'conditional' if messages[0].content.startswith('Judge the focus utterance') else 'response')
        state['calls'].append(stage)
        if state['failure'] == stage + '-timeout':
            raise TimeoutError('prescribed unavailable transport')
        if stage == 'source':
            payload = json.loads(messages[1].content)
            factory = state['analysis_factory']
            value = factory(payload) if factory else dict(coverage='enumerated', readings=[dict(bindings=[], events=[])])
        elif stage == 'conditional':
            payload = json.loads(messages[1].content)
            factory = state['conditional_factory']
            text = payload['source']['turns'][payload['source']['focus_turn']]['text']
            labels = [('match' if ('question' in description.lower() or '问题' in description) == text.endswith(('?', '？')) else 'nonmatch')
                      for description in payload['descriptions']]
            value = factory(payload) if factory else dict(labels=[labels]*len(payload['interpretation']['readings']), open_labels=None)
        else:
            value = None
        content = 'What small step would you try?' if value is None else json.dumps(value, ensure_ascii=False)
        if state['failure'] == stage + '-malformed':
            content = '{}'
        metadata = dict(local_only=True, done=True, response_model='authored-test',
            model_attestation=dict(digest=provider.expected_digest), context_length=8192, seed=0, json_mode=True)
        if state['failure'] == stage + '-attestation':
            metadata['model_attestation']['digest'] = 'f'*64
        return replace(completions(), content=content, tier=tier, metadata=metadata)

    monkeypatch.setattr(provider, 'chat', chat)
    cache = cache or LLMCache(tmp_path/'cache', derived_max_entries=32, derived_max_bytes=262144)
    candidate = LocalChatMembershipAssessor(router, assessment_kind='interpreted-membership-v2',
        format_version=4, max_input=7168, max_tokens=1024, reuse_cache=cache, reuse_scope=(application, user))
    return candidate, router, state, cache


def fragments(*, group='dialogue', session='one', texts=('A question?', 'An answer.')):
    return [source(i, group=group, session=session, text=text) for i, text in enumerate(texts)]


def assess(candidate, fs=None, query=QUERY, *, limits=LIMITS):
    session = ReportingMembershipSession(candidate.reuse_scope[0], query, candidate, user_id=candidate.reuse_scope[1])
    with candidate.router.conversation_turn(limits=limits) as budget:
        result = session.required_occurrences(fs if fs is not None else fragments())
    return result, budget.snapshot(), session


def test_two_stage_reuse_avoids_repeat_calls_but_new_descriptions_are_rejudged(tmp_path, monkeypatch):
    candidate, router, state, cache = setup(tmp_path, monkeypatch)
    cold, first, _ = assess(candidate)
    assert state['calls'] == ['source', 'conditional', 'source', 'conditional']
    assert first['usage']['auxiliary']['attempts'] == 4
    assert candidate.last_reuse['planned_calls'] == 4
    warm, second, _ = assess(candidate, limits=ConversationLimits())
    assert cold == warm and len(state['calls']) == 4
    assert candidate.last_calls == []
    assert candidate.last_reuse['source_hits'] == candidate.last_reuse['conditional_hits'] == 2
    assert second['usage']['auxiliary']['attempts'] == 0 and second['usage']['auxiliary']['cache_hits'] == 4
    assess(candidate, query=SWAPPED)
    assert state['calls'][4:] == ['conditional', 'conditional']
    assert candidate.last_reuse['source_hits'] == 2 and candidate.last_reuse['planned_calls'] == 2


@pytest.mark.parametrize('limits', [ConversationLimits(), ConversationLimits(auxiliary=InferenceLimit(3, 65536)),
                                   ConversationLimits(auxiliary=InferenceLimit(8, 100))])
def test_oversized_complete_pool_is_refused_before_any_metadata_or_inference(tmp_path, monkeypatch, limits):
    candidate, router, state, cache = setup(tmp_path, monkeypatch)
    with router.conversation_turn(limits=limits) as budget:
        with pytest.raises(ExecutionDeferred, match='plan'):
            ReportingMembershipSession('tutor', QUERY, candidate, user_id='alice').required_occurrences(fragments())
        assert not state['calls'] and not state['metadata']
        assert budget.snapshot()['usage']['auxiliary']['attempts'] == 0
        assert make_llm_responder(router)('Policy.', '', 'Hello')
        assert budget.snapshot()['usage']['response']['attempts'] == 1
    assert cache.derived_snapshot()['entries'] == 0


@pytest.mark.parametrize('change', ['text', 'prior_text', 'role', 'context_scope', 'occurrence', 'session', 'source_group'])
def test_changed_dependencies_invalidate_without_aliasing_prior_cached_work(tmp_path, monkeypatch, change):
    candidate, router, state, cache = setup(tmp_path, monkeypatch)
    assess(candidate)
    original = len(state['calls'])
    if change in ('text', 'prior_text'):
        fs = fragments(texts=('A different question?', 'An answer.') if change == 'prior_text' else ('A question?', 'A different answer.'))
    elif change == 'session':
        fs = fragments(session='another')
    elif change == 'source_group':
        fs = fragments(group='another')
    elif change == 'occurrence':
        fs = [replace(f, id='different-' + f.id) for f in fragments()]
    elif change == 'role':
        fs = [replace(f, tags=tuple('speaker:tutor' if tag == 'speaker:user' else tag for tag in f.tags)) for f in fragments()]
    else:
        # Explicit direct-call control: source context policy is a dependency.
        original_call=candidate._assess_reusing
        def changed(requests, hypotheses, labels):
            requests=deepcopy(requests)
            from psychology_base.applications.measurement.local_query_binding import reference_messages
            for i,(active,messages,focus) in enumerate(requests):
                focus['context_scope']+=' Revised local context contract.'
                requests[i]=(active,reference_messages(focus,interpretation=True),focus)
            return original_call(requests,hypotheses,labels)
        monkeypatch.setattr(candidate,'_assess_reusing',changed)
        fs=fragments()
    assess(candidate, fs)
    assert len(state['calls']) > original
    if change == 'text':
        assert state['calls'][original:] == ['source', 'conditional']


def test_growing_pool_reuses_old_occurrences_but_assesses_the_complete_new_frontier(tmp_path, monkeypatch):
    candidate, router, state, cache = setup(tmp_path, monkeypatch)
    assess(candidate)
    _, budget, _ = assess(candidate, fragments(texts=('A question?', 'An answer.', 'Another answer.')),
                          limits=ConversationLimits(auxiliary=InferenceLimit(2, 20000)))
    assert len(state['calls']) == 6
    assert candidate.last_reuse['source_hits'] == 2 and candidate.last_reuse['conditional_hits'] == 2
    assert budget['usage']['auxiliary']['attempts'] == 2
    # More than the allowance never produces a proof over a clipped prefix.
    with pytest.raises(ExecutionDeferred, match='plan'):
        assess(candidate, fragments(texts=('A question?', 'An answer.', 'Another answer.', 'Fourth.', 'Fifth.')),
               limits=ConversationLimits(auxiliary=InferenceLimit(2, 20000)))
    assert len(state['calls']) == 6


def test_full_32_record_cold_pool_does_not_start_64_generations(tmp_path,monkeypatch):
    candidate,router,state,cache=setup(tmp_path,monkeypatch)
    fs=fragments(texts=tuple(f'Question {i}?' for i in range(32)))
    with pytest.raises(ExecutionDeferred,match='plan'):assess(candidate,fs)
    assert candidate.last_reuse['planned_calls']==64
    assert not state['calls'] and not state['metadata']


@pytest.mark.parametrize('change', ['user', 'application', 'model', 'revision'])
def test_shared_cache_never_crosses_user_application_model_or_protocol_identity(tmp_path, monkeypatch, change):
    candidate, router, state, cache = setup(tmp_path, monkeypatch)
    assess(candidate)
    other, other_router, other_state, _ = setup(tmp_path, monkeypatch, cache=cache,
        application='companion' if change == 'application' else 'tutor', user='bob' if change == 'user' else 'alice')
    if change == 'model':
        other.provider.expected_digest = 'b'*64
    if change == 'revision':
        from psychology_base.applications.measurement import reference_readings
        monkeypatch.setattr(reference_readings, 'INTERPRETATION_VALIDATION_REVISION', 'authored-new-revision')
    if change in ('model', 'revision'):
        with pytest.raises(ValueError, match='configuration'):
            assess(other)
        other=LocalChatMembershipAssessor(other_router, assessment_kind='interpreted-membership-v2',
            format_version=4,max_input=7168,max_tokens=1024,reuse_cache=cache,reuse_scope=('tutor','alice'))
    assess(other)
    assert len(other_state['calls']) == 4
    assert other.last_reuse['source_hits'] == 0


def test_wrong_application_user_and_missing_provenance_are_refused_before_cache_or_inference(tmp_path, monkeypatch):
    candidate, router, state, cache = setup(tmp_path, monkeypatch)
    with router.conversation_turn(limits=LIMITS):
        for application,user in [('tutor',None),('tutor','bob'),('companion','alice')]:
            with pytest.raises(ValueError, match='another application or user'):
                ReportingMembershipSession(application,QUERY,candidate,user_id=user).required_occurrences(fragments())
    from tests.test_interpretation_local_integration import rows
    from tests.test_interpretation_contract import software_example
    focus,_=software_example()
    with router.conversation_turn(limits=LIMITS):
        with pytest.raises(ValueError, match='provenance'):
            candidate(rows(focus))
    assert not state['calls']


@pytest.mark.parametrize('failure', ['source-timeout','source-malformed','source-attestation',
                                    'conditional-timeout','conditional-malformed','conditional-attestation'])
def test_failed_results_are_not_cached_and_same_session_never_retries(tmp_path, monkeypatch, failure):
    candidate, router, state, cache = setup(tmp_path, monkeypatch)
    state['failure']=failure
    session=ReportingMembershipSession('tutor',QUERY,candidate,user_id='alice')
    with router.conversation_turn(limits=LIMITS):
        with pytest.raises((ExecutionDeferred,ValueError)):
            session.required_occurrences(fragments())
        calls=len(state['calls'])
        with pytest.raises(ExecutionDeferred,match='not retried'):
            session.required_occurrences(fragments())
        assert len(state['calls'])==calls
    assert cache.derived_snapshot()['entries']==int(failure.startswith('conditional'))
    state['failure']=None
    assess(candidate)
    assert candidate.last_reuse['source_hits']==int(failure.startswith('conditional'))


def test_reuse_requires_active_turn_and_configuration_stays_pinned(tmp_path, monkeypatch):
    candidate, router, state, cache=setup(tmp_path,monkeypatch)
    with pytest.raises(ExecutionDeferred,match='whole-turn'):
        ReportingMembershipSession('tutor',QUERY,candidate,user_id='alice').required_occurrences(fragments())
    candidate.reuse_scope=('tutor','bob')
    with pytest.raises(ValueError,match='configuration'):
        assess(candidate)
    assert not state['calls']


@pytest.mark.parametrize('application',['tutor','companion'])
@pytest.mark.parametrize('version',[4,7])
@pytest.mark.parametrize('language',['en','zh'])
def test_actual_agents_reuse_semantics_but_recompute_earlier_later_binding(application,version,language,tmp_path,monkeypatch):
    candidate,router,state,cache=setup(tmp_path,monkeypatch,application=application)
    system=CompanionSystem(byte_cap=None,top_k=6)
    texts=('A question?','An answer.') if language=='en' else ('一个问题？','一个回答。')
    query=QUERY if language=='en' else '原对话中，第一个问题和第一个回答，哪一句出现得较早？'
    later=query.replace('earlier','later') if language=='en' else query.replace('较早','较晚')
    for fragment in fragments(texts=texts):system._ingest_fragment(fragment)
    agent=build_agent(application,system,make_llm_responder(router))
    agent.local_reporting_order=True;agent.max_request_input=8000;agent.context_format_version=version
    agent.reporting_assessor=candidate
    # Isolate serving/reconstruction from history growth in this comparison.
    original=deepcopy(system)
    with router.conversation_turn(limits=LIMITS):
        first=agent.respond(query)
    assert state['calls']==['source','conditional','source','conditional']
    agent.companion_system=original
    with router.conversation_turn() as budget:
        second=agent.respond(later)
    assert len(state['calls'])==4 and budget.snapshot()['usage']['auxiliary']['attempts']==0
    first_text=response_text(first)
    second_text=response_text(second)
    if language=='en':assert 'earlier' in first_text and 'later' in second_text
    assert first_text != second_text


@pytest.mark.parametrize('case', CASES, ids=lambda case:case['id'])
def test_reused_bilingual_readings_preserve_correlated_alternatives_negation_and_unknowns(case,tmp_path,monkeypatch):
    from tests.test_interpretation_contract import software_example, BANK
    from tests.test_interpretation_local_integration import rows
    focus,raw=software_example(case)
    reference=BANK['membership_references'][case['id']]
    candidate,router,state,cache=setup(tmp_path,monkeypatch)
    state['analysis_factory']=lambda payload:dict(coverage=raw['coverage'],readings=raw['readings'])
    state['conditional_factory']=lambda payload:dict(labels=reference['labels'],open_labels=reference['open_labels'])
    turns=[*focus['prior_utterances'],focus['focus_utterance']]
    provenance=[dict(occurrence_id=f"{case['id']}/{i}",record=dict(speaker=turn['speaker'],
        session='authored-reference',text=turn['text'],source_order=None)) for i,turn in enumerate(turns)]
    inputs=rows(focus)
    for i,row in enumerate(inputs):
        row['pair']=(row['pair'][0],reference['descriptions'][i])
        row['reuse_provenance']=dict(focus=provenance[-1],prior=provenance[:-1])
    before=deepcopy(inputs)
    with router.conversation_turn(limits=LIMITS):assert candidate(inputs)==reference['reduced']
    with router.conversation_turn() as budget:assert candidate(inputs)==reference['reduced']
    assert inputs==before and len(state['calls'])==2
    assert budget.snapshot()['usage']['auxiliary']['attempts']==0


def test_unsupported_interpretations_never_become_cached_certainty(tmp_path,monkeypatch):
    candidate,router,state,cache=setup(tmp_path,monkeypatch)
    state['analysis_factory']=lambda payload:dict(coverage='unsupported',readings=[dict(bindings=[],events=[])])
    with pytest.raises(ExecutionDeferred,match='does not support'):assess(candidate)
    assert cache.derived_snapshot()['entries']==0 and state['calls']==['source']


def test_invalid_cached_attestation_is_discarded_and_recomputed_only_with_allowance(tmp_path,monkeypatch):
    candidate,router,state,cache=setup(tmp_path,monkeypatch)
    assess(candidate)
    # Deliberate corruption injection into this owner's public completion cache.
    keys=list(cache._derived)
    for key in keys:
        value=cache.get_derived(key)
        value.metadata['model_attestation']['digest']='f'*64
        cache.put_derived(key,value)
    with pytest.raises(ExecutionDeferred,match='plan'):assess(candidate,limits=ConversationLimits())
    assert len(state['calls'])==4 and candidate.last_reuse['invalidations']==2
    assess(candidate)
    assert len(state['calls'])==8


def test_copied_closed_turn_cannot_publish_reused_results(tmp_path,monkeypatch):
    from contextvars import copy_context
    candidate,router,state,cache=setup(tmp_path,monkeypatch)
    assess(candidate)
    with router.conversation_turn():context=copy_context()
    with pytest.raises(ExecutionDeferred,match='plan'):
        context.run(assess,candidate,limits=ConversationLimits())
    assert len(state['calls'])==4


@pytest.mark.parametrize('capacity',[(1,262144),(32,128)])
def test_eviction_and_oversized_entries_preserve_results_and_do_not_bypass_budget(capacity,tmp_path,monkeypatch):
    cache=LLMCache(tmp_path/'cache',derived_max_entries=capacity[0],derived_max_bytes=capacity[1])
    candidate,router,state,_=setup(tmp_path,monkeypatch,cache=cache)
    first,_,_=assess(candidate)
    second,_,_=assess(candidate)
    assert first==second and len(state['calls'])==8
    assert cache.derived_snapshot()['entries']<=capacity[0] and cache.derived_snapshot()['bytes']<=capacity[1]
    with pytest.raises(ExecutionDeferred,match='plan'):assess(candidate,limits=ConversationLimits())
    assert len(state['calls'])==8


def test_actual_session_logs_deferred_input_without_fabricated_reply(tmp_path,monkeypatch):
    import io
    from psychology_base.applications.pilot.session_runner import run_session,SessionLogWriter
    candidate,router,state,cache=setup(tmp_path,monkeypatch)
    system=CompanionSystem(byte_cap=None,top_k=6)
    for fragment in fragments():system._ingest_fragment(fragment)
    agent=build_agent('tutor',system,make_llm_responder(router))
    agent.local_reporting_order=True;agent.max_request_input=8000;agent.reporting_assessor=candidate
    path=tmp_path/'session.jsonl'
    with SessionLogWriter(path) as log:
        turns=run_session(user_id='alice',session_idx=1,
            agent_fn=lambda text:(response_text(agent.respond(text)),{}),log_writer=log,
            on_export=lambda path:0,input_stream=io.StringIO(QUERY+'\n/quit\n'),output_stream=io.StringIO())
    events=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    assert turns==0 and any(row['event']=='user_turn' and row['text']==QUERY for row in events)
    assert sum(row['event']=='turn_deferred' for row in events)==1
    assert not any(row['event']=='agent_turn' for row in events)
    assert not state['calls'] and len(system.companion.store.all())==2


@pytest.mark.parametrize('limits',[(-1,1),(True,1),(1,0),(0,1),(1,1.5)])
def test_cache_limits_are_explicit_and_strict(tmp_path,limits):
    with pytest.raises(ValueError):LLMCache(tmp_path/'cache',derived_max_entries=limits[0],derived_max_bytes=limits[1])


def test_cache_is_bounded_lru_detached_and_rebuildable(tmp_path):
    cache=LLMCache(tmp_path/'cache',derived_max_entries=2,derived_max_bytes=4096)
    keys=[hashlib.sha256(str(i).encode()).hexdigest() for i in range(3)]
    cache.put_derived(keys[0],completions());cache.put_derived(keys[1],completions())
    result=cache.get_derived(keys[0]);result.metadata['caller_mutation']=True
    assert cache.get_derived(keys[0]).metadata=={}
    cache.put_derived(keys[2],completions())
    assert cache.get_derived(keys[1]) is None and cache.get_derived(keys[0]) is not None
    assert cache.derived_snapshot()['entries']==2 and cache.derived_snapshot()['bytes']<=4096
    assert not cache.put_derived(keys[0],replace(completions(),content='大'*4096))
    restarted=LLMCache(tmp_path/'cache',derived_max_entries=2,derived_max_bytes=4096)
    assert restarted.get_derived(keys[0]) is None
    cache.clear_derived()
    assert cache.derived_snapshot()['entries']==cache.derived_snapshot()['bytes']==0


def test_concurrent_cache_access_preserves_capacity_and_returns_independent_values(tmp_path):
    cache=LLMCache(tmp_path/'cache',derived_max_entries=4,derived_max_bytes=2048)
    def use(i):
        key=hashlib.sha256(str(i%8).encode()).hexdigest()
        cache.put_derived(key,completions())
        value=cache.get_derived(key)
        if value is not None:value.metadata['mutation']=i
    with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(use,range(128)))
    state=cache.derived_snapshot()
    assert state['entries']<=4 and state['bytes']<=2048
    for i in range(8):
        value=cache.get_derived(hashlib.sha256(str(i).encode()).hexdigest())
        assert value is None or value.metadata=={}


@pytest.mark.parametrize('limits',[(0,0),(2,4096)])
def test_existing_cache_deepcopy_remains_supported_with_independent_derived_state(limits,tmp_path):
    cache=LLMCache(tmp_path/'cache',derived_max_entries=limits[0],derived_max_bytes=limits[1])
    key=hashlib.sha256(b'copy-control').hexdigest()
    cache.put_chat(key,completions())
    retained=cache.put_derived(key,completions())
    copied=deepcopy(cache)
    assert copied.get_chat(key)==cache.get_chat(key)
    assert copied._derived_lock is not cache._derived_lock
    assert copied.derived_snapshot()['entries']==int(retained)
    copied.clear_derived()
    assert copied.derived_snapshot()['entries']==0
    assert cache.derived_snapshot()['entries']==int(retained)


@pytest.mark.parametrize('application',['tutor','companion'])
@pytest.mark.parametrize('language',['en','zh'])
@pytest.mark.parametrize('version',[4,7])
def test_actual_application_reuse_workload_records_calls_and_software_latency(
        application,language,version,tmp_path,monkeypatch,request):
    """Fixed two-occurrence history, 21 comparisons + 20 normal turns per arm.

    State copying happens before timing. Scripted transport excludes real
    inference/queues; reported latency is only this software control workload.
    """
    from collections import Counter
    from math import ceil
    from statistics import median
    from time import perf_counter

    candidate,router,state,cache=setup(tmp_path,monkeypatch,application=application)
    baseline=LocalChatMembershipAssessor(router,assessment_kind='interpreted-membership-v2',
        format_version=4,max_input=7168,max_tokens=1024)
    texts=('A question?','An answer.') if language=='en' else ('一个问题？','一个回答。')
    query=QUERY if language=='en' else '原对话中，第一个问题和第一个回答，哪一句出现得较早？'
    later=query.replace('earlier','later') if language=='en' else query.replace('较早','较晚')
    template=CompanionSystem(byte_cap=None,top_k=6)
    for fragment in fragments(texts=texts):template._ingest_fragment(fragment)
    measured={}
    for arm,assessor in [('baseline',baseline),('reuse',candidate)]:
        agent=build_agent(application,deepcopy(template),make_llm_responder(router))
        agent.local_reporting_order=True;agent.max_request_input=8000
        agent.context_format_version=version;agent.reporting_assessor=assessor
        start=len(state['calls']);times=[]
        for turn in range(21):
            agent.companion_system=deepcopy(template)
            limits=LIMITS if arm=='baseline' or turn==0 else ConversationLimits()
            began=perf_counter()
            with router.conversation_turn(limits=limits) as budget:
                response=agent.respond(query if turn%2==0 else later)
            times.append((perf_counter()-began)*1000)
            text=response_text(response)
            expected=(texts[turn%2] if application=='companion' else
                      f'source position {turn%2}' if language=='en' else f'位置为 {turn%2}')
            assert expected in text
            assert budget.snapshot()['usage']['auxiliary']['attempts']==(4 if arm=='baseline' or turn==0 else 0)
        counts=Counter(state['calls'][start:])
        assert counts['source']==counts['conditional']==(42 if arm=='baseline' else 2)
        assert counts['response']==0
        normal_start=len(state['calls'])
        # Consecutive ordinary turns exercise the live, growing application memory.
        agent.companion_system=deepcopy(template)
        for turn in range(20):
            with router.conversation_turn() as budget:
                response=agent.respond(f'I would like to practice a little today, step {turn}.')
            assert budget.snapshot()['usage']['response']['attempts']==1
            assert budget.snapshot()['usage']['auxiliary']['attempts']==0
        assert state['calls'][normal_start:]==['response']*20
        warm=times[1:]
        measured[arm]=dict(comparison_turns=21,source_dispatches=counts['source'],
            conditional_dispatches=counts['conditional'],normal_turns=20,normal_reply_dispatches=20,
            cold_software_ms=times[0],warm_software_p50_ms=median(warm),
            warm_software_p95_ms=sorted(warm)[ceil(.95*len(warm))-1])
    measured.update(application=application,language=language,format_version=version,
        cache=cache.derived_snapshot(),scripted_transport=True,semantic_accuracy_qualified=False,
        latency_scope='Application software with prescribed completions; excludes real inference, queues and state-copy setup.')
    assert measured['cache']['bytes']<=measured['cache']['max_bytes']
    request.node.user_properties.append(('semantic_reuse_workload',json.dumps(measured,ensure_ascii=False)))
