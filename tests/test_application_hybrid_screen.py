"""Hybrid transport and qualification controls, never application-quality evidence.

Injected HTTP responses test the real SDK/provider/router/audit owners. Live judge
qualification is recorded separately and cannot be established by these controls.
"""
import copy
import json
from argparse import Namespace
from decimal import Decimal
from types import SimpleNamespace

import pytest

from psychology_base.applications.pilot.session_runner import SessionLogWriter
from psychology_base.eval.benchmarks import application_context as study
from psychology_base.eval.benchmarks.application_calls import Budget, validate_screening_call
from psychology_base.llm import load_config
from psychology_base.llm.providers.openai import OpenAIProvider
from psychology_base.llm.types import ChatMessage, TaskTier
from tests.test_application_context_study import real_sources


PROFILE = study.FIXTURES/'hybrid-screen-v1.json'


def test_hybrid_reuses_frozen_sources_and_192_response_scope(real_sources, tmp_path):
    base, cases, _ = real_sources
    original = copy.deepcopy(base)
    hybrid = study.local_profile(PROFILE, base)
    local = study.local_profile(study.FIXTURES/'local-screen-template-repair-v2.json', base)
    assert base == original and hybrid['generator'] == local['generator']
    assert hybrid['screening'] and not hybrid['local_only'] and hybrid['samples'] == 1
    assert hybrid['budget'] == {'max_calls':600, 'max_usd':'1', 'local_only':False}
    for field in ('gates','source_pins','annotations','catalog_sha256','scoring_prompt_sha256'):
        assert hybrid[field] == base[field]
    selected = [c for c in cases if c['split']=='development']
    rows = [dict(case=c['id'], app=c['app'], query_index=q, absent=a, candidate=arm, sample=0)
            for c in selected for q,a,arm in study.itertools.product(range(2),(False,True),study.ARMS)]
    assert study.matrix(rows, selected, hybrid['samples']) == {'expected':192,'actual':192,'unique':192}
    with pytest.raises(ValueError, match='held-out'):
        study.run(Namespace(command='application-evaluation', application_profile=PROFILE))


@pytest.mark.parametrize('change', ['budget','attempts','split','samples','model','rates','reasoning','format','provider','generator'])
def test_hybrid_rejects_unapproved_model_transport_cost_and_scope(real_sources, tmp_path, change):
    profile = study.read(PROFILE)
    if change == 'budget': profile['budget']['max_usd'] = '50'
    elif change == 'attempts': profile['budget']['max_calls'] = 601
    elif change == 'split': profile['split'] = 'evaluation'
    elif change == 'samples': profile['samples'] = 3
    elif change == 'model': profile['judge']['model'] = 'gpt-6-astra'
    elif change == 'rates': profile['judge']['input_rate'] = '0'
    elif change == 'reasoning': profile['judge']['reasoning_effort'] = 'medium'
    elif change == 'format': profile['judge']['json_mode'] = False
    elif change == 'provider': profile['judge']['provider'] = 'ollama'
    else: profile['generator']['base_url'] = 'https://example.com'
    path = tmp_path/'profile.json'; path.write_text(json.dumps(profile), encoding='utf-8')
    with pytest.raises(ValueError): study.local_profile(path, real_sources[0])


def hybrid_judge(tmp_path, *, dollars='1', response_change=None, status=200):
    openai = pytest.importorskip('openai')
    httpx = pytest.importorskip('httpx')
    base, _ = study.load_inputs()
    protocol = study.local_profile(PROFILE, base)
    budget = Budget(SessionLogWriter(tmp_path/'events.jsonl'), max_calls=600, max_usd=dollars)
    generation, judge = study.make_routers(protocol, load_config(), budget)
    requests = []
    def respond(request):
        requests.append(json.loads(request.content))
        if status != 200:
            return httpx.Response(status, json={'error':{'message':'Declared transport failure','type':'server_error'}})
        data = {'id':'chatcmpl-declared-unit-control', 'object':'chat.completion', 'created':1,
                'model':'gpt-6-luna', 'service_tier':'default',
                'choices':[{'index':0,'message':{'role':'assistant','content':'{"ready":true}'},'finish_reason':'stop'}],
                'usage':{'prompt_tokens':12,'completion_tokens':4,'total_tokens':16,
                         'completion_tokens_details':{'reasoning_tokens':0}}}
        if response_change: response_change(data)
        return httpx.Response(200, json=data)
    judge.router.providers['openai']._client = openai.OpenAI(api_key='declared-unit-key', max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    return generation, judge, budget, requests, protocol


def test_hybrid_real_sdk_request_uses_bounded_nonreasoning_json_and_shared_ledger(tmp_path, monkeypatch):
    generation, judge, budget, requests, protocol = hybrid_judge(tmp_path)
    from tests.test_application_local_screen import local_wrapper
    local, _, local_requests, _ = local_wrapper(tmp_path, monkeypatch, budget=budget)
    monkeypatch.setattr(generation.router.providers['ollama'], '_open', local.router.providers['ollama']._open)
    generation.chat([ChatMessage('user','Declared local request')], tier=TaskTier.ADVANCED, max_tokens=20)
    judge.chat(study.preflight_requests(protocol,'judge')[1], tier=TaskTier.ADVANCED,
               max_tokens=16, temperature=0, use_cache=False)
    assert len(local_requests) == len(requests) == 1 and budget.attempts == 2
    assert set(generation.router.providers) == {'ollama'} and set(judge.router.providers) == {'openai'}
    assert generation.router.cache is judge.router.cache is None
    assert generation.last_call['local_only'] and generation.last_call['accounted_usd'] == '0'
    assert requests[0]['model'] == 'gpt-6-luna' and requests[0]['max_completion_tokens'] == 16
    assert 'max_tokens' not in requests[0] and requests[0]['reasoning_effort'] == 'none'
    assert requests[0]['response_format'] == {'type':'json_object'} and requests[0]['service_tier'] == 'default'
    assert requests[0]['temperature'] == 0 and '就绪' in requests[0]['messages'][1]['content']
    validate_screening_call(judge.last_call, protocol['judge'])
    assert budget.spent == Decimal('0.0000035')
    budget.writer.close()
    restored = Budget(SessionLogWriter(tmp_path/'events.jsonl'), max_calls=600, max_usd='1')
    assert restored.spent == budget.spent and restored.attempts == 2 and not restored.uncertain
    restored.writer.close()


def test_hybrid_dollar_limit_blocks_before_http_and_does_not_reset_spend(tmp_path):
    _, judge, budget, requests, _ = hybrid_judge(tmp_path, dollars='0.000001')
    with pytest.raises(ValueError, match='before dispatch'):
        judge.chat([ChatMessage('user','JSON unit request')], tier=TaskTier.ADVANCED, max_tokens=300)
    assert not requests and budget.attempts == 0
    budget.writer.close()


def test_hybrid_sdk_failure_is_one_attempt_and_retains_reservation(tmp_path):
    _, judge, budget, requests, _ = hybrid_judge(tmp_path, status=500)
    with pytest.raises(Exception, match='Declared transport failure'):
        judge.chat([ChatMessage('user','JSON unit request')], tier=TaskTier.ADVANCED, max_tokens=300)
    assert len(requests) == budget.attempts == 1 and budget.uncertain
    assert budget.spent == Decimal('0.00165')
    budget.writer.close()


@pytest.mark.parametrize('change', ['returned_model','reasoning_tokens','missing_usage','output_overrun','tier'])
def test_hybrid_rejects_actual_response_identity_decoding_and_usage_drift(tmp_path, change):
    def mutate(data):
        if change == 'returned_model': data['model'] = 'another-model'
        elif change == 'reasoning_tokens': data['usage']['completion_tokens_details']['reasoning_tokens'] = 1
        elif change == 'missing_usage': data.pop('usage')
        elif change == 'output_overrun': data['usage']['completion_tokens'] = 301
        else: data['service_tier'] = 'priority'
    _, judge, budget, requests, _ = hybrid_judge(tmp_path, response_change=mutate)
    with pytest.raises(ValueError):
        judge.chat([ChatMessage('user','JSON unit request')], tier=TaskTier.ADVANCED, max_tokens=300)
    assert len(requests) == budget.attempts == 1
    budget.writer.close()


@pytest.mark.parametrize('change', ['cache','reasoning','format','endpoint'])
def test_hybrid_rejects_changed_provider_before_http(tmp_path, change):
    _, judge, budget, requests, _ = hybrid_judge(tmp_path)
    provider = judge.router.providers['openai']
    if change == 'cache': judge.router.cache = object()
    elif change == 'reasoning': provider.reasoning_effort = 'low'
    elif change == 'format': provider.json_mode = False
    else: provider._client.base_url = 'http://api.openai.com/v1'
    with pytest.raises(ValueError):
        judge.chat([ChatMessage('user','JSON unit request')], tier=TaskTier.ADVANCED, max_tokens=300)
    assert not requests and budget.attempts == 0
    budget.writer.close()


def test_existing_openai_defaults_preserve_request_contract():
    requests = []
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='unit'), finish_reason='stop')], usage=None)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: requests.append(kw) or response)))
    OpenAIProvider(_client=client).chat([ChatMessage('user','unit')], max_tokens=20, temperature=0)
    assert requests == [{'model':'gpt-4o-mini', 'messages':[{'role':'user','content':'unit'}], 'max_tokens':20, 'temperature':0}]
    with pytest.raises(ValueError, match='Temperature'):
        OpenAIProvider(_client=client, reasoning_effort='low').chat([], temperature=0)
    assert len(requests) == 1


def test_hybrid_failed_calibration_remains_failure_and_validates_actual_call(real_sources, tmp_path, monkeypatch):
    _, _, sources = real_sources
    _, judge, budget, requests, protocol = hybrid_judge(tmp_path)
    # The explicit transport control returns a readiness object, not a rubric.
    # It must fail qualification rather than becoming a synthetic success.
    report = study.calibration(protocol, sources, judge)
    report.update(code_sha256=study.fingerprints(), protocol_sha256=study.protocol_sha(protocol))
    assert len(requests) == 1 and not report['passed'] and report['stopped_early']
    assert report['remaining_controls'] == {'memory':23,'tutor':24}
    assert study.validate_calibration(report, protocol, sources) is False
    for change in ('passed','raw','cost','identity','finish'):
        bad = copy.deepcopy(report)
        if change == 'passed': bad['passed'] = True
        elif change == 'raw': bad['memory_controls'][0]['call']['completion']['content'] = '{}'
        elif change == 'cost': bad['memory_controls'][0]['call']['accounted_usd'] = '0'
        elif change == 'identity': bad['memory_controls'][0]['call']['completion']['metadata']['response_model'] = 'different'
        else: bad['memory_controls'][0]['call']['completion']['finish_reason'] = 'length'
        with pytest.raises(ValueError): study.validate_calibration(bad, protocol, sources)
    path = tmp_path/'failed-calibration.json'
    report.update(study.runtime_identity(load_config(), cjk_tokenizer=protocol['tokenizer']))
    path.write_text(json.dumps(report), encoding='utf-8')
    with pytest.raises(ValueError, match='Calibration failed'):
        study.require_calibration(Namespace(application_calibration=path), protocol, sources, load_config())
    budget.writer.close()
