"""Real SDK/mock HTTP and durable budget controls; no model-quality evidence."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from psychology_base.llm.execution import ExecutionDeferred
from psychology_base.llm.providers.openai import OpenAIProvider


@pytest.fixture
def screen(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    path = root / 'docs/experiments/phase-9-memory-navigation/run-tongyi-interpretation-screen-v1.py'
    spec = importlib.util.spec_from_file_location('test_tongyi_interpretation_entry', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'OUT', tmp_path / 'out')
    module.OUT.mkdir()
    monkeypatch.setattr(module, 'PLAN', tmp_path / 'plan.json')
    monkeypatch.setattr(module, 'RESULT', tmp_path / 'result.json')
    monkeypatch.setattr(module, 'AUDIT', tmp_path / 'audit.json')
    return module


def test_source_only_bilingual_control_packet_and_reference_replay(screen):
    inputs, references = screen.build_packet()
    assert [r['application'] for r in inputs['requests']] == ['tutor', 'tutor', 'companion', 'companion']
    assert [r['language'] for r in inputs['requests']] == ['zh', 'en', 'zh', 'en']
    observations = []
    for request in inputs['requests']:
        user = json.loads(request['messages'][1]['content'])
        assert 'descriptions' not in user and 'readings' not in user and 'coverage' not in user
        assert user['turns'][0]['speaker'] == request['application']
        observations.append(dict(id=request['id'], analysis=references[request['case']]['expected']))
    grade = screen.grade_interpretation_packet(inputs['cases'], references, inputs['schedule'], observations)
    assert grade['summary']['all']['correct'] == 4
    assert grade['candidate_screen_passed'] and not grade['qualified']


@pytest.mark.parametrize('mode', ['valid', 'invalid-coverage', 'missing-alternative', 'http-failure', 'redirect', 'truncated', 'invalid-usage'])
def test_actual_paid_router_sdk_capture_and_grading(screen, monkeypatch, mode):
    openai = pytest.importorskip('openai')
    httpx = pytest.importorskip('httpx')
    inputs, references = screen.build_packet()
    screen.save_new(screen.OUT / 'inputs.json', inputs)
    screen.save_new(screen.OUT / 'references.json', references)
    monkeypatch.setattr(screen, 'validate_plan', lambda: None)
    monkeypatch.setattr(screen, 'load_config', lambda: SimpleNamespace(
        get_provider=lambda name: SimpleNamespace(api_key=lambda: 'AUTHORED-KEY')))
    seen = []
    def respond(request):
        payload = json.loads(request.content)
        seen.append(payload)
        if mode in ('http-failure', 'redirect'):
            return httpx.Response(500 if mode == 'http-failure' else 307,
                json={'error': {'message': 'AUTHORED-KEY'}}, headers={'Location': 'https://example.com/forbidden'})
        declared = inputs['requests'][len(seen) - 1]
        expected = references[declared['case']]['expected']
        target = deepcopy({key: expected[key] for key in ('coverage', 'readings')})
        if mode == 'invalid-coverage':
            target['coverage'] = 'open'
        if mode == 'missing-alternative' and len(target['readings']) > 1:
            target['readings'] = target['readings'][:1]
        return httpx.Response(200, json=dict(id='authored-' + str(len(seen)), object='chat.completion',
            created=1, model=screen.SPEC['model'], choices=[dict(index=0,
                finish_reason='length' if mode == 'truncated' else 'stop',
                message=dict(role='assistant', content=json.dumps(target), reasoning_content=''))],
            usage=dict(prompt_tokens='100' if mode == 'invalid-usage' else 100,
                completion_tokens=100, total_tokens=200)))
    def ensure(provider):
        if provider._client is None:
            provider._client = openai.OpenAI(api_key=provider.api_key, base_url=provider.base_url,
                max_retries=0, timeout=60, http_client=httpx.Client(
                    transport=httpx.MockTransport(respond), follow_redirects=False))
        return provider._client
    monkeypatch.setattr(OpenAIProvider, '_ensure_client', ensure)
    screen.run()
    screen.audit()
    result = screen.read(screen.RESULT)
    assert not result['model_qualified']
    assert result['physical_http_attempts'] == len(seen) == result['ledger_calls']
    assert 'AUTHORED-KEY' not in (screen.OUT / 'transport.jsonl').read_text()
    for payload in seen:
        assert payload['enable_thinking'] is False
        assert payload['max_completion_tokens'] == 4096
        assert payload['response_format'] == {'type': 'json_object'}
    counts = result['grade']['summary']['all']
    if mode == 'valid':
        assert counts['correct'] == 4
    elif mode == 'missing-alternative':
        assert counts['completed'] == 4 and counts['correct'] == 2
    elif mode == 'invalid-coverage':
        assert counts['completed'] == counts['correct'] == 0 and len(seen) == 4
    else:
        assert counts['completed'] == counts['correct'] == 0 and len(seen) == 1
    with pytest.raises(FileExistsError, match='cannot be retried'):
        screen.run()


def test_fifth_dispatch_cannot_reset_durable_allowance(screen):
    from psychology_base.llm.accounting import RuntimeBudget
    for _ in range(4):
        budget = RuntimeBudget(screen.policy())
        budget.reserve('0.001', session='same', operation='same', purpose='control',
            provider='tongyi', model=screen.SPEC['model'], kind='chat')
    with pytest.raises(ExecutionDeferred):
        RuntimeBudget(screen.policy()).reserve('0.001', session='other', operation='other',
            purpose='control', provider='tongyi', model=screen.SPEC['model'], kind='chat')


def test_existing_artifacts_cannot_be_overwritten(screen):
    screen.save_new(screen.RESULT, {'original': True})
    with pytest.raises(FileExistsError):
        screen.save_new(screen.RESULT, {'original': False})
    assert screen.read(screen.RESULT) == {'original': True}
