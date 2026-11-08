"""Authored software/transport checks; no model-accuracy claim or model access."""
from copy import deepcopy
import io
import json
from pathlib import Path
from urllib.request import Request

import pytest

from psychology_base.eval.benchmarks.application_calls import LocalNativeCapture
from psychology_base.eval.benchmarks.membership_focus import (
    build_cases, build_schedule, condition, focus_rows, grade, normalize,
)
from psychology_base.llm.providers.ollama import OllamaProvider

FIXTURE = Path(__file__).parents[1] / 'benchmarks/fixtures/application_context/membership-focus-contrasts-v1.json'


def packet():
    cases, refs = build_cases(json.loads(FIXTURE.read_text(encoding='utf-8')))
    return cases, refs, build_schedule(cases)


def test_bank_complete_references_fixed_schedule_and_label_free_inference():
    cases, refs, schedule = packet()
    assert len(cases) == 20 and len(schedule) == 42
    assert sum(len(r['judgments']) for r in refs.values()) == 88
    assert {lang: sum(e['language'] == lang for e in schedule) for lang in ('en', 'zh')} == {'en': 21, 'zh': 21}
    assert sum(e['repeat'] for e in schedule) == 2
    assert sum(e['prompt_profile'] == 'example-reversed-v1' for e in schedule) == 4
    for entry in schedule:
        case = cases[entry['case']]
        rows = focus_rows(case, entry)
        assert [r['slot'] for r in rows] == [0, 1]
        assert all(r['record'] == 2 and r['constraint'] is None for r in rows)
        assert all('rationale' not in r and 'label' not in r for r in rows)
        p = json.loads(rows[0]['pair'][0])
        assert len(p['prior_utterances']) in (0, 2)
        assert p['focus_utterance']['text'] == case['records'][-1]['text']
        assert [t['text'] for t in p['prior_utterances']] == [r['text'] for r in case['records'][:-1]]
    assert {e['language'] for e in schedule[:8]} == {'en', 'zh'}
    assert any(e['case'] == 'both-en' for e in schedule[:8])
    assert any(e['case'] == 'neither-zh' for e in schedule[:8])
    assert any(e['case'] == 'referent-zh-ambiguous' for e in schedule[:8])
    assert packet()[2] == schedule


def test_description_reversal_reformats_payload_and_normalizes_semantics():
    cases, refs, _ = packet()
    for case in cases.values():
        fwd, rev = condition(case), condition(case, reverse=True)
        a, b = focus_rows(case, fwd), focus_rows(case, rev)
        assert a[0]['pair'][0] == b[0]['pair'][0]
        assert [r['pair'][1] for r in a] == [r['pair'][1] for r in reversed(b)]
        assert normalize(fwd, ['match', 'nonmatch']) == normalize(rev, ['nonmatch', 'match'])
        changed = deepcopy(rev); changed['description_ids'].reverse()
        with pytest.raises(ValueError): focus_rows(case, changed)


def test_grading_preserves_missing_failed_and_complete_contrast_denominators():
    cases, refs, schedule = packet()
    empty = grade(cases, refs, schedule, [])
    assert empty['summary']['all']['missing'] == 42 and empty['summary']['all']['planned_pairs'] == 84
    assert not any(c['complete'] for c in empty['contrasts'])
    observations = []
    for e in schedule:
        case = cases[e['case']]
        truth = {r['slot']: r['label'] for r in refs[e['case']]['judgments'] if r['record'] == 2}
        observations.append(dict(id=e['id'], labels=[truth[i] for i in e['description_order']]))
    full = grade(cases, refs, schedule, observations)
    assert full['summary']['all']['correct_pairs'] == 84
    assert full['summary']['all']['correct_conditions'] == 42
    assert all(c['same'] == (c['expected_relation'] == 'same') for c in full['contrasts'])
    observations[0] = dict(id=observations[0]['id'], error='authored format failure')
    observations.pop()
    partial = grade(cases, refs, schedule, observations)['summary']['all']
    assert partial['failed'] == partial['missing'] == 1 and partial['planned'] == 42
    with pytest.raises(ValueError): grade(cases, refs, schedule, observations*2)


@pytest.mark.parametrize('labels', [['match', 'nonmatch'], ['nonmatch', 'match'], ['uncertain', 'uncertain']])
def test_constant_vectors_cannot_pass_controls(labels):
    cases, refs, schedule = packet()
    scored = grade(cases, refs, schedule, [dict(id=e['id'], labels=labels) for e in schedule])
    assert scored['summary']['all']['correct_conditions'] < 42
    assert any(not r['correct'] for r in scored['records'] if r['id'].startswith(('both-', 'neither-')))


def capture_setup(*, fail=False, max_attempts=1, clock=lambda: 0):
    provider = OllamaProvider(base_url='http://127.0.0.1:11434', local_only=True)
    seen, events = [], []
    def opened(request, *, timeout):
        seen.append((request, timeout))
        if fail: raise TimeoutError('authored transport failure')
        return io.BytesIO(b'{"authored_transport_fixture":true}')
    provider._open = opened
    def validate(payload):
        if payload != {'authored_request': True}:
            raise ValueError('authored payload differs')
    cap = LocalNativeCapture(provider, max_attempts=max_attempts, deadline=10,
        observer=lambda event, phase: events.append((event, phase)), validate_chat=validate, clock=clock)
    return provider, cap, seen, events, opened


def chat_request():
    return Request('http://127.0.0.1:11434/api/chat', data=b'{"authored_request":true}')


def test_capture_exact_response_attempt_before_transport_and_restore():
    p, cap, seen, events, opened = capture_setup()
    with cap:
        with p._open(chat_request(), timeout=180) as response:
            assert json.load(response) == {'authored_transport_fixture': True}
        assert cap.attempts == 1 and seen[0][1] == 10
        with pytest.raises(ValueError, match='attempt cap'): p._open(chat_request(), timeout=1)
    assert p._open is opened and len(seen) == 1
    assert [phase for _, phase in events] == ['started', 'finished']
    assert 'response' not in events[0][0] and events[1][0]['response_sha256']


def test_failed_transport_is_counted_and_emitted_without_retry():
    p, cap, seen, events, opened = capture_setup(fail=True)
    with pytest.raises(TimeoutError), cap: p._open(chat_request(), timeout=180)
    assert p._open is opened and cap.attempts == len(seen) == 1
    assert 'error' in events[-1][0]


@pytest.mark.parametrize('endpoint', ['http://example.com/api/chat',
    'http://127.0.0.1:11434/api/embed', 'http://127.0.0.1:11434/api/chat?extra=1'])
def test_capture_rejects_other_endpoints_before_transport(endpoint):
    p, cap, seen, _, _ = capture_setup()
    with cap, pytest.raises(ValueError): p._open(Request(endpoint), timeout=10)
    assert seen == [] and cap.attempts == 0


def test_capture_deadline_and_metadata_bound():
    p, cap, seen, _, _ = capture_setup(clock=lambda: 11)
    with cap, pytest.raises(TimeoutError): p._open(chat_request(), timeout=1)
    assert seen == []
    p, cap, seen, _, _ = capture_setup()
    cap.max_metadata = 1
    with cap:
        with p._open(Request(p.base_url+'/api/ps'), timeout=1): pass
        with pytest.raises(ValueError, match='metadata attempt cap'):
            p._open(Request(p.base_url+'/api/tags'), timeout=1)
    assert cap.metadata_attempts == 1 and cap.attempts == 0 and len(seen) == 1
