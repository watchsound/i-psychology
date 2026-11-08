"""Portable transport/contract tests; authored labels are not accuracy evidence."""
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
import json
from itertools import product
from types import SimpleNamespace

import pytest

from psychology_base.applications.context import render_context_block
from psychology_base.applications.measurement.local_query_binding import LocalChatMembershipAssessor, LocalQueryProposer
from psychology_base.applications.measurement.semantic_membership import membership_inputs
from psychology_base.applications.reporting_order import ReportingMembershipSession, bind_reporting_order
from psychology_base.llm.execution import ExecutionDeferred, ExecutionPolicy
from psychology_base.llm.providers.ollama import OllamaProvider
from psychology_base.llm.types import Completion, TaskTier
from tests.test_source_order_visibility import source

QUERY = 'Which came earlier in the conversation: the first question or the first answer?'


def inputs():
    case = dict(language='en', descriptions=['the first question', 'the first answer'],
        records=[dict(id=i, speaker='user', text=text, source_order=dict(group='a'*32, position=i))
            for i, text in enumerate(['Question?', 'Answer.', 'Future text not allowed for earlier focuses.'])])
    return membership_inputs(case, format_version=3)


def router(response=None):
    provider = OllamaProvider(base_url='http://127.0.0.1:11434', simple_model='authored-test',
        advanced_model='authored-test', local_only=True, expected_digest='a'*64,
        context_length=8192, seed=0, json_mode=True)
    seen = []
    def chat(messages, **kwargs):
        seen.append((messages, kwargs))
        payload = json.loads(messages[1].content)
        text = payload['focus_input']['focus_utterance']['text']
        labels = ['match' if text == ('Question?' if 'first question' in d else 'Answer.') else 'nonmatch'
                  for d in payload['descriptions']]
        value = Completion(content=json.dumps(dict(labels=labels)), provider='ollama', model_id='authored-test',
            tier=TaskTier.ADVANCED, prompt_tokens=100, completion_tokens=12, finish_reason='stop',
            metadata=dict(local_only=True, done=True, response_model='authored-test',
                model_attestation=dict(digest='a'*64), context_length=8192, seed=0, json_mode=True))
        if isinstance(response, Exception):
            raise response
        return response(value) if response else value
    value = SimpleNamespace(execution=ExecutionPolicy(mode='local'), providers={'ollama': provider},
                            operation=lambda *args: nullcontext(), chat=chat)
    return value, seen


def test_groups_both_descriptions_per_focus_without_other_candidates_or_future_context():
    value, seen = router()
    rows = inputs()
    labels = LocalChatMembershipAssessor(value)(rows)
    assert labels == ['match', 'nonmatch', 'nonmatch', 'nonmatch', 'match', 'nonmatch']
    assert len(seen) == 3
    for i, (messages, kwargs) in enumerate(seen):
        payload = json.loads(messages[1].content)
        assert payload['focus_input'] == json.loads(rows[i]['pair'][0])
        assert payload['descriptions'] == [rows[i]['pair'][1], rows[i+3]['pair'][1]]
        assert kwargs == dict(tier=TaskTier.ADVANCED, max_tokens=64, temperature=0, use_cache=False)
        if i < 2:
            assert 'Future text not allowed' not in messages[1].content


def test_actor_constraints_make_no_calls_or_fake_probabilities():
    value, seen = router()
    rows = inputs()
    for row in rows:
        row['constraint'] = 'uncertain' if row['record'] == 0 else 'nonmatch'
    assert LocalChatMembershipAssessor(value)(rows) == ['uncertain', 'nonmatch', 'nonmatch'] * 2
    assert seen == []


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'boolean', 'context', 'description', 'oversize', 'nonobject'])
def test_validate_the_whole_matrix_before_any_dispatch(damage):
    value, seen = router()
    rows = inputs()
    if damage == 'missing': rows.pop()
    elif damage == 'duplicate': rows[-1] = deepcopy(rows[0])
    elif damage == 'boolean': rows[-1]['record'] = True
    elif damage == 'context': rows[-1]['pair'] = ('{}', rows[-1]['pair'][1])
    elif damage == 'description': rows[-1]['pair'] = (rows[-1]['pair'][0], 'changed description')
    elif damage == 'nonobject': rows[-1] = None
    else:
        for row in rows:
            if row['record'] == 2:
                p = json.loads(row['pair'][0]); p['focus_utterance']['text'] = 'x'*9000
                row['pair'] = (json.dumps(p), row['pair'][1])
    with pytest.raises(ValueError): LocalChatMembershipAssessor(value)(rows)
    assert seen == []


@pytest.mark.parametrize('content', ['{"labels":["match"]}', '{"labels":["match","bad"]}',
    '{"labels":[true,false]}', '{"labels":["match","nonmatch"],"extra":true}',
    '{"labels":[],"labels":["match","nonmatch"]}', '[]', 'not json'])
def test_malformed_outputs_cannot_become_partial_matrices(content):
    value, seen = router(lambda result: replace(result, content=content))
    with pytest.raises(ValueError): LocalChatMembershipAssessor(value)(inputs())
    assert len(seen) == 1


@pytest.mark.parametrize('damage', ['finish', 'provider', 'model', 'digest', 'usage', 'context', 'json'])
def test_response_identity_and_completion_must_be_valid(damage):
    def corrupt(value):
        if damage == 'finish': return replace(value, finish_reason='length')
        if damage == 'provider': return replace(value, provider='remote')
        if damage == 'model': return replace(value, model_id='other')
        if damage == 'usage': return replace(value, prompt_tokens=0)
        metadata = dict(value.metadata)
        if damage == 'digest': metadata['model_attestation'] = dict(digest='b'*64)
        elif damage == 'context': metadata['context_length'] = 4096
        else: metadata['json_mode'] = False
        return replace(value, metadata=metadata)
    value, seen = router(corrupt)
    with pytest.raises(ValueError): LocalChatMembershipAssessor(value)(inputs())
    assert len(seen) == 1


@pytest.mark.parametrize('damage', ['mode', 'model', 'digest', 'seed', 'format', 'bounds'])
def test_changed_configuration_fails_before_dispatch(damage):
    value, seen = router()
    assessor = LocalChatMembershipAssessor(value)
    if damage == 'mode': value.execution = ExecutionPolicy(mode='research')
    elif damage == 'model': value.providers['ollama'].advanced_model = 'changed'
    elif damage == 'digest': value.providers['ollama'].expected_digest = 'b'*64
    elif damage == 'seed': value.providers['ollama'].seed = 1
    elif damage == 'format': assessor.format_version = 2
    else: assessor.max_input = 9000
    with pytest.raises(ValueError): assessor(inputs())
    assert seen == []


def test_transport_failure_has_no_retry_or_fallback():
    value, seen = router(ExecutionDeferred('Authored local failure'))
    with pytest.raises(ExecutionDeferred): LocalChatMembershipAssessor(value)(inputs())
    assert len(seen) == 1


def test_pool_assessment_is_reused_at_binding_without_any_more_chat_calls():
    value, seen = router()
    assessor = LocalChatMembershipAssessor(value)
    fragments = [source(i, text=text) for i, text in enumerate(['Question?', 'Answer.', 'Other.'])]
    session = ReportingMembershipSession('tutor', QUERY, assessor)
    session.required_occurrences(fragments)
    calls = len(seen)
    result = bind_reporting_order('tutor', QUERY, render_context_block(fragments,
        assistant_speaker='tutor', format_version=4), membership_session=session)
    assert result.selected.source_position == 0 and len(seen) == calls == 3


def test_old_proposer_cannot_switch_to_paid_execution_after_construction():
    value, seen = router()
    proposer = LocalQueryProposer(value, model_identity='authored-test')
    value.execution = ExecutionPolicy(mode='research')
    with pytest.raises(ValueError): proposer('tutor', QUERY, ('a', 'b'), [])
    assert seen == []


@pytest.mark.parametrize('vector', list(product(('match', 'nonmatch', 'uncertain'), repeat=2)))
def test_all_label_vectors_follow_actual_interleaved_row_order(vector):
    value, seen = router(lambda result: replace(result, content=json.dumps(dict(labels=list(vector)))))
    original = inputs()
    # Reverse active slots and interleave records: the response follows the
    # descriptions actually sent, never assumed slot-major offsets.
    rows = [original[i] for i in (3, 0, 4, 1, 5, 2)]
    assert LocalChatMembershipAssessor(value)(rows) == list(vector)*3
    assert len(seen) == 3
    assert json.loads(seen[0][0][1].content)['descriptions'] == [original[3]['pair'][1], original[0]['pair'][1]]


def test_one_active_description_does_not_shift_actor_constrained_row():
    value, seen = router(lambda result: replace(result, content='{"labels":["match"]}'))
    rows = [inputs()[0], inputs()[3]]
    rows[0]['constraint'] = 'nonmatch'
    assert LocalChatMembershipAssessor(value)(rows) == ['nonmatch', 'match']
    assert len(json.loads(seen[0][0][1].content)['descriptions']) == 1


def test_prompt_variant_changes_only_example_and_has_a_separate_identity():
    from psychology_base.applications.measurement.local_query_binding import MEMBERSHIP_PROMPT, membership_prompt
    value, seen = router()
    baseline = LocalChatMembershipAssessor(value)
    variant = LocalChatMembershipAssessor(value, prompt_profile='example-reversed-v1')
    assert baseline.identity != variant.identity
    assert membership_prompt() == MEMBERSHIP_PROMPT
    assert membership_prompt('example-reversed-v1').replace(
        '{"labels":["nonmatch","match"]}', '{"labels":["match","nonmatch"]}') == MEMBERSHIP_PROMPT
    baseline(inputs()); variant(inputs())
    assert seen[0][0][0].content == MEMBERSHIP_PROMPT
    assert seen[3][0][1].content == seen[0][0][1].content
    variant.prompt_profile = 'baseline-v1'
    before = len(seen)
    with pytest.raises(ValueError): variant(inputs())
    assert len(seen) == before
    with pytest.raises(ValueError): LocalChatMembershipAssessor(value, prompt_profile='untracked')
