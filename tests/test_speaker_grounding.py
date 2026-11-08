"""Rendering/request regressions; authored criteria are not model-quality results."""
import copy
from datetime import datetime, timedelta, timezone
import importlib
import json

import pytest

from psychology_base.applications.emotional import companion_agent as companion
from psychology_base.applications.learning import tutor_agent as tutor
from psychology_base.companion.turn_provenance import exchange_tag, new_exchange_id, sequence_tag
from psychology_base.eval.adapters import make_fragment
from psychology_base.eval.benchmarks import application_context as study
from tests.test_application_context_study import real_sources


STAMP = datetime(2026, 10, 20, tzinfo=timezone.utc)
BANK = study.read(study.FIXTURES / 'memory-query-boundaries-v1.json')
OBSERVED = [c for c in BANK['cases'] if c['origin'] == 'observed-real-smoke']


def fragments(case):
    result = []
    for f in case['fragments']:
        fragment = make_fragment(f['content'], datetime.fromisoformat(f['timestamp']), f['affect'],
                                 tags=tuple(f['tags']), parent_ids=tuple(f['parent_ids']))
        assert fragment.id == f['id']
        result.append(fragment)
    return result


def test_actual_tutor_source_roles_survive_rendering_without_mutating_source_memory():
    current = study.read(study.FIXTURES / 'memory-query-boundaries-v2.json')
    case = next(c for c in current['cases'] if c['id'] == 'observed-tutor-explicit-scaled_focus')
    source = fragments(case)
    before = copy.deepcopy(source)
    common = importlib.import_module('psychology_base.applications.context')
    assert common.render_context_block(source, assistant_speaker='tutor', format_version=2) == case['context']
    assert [row['text'] for row in map(json.loads, tutor._render_context_block(source).splitlines())] == [f.content for f in source]
    assert source == before


@pytest.mark.parametrize('owner,self_role,other_role', [(companion, 'companion', 'tutor'), (tutor, 'tutor', 'companion')])
def test_both_apps_use_one_shared_renderer(owner, self_role, other_role):
    common = importlib.import_module('psychology_base.applications.context')
    assert owner.render_context_block is common.render_context_block
    source = [make_fragment('same source words', STAMP, tags=(f'speaker:{role}', 'session:earlier'))
              for role in ('user', self_role, other_role)]
    expected = [{'speaker': actor, 'session': 'earlier', 'text': 'same source words', 'source_order': None}
                for actor in ('user', self_role, other_role)]
    assert list(map(json.loads, owner._render_context_block(iter(source)).splitlines())) == expected


@pytest.mark.parametrize('owner', [companion, tutor])
@pytest.mark.parametrize('tags,label,suffix', [
    ((), 'fragment', ''),
    (('speaker:user',), 'user said', ''),
    (('speaker:user', 'speaker:user'), 'user said', ''),
    (('speaker:user', 'speaker:tutor'), 'fragment', ''),
    (('speaker:user', 'speaker:'), 'fragment', ''),
    (('speaker:source-actor-0', 'session:source'), 'fragment', ' (session source)'),
    (('speaker:user', 'session:a', 'session:b'), 'user said', ''),
    (('speaker:user', 'session:', 'session:a'), 'user said', ''),
    (('speaker:user', 'session:a', 'session:a'), 'user said', ' (session a)'),
    (('speaker:user', 'pb-turn:v99:invalid'), 'fragment', ''),
])
def test_ambiguous_or_unknown_metadata_is_never_reassigned(owner, tags, label, suffix):
    source = make_fragment('literal text', STAMP, tags=tags)
    actor = label[:-5] if label.endswith(' said') else None
    session = suffix[len(' (session '):-1] if suffix else None
    assert json.loads(owner._render_context_block([source])) == {'speaker': actor, 'session': session, 'text': 'literal text', 'source_order': None}
    assert owner._render_context_block(()) == ''


@pytest.mark.parametrize('owner', [companion, tutor])
def test_existing_canonical_actor_evidence_is_used_and_conflicts_fail_closed(owner):
    descriptor = exchange_tag(new_exchange_id(), 'user', 0)
    source = make_fragment('source statement', STAMP, tags=(descriptor, 'session:a'))
    assert json.loads(owner._render_context_block([source])) == {'speaker': 'user', 'session': 'a', 'text': 'source statement', 'source_order': None}
    conflict = make_fragment('source statement', STAMP, tags=(descriptor, 'speaker:tutor', 'session:a'))
    assert json.loads(owner._render_context_block([conflict])) == {'speaker': None, 'session': 'a', 'text': 'source statement', 'source_order': None}
    missing_actor = sequence_tag('source', 'dialogue', 0)
    unknown = make_fragment('source statement', STAMP, tags=(missing_actor, 'session:a'))
    assert json.loads(owner._render_context_block([unknown])) == {'speaker': None, 'session': 'a', 'text': 'source statement', 'source_order': None}


def test_companion_names_speakers_explicitly_and_preserves_literal_content():
    source = [make_fragment('source statement', STAMP, tags=(f'speaker:{role}', 'session:a'))
              for role in ('user', 'companion', 'source-actor-0')]
    assert list(map(json.loads, companion._render_context_block(source).splitlines())) == [
        {'speaker': actor, 'session': 'a', 'text': 'source statement', 'source_order': None} for actor in ('user', 'companion', None)]


def test_tutor_normal_exchange_passes_roles_to_the_real_responder_boundary():
    from tests.test_learning_tutor_agent import _build
    from psychology_base.applications.safety import AgeAttestation
    contexts = []
    def software_responder(prompt, context, utterance):
        contexts.append(context)
        return 'Which step would you inspect?'
    # This injected reply tests transport/composition only, never judge quality.
    agent = _build(responder=software_responder)
    agent.respond('I used fifteen days in my calculation.', timestamp=STAMP, attestation=AgeAttestation(30, STAMP.timestamp(), 'test'))
    now = STAMP + timedelta(minutes=1)
    agent.respond('Can we revisit my fifteen-day calculation?', timestamp=now, attestation=AgeAttestation(30, now.timestamp(), 'test'))
    assert contexts[0] == ''
    rows = [{k: v for k, v in row.items() if k != 'source_order'}
            for row in map(json.loads, contexts[1].splitlines())]
    assert {'speaker': 'user', 'session': 's1', 'text': 'I used fifteen days in my calculation.'} in rows
    assert {'speaker': 'tutor', 'session': 's1', 'text': 'Which step would you inspect?'} in rows
    assert 'Can we revisit' not in contexts[1]


class CapturedRequest(Exception):
    """No model output exists in a request-construction test."""


class RequestCapture:
    controls = False

    def chat(self, messages, **kwargs):
        self.messages, self.kwargs = messages, kwargs
        raise CapturedRequest()


def test_current_query_clarification_is_opt_in_nonmutating_and_idempotent():
    old = study.revised_grading_protocol({'memory_prompt': study.MEMORY_PROMPT, 'profile_sha256': 'a'*64})
    before = copy.deepcopy(old)
    new = study.revised_grading_protocol(old, current_query_evidence=True)
    assert old == before and new['grading_contract_version'] == 2
    assert new['memory_prompt'].startswith(old['memory_prompt'])
    assert new['memory_prompt'].endswith(study.MEMORY_CURRENT_QUERY_CLARIFICATION_V1)
    assert study.revised_grading_protocol(new, current_query_evidence=True) == new
    assert study.revised_grading_protocol(new) == new
    assert study.protocol_sha(old) != study.protocol_sha(new)
    assert study.grading_identity(old)['tutor_contract_prompt_sha256'] == study.grading_identity(new)['tutor_contract_prompt_sha256']


@pytest.mark.parametrize('value', [None, 'true', 1])
def test_current_query_switch_rejects_ambiguous_configuration(value):
    with pytest.raises(ValueError):
        study.revised_grading_protocol({}, current_query_evidence=value)


@pytest.mark.parametrize('case', OBSERVED, ids=lambda c: c['id'])
def test_observed_case_reaches_existing_judge_without_hidden_roles_or_reference_leak(case):
    protocol = study.revised_grading_protocol({}, current_query_evidence=True)
    router = RequestCapture()
    with pytest.raises(CapturedRequest):
        study.judge_memory(router, {'app': case['application']}, case['source_absent'],
                           {'query': case['query'], 'context': case['context']}, case['reply'], case['reference'], protocol=protocol)
    assert router.messages[0].content == protocol['memory_prompt']
    payload = json.loads(router.messages[1].content)
    assert payload == {'APPLICATION': case['application'], 'QUERY': case['query'], 'VISIBLE_CONTEXT': case['context'],
                       'REPLY': case['reply'], 'SOURCE_ABSENT': case['source_absent'], 'WITHHELD_REFERENCE': case['reference']}
    assert router.kwargs['temperature'] == 0 and router.kwargs['max_tokens'] == 300
    # The observed wrong judgment is retained; parsing is not a semantic repair.
    old = case['observed_judgment']
    assert study.parse_memory(old['raw'], old['finish_reason'], old['proxy'], contract_version=2) == old


@pytest.mark.parametrize('case', BANK['cases'], ids=lambda c: c['id'])
def test_reviewed_boundary_criteria_are_coherent_not_automated_semantic_results(case):
    result = study.parse_memory(json.dumps(case['expected_criteria']), 'stop', True, contract_version=2)
    assert result['proxy'] and study.judgment_resolved(result)


def test_frozen_v4_prompt_stays_exact_and_old_qualification_cannot_cover_new_evidence(real_sources):
    old = study.local_profile(study.FIXTURES/'hybrid-screen-v4.json', real_sources[0])
    pin = study.read(study.FIXTURES/'grading-contract-v2.json')
    assert study.grading_identity(old) == {k: v for k, v in pin.items() if k != 'schema'}
    new = study.revised_grading_protocol(old, current_query_evidence=True)
    assert study.qualification_identity(old) != study.qualification_identity(new)
    # The frozen catalog must reject the new prompt even without a local run archive.
    with pytest.raises(ValueError, match='pre-model freeze'):
        study.calibration_items(new, real_sources[2])
    path = study.ROOT/'.data/application_context/grading-contract-v4-qualification/calibration-result.json'
    if path.exists():
        with pytest.raises(ValueError, match='pre-model freeze'):
            study.validate_calibration(study.read(path), new, real_sources[2])
