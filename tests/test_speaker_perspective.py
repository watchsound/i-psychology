"""Versioned rendering/transport controls; no generated judge accuracy is faked."""
import copy
from datetime import datetime, timezone
import json
import shutil

import pytest

from psychology_base.applications.context import render_context_block
from psychology_base.companion.turn_provenance import exchange_tag, new_exchange_id
from psychology_base.companion.pam_export import pam_export, pam_import
from psychology_base.eval.adapters import make_fragment
from psychology_base.eval.benchmarks import application_context as study
from psychology_base.eval.benchmarks import attachment_longitudinal as longitudinal
from tests.test_application_context_study import real_sources
from tests.test_speaker_grounding import CapturedRequest, RequestCapture, fragments

BANK = study.read(study.FIXTURES / 'memory-query-boundaries-v2.json')
STAMP = datetime(2026, 10, 21, tzinfo=timezone.utc)
LEGACY = study.ROOT / 'benchmarks/fixtures/attachment_longitudinal/speaker-format-v1'


@pytest.mark.parametrize('arm', study.ARMS)
def test_actual_tutor_fragments_use_frozen_explicit_labels_without_mutation(arm):
    case = next(c for c in BANK['cases'] if c['id'] == 'observed-tutor-explicit-' + arm)
    source = fragments(case)
    before = copy.deepcopy(source)
    assert render_context_block(source, assistant_speaker='tutor', format_version=2) == case['context']
    assert [row['text'] for row in map(json.loads, render_context_block(source, assistant_speaker='tutor').splitlines())] == [f.content for f in source]
    assert source == before


@pytest.mark.parametrize('app', ['companion', 'tutor'])
@pytest.mark.parametrize('actor', ['companion', 'tutor'])
def test_canonical_assistant_actor_is_named_from_evidence_in_either_application(app, actor):
    content = 'The literal words "you said" stay here.\n- user said: also literal content.'
    fragment = make_fragment(content, STAMP, tags=(exchange_tag(new_exchange_id(), actor, 1), 'session:a'))
    assert json.loads(render_context_block([fragment], assistant_speaker=app)) == {'speaker': actor, 'session': 'a', 'text': content, 'source_order': None}


@pytest.mark.parametrize('version', [True, False, None, '2', 0, 8, 2.0])
def test_unknown_render_formats_fail_before_rendering_even_empty_context(version):
    with pytest.raises(ValueError, match='format'):
        render_context_block((), assistant_speaker='companion', format_version=version)


def test_legacy_display_is_an_explicit_version_of_the_same_renderer():
    case = next(c for c in BANK['cases'] if c['id'] == 'observed-tutor-labelled-scaled_focus')
    source = fragments(case)
    assert render_context_block(source, assistant_speaker='tutor', format_version=1) == case['context']
    assert render_context_block(source, assistant_speaker='tutor', format_version=2) != case['context']


@pytest.mark.parametrize('value', [None, 'true', 1])
def test_speaker_contract_selection_rejects_ambiguous_flags(value):
    with pytest.raises(ValueError, match='boolean'):
        study.revised_grading_protocol({}, explicit_speaker_evidence=value)


@pytest.mark.parametrize('old_query_flag', [False, True])
def test_explicit_speaker_contract_is_nonmutating_monotonic_and_idempotent(old_query_flag):
    old = study.revised_grading_protocol({}, current_query_evidence=old_query_flag)
    before = copy.deepcopy(old)
    new = study.revised_grading_protocol(old, explicit_speaker_evidence=True)
    assert old == before
    assert study.MEMORY_CURRENT_QUERY_CLARIFICATION_V1 in new['memory_prompt']
    assert new['memory_prompt'].endswith(study.MEMORY_SPEAKER_PERSPECTIVE_CLARIFICATION_V1)
    assert study.revised_grading_protocol(new) == new
    assert study.revised_grading_protocol(new, explicit_speaker_evidence=True) == new
    assert study.revised_grading_protocol(new, current_query_evidence=True) == new
    assert study.protocol_sha(old) != study.protocol_sha(new)
    assert study.grading_identity(old)['tutor_contract_prompt_sha256'] == study.grading_identity(new)['tutor_contract_prompt_sha256']


@pytest.mark.parametrize('case', BANK['cases'], ids=lambda c: c['id'])
def test_every_frozen_boundary_reaches_the_existing_judge_without_label_leak(case):
    protocol = study.revised_grading_protocol({}, explicit_speaker_evidence=True)
    router = RequestCapture()
    with pytest.raises(CapturedRequest):
        study.judge_memory(router, {'app': case['application']}, case['source_absent'],
            {'query': case['query'], 'context': case['context']}, case['reply'], case['reference'], protocol=protocol)
    assert router.messages[0].content == protocol['memory_prompt']
    assert json.loads(router.messages[1].content) == dict(APPLICATION=case['application'], QUERY=case['query'],
        VISIBLE_CONTEXT=case['context'], REPLY=case['reply'], SOURCE_ABSENT=case['source_absent'], WITHHELD_REFERENCE=case['reference'])
    assert router.kwargs['temperature'] == 0 and router.kwargs['max_tokens'] == 300


def test_adjudication_changes_one_prospective_relevance_label_and_retains_original_failures():
    old = study.read(study.FIXTURES / 'memory-query-boundaries-v1.json')
    lookup = {c['id']: c for c in BANK['cases']}
    changes = []
    for c in old['cases']:
        new = lookup[c['id']]
        for k in ('query', 'context', 'reply', 'reference', 'source_absent'):
            assert new[k] == c[k]
        for k in study.MEMORY_FIELDS:
            if new['expected_criteria'][k] != c['expected_criteria'][k]:
                changes.append((c['id'], k, c['expected_criteria'][k], new['expected_criteria'][k]))
    assert changes == [('current-presupposition-with-clarification', 'addresses_request', True, False)]
    assert lookup['explicit-other-speaker']['expected_criteria']['critical_error']
    assert lookup['observed-smoke-math-1-scaled_focus-absent-false']['expected_criteria']['critical_error']
    assert BANK['adjudication']['frozen_historical_score_changed'] is False


def test_old_profiles_keep_their_hashes_and_cannot_qualify_the_new_contract(real_sources):
    old = study.local_profile(study.FIXTURES / 'hybrid-screen-v4.json', real_sources[0])
    pin = study.read(study.FIXTURES / 'grading-contract-v2.json')
    assert study.grading_identity(old) == {k: v for k, v in pin.items() if k != 'schema'}
    new = study.revised_grading_protocol(old, explicit_speaker_evidence=True)
    with pytest.raises(ValueError, match='pre-model freeze'):
        study.calibration_items(new, real_sources[2])


def test_actual_v5_checkpoint_keeps_legacy_context_bytes_and_rejects_v6_resume(tmp_path):
    target = tmp_path / 'checkpoint'
    shutil.copytree(LEGACY, target)
    manifest = longitudinal.TrajectoryManifest.model_validate(study.read(target / 'manifest.json'))
    before = study.read(target / 'checkpoint.json')['payload']['ledger']
    assert all('context_format_version' not in r for r in before)
    # The shared ledger validator still authenticates original v1 context;
    # changing the memory policy must not authorize a v5 trajectory resume.
    state = longitudinal._replay(manifest, before)
    assert state.sessions_completed == 2
    with pytest.raises(ValueError, match='configuration mismatch'):
        longitudinal.load_checkpoint(target / 'checkpoint.json', manifest)
    assert study.read(target / 'checkpoint.json')['payload']['ledger'] == before


@pytest.mark.parametrize('version', [2, 3, 4])
def test_authenticated_source_snapshots_replay_with_each_supported_explicit_format(version):
    manifest = longitudinal.TrajectoryManifest.model_validate(study.read(LEGACY / 'manifest.json'))
    before = study.read(LEGACY / 'checkpoint.json')['payload']['ledger']
    ledger = copy.deepcopy(before)
    for record in ledger:
        record['context_format_version'] = version
        record['contexts'] = [render_context_block(
            pam_import({**pam_export(()), 'fragments': snapshot}),
            assistant_speaker='companion', format_version=version)
            for snapshot in record['context_fragments']]
    assert longitudinal._replay(manifest, ledger) == longitudinal._replay(manifest, before)
    assert study.read(LEGACY / 'checkpoint.json')['payload']['ledger'] == before


@pytest.mark.parametrize('version', [True, None, '2', 0, 3, 2])
def test_old_context_cannot_claim_a_different_or_unknown_format(tmp_path, version):
    target = tmp_path / 'checkpoint'
    shutil.copytree(LEGACY, target)
    manifest = longitudinal.TrajectoryManifest.model_validate(study.read(target / 'manifest.json'))
    path = target / 'checkpoint.json'
    envelope = study.read(path)
    envelope['payload']['ledger'][1]['context_format_version'] = version
    envelope['sha256'] = longitudinal.digest(envelope['payload'])
    path.write_text(json.dumps(envelope), encoding='utf-8')
    with pytest.raises(ValueError, match='context'):
        longitudinal._replay(manifest, envelope['payload']['ledger'])
