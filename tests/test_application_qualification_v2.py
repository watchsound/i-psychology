"""Source-disjoint qualification and rubric routing; controls are not model evidence."""
import copy
import importlib.util
import json
from argparse import Namespace

import pytest

from psychology_base.eval.benchmarks import application_context as study
from psychology_base.eval.benchmarks import chinese_retrieval as shared
from psychology_base.llm import load_config
from tests.test_application_context_study import real_sources


PROFILE = study.FIXTURES/'hybrid-screen-v2.json'


def revised(real_sources):
    return study.local_profile(PROFILE, real_sources[0])


def test_revised_rubric_keeps_original_rules_thresholds_model_and_budget(real_sources):
    old = study.local_profile(study.FIXTURES/'hybrid-screen-v1.json', real_sources[0])
    new = revised(real_sources)
    assert shared.stable_key(study.MEMORY_PROMPT) == real_sources[0]['scoring_prompt_sha256']
    assert 'memory_prompt' not in old and new['memory_prompt'].startswith(study.MEMORY_PROMPT+'\n')
    for key in ('generator','judge','budget','samples','gates','source_pins','annotations','catalog_sha256'):
        assert new[key] == old[key]
    assert new['calibration_revision'] == 2 and study.protocol_sha(new) != study.protocol_sha(old)
    identity = study.qualification_identity(new)
    assert identity['judge_scoring_prompt_sha256'] == shared.stable_key(new['memory_prompt'])
    assert identity['qualification_catalog_sha256'] == shared.digest(study.FIXTURES/'qualification-v2.json')
    assert study.qualification_identity(old) == {}


def test_fresh_controls_are_balanced_and_disjoint_from_every_reserved_family(real_sources):
    base, cases, sources = real_sources
    memory, tutor = study.calibration_items(revised(real_sources), sources)
    old_memory, old_tutor = study.calibration_items(base, sources)
    old_smoke = study.smoke_cases(base, sources)
    assert len(memory) == len(tutor) == 24
    assert len({r['source'] for r in memory}) == 12
    main_msc = {sources.data[c['source']][c['row']].conversation_id for c in cases if '/msc/' in c['source']}
    assert {r['source'] for r in memory}.isdisjoint(main_msc | {r['source'] for r in old_memory})
    normalize = lambda text: ' '.join(text.casefold().split())
    assert {normalize(r['reference']) for r in memory}.isdisjoint(normalize(r['reference']) for r in old_memory)
    reserved_problems = {normalize(sources.data[c['source']][c['row']].problem)
                         for c in cases+old_smoke if '/mathdial/' in c['source']}
    reserved_problems.update(normalize(r['problem']) for r in old_tutor)
    reserved_math_ids = {sources.data[c['source']][c['row']].conversation_id
                         for c in cases+old_smoke if '/mathdial/' in c['source']}
    reserved_math_ids.update(r['source'] for r in old_tutor)
    assert len({normalize(r['problem']) for r in tutor}) == 24
    assert {normalize(r['problem']) for r in tutor}.isdisjoint(reserved_problems)
    assert len({r['source'] for r in tutor})==24
    assert {r['source'] for r in tutor}.isdisjoint(reserved_math_ids)
    for lang in ('en','zh'):
        rows = [r for r in memory if r['language']==lang]
        assert len(rows) == 12 and sum(r['expected_success'] for r in rows) == 6
        assert sum(not r['expected_success'] for r in rows) == 6
    assert sum(r['gold']=='socratic' for r in tutor) == sum(r['gold']=='direct_answer' for r in tutor) == 12
    assert study.calibration_catalog(memory,tutor) == study.read(study.FIXTURES/'qualification-v2.json')


def test_fresh_controls_preserve_actual_source_words_and_native_tutor_labels(real_sources):
    from psychology_base.eval.benchmarks.public_datasets.mathdial_empath_scorer import gold_mathdial_to_compliance
    from psychology_base.eval.benchmarks.public_datasets.mathdial_fade_harness import extract_final_number
    _, _, sources = real_sources
    memory, tutor = study.calibration_items(revised(real_sources), sources)
    msc = sources.data['.data/msc/train.jsonl']
    math = sources.data['.data/mathdial/train.jsonl']
    for row in memory:
        assert row['authored_control'] is True
        source=msc[row['source_row']]; turn=source.sessions[0].turns[row['source_turn']]
        assert source.conversation_id==row['source'] and turn.speaker_idx==0 and turn.utterance==row['reference']
        assert row['capture']['context'] == ('' if row['source_absent'] else
            ('assistant said: ' if row['control_kind']=='speaker_reversal' else 'user said: ')+row['reference'])
    for row in tutor:
        source = math[row['source_row']]
        assert source.conversation_id==row['source'] and row['problem'] == source.problem
        if row['origin']=='native_mathdial_move':
            turn=source.turns[row['source_turn']]
            assert turn.role=='teacher' and turn.utterance==row['response'] and turn.move is not None
            assert gold_mathdial_to_compliance(turn.move).value==row['gold']
        else:
            assert row['origin']=='authored_chinese_control_on_disjoint_problem'
            assert row['source_turn'] is None
            assert row['response'] == ('请解释一下你为什么选择这一步，好吗？' if row['gold']=='socratic' else
                f'最终答案就是{extract_final_number(source.ground_truth)}。')


def test_original_48_controls_match_the_archived_pre_revision_owner(real_sources):
    path = study.ROOT/'.data/application_context/hybrid-screen-inputs-v1/src/psychology_base/eval/benchmarks/application_context.py'
    if not path.exists(): pytest.skip('Archived local source snapshot unavailable')
    freeze = study.read(study.ROOT/'docs/experiments/phase-9-application-context/hybrid-screen-input-freeze.json')
    assert shared.digest(path) == freeze['code_sha256']['src/psychology_base/eval/benchmarks/application_context.py']
    spec = importlib.util.spec_from_file_location('archived_application_context', path)
    legacy = importlib.util.module_from_spec(spec); spec.loader.exec_module(legacy)
    legacy.ROOT, legacy.FIXTURES = study.ROOT, study.FIXTURES
    legacy.load_inputs.__defaults__=(study.FIXTURES/'protocol.json',study.FIXTURES/'catalog.json')
    base, _, sources = real_sources
    assert study.calibration_items(base,sources) == legacy.calibration_items(base,sources)


@pytest.mark.parametrize('mutation',['rubric_hash','catalog_hash','path','legacy_schema','budget'])
def test_revised_profile_rejects_unfrozen_inputs_before_execution(real_sources,tmp_path,mutation):
    profile = study.read(PROFILE)
    if mutation=='rubric_hash': profile['memory_rubric']['sha256'] = '0'*64
    elif mutation=='catalog_hash': profile['qualification']['sha256'] = '0'*64
    elif mutation=='path': profile['qualification']['path'] = '../other-controls.json'
    elif mutation=='legacy_schema': profile['schema'] = study.HYBRID_SCHEMA
    else: profile['budget']['max_usd'] = '2'
    path=tmp_path/'profile.json'; path.write_text(json.dumps(profile),encoding='utf-8')
    with pytest.raises(ValueError): study.local_profile(path,real_sources[0])


@pytest.mark.parametrize('mutation',['label','source','hash','count'])
def test_catalog_cannot_relabel_replace_or_omit_actual_controls(real_sources,monkeypatch,mutation):
    new = revised(real_sources); original_read=study.read
    wrong=study.read(study.FIXTURES/'qualification-v2.json')
    if mutation=='label': wrong['memory_controls'][0]['expected_success'] = not wrong['memory_controls'][0]['expected_success']
    elif mutation=='source': wrong['memory_controls'][0]['source'] = 'different-source'
    elif mutation=='hash': wrong['tutor_controls'][0]['control_sha256'] = '0'*64
    else: wrong['memory_controls'].pop()
    monkeypatch.setattr(study,'read',lambda path: wrong if str(path).endswith('qualification-v2.json') else original_read(path))
    with pytest.raises(ValueError,match='pre-model freeze'): study.calibration_items(new,real_sources[2])


@pytest.mark.skipif(importlib.util.find_spec('jieba') is None,reason='CLI requires the frozen jieba runtime')
def test_corrupted_control_catalog_blocks_preflight_before_provider_creation(real_sources,tmp_path,monkeypatch):
    def reject(*args): raise ValueError('Qualification controls differ from their pre-model freeze')
    monkeypatch.setattr(study,'calibration_items',reject)
    monkeypatch.setattr(study,'make_routers',lambda *a,**kw: pytest.fail('Corrupt controls reached provider creation'))
    with pytest.raises(ValueError,match='pre-model freeze'):
        study.run(Namespace(command='application-preflight',application_profile=PROFILE,
            output=tmp_path/'preflight.json',llm_config=None,application_resume=None))


def test_new_calibration_uses_revised_prompt_and_cannot_promote_a_control_response(real_sources,tmp_path):
    from tests.test_application_hybrid_screen import hybrid_judge
    _, judge, budget, requests, _ = hybrid_judge(tmp_path)
    new = revised(real_sources)
    report = study.calibration(new,real_sources[2],judge)
    report.update(code_sha256=study.fingerprints(),protocol_sha256=study.protocol_sha(new))
    assert requests[0]['messages'][0]['content'] == new['memory_prompt']
    assert not report['passed'] and report['stopped_early'] and len(requests)==1
    assert study.validate_calibration(report,new,real_sources[2]) is False
    for key in study.qualification_identity(new):
        altered=copy.deepcopy(report); altered[key]='wrong-identity'
        with pytest.raises(ValueError,match='rubric/qualification'):
            study.validate_calibration(altered,new,real_sources[2])
    budget.writer.close()


@pytest.mark.skipif(importlib.util.find_spec('jieba') is None,reason='Real jieba backend required')
@pytest.mark.parametrize('app',['companion','tutor'])
def test_normal_application_trial_forwards_the_revised_rubric(real_sources,tmp_path,app):
    from tests.test_application_call_audit import audited
    new=revised(real_sources); _,cases,sources=real_sources
    case=next(c for c in cases if c['app']==app and c['stratum']=='native_zh_bridge')
    generation,_,gb=audited(tmp_path/'g',calls=2)
    judge,_,jb=audited(tmp_path/'j',calls=2)
    generation.max_input=judge.max_input=12000
    row=study.trial(case,0,0,'scaled_focus',False,sources,load_config(),generation,judge,tmp_path,protocol=new)
    gb.writer.close(); jb.writer.close()
    reservations=[r for line in jb.writer.path.read_text(encoding='utf-8').splitlines() if (r:=json.loads(line))['event']=='reservation']
    assert reservations[0]['messages'][0]['content']==new['memory_prompt']
    assert row['capture']['fragments'] and row['exchange_id'] and row['proxy']
    assert row['memory_judgment']['parse_failed'] and not row['scores']['resolved']


@pytest.mark.skipif(importlib.util.find_spec('jieba') is None,reason='Real jieba backend required')
def test_smoke_uses_same_rubric_and_reports_controls_as_unqualified(real_sources,tmp_path):
    from tests.test_application_call_audit import audited
    new=copy.deepcopy(revised(real_sources)); sources=real_sources[2]
    # Explicit software-only providers permit replay of real source/application
    # plumbing. Proxy markers and failed parsing prohibit quality claims.
    new['screening']=False
    new['generator'].update(provider='control',model='declared-control')
    generation,_,gb=audited(tmp_path/'g',calls=8)
    judge,_,jb=audited(tmp_path/'j',calls=10)
    generation.max_input=judge.max_input=12000
    result=study.smoke_run(new,sources,load_config(),generation,judge,tmp_path)
    gb.writer.close(); jb.writer.close()
    requests=[r for line in jb.writer.path.read_text(encoding='utf-8').splitlines() if (r:=json.loads(line))['event']=='reservation']
    memory=[r for r in requests if r['messages'][0]['content'].startswith(study.MEMORY_PROMPT)]
    assert len(memory)==6 and all(r['messages'][0]['content']==new['memory_prompt'] for r in memory)
    assert len(requests)==8 and all(r['proxy'] for r in result['trials']) and not result['passed']
