"""Context/reuse regression controls; injected outputs are not judge evidence."""
import copy
import importlib.util
import json
from argparse import Namespace

import pytest

from psychology_base.eval.benchmarks import application_context as study
from psychology_base.eval.benchmarks import chinese_retrieval as shared
from psychology_base.eval.benchmarks.public_datasets.mathdial import (
    MathDialConversation, MathDialMove, MathDialTurn, iter_mathdial_teacher_contexts,
)
from psychology_base.eval.benchmarks.public_datasets.mathdial_empath_scorer import gold_mathdial_to_compliance
from tests.test_application_context_study import real_sources


PROFILE=study.FIXTURES/'hybrid-screen-v3.json'


def revised(real_sources):
    return study.local_profile(PROFILE,real_sources[0])


def test_teacher_context_is_chronological_and_never_uses_future_or_annotated_work():
    conversation=MathDialConversation('repeated-id','problem','answer','withheld annotation',(
        MathDialTurn('teacher','opening',MathDialMove.FOCUS),
        MathDialTurn('student','first attempt'),
        MathDialTurn('teacher','unlabelled'),
        MathDialTurn('teacher','hint',MathDialMove.PROBING),
        MathDialTurn('student','second attempt'),
        MathDialTurn('student','latest attempt'),
        MathDialTurn('teacher','correction',MathDialMove.TELLING),
        MathDialTurn('teacher','followup',MathDialMove.FOCUS),
        MathDialTurn('student','future attempt'),
    ))
    contexts=list(iter_mathdial_teacher_contexts(conversation))
    assert [c.turn_index for c in contexts]==[0,3,6,7]
    assert [c.student_utterance for c in contexts]==['','first attempt','latest attempt','latest attempt']
    other=copy.copy(conversation)
    assert list(iter_mathdial_teacher_contexts(other))[0].student_utterance==''
    assert all(c.turn is conversation.turns[c.turn_index] for c in contexts)


def test_v3_keeps_memory_rubric_gate_native_mapping_and_cumulative_budget(real_sources):
    new=revised(real_sources)
    old=study.local_profile(study.FIXTURES/'hybrid-screen-v2.json',real_sources[0])
    for key in ('memory_prompt','memory_rubric','generator','judge','budget','gates','samples','source_pins'):
        assert new[key]==old[key]
    assert new['calibration_revision']==3
    assert new['budget']=={'max_calls':600,'max_usd':'1','local_only':False}
    memory,tutor=study.calibration_items(new,real_sources[2])
    assert memory==study.calibration_items(old,real_sources[2])[0]
    for row in tutor[:12]:
        conversation=real_sources[2].data['.data/mathdial/train.jsonl'][row['source_row']]
        context=next(c for c in iter_mathdial_teacher_contexts(conversation) if c.turn_index==row['source_turn'])
        assert row['response']==context.turn.utterance
        assert row['student_utterance']==context.student_utterance
        assert row['native_move']==context.turn.move.value
        assert row['gold']==gold_mathdial_to_compliance(context.turn.move).value
    assert sum(bool(r['student_utterance']) for r in tutor[:12])==5
    assert all(r['student_utterance']=='' and r['native_move'] is None for r in tutor[12:])


def test_fresh_tutoring_excludes_both_observed_qualifications_and_main_smoke(real_sources):
    base,cases,sources=real_sources
    _,tutor=study.calibration_items(revised(real_sources),sources)
    prior=[r for protocol in (base,{'calibration_revision':2,**base})
           for r in study.calibration_items(protocol,sources)[1]]
    reserved=cases+study.smoke_cases(base,sources)
    ids={r['source'] for r in prior}
    normalize=lambda t:' '.join(t.casefold().split())
    problems={normalize(r['problem']) for r in prior}
    for case in reserved:
        if '/mathdial/' in case['source']:
            conversation=sources.data[case['source']][case['row']]
            ids.add(conversation.conversation_id); problems.add(normalize(conversation.problem))
    assert len(tutor)==len({r['source'] for r in tutor})==len({normalize(r['problem']) for r in tutor})==24
    assert {r['source'] for r in tutor}.isdisjoint(ids)
    assert {normalize(r['problem']) for r in tutor}.isdisjoint(problems)
    assert study.calibration_catalog(*study.calibration_items(revised(real_sources),sources))==study.read(study.FIXTURES/'qualification-v3.json')


def test_v2_control_bytes_match_the_tested_archived_builder(real_sources):
    freeze=study.read(study.ROOT/'docs/experiments/phase-9-application-context/hybrid-screen-v2-input-freeze.json')
    owner='src/psychology_base/eval/benchmarks/application_context.py'
    path=study.ROOT/freeze['snapshot_directory']/owner
    assert shared.digest(path)==freeze['code_sha256'][owner]
    spec=importlib.util.spec_from_file_location('archived_v2_context',path)
    archive=importlib.util.module_from_spec(spec); spec.loader.exec_module(archive)
    archive.ROOT,archive.FIXTURES=study.ROOT,study.FIXTURES
    archive.load_inputs.__defaults__=(study.FIXTURES/'protocol.json',study.FIXTURES/'catalog.json')
    protocol=study.local_profile(study.FIXTURES/'hybrid-screen-v2.json',real_sources[0])
    assert study.calibration_items(protocol,real_sources[2])==archive.calibration_items(protocol,real_sources[2])


def test_original_memory_evidence_remains_valid_but_cannot_qualify_changed_owners(real_sources):
    protocol=revised(real_sources)
    original=study.read(study.ROOT/protocol['memory_evidence']['report'])
    prior=study.read(study.ROOT/protocol['memory_evidence']['freeze'])
    old=study.local_profile(study.FIXTURES/'hybrid-screen-v2.json',real_sources[0])
    assert study.validate_calibration(original,old,real_sources[2],expected_code=prior['code_sha256']) is False
    assert len(original['memory_controls'])==24
    assert all(not r['judgment']['parse_failed'] for r in original['memory_controls'])
    assert original['passed'] is False and original['tutor_kappa']<.60
    with pytest.raises(ValueError,match='owners changed'):
        study.reused_memory_controls(protocol,real_sources[2])


@pytest.mark.parametrize('change',['provider','parser','archive','report','runtime'])
def test_reuse_rejects_changed_owners_outputs_or_environment(real_sources,monkeypatch,change):
    protocol=revised(real_sources)
    if change=='provider':
        actual=study.fingerprints(); actual['src/psychology_base/llm/providers/openai.py']='0'*64
        monkeypatch.setattr(study,'fingerprints',lambda:actual)
    elif change=='parser':
        protocol['memory_prompt']+=' changed parser contract'
    elif change in ('archive','report'):
        original=shared.digest
        target=(study.ROOT/protocol['memory_evidence']['report'] if change=='report' else
                study.ROOT/'.data/application_context/hybrid-screen-inputs-v2/src/psychology_base/eval/benchmarks/application_context.py')
        monkeypatch.setattr(shared,'digest',lambda p:'0'*64 if p==target else original(p))
    else:
        monkeypatch.setattr(study,'runtime_identity',lambda cfg, **kwargs:{'config_sha256':'changed'})
    with pytest.raises(ValueError):
        study.reused_memory_controls(protocol,real_sources[2],object() if change=='runtime' else None)


def test_v3_stale_memory_stops_before_any_new_judgment(real_sources,tmp_path):
    from tests.test_application_hybrid_screen import hybrid_judge
    seen=[]
    def response(data):
        seen.append(True)
        if len(seen)==1:
            data['choices'][0]['message']['content']=json.dumps({
                'requests_reasoning':True,'states_final_answer':False,'provides_scaffold_hint':False,
                'corrects_with_fact':False,'affirms_or_greets':False})
    _,judge,budget,requests,_=hybrid_judge(tmp_path,response_change=response)
    protocol=revised(real_sources)
    with pytest.raises(ValueError,match='owners changed'):
        study.calibration(protocol,real_sources[2],judge)
    assert not requests and not seen and budget.attempts==0
    budget.writer.close()


@pytest.mark.parametrize('key',['memory_evidence','qualification'])
def test_v3_profile_rejects_unpinned_evidence(real_sources,tmp_path,key):
    profile=study.read(PROFILE)
    profile[key]['report_sha256' if key=='memory_evidence' else 'sha256']='0'*64
    path=tmp_path/'profile.json'; path.write_text(json.dumps(profile),encoding='utf-8')
    with pytest.raises(ValueError): study.local_profile(path,real_sources[0])


@pytest.mark.skipif(importlib.util.find_spec('jieba') is None,reason='Frozen CLI requires jieba')
def test_reuse_is_checked_before_provider_creation(real_sources,tmp_path,monkeypatch):
    monkeypatch.setattr(study,'reused_memory_controls',lambda *a,**kw:(_ for _ in ()).throw(ValueError('memory runtime changed')))
    monkeypatch.setattr(study,'make_routers',lambda *a,**kw:pytest.fail('Invalid reuse reached provider initialization'))
    with pytest.raises(ValueError,match='memory runtime changed'):
        study.run(Namespace(command='application-preflight',application_profile=PROFILE,
            output=tmp_path/'preflight.json',llm_config=None,application_resume=None))
