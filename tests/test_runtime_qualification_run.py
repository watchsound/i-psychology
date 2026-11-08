"""Offline authorization, factory and progress checks; no provider quality claims."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

from psychology_base.eval.benchmarks import application_context as study
from psychology_base.llm.providers.openai import OpenAIProvider
from psychology_base.applications.measurement.memory_grounding import runtime_memory_protocol, parse_runtime_memory
from tests.test_runtime_qualification import scripted_review

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def runner(tmp_path,monkeypatch):
    path=ROOT/'benchmarks/scripts/phase9_runtime_qualification_run.py'
    spec=importlib.util.spec_from_file_location('runtime_qualification_run_test',path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    # Pure entry-point tests use a verified historical ledger prefix and a new
    # temporary plan. A consumed live plan must remain rejected, not become a
    # prerequisite for testing nine unrelated drift fields.
    if module.PLAN.exists():
        frozen=module.read(module.PLAN)
        original=module.LEDGER.read_bytes().splitlines(keepends=True)
        ledger=tmp_path/'historical-ledger.jsonl'
        ledger.write_bytes(b''.join(line for line in original if json.loads(line)['dispatch']<=831))
        assert module.sha(ledger)==frozen['baseline_ledger_sha256']
        monkeypatch.setattr(module,'LEDGER',ledger)
        packet,controls,review=module.checked_inputs()
        plan=dict(module.plan_fields(packet,study.load_config(),controls,review,module.baseline()),
                  timestamp_utc=frozen['timestamp_utc'])
        path=tmp_path/'isolated-test-plan.json'
        module.save_new(path,plan)
        monkeypatch.setattr(module,'PLAN',path)
        monkeypatch.setattr(module.validate,'__defaults__',(path,))
        monkeypatch.setattr(module.run,'__defaults__',(path,))
    return module


def require_frozen_environment(runner):
    if not runner.PLAN.exists(): pytest.skip('Local hosted plan has not been frozen')
    expected=runner.read(runner.PLAN)['runtime_identity']
    if study.runtime_identity(study.load_config())!=expected:
        # The actual run must reject another SDK/tokenizer environment. Tests
        # needing the live freeze run in the pinned environment instead.
        with pytest.raises(ValueError): runner.validate()
        pytest.skip('Hosted plan pins the Chinese-enabled environment and installed provider SDK')


def test_consumed_live_plan_rejects_without_a_provider_call():
    path=ROOT/'benchmarks/scripts/phase9_runtime_qualification_run.py'
    spec=importlib.util.spec_from_file_location('consumed_runtime_qualification_test',path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    if not module.PLAN.exists(): pytest.skip('Optional frozen hosted plan unavailable')
    with pytest.raises(ValueError,match='baseline|drift|budget'):
        module.validate()


def test_judge_only_factory_never_requires_or_constructs_a_generator(monkeypatch):
    created=[]
    class FactoryCapture:
        def __init__(self,cfg):
            created.append(cfg)
            self.providers={'openai':OpenAIProvider()}
    monkeypatch.setattr(study,'LLMRouter',FactoryCapture)
    spec=study.read(study.FIXTURES/'hybrid-screen-v4.json')['judge']
    result=study.make_routers(dict(screening=True,judge=spec),study.load_config(),object(),roles=('judge',))
    assert len(result)==len(created)==1
    assert result[0].model==spec['model'] and result[0].remote_spec==spec
    assert created[0].active_provider=='openai' and set(created[0].providers)=={'openai'}
    assert not created[0].cache.enabled and not created[0].local_ollama.enabled


@pytest.mark.parametrize('roles',[(),('judge','judge'),('unknown',)])
def test_bad_factory_role_scope_stops_before_configuration_access(roles):
    with pytest.raises(ValueError,match='roles'):
        study.make_routers({},object(),object(),roles=roles)


def test_progress_callback_gets_a_copy_and_cannot_change_the_gate(monkeypatch):
    path=ROOT/'.data/application_context/runtime-qualification-v1/sources.jsonl'
    if not path.exists(): pytest.skip('Optional real source packet unavailable')
    controls=study.runtime_qualification_items(); protocol=runtime_memory_protocol()
    protocol['judge']=study.read(study.FIXTURES/'hybrid-screen-v4.json')['judge']
    judge=object.__new__(study.RecordingRouter)
    judge.controls=False; judge.provider='openai'; judge.model=protocol['judge']['model']; judge.max_input=12000
    judge.remote_spec=protocol['judge']; judge.local_spec=None
    def assessed(*args,**kwargs):
        return dict(judgment=parse_runtime_memory('{}','stop',False),observed_success=False,call={'software_control':True})
    monkeypatch.setattr(study,'assess_calibration_control',assessed)
    monkeypatch.setattr(study,'validate_runtime_qualification',lambda *args,**kwargs:False)
    seen=[]
    def save_progress(row):
        seen.append(copy.deepcopy(row))
        row['judgment']['parse_failed']=False
    report=study.runtime_qualification(controls,protocol,judge,review=scripted_review(controls),on_assessment=save_progress)
    assert len(seen)==len(report['rows'])==1
    assert report['rows'][0]['judgment']['parse_failed'] is True
    assert not report['passed'] and report['metrics']['irreversible_failure']


@pytest.mark.parametrize('approval',[{'approved':False},{'approved':'true'},{'approved':True,'plan_sha256':'wrong'}])
def test_unapproved_run_never_enters_validation_or_provider_code(tmp_path,monkeypatch,runner,approval):
    plan=tmp_path/'plan.json'; plan.write_text('{}',encoding='utf-8')
    receipt=tmp_path/'approval.json'; receipt.write_text(json.dumps(approval),encoding='utf-8')
    def forbidden(*args,**kwargs): pytest.fail('Approval must be checked first')
    monkeypatch.setattr(runner,'validate',forbidden)
    with pytest.raises(ValueError,match='approval'):
        runner.run(receipt,plan)


def test_frozen_plan_rebuilds_the_exact_reviewed_messages_and_budget(runner):
    require_frozen_environment(runner)
    plan,_,controls,review,protocol=runner.validate()
    assert len(controls)==40 and len(study.runtime_qualification_schedule(controls))==64
    assert study.validate_runtime_qualification_review(controls,review)
    assert protocol['budget']['max_calls']==895
    assert runner.Decimal(plan['additional_conservative_usd_cap'])==runner.Decimal('0.1056')
    assert plan['endpoint']=='https://api.openai.com/v1'
    assert protocol['judge']['reasoning_effort']=='none' and protocol['judge']['json_mode'] is True


@pytest.mark.parametrize('field,value',[
    ('additional_attempts',65),('shared_attempt_cap',896),
    ('additional_conservative_usd_cap','1'),('endpoint','https://example.com/v1'),
    ('runner_sha256','wrong'),('baseline_ledger_sha256','wrong'),('review_sha256','wrong'),
    ('requests_sha256','wrong'),('candidate',{'model':'different'}),
])
def test_plan_drift_rejects_offline(tmp_path,runner,field,value):
    require_frozen_environment(runner)
    plan=runner.read(runner.PLAN); plan[field]=value
    path=tmp_path/'drift.json'; path.write_text(json.dumps(plan),encoding='utf-8')
    with pytest.raises(ValueError): runner.validate(path)


@pytest.mark.parametrize('abort',[False,True])
def test_scripted_execution_persists_progress_or_abort_without_a_provider_call(tmp_path,monkeypatch,runner,abort):
    """Entry-point control flow only; the qualification owner has separate tests."""
    require_frozen_environment(runner)
    from contextlib import contextmanager
    from types import SimpleNamespace
    frozen=runner.validate()
    plan,cfg,controls,review,protocol=frozen
    monkeypatch.setattr(runner,'validate',lambda *args:frozen)
    directory=tmp_path/'scripted-run'
    monkeypatch.setattr(runner,'RUN_DIR',directory)
    before=plan['baseline']
    budget=SimpleNamespace(attempts=before['provider_attempts'],dispatches=before['dispatches'],
        pending=set(),spent=runner.Decimal(before['accounted_usd']),cache_hits=before['cache_hits'])
    budget.summary=lambda:dict(scripted_control=True,provider_attempts=budget.attempts,accounted_usd=str(budget.spent))
    @contextmanager
    def locked(*args): yield budget
    monkeypatch.setattr(study,'study_budget',locked)
    monkeypatch.setattr(study,'make_routers',lambda *args,**kwargs:[object()])
    def forbidden(*args,**kwargs): pytest.fail('A scripted execution test must never call a provider')
    monkeypatch.setattr(OpenAIProvider,'chat',forbidden)
    def simulated(*args,**kwargs):
        row=dict(index=0,control='SCRIPTED-ONLY',repeat=1,software_control=True)
        kwargs['on_assessment'](row)
        budget.attempts+=1; budget.dispatches+=1; budget.spent+=runner.Decimal('0.000001')
        if abort: raise RuntimeError('placeholder exception text must not be copied to an abort receipt')
        return dict(rows=[row],passed=False,software_control=True)
    monkeypatch.setattr(study,'runtime_qualification',simulated)
    approval=tmp_path/'scripted-approval.json'
    approval.write_text(json.dumps(dict(approved=True,plan_sha256=runner.sha(runner.PLAN))),encoding='utf-8')
    if abort:
        with pytest.raises(RuntimeError): runner.run(approval)
        result=json.loads((directory/'abort.json').read_text(encoding='utf-8'))
        assert result['exception_type']=='RuntimeError' and result['automatic_retry'] is False
        assert 'placeholder' not in json.dumps(result)
    else:
        runner.run(approval)
        result=json.loads((directory/'result.json').read_text(encoding='utf-8'))
        assert result['judgments']==1 and result['passed'] is False
        assert json.loads((directory/'report.json').read_text(encoding='utf-8'))['software_control'] is True
    assert len((directory/'judgments.jsonl').read_text(encoding='utf-8').splitlines())==1
