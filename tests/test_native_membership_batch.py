"""Declared transport fixtures exercise actual deadlines; no model calls."""
import pytest

from psychology_base.eval.benchmarks import local_membership_diagnostic as executor
from psychology_base.eval.benchmarks.membership_focus import grade
from psychology_base.llm.execution import current_operation
from psychology_base.llm.local_backend import OwnedLlamaServer
from psychology_base.llm.providers.llama_cpp import LlamaCppProvider
from tests.test_local_membership_diagnostic import setup
from tests.test_native_local_backend import native_spec, software_backend


def batch(tmp_path, monkeypatch, native_spec, *, operation_scope='condition', wall=3600):
    state = setup(tmp_path, monkeypatch)
    packet = state.read(state.out/'inputs.json')
    first = packet['schedule'][0]
    packet['schedule'] = [first, dict(first, id=first['id']+'/second')]
    state.save(state.out/'inputs.json', packet)
    plan = state.read(state.plan_path)
    plan.update(transport='llama_cpp', native_backend=native_spec.record(), think=True,
        reasoning_budget=True, model=native_spec.model.model_id, digest=native_spec.model.source_digest,
        input_sha256=state.args['archive'].base.sha(state.out/'inputs.json'))
    if operation_scope is not None:
        plan['native_operation_scope'] = operation_scope
    plan['limits'].update(max_native_attempts=2, max_metadata_attempts=14,
        max_wall_s=wall, metadata_timeout_s=60)
    state.save(state.plan_path, plan)
    state.args['grade'] = lambda c,r,s,o: grade(c,r,s,o,contrast_pairs=[])
    provider, wire, _ = software_backend(native_spec, tmp_path, monkeypatch)
    lifecycle = []
    def start(self):
        lifecycle.append('start'); self.started = True
        self.memory_admission = dict(available_bytes=10**10,
            required_bytes=native_spec.cold_memory_minimum_bytes,
            exact_model_context_resident=False, admitted=True)
        return self
    def close(self):
        lifecycle.append('close'); self.closed = True
        self.cleanup = dict(verified_exit=True, job_verified_empty=True)
    monkeypatch.setattr(OwnedLlamaServer, 'start', start)
    monkeypatch.setattr(OwnedLlamaServer, 'close', close)
    return state, wire, lifecycle


@pytest.mark.parametrize('scope,complete,attempts', [('condition',True,2), ('scope',False,2), (None,False,2)])
def test_two_slow_conditions_get_distinct_operations_only_when_explicitly_enabled(
        tmp_path, monkeypatch, native_spec, scope, complete, attempts):
    state, wire, lifecycle = batch(tmp_path,monkeypatch,native_spec,operation_scope=scope)
    clock = [executor.time.monotonic()]
    monkeypatch.setattr(executor.time, 'monotonic', lambda:clock[0])
    original = LlamaCppProvider._open
    operations = []
    def advance(self, request, *, timeout):
        result = original(self,request,timeout=timeout)
        if request.full_url.endswith('/v1/chat/completions'):
            op = current_operation.get()
            operations.append((op.id, op.deadline-clock[0], timeout))
            clock[0] += 160
        return result
    monkeypatch.setattr(LlamaCppProvider,'_open',advance)
    executor.run_local_membership_diagnostic(**state.args)
    result = state.read(state.result_path)
    assert result['completed'] is complete and result['native_attempts'] == attempts
    assert lifecycle == ['start','close']
    if complete:
        assert operations[0][0] != operations[1][0]
        assert all(remaining == 300 and 0 < timeout <= 180 for _,remaining,timeout in operations)
        assert result['graded']['summary']['all']['correct_conditions'] == 2
    else:
        assert result['graded']['summary']['all']['failed'] == 1
        assert operations[0][0] == operations[1][0]
        assert operations[1][1] == 140


def test_per_condition_deadlines_do_not_reset_the_total_scope_limit(tmp_path,monkeypatch,native_spec):
    state, wire, lifecycle = batch(tmp_path,monkeypatch,native_spec,wall=630)
    clock = [executor.time.monotonic()]
    monkeypatch.setattr(executor.time,'monotonic',lambda:clock[0])
    original = LlamaCppProvider._open
    def advance(self,request,*,timeout):
        result = original(self,request,timeout=timeout)
        if request.full_url.endswith('/v1/chat/completions'):clock[0] += 160
        return result
    monkeypatch.setattr(LlamaCppProvider,'_open',advance)
    executor.run_local_membership_diagnostic(**state.args)
    result = state.read(state.result_path)
    assert result['completed'] is False and result['native_attempts'] == 1
    assert result['graded']['summary']['all']['missing'] == 1
    assert 'Insufficient allowance' in result['error']
    assert lifecycle == ['start','close']


@pytest.mark.parametrize('value', [True,False,None,'all','conditions',1,[]])
def test_invalid_operation_scope_fails_before_scope_consumption(tmp_path,monkeypatch,native_spec,value):
    state,wire,lifecycle = batch(tmp_path,monkeypatch,native_spec)
    plan = state.read(state.plan_path); plan['native_operation_scope'] = value
    state.save(state.plan_path,plan)
    with pytest.raises(ValueError):executor.run_local_membership_diagnostic(**state.args)
    assert not wire and not lifecycle and not (state.out/'started.json').exists()


def test_ollama_cannot_opt_into_native_batch_deadlines(tmp_path,monkeypatch):
    state = setup(tmp_path,monkeypatch)
    plan = state.read(state.plan_path); plan['native_operation_scope'] = 'condition'
    with pytest.raises(ValueError):executor.diagnostic_router(plan)
    assert not state.wire


def test_actual_stage_executor_stops_on_wrong_labels_and_keeps_all_control_denominators(
        tmp_path,monkeypatch,native_spec):
    from tests.test_native_semantics_packet import scope, sources
    state,wire,lifecycle = batch(tmp_path,monkeypatch,native_spec)
    packet,refs = scope.build_packet(*sources())
    state.save(state.out/'inputs.json',packet); state.save(state.out/'references.json',refs)
    plan = state.read(state.plan_path)
    plan.update(input_sha256=state.args['archive'].base.sha(state.out/'inputs.json'),
        references_sha256=state.args['archive'].base.sha(state.out/'references.json'),stop_on_label_error=True)
    plan['limits'].update(max_native_attempts=6,max_metadata_attempts=22)
    state.save(state.plan_path,plan)
    state.args.update(rows_for=scope.rows_for,grade=scope.score_owner(packet))
    executor.run_local_membership_diagnostic(**state.args)
    result = state.read(state.result_path)
    assert result['completed'] is False and result['stop_reason'] == 'semantic_gate'
    assert result['native_attempts'] == 1 and lifecycle == ['start','close']
    assert result['graded']['stage'] == dict(planned=6,correct=0,passed=False)
    assert result['graded']['summary']['all']['planned_pairs'] == 72
    assert result['graded']['summary']['all']['missing'] == 35
    assert result['graded']['model_only']['planned'] == 70
    assert result['graded']['model_only']['measured'] == 2
    assert result['graded']['model_only']['unsafe_inclusions'] == 1
    assert len([e for e in wire if e[0]=='/v1/chat/completions']) == 1
