"""No model substitutes: packet identity, source scope and executor contracts."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def run():
    path = Path('docs/experiments/phase-9-memory-navigation/run-prompt-interference-v1.py')
    spec = importlib.util.spec_from_file_location('prompt_interference_controls', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_only_companion_instructions_are_removed_from_exact_paired_messages(run):
    from psychology_base.applications.emotional.companion_agent import CompanionSystemPrompt
    from psychology_base.applications.context import RUNTIME_AMBIGUITY_POLICY_V6
    probes, references, packets = run.build()
    assert len(probes) == 20 and len(references) == len(packets) == 8
    assert sum(p['repeat'] == 1 for p in probes) == 4
    for packet in packets:
        full, minimal = packet['messages'], packet['minimal_messages']
        assert full[1:] == minimal[1:]
        assert full[0]['content'] == CompanionSystemPrompt.SYSTEM_PROMPT + minimal[0]['content']
        assert minimal[0]['content'] == '\n\n' + RUNTIME_AMBIGUITY_POLICY_V6 + '\nResponding application role: companion.'
        assert packet['id'] not in full[1]['content']
    assert {p['arm'] for p in probes[:2]} == {'full', 'minimal'}
    assert all(p['max_tokens'] == 160 for p in probes)


def test_layout_counterfactual_keeps_actual_fragment_ids_and_source_coordinates(run):
    import json
    probes, refs, packets = run.build()
    for source, _ in run.SOURCES[:2]:
        a, b = [p for p in packets if p['parent_case'] == source]
        assert a['selected_ids'] == b['selected_ids'][::-1]
        def records(packet):
            block = packet['messages'][1]['content'].split('[CONTEXT]\n', 1)[1].rsplit('\n\n[USER]\n', 1)[0]
            return [json.loads(line) for line in block.splitlines()]
        old, reverse = records(a), records(b)
        # Computed row references must move; original source identities cannot.
        assert [{k:v for k,v in r.items() if k not in ('row','source_relations')} for r in old] == [
            {k:v for k,v in r.items() if k not in ('row','source_relations')} for r in reverse[::-1]]
        assert [r['row'] for r in old] == [r['row'] for r in reverse] == [0, 1]
        assert old[0]['source_relations'] != reverse[0]['source_relations']
        expected = [r['expectation'] for r in refs if r['parent_case'] == source]
        assert expected[0] == expected[1]


@pytest.mark.parametrize('mutation', ['query', 'coordinates', 'policy', 'role'])
def test_extra_paired_treatments_cannot_pass_packet_validation(run, mutation):
    packets = run.build()[2]
    run.validate_pairs(packets)
    changed = deepcopy(packets)
    target = changed[0]['minimal_messages']
    if mutation == 'query': target[1]['content'] += ' altered query'
    elif mutation == 'coordinates': target[1]['content'] = target[1]['content'].replace('"position":8', '"position":7')
    elif mutation == 'policy': target[0]['content'] += '\nAssume the last row was later.'
    else: target[0]['role'] = 'user'
    with pytest.raises(ValueError, match='Only the companion'):
        run.validate_pairs(changed)


def test_shared_scope_and_operation_restore_on_executor_failure(run, tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'OUT', tmp_path / 'scope')
    monkeypatch.setattr(run, 'PLAN', tmp_path / 'plan.json')
    run.freeze()
    run.load_frozen()
    parent, base = run.parent, run.base
    old = (parent.OUT, parent.PLAN, parent.RESULT, parent.MAX_CALLS, base.OUT, base.PLAN, base.SPEC, base.save,
           parent.profile.OUT, parent.profile.PLAN, parent.profile.RESULT, parent.profile.recording_transport)
    def fail():
        assert run.current_operation.get().id == 'prompt-interference-v1'
        assert parent.profile.OUT == run.OUT
        raise RuntimeError('Declared software failure before any model call')
    monkeypatch.setattr(parent.profile, 'execute', fail)
    with pytest.raises(RuntimeError): run.execute()
    assert run.current_operation.get() is None
    assert (parent.OUT, parent.PLAN, parent.RESULT, parent.MAX_CALLS, base.OUT, base.PLAN, base.SPEC, base.save,
           parent.profile.OUT, parent.profile.PLAN, parent.profile.RESULT, parent.profile.recording_transport) == old


@pytest.mark.parametrize('mutation', ['tokens', 'budget', 'policy', 'grading', 'parent', 'archive', 'inputs'])
def test_changed_scope_rejects_before_provider_construction(run, tmp_path, monkeypatch, mutation):
    import json
    monkeypatch.setattr(run, 'OUT', tmp_path / 'scope')
    monkeypatch.setattr(run, 'PLAN', tmp_path / 'plan.json')
    run.freeze()
    run.load_frozen()
    plan = run.base.read(run.PLAN)
    if mutation == 'tokens': plan['probes'][0]['max_tokens'] = 161
    elif mutation == 'budget': plan['protocol']['budget']['max_calls'] = 21
    elif mutation == 'policy': plan['probes'][0]['messages'][0]['content'] += ' altered'
    elif mutation == 'grading': plan['grading']['binding'] = ['correct']
    elif mutation == 'parent': plan['prior_pins'] = {}
    elif mutation == 'archive':
        (run.OUT / 'executed-source' / next(iter(plan['source']))).write_text('changed', encoding='utf-8')
    else: (run.OUT / 'inputs.json').write_text('[]', encoding='utf-8')
    run.PLAN.write_text(json.dumps(plan), encoding='utf-8')
    def forbidden(*args, **kwargs): pytest.fail('Altered inference must not construct a provider')
    monkeypatch.setattr(run.base.study, 'make_routers', forbidden)
    with pytest.raises(ValueError): run.execute()


def test_consumed_diagnostic_cannot_dispatch_twice(run, tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'OUT', tmp_path / 'scope')
    monkeypatch.setattr(run, 'PLAN', tmp_path / 'plan.json')
    run.freeze()
    run.load_frozen()
    (run.OUT / 'started.json').write_text('{}', encoding='utf-8')
    with pytest.raises(FileExistsError): run.execute()
    assert run.current_operation.get() is None
