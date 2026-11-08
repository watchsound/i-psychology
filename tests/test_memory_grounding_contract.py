"""Compatibility against the untouched owner; no model-quality evidence."""
import copy
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SHARED = 'psychology_base.applications.measurement.memory_grounding'
LEGACY = 'psychology_base.eval.benchmarks.application_context'
FIXTURE = json.loads((ROOT / 'benchmarks/fixtures/application_context/memory-grounding-compatibility-v1.json').read_text(encoding='utf-8'))


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()


def outcome(fn, *args, **kwargs):
    try:
        return dict(result_sha256=canonical_sha(fn(*args, **kwargs)))
    except Exception as exc:
        return dict(exception_type=type(exc).__name__, message=str(exc))


@pytest.fixture(params=['legacy', 'shared'])
def owner(request):
    return importlib.import_module(LEGACY if request.param == 'legacy' else SHARED)


def test_all_prompt_constants_keep_frozen_bytes(owner):
    for name, expected in FIXTURE['constants'].items():
        actual = getattr(owner, name)
        if expected['kind'] == 'tuple':
            assert isinstance(actual, tuple) and list(actual) == expected['value'], name
        else:
            assert hashlib.sha256(actual.encode('utf-8')).hexdigest() == expected['utf8_sha256'], name


def test_signatures_keep_the_legacy_defaults_and_keyword_contract(owner):
    for name, signature in FIXTURE['function_signatures'].items():
        assert str(inspect.signature(getattr(owner, name))) == signature


def test_parser_keeps_all_frozen_results_and_rejections(owner):
    for index, case in enumerate(FIXTURE['parser_cases']):
        actual = outcome(owner.parse_memory, case['raw'], case['finish_reason'], case['proxy'],
                         contract_version=case['contract_version'])
        assert actual == case['expected'], (index, case, actual)


def test_protocol_composition_upgrade_and_downgrade_keep_frozen_behavior(owner):
    for index, case in enumerate(FIXTURE['protocol_cases']):
        base = dict(nested=dict(items=[1, True, None]), **case['base_extra'])
        initial = case['initial_options']
        protocol = owner.revised_grading_protocol(base, **initial) if initial is not None else base
        before = copy.deepcopy(protocol)
        actual = outcome(owner.revised_grading_protocol, protocol, **case['options'])
        assert actual == case['expected'], (index, case, actual)
        assert protocol == before
        if 'result_sha256' in actual:
            result = owner.revised_grading_protocol(protocol, **case['options'])
            result['nested']['items'].append('result mutation')
            assert protocol == before


def test_all_boolean_consistency_patterns_keep_the_frozen_contract(owner):
    for case in FIXTURE['consistency_cases']:
        criteria = copy.deepcopy(case['criteria'])
        assert owner.memory_consistency_errors(criteria) == case['expected']
        assert criteria == case['criteria']


def test_contract_version_selection_keeps_strict_types_and_defaults(owner):
    for case in FIXTURE['version_cases']:
        assert outcome(owner.grading_version, case['protocol']) == case['expected']


def test_old_import_path_reexports_the_single_shared_implementation():
    legacy = importlib.import_module(LEGACY)
    shared = importlib.import_module(SHARED)
    for name in (*FIXTURE['constants'], *FIXTURE['function_signatures']):
        assert getattr(legacy, name) is getattr(shared, name), name
    for name in FIXTURE['function_signatures']:
        assert getattr(shared, name).__module__ == SHARED


def test_shared_owner_is_covered_by_existing_source_fingerprints():
    legacy = importlib.import_module(LEGACY)
    path = 'src/psychology_base/applications/measurement/memory_grounding.py'
    assert legacy.fingerprints()[path] == hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def test_shared_owner_imports_without_benchmarks_datasets_sdk_or_network():
    code = '''
import importlib.abc
import socket
import sys

class ForbiddenDependency(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith('psychology_base.eval.benchmarks') or fullname.split('.')[0] in ('openai', 'anthropic'):
            raise AssertionError('Runtime contract imported forbidden dependency: ' + fullname)

def forbidden_network(*args, **kwargs):
    raise AssertionError('Contract import attempted a network connection')

sys.meta_path.insert(0, ForbiddenDependency())
socket.socket.connect = forbidden_network
from psychology_base.applications.measurement.memory_grounding import parse_memory, revised_grading_protocol
assert not parse_memory('{}', 'stop', False)['criteria']
assert revised_grading_protocol({}, speaker_evidence_version=4)['grading_contract_version'] == 2
assert not any(name.startswith('psychology_base.eval.benchmarks') for name in sys.modules)
'''
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True,
                            text=True, encoding='utf-8', timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
