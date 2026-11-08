"""Real policy/router/accounting checks without live provider traffic."""
from dataclasses import asdict, replace
import json
import sqlite3
from types import SimpleNamespace

import pytest

from psychology_base.llm.accounting import RuntimeBudget
from psychology_base.llm.execution import ExecutionDeferred, ExecutionPolicy, current_operation
from psychology_base.llm.types import ChatMessage
from tests.test_runtime_budget import allowance


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import socket
    def reject(*args,**kwargs):
        raise AssertionError('Policy tests must not contact a provider')
    monkeypatch.setattr(socket.socket,'connect',reject)
    monkeypatch.setattr(socket,'create_connection',reject)


@pytest.mark.parametrize('mode',['local','paid','research'])
def test_long_deadlines_require_explicit_local_constructor_opt_in(mode):
    with pytest.raises(ValueError,match='300'):
        ExecutionPolicy(mode=mode,request_timeout_s=1200,operation_timeout_s=1200)
    if mode!='local':
        with pytest.raises(ValueError,match='local-only'):
            ExecutionPolicy(mode=mode,offline_local_diagnostic=True)


@pytest.mark.parametrize('value',[None,1,0,'true'])
def test_constructor_opt_in_is_strict_boolean(value):
    with pytest.raises(ValueError):
        ExecutionPolicy(offline_local_diagnostic=value)


@pytest.mark.parametrize('value',[True,0,-1,3600.001,float('inf'),float('nan')])
def test_opt_in_remains_finite_and_bounded(value):
    with pytest.raises(ValueError):
        ExecutionPolicy(offline_local_diagnostic=True,request_timeout_s=value)


@pytest.mark.parametrize('field',['request_timeout_s','operation_timeout_s'])
def test_hour_long_cpu_budget_remains_an_explicit_local_only_constructor_setting(field):
    value={field:3600}
    assert getattr(ExecutionPolicy(offline_local_diagnostic=True,**value),field)==3600
    for mode in ('local','paid','research'):
        with pytest.raises(ValueError):ExecutionPolicy(mode=mode,**value)
        with pytest.raises(ValueError):ExecutionPolicy.from_mapping(dict(mode=mode,**value))


def test_long_policy_is_not_a_serving_configuration_or_implicit_replace_default():
    policy=ExecutionPolicy(request_timeout_s=1200,operation_timeout_s=1200,offline_local_diagnostic=True)
    assert 'offline_local_diagnostic' not in asdict(policy)
    assert policy.mode=='local'
    with pytest.raises(ValueError):
        ExecutionPolicy.from_mapping(dict(mode='local',offline_local_diagnostic=True))
    with pytest.raises(ValueError):
        ExecutionPolicy.from_mapping(asdict(policy))
    with pytest.raises(ValueError):
        replace(policy)
    assert replace(policy,offline_local_diagnostic=True)==policy
    normal=ExecutionPolicy.from_mapping(dict(mode='local',request_timeout_s=300,operation_timeout_s=300))
    assert normal.request_timeout_s==normal.operation_timeout_s==300


def test_existing_paid_ledger_contract_stays_byte_compatible(tmp_path):
    policy=allowance(tmp_path)
    data=policy.to_mapping()
    assert set(data)=={'mode','ledger','budget_id','max_calls','max_usd','session_max_calls',
        'session_max_usd','operation_max_calls','operation_max_usd','request_timeout_s','operation_timeout_s','rates'}
    data.pop('ledger')
    legacy=json.dumps(data,sort_keys=True,default=str)
    with sqlite3.connect(policy.ledger) as db:
        db.execute('CREATE TABLE policies (budget TEXT PRIMARY KEY,contract TEXT NOT NULL)')
        db.execute('INSERT INTO policies VALUES (?,?)',(policy.budget_id,legacy))
    assert RuntimeBudget(policy).summary()['provider_attempts']==0
    with sqlite3.connect(policy.ledger) as db:
        assert db.execute('SELECT contract FROM policies').fetchone()[0]==legacy
    with pytest.raises(ExecutionDeferred,match='contract differs'):
        RuntimeBudget(replace(policy,max_calls=policy.max_calls+1))


def test_real_router_honors_long_deadline_and_never_extends_shorter_parent(monkeypatch):
    from psychology_base.llm import execution,router as module
    from psychology_base.eval.benchmarks.local_membership_diagnostic import diagnostic_router
    from tests.test_membership_local_sentinel import sentinel
    long=diagnostic_router(sentinel.configuration())
    normal=diagnostic_router(dict(sentinel.configuration(),offline_local_diagnostic=False,
        limits=dict(sentinel.LIMITS,request_timeout_s=300,operation_timeout_s=300)))
    clock=SimpleNamespace(monotonic=lambda:100)
    monkeypatch.setattr(execution,'time',clock)
    monkeypatch.setattr(module,'time',clock)
    with normal.operation('ordinary',timeout_s=1200):
        assert current_operation.get().deadline==400
    with long.operation('offline',timeout_s=5000):
        assert current_operation.get().deadline==1300
    with long.operation('short-parent',timeout_s=.25) as outer:
        with long.operation('long-child',timeout_s=1200) as inner:
            assert outer==inner and current_operation.get().deadline==100.25
    assert current_operation.get() is None
    assert long.providers['ollama'].request_timeout_s==1200


def test_offline_deadline_does_not_enable_remote_dispatch(monkeypatch):
    from psychology_base.llm.config import default_config
    from psychology_base.llm.providers.alibaba_tongyi import AlibabaTongyiProvider
    from psychology_base.llm.router import LLMRouter
    remote=AlibabaTongyiProvider()
    def forbidden(*args,**kwargs):
        raise AssertionError('A long local deadline cannot authorize cloud traffic')
    monkeypatch.setattr(remote,'chat',forbidden)
    cfg=replace(default_config(),execution=ExecutionPolicy(offline_local_diagnostic=True,
        request_timeout_s=1200,operation_timeout_s=1200),cache=replace(default_config().cache,enabled=False))
    router=LLMRouter(cfg,providers={'tongyi':remote})
    with pytest.raises(ExecutionDeferred):
        router.chat([ChatMessage('user','local-only control')])
