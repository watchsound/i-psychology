"""Offline durable user-period admission; no model or paid requests."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime, timezone
from decimal import Decimal
import json
import sqlite3

import pytest

from psychology_base.llm import accounting
from psychology_base.llm.accounting import RuntimeBudget
from psychology_base.llm.config import CacheConfig, LLMConfig, ProviderConfig
from psychology_base.llm.execution import ExecutionDeferred, ExecutionPolicy, UserPeriodLimits
from psychology_base.llm.router import LLMRouter
from tests.test_runtime_budget import allowance, call, paid_router, no_network


def user_policy(root, **overrides):
    limits = dict(daily_max_calls=3, daily_max_usd=".01", monthly_max_calls=5, monthly_max_usd=".02")
    limits.update(overrides)
    return allowance(root, user_period_limits=UserPeriodLimits(**limits))


def reserve(budget, *, user="alice", session="session", ceiling=".001"):
    return budget.reserve(ceiling, session=session, operation="operation", purpose="offline-test",
                          provider="control", model="none", kind="chat", user_id=user)


def clock(monkeypatch, instant):
    value = datetime.fromisoformat(instant).replace(tzinfo=timezone.utc)
    monkeypatch.setattr(accounting, "_utc_now", lambda: value)


def _process_reservations(root, crash=False, ready=None):
    """Spawn-compatible independent ledger connection and user session."""
    import os
    from pathlib import Path
    if ready is not None:
        ready.set()
    budget = RuntimeBudget(user_policy(Path(root)))
    count = 0
    for _ in range(4):
        try:
            reserve(budget, session=f"process-{os.getpid()}")
            count += 1
            if crash:
                os._exit(0)
        except ExecutionDeferred:
            break
    return count


@pytest.mark.parametrize("field,value", [
    ("daily_max_calls", 0), ("monthly_max_calls", True), ("daily_max_calls", 1.5),
    ("daily_max_usd", "NaN"), ("monthly_max_usd", "Infinity"), ("daily_max_usd", "0"),
    ("monthly_max_usd", "-.01"),
])
def test_period_limits_require_explicit_finite_positive_caps(tmp_path, field, value):
    with pytest.raises(ValueError):
        user_policy(tmp_path, **{field: value})


def test_period_configuration_round_trip_and_strict_fields(tmp_path):
    policy = user_policy(tmp_path)
    parsed = ExecutionPolicy.from_mapping(asdict(policy))
    assert parsed == policy
    for limits in ({}, [], {"daily_max_calls": 1}, {**asdict(policy.user_period_limits), "timezone": "local"}):
        with pytest.raises(ValueError):
            ExecutionPolicy.from_mapping({"user_period_limits": limits})
    with pytest.raises(ValueError):
        replace(policy, user_period_limits={})
    assert ExecutionPolicy.from_mapping({}).user_period_limits is None


def test_configuration_identity_preserves_legacy_and_pins_enabled_caps(tmp_path):
    import hashlib
    from psychology_base.eval.benchmarks.application_context import runtime_identity
    legacy = LLMConfig("tongyi", execution=allowance(tmp_path))
    historical = asdict(legacy)
    historical["execution"].pop("user_period_limits")
    serialized = json.dumps(historical, sort_keys=True, default=str)
    expected = hashlib.sha256(json.dumps(serialized, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    assert runtime_identity(legacy)["config_sha256"] == expected
    configured = replace(legacy, execution=user_policy(tmp_path))
    assert runtime_identity(configured)["config_sha256"] != expected
    assert ExecutionPolicy.from_mapping(configured.execution.to_mapping()) == configured.execution


@pytest.mark.parametrize("identity", [None, "", " ", 1, " alice", "alice "])
def test_missing_or_ambiguous_identity_cannot_reserve(tmp_path, identity):
    budget = RuntimeBudget(user_policy(tmp_path))
    with pytest.raises(ExecutionDeferred, match="stable user"):
        reserve(budget, user=identity)
    assert budget.summary()["provider_attempts"] == 0


def test_user_allowance_shared_across_sessions_and_restart(tmp_path):
    policy = user_policy(tmp_path, daily_max_calls=2)
    reserve(RuntimeBudget(policy), session="tutor:one")
    reserve(RuntimeBudget(policy), session="companion:two")
    restarted = RuntimeBudget(policy)
    with pytest.raises(ExecutionDeferred):
        reserve(restarted, session="tutor:three")
    assert restarted.user_summary("alice")["daily"]["provider_attempts"] == 2
    reserve(restarted, user="bob", session="bob:one")
    assert restarted.user_summary("bob")["daily"]["provider_attempts"] == 1


def test_session_cannot_be_reassigned_to_avoid_a_user_limit(tmp_path):
    budget = RuntimeBudget(user_policy(tmp_path))
    reserve(budget)
    with pytest.raises(ExecutionDeferred, match="another user"):
        reserve(budget, user="bob")
    with pytest.raises(ExecutionDeferred, match="another user"):
        RuntimeBudget(user_policy(tmp_path)).bind_session(session="session", user_id="bob")
    assert budget.summary()["provider_attempts"] == 1


@pytest.mark.parametrize("scope", ["daily", "monthly"])
def test_user_spend_bound_is_checked_before_reserving(tmp_path, scope):
    budget = RuntimeBudget(user_policy(tmp_path, **{scope + "_max_usd": ".0015"}))
    reserve(budget)
    with pytest.raises(ExecutionDeferred):
        reserve(budget, session="another")
    assert budget.summary()["provider_attempts"] == 1
    assert Decimal(budget.user_summary("alice")[scope]["accounted_usd"]) == Decimal(".001")


@pytest.mark.parametrize("scope", ["daily", "monthly"])
def test_subledger_unit_period_cap_never_rounds_up(tmp_path, scope):
    budget = RuntimeBudget(user_policy(tmp_path, **{scope + "_max_usd": ".0000000000001"}))
    with pytest.raises(ExecutionDeferred):
        reserve(budget, ceiling=".0000000000001")
    assert budget.summary()["provider_attempts"] == 0


def test_daily_rollover_preserves_monthly_and_project_usage(tmp_path, monkeypatch):
    policy = user_policy(tmp_path, daily_max_calls=1, monthly_max_calls=2)
    clock(monkeypatch, "2026-11-01T23:59:59")
    budget = RuntimeBudget(policy)
    reserve(budget)
    with pytest.raises(ExecutionDeferred):
        reserve(budget)
    clock(monkeypatch, "2026-11-02T00:00:00")
    reserve(RuntimeBudget(policy), session="second-day")
    clock(monkeypatch, "2026-11-03T00:00:00")
    with pytest.raises(ExecutionDeferred):
        reserve(RuntimeBudget(policy), session="third-day")
    assert budget.user_summary("alice")["daily"]["provider_attempts"] == 0
    assert budget.user_summary("alice")["monthly"]["provider_attempts"] == 2
    assert budget.summary()["provider_attempts"] == 2


@pytest.mark.parametrize("before,after", [
    ("2026-11-30T23:59:59", "2026-12-01T00:00:00"),
    ("2026-12-31T23:59:59", "2027-01-01T00:00:00"),
    ("2028-02-29T23:59:59", "2028-03-01T00:00:00"),
])
def test_month_rollover_keeps_original_reservation_period(tmp_path, monkeypatch, before, after):
    policy = user_policy(tmp_path, daily_max_calls=1, monthly_max_calls=1)
    clock(monkeypatch, before)
    budget = RuntimeBudget(policy)
    dispatch = reserve(budget)
    clock(monkeypatch, after)
    # A late result changes the old reservation, never the settlement-day pool.
    budget.settle(dispatch, actual=".0002")
    reserve(RuntimeBudget(policy), session="new-month")
    assert budget.user_summary("alice")["monthly"]["provider_attempts"] == 1
    assert Decimal(budget.user_summary("alice")["monthly"]["accounted_usd"]) == Decimal(".001")
    assert budget.summary()["provider_attempts"] == 2
    assert Decimal(budget.summary()["accounted_usd"]) == Decimal(".0012")
    with sqlite3.connect(policy.ledger) as db:
        assert db.execute("SELECT period_day FROM calls WHERE id=?", (dispatch,)).fetchone()[0] == before[:10]


@pytest.mark.parametrize("limit", ["project", "session", "operation"])
def test_rollover_never_resets_existing_scopes(tmp_path, monkeypatch, limit):
    overrides = {"max_calls" if limit == "project" else limit + "_max_calls": 1}
    policy = replace(user_policy(tmp_path), **overrides)
    clock(monkeypatch, "2026-11-30T23:59:59")
    reserve(RuntimeBudget(policy))
    clock(monkeypatch, "2026-12-01T00:00:00")
    with pytest.raises(ExecutionDeferred):
        reserve(RuntimeBudget(policy))
    assert RuntimeBudget(policy).summary()["provider_attempts"] == 1


def test_backward_or_naive_clock_cannot_allocate_an_earlier_period(tmp_path, monkeypatch):
    policy = user_policy(tmp_path)
    clock(monkeypatch, "2026-12-01T00:00:00")
    reserve(RuntimeBudget(policy))
    clock(monkeypatch, "2026-11-30T00:00:00")
    with pytest.raises(ExecutionDeferred, match="earlier day"):
        reserve(RuntimeBudget(policy))
    monkeypatch.setattr(accounting, "_utc_now", lambda: datetime(2026, 12, 1))
    with pytest.raises(ExecutionDeferred, match="aware UTC"):
        reserve(RuntimeBudget(policy))
    assert RuntimeBudget(policy).summary()["provider_attempts"] == 1


def test_known_usage_releases_money_but_never_a_call(tmp_path):
    budget = RuntimeBudget(user_policy(tmp_path, daily_max_usd=".0015", daily_max_calls=2))
    dispatch = reserve(budget)
    budget.settle(dispatch, actual=".0002")
    dispatch = reserve(budget)
    budget.settle(dispatch)  # Unknown billed outcome retains the full bound.
    daily = budget.user_summary("alice")["daily"]
    assert daily["provider_attempts"] == 2 and daily["usage_uncertain"]
    assert Decimal(daily["accounted_usd"]) == Decimal(".0012")
    with pytest.raises(ExecutionDeferred):
        reserve(budget)


def test_provider_overrun_holds_all_users_after_rollover(tmp_path, monkeypatch):
    clock(monkeypatch, "2026-11-30T23:59:59")
    policy = user_policy(tmp_path)
    budget = RuntimeBudget(policy)
    dispatch = reserve(budget)
    with pytest.raises(ExecutionDeferred, match="exceeded"):
        budget.settle(dispatch, actual=".002")
    clock(monkeypatch, "2026-12-01T00:00:00")
    with pytest.raises(ExecutionDeferred, match="overrun"):
        reserve(RuntimeBudget(policy), user="bob", session="bob")


@pytest.mark.parametrize("limit", ["daily_calls", "monthly_calls", "daily_usd", "monthly_usd", "project_calls"])
def test_parallel_sessions_cannot_oversubscribe_user_or_project(tmp_path, limit):
    overrides = {
        "daily_calls": {}, "monthly_calls": {"monthly_max_calls": 3},
        "daily_usd": {"daily_max_usd": ".003"}, "monthly_usd": {"monthly_max_usd": ".003"},
        "project_calls": {},
    }[limit]
    policy = user_policy(tmp_path, daily_max_calls=10, **overrides) if limit != "daily_calls" else user_policy(tmp_path)
    if limit == "project_calls":
        policy = replace(policy, max_calls=3)
    RuntimeBudget(policy)
    def attempt(index):
        try:
            reserve(RuntimeBudget(policy), session=f"thread-{index}")
            return 1
        except ExecutionDeferred:
            return 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(attempt, range(16))) == 3
    assert RuntimeBudget(policy).summary()["provider_attempts"] == 3


def test_independent_processes_share_user_allowance(tmp_path):
    import multiprocessing
    RuntimeBudget(user_policy(tmp_path))
    with multiprocessing.get_context("spawn").Pool(3) as pool:
        assert sum(pool.map(_process_reservations, [str(tmp_path)] * 3)) == 3
    assert RuntimeBudget(user_policy(tmp_path)).user_summary("alice")["daily"]["provider_attempts"] == 3


def test_exited_process_keeps_user_reservation_on_restart(tmp_path):
    import multiprocessing
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    process = context.Process(target=_process_reservations, args=(str(tmp_path), True, ready))
    process.start()
    try:
        assert ready.wait(60), "Child did not initialize"
        process.join(15)
        assert not process.is_alive(), "Child did not finish its reservation"
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
    assert process.exitcode == 0
    summary = RuntimeBudget(user_policy(tmp_path)).user_summary("alice")["daily"]
    assert summary["provider_attempts"] == 1 and summary["usage_uncertain"]
    assert Decimal(summary["accounted_usd"]) == Decimal(".001")


def test_legacy_schema_upgrade_preserves_contract_and_unfinished_usage(tmp_path):
    policy = allowance(tmp_path)
    contract = asdict(policy)
    contract.pop("ledger")
    contract.pop("user_period_limits")
    encoded = json.dumps(contract, sort_keys=True, default=str)
    with sqlite3.connect(policy.ledger) as db:
        db.execute("CREATE TABLE policies (budget TEXT PRIMARY KEY, contract TEXT NOT NULL)")
        db.execute("INSERT INTO policies VALUES (?,?)", (policy.budget_id, encoded))
        db.execute("""CREATE TABLE calls (id TEXT PRIMARY KEY, budget TEXT NOT NULL, session TEXT NOT NULL,
            operation TEXT NOT NULL, purpose TEXT NOT NULL, provider TEXT NOT NULL, model TEXT NOT NULL,
            kind TEXT NOT NULL, units INTEGER NOT NULL, state TEXT NOT NULL, input_tokens INTEGER,
            output_tokens INTEGER, latency_ms REAL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""")
        db.execute("INSERT INTO calls(id,budget,session,operation,purpose,provider,model,kind,units,state) "
                   "VALUES ('old',?,'s','o','p','p','m','chat',1000000000,'reserved')", (policy.budget_id,))
    budget = RuntimeBudget(policy)
    assert budget.summary()["provider_attempts"] == 1 and budget.summary()["usage_uncertain"]
    with sqlite3.connect(policy.ledger) as db:
        assert db.execute("SELECT contract FROM policies").fetchone()[0] == encoded
        assert db.execute("SELECT units,user_id,period_day,period_month FROM calls").fetchone() == (1000000000, None, None, None)
    budget.settle("old", actual=".0002")
    reserve(budget, session="new")
    # Opt-in may not reinterpret an existing allowance or erase old usage.
    with pytest.raises(ExecutionDeferred, match="contract differs"):
        RuntimeBudget(user_policy(tmp_path))
    assert RuntimeBudget(policy).summary()["provider_attempts"] == 2


def test_period_contract_cannot_be_changed_or_removed(tmp_path):
    policy = user_policy(tmp_path)
    RuntimeBudget(policy)
    for changed in (replace(policy, user_period_limits=None),
                    replace(policy, user_period_limits=replace(policy.user_period_limits, daily_max_calls=4))):
        with pytest.raises(ExecutionDeferred, match="contract differs"):
            RuntimeBudget(changed)


@pytest.mark.parametrize("kind", ["chat", "embed", "rerank"])
def test_router_checks_user_allowance_before_any_remote_transport(tmp_path, monkeypatch, kind):
    from psychology_base.llm.types import ChatMessage
    policy = user_policy(tmp_path, daily_max_calls=1)
    router, calls = paid_router(tmp_path, monkeypatch, policy=policy, user="alice")
    other, other_calls = paid_router(tmp_path, monkeypatch, policy=policy, session="other", user="alice")
    def dispatch(target):
        with target.operation("period-control"):
            if kind == "embed":
                return target.embed("a", backend="cloud", use_cache=False)
            if kind == "rerank":
                return target.rerank("q", ["a", "b"])
            return target.chat([ChatMessage("user", "hello")], max_tokens=50, use_cache=False)
    dispatch(router)
    with pytest.raises(ExecutionDeferred):
        dispatch(other)
    assert calls == [kind] and other_calls == []
    assert other.budget.user_summary("alice")["daily"]["provider_attempts"] == 1


def test_router_requires_user_before_provider_construction(tmp_path, monkeypatch):
    import psychology_base.llm.router as module
    def forbidden(*args, **kwargs):
        raise AssertionError("Provider constructed before identity admission")
    monkeypatch.setattr(module, "_build_provider", forbidden)
    config = LLMConfig("tongyi", cache=CacheConfig(False), execution=user_policy(tmp_path))
    with pytest.raises(ExecutionDeferred, match="stable user"):
        LLMRouter(config, session_id="session")


def test_cache_hits_remain_free_after_user_allowance_exhaustion(tmp_path, monkeypatch):
    policy = user_policy(tmp_path, daily_max_calls=1)
    router, calls = paid_router(tmp_path, monkeypatch, policy=policy, cache=True, user="alice")
    assert call(router).content == call(router).content
    with pytest.raises(ExecutionDeferred):
        call(router, "different")
    assert calls == ["chat"]
    assert router.budget.summary()["cache_hits"] == 1
    assert router.budget.user_summary("alice")["daily"]["provider_attempts"] == 1


def test_failed_provider_retains_user_money_and_call_count(tmp_path, monkeypatch):
    policy = user_policy(tmp_path, daily_max_calls=1)
    router, calls = paid_router(tmp_path, monkeypatch, policy=policy, user="alice")
    def fail(*args, **kwargs):
        calls.append("failure")
        raise TimeoutError("unknown billing")
    monkeypatch.setattr(router.providers["tongyi"], "chat", fail)
    with pytest.raises(ExecutionDeferred):
        call(router)
    with pytest.raises(ExecutionDeferred):
        call(router, "another")
    assert calls == ["failure"]
    daily = RuntimeBudget(policy).user_summary("alice")["daily"]
    assert daily["provider_attempts"] == 1 and daily["usage_uncertain"]
    assert Decimal(daily["accounted_usd"]) > 0


def test_both_actual_cli_sessions_share_enrolled_user_allowance(tmp_path, monkeypatch):
    """Real enrollment, factories, agents, session logs and PAM; offline provider."""
    import io
    from contextlib import redirect_stdout, redirect_stderr
    from psychology_base.applications.learning import tutor_cli
    from psychology_base.applications.emotional import companion_cli
    from psychology_base.applications.pilot.layout import PilotApp, PilotLayout, session_log_path
    from psychology_base.llm import config as config_module, router as router_module

    policy = user_policy(tmp_path, daily_max_calls=1)
    policy = replace(policy, rates={key: replace(rate, max_input_tokens=15000, max_output_tokens=400)
                                    for key, rate in policy.rates.items()})
    original, calls = paid_router(tmp_path, monkeypatch, policy=policy, user="alice", session="control")
    cfg = replace(original.config, providers={"tongyi": ProviderConfig("tongyi", "small", "large")})
    monkeypatch.setattr(config_module, "load_config", lambda: cfg)
    monkeypatch.setattr(router_module, "_build_provider", lambda slot: original.providers[slot.name])
    captured = []
    for module, app in ((tutor_cli, PilotApp.LEARNING_TUTOR), (companion_cli, PilotApp.EMOTIONAL_COMPANION)):
        prefix = ["--pilot-root", str(tmp_path / "pilot")]
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            assert module.main(prefix + ["enroll", "--user", "alice", "--policy", "S1",
                                         "--age-bracket", "26-45"]) == 0
            monkeypatch.setattr("sys.stdin", io.StringIO("I want to practice a little today.\n/quit\n"))
            assert module.main(prefix + ["session", "--user", "alice", "--mode", "cloud"]) == 0, err.getvalue()
        layout = PilotLayout(pilot_root=tmp_path / "pilot", app=app)
        events = [json.loads(line) for line in session_log_path(layout, "alice", 1).read_text(encoding="utf-8").splitlines()]
        captured.append([event["event"] for event in events])
    assert calls == ["chat"]
    assert "agent_turn" in captured[0] and "turn_deferred" not in captured[0]
    assert "turn_deferred" in captured[1] and "agent_turn" not in captured[1]
    budget = RuntimeBudget(policy)
    assert budget.user_summary("alice")["daily"]["provider_attempts"] == 1
    with sqlite3.connect(policy.ledger) as db:
        assert set(db.execute("SELECT session,user_id FROM session_users WHERE session!='control'")) == {
            ("tutor:alice-s0001", "alice"), ("companion:alice-s0001", "alice")}


def test_router_from_config_accepts_explicit_session_and_user(tmp_path, monkeypatch):
    from psychology_base.llm import config as config_module, router as router_module
    policy = user_policy(tmp_path)
    original, calls = paid_router(tmp_path, monkeypatch, policy=policy, user="alice")
    monkeypatch.setattr(config_module, "load_config", lambda path: original.config)
    router = router_module.router_from_config(providers=original.providers, session_id="new-session", user_id="alice")
    call(router)
    assert calls == ["chat"]
    assert router.budget.user_summary("alice")["daily"]["provider_attempts"] == 1
