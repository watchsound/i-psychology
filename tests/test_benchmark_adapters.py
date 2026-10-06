"""Step-1 gate for the benchmark adapter plumbing.

Per [docs/phases/phase-1-benchmark-adapters-plan.md] §5 step 1:
  gate = "empty-run on all three benchmarks loads instances without errors;
          cache round-trips a judgment"

Deliberately deterministic, no network, no judge-model calls. Real loader +
scorer tests arrive in step 2+.
"""

from __future__ import annotations

import pytest

from psychology_base.eval.benchmarks.instance import (
    BenchmarkInstance,
    BenchmarkLoader,
    BenchmarkQuestion,
    BenchmarkTurn,
)
from psychology_base.eval.benchmarks.judge import (
    Judgment,
    JudgmentCache,
    JudgmentKey,
    NullJudge,
)
from psychology_base.eval.scripts.run_benchmark import (
    build_parser,
    get_loader,
    main,
    register_loader,
)


# ---------- loader scaffold ----------


@pytest.mark.parametrize("name", ["longmemeval", "locomo", "epbench"])
def test_loader_registered_for_every_benchmark(name):
    loader = get_loader(name)
    assert isinstance(loader, BenchmarkLoader)
    assert loader.name == name
    # Default-registered loaders yield no instances until a data path is wired.
    assert list(loader.iter_instances()) == []


def test_unknown_benchmark_name_raises():
    with pytest.raises(KeyError):
        get_loader("made-up-benchmark")  # type: ignore[arg-type]


def test_register_loader_overrides():
    class _OneShotLoader:
        name = "epbench"

        def iter_instances(self, root=None):
            yield BenchmarkInstance(
                name="epbench",
                instance_id="demo-0",
                history=(),
                questions=(
                    BenchmarkQuestion(id="q0", cue="what/when/where/who?", gold="x"),
                ),
            )

    original = get_loader("epbench")
    register_loader("epbench", _OneShotLoader())
    try:
        instances = list(get_loader("epbench").iter_instances())
        assert len(instances) == 1
        assert instances[0].instance_id == "demo-0"
    finally:
        # Restore whichever loader was registered at module-import time.
        register_loader("epbench", original)


# ---------- cache round-trip ----------


def _make_key(sha_suffix: str = "a") -> JudgmentKey:
    return JudgmentKey(
        benchmark="epbench",
        instance_id="inst-1",
        question_id="q-1",
        gold="answer-a",
        candidate="answer-a",
        judge_id=f"test-judge-{sha_suffix}",
        judge_prompt_sha="v1",
    )


def test_judgment_cache_round_trip(tmp_path):
    cache = JudgmentCache(cache_dir=tmp_path)
    key = _make_key()
    assert cache.get(key) is None
    assert cache.misses == 1

    cache.put(key, Judgment(score=1.0, verdict="correct", rationale="matches gold"))
    recalled = cache.get(key)
    assert recalled == Judgment(score=1.0, verdict="correct", rationale="matches gold")
    assert cache.hits == 1


def test_judgment_cache_key_sensitivity(tmp_path):
    """Changing any key field must miss the cache. Guards against silent reuse
    of stale judgments after a judge or prompt change."""
    cache = JudgmentCache(cache_dir=tmp_path)
    key_a = _make_key("a")
    cache.put(key_a, Judgment(score=1.0, verdict="correct"))

    for field_override in [
        {"benchmark": "longmemeval"},
        {"instance_id": "inst-2"},
        {"question_id": "q-2"},
        {"gold": "answer-b"},
        {"candidate": "answer-b"},
        {"judge_id": "test-judge-b"},
        {"judge_prompt_sha": "v2"},
    ]:
        new_key = JudgmentKey(
            benchmark=field_override.get("benchmark", key_a.benchmark),
            instance_id=field_override.get("instance_id", key_a.instance_id),
            question_id=field_override.get("question_id", key_a.question_id),
            gold=field_override.get("gold", key_a.gold),
            candidate=field_override.get("candidate", key_a.candidate),
            judge_id=field_override.get("judge_id", key_a.judge_id),
            judge_prompt_sha=field_override.get("judge_prompt_sha", key_a.judge_prompt_sha),
        )
        assert cache.get(new_key) is None, f"stale hit when {field_override} changed"


def test_judgment_cache_atomic_rewrite(tmp_path):
    """put() overwrites in-place; a second put with the same key but different
    verdict wins without leaving the .tmp file behind."""
    cache = JudgmentCache(cache_dir=tmp_path)
    key = _make_key()
    cache.put(key, Judgment(score=0.5, verdict="partial"))
    cache.put(key, Judgment(score=1.0, verdict="correct"))
    assert cache.get(key).verdict == "correct"
    # No orphan .tmp files anywhere under cache dir.
    tmps = list(tmp_path.rglob("*.tmp"))
    assert tmps == [], f"orphan tmp files: {tmps}"


def test_null_judge_refuses_free_form():
    judge = NullJudge()
    with pytest.raises(RuntimeError, match="NullJudge cannot judge"):
        judge.judge(benchmark="locomo", question="?", gold="x", candidate="y")


def test_judge_cached_hits_cache_on_second_call(tmp_path):
    """Verifies the full cached-judgment path: miss → populate → hit without
    re-invoking the client. Uses a stub client whose call-count is observable."""

    class _CountingClient:
        judge_id = "stub-1"
        prompt_sha = "stub-prompt-sha"

        def __init__(self) -> None:
            self.calls = 0

        def judge(self, *, benchmark, question, gold, candidate, question_type=""):
            self.calls += 1
            return Judgment(score=1.0, verdict="correct")

    cache = JudgmentCache(cache_dir=tmp_path)
    client = _CountingClient()
    kwargs = dict(
        client=client,
        benchmark="longmemeval",
        instance_id="inst-x",
        question_id="q-x",
        question="what day did she arrive?",
        gold="Tuesday",
        candidate="Tuesday",
    )
    first = cache.judge_cached(**kwargs)
    second = cache.judge_cached(**kwargs)
    assert first == second
    assert client.calls == 1, "second call should be a cache hit, not a re-judge"


# ---------- CLI skeleton ----------


def test_cli_parser_requires_benchmark():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


@pytest.mark.parametrize("name", ["epbench", "longmemeval", "locomo"])
def test_cli_real_loader_without_data_prints_data_hint(name, capsys):
    """All three benchmarks have real loaders as of step 4. Without
    --data-root the CLI should print the 'real loader, no data configured'
    hint rather than the (now-retired) step-1 stub message."""
    rc = main(["--benchmark", name, "--mode", "direct", "--split", "smoke"])
    out = capsys.readouterr().out
    assert rc == 0
    assert f"benchmark={name}" in out
    assert "instances=0" in out
    assert "stub (step 1)" not in out
    assert "zero instances" in out


def test_cli_rejects_unknown_mode():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--benchmark", "epbench", "--mode", "invented-mode"])
