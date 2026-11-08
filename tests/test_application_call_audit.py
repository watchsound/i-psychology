"""Budget/failure controls use an explicitly injected provider, never quality labels."""

from psychology_base.llm.execution import ExecutionPolicy
from decimal import Decimal
from pathlib import Path

import pytest

from psychology_base.eval.benchmarks.application_calls import Budget, RecordingRouter
from psychology_base.applications.pilot.session_runner import SessionLogWriter
from psychology_base.llm.config import LLMConfig, ProviderConfig, CacheConfig, LocalOllamaConfig
from psychology_base.llm.router import LLMRouter
from psychology_base.llm.types import ChatMessage, Completion, TaskTier


class ControlProvider:
    name = "control"
    calls = 0
    fail = False

    def model_for(self, tier):
        return "declared-control"

    def chat(self, messages, **kwargs):
        self.calls += 1
        if self.fail:
            raise RuntimeError("Injected provider failure")
        return Completion("control response", self.name, self.model_for(None), TaskTier.ADVANCED,
                          prompt_tokens=10, completion_tokens=2, finish_reason="stop")


def audited(tmp_path, *, calls=2, dollars="1", cache=False):
    cfg = LLMConfig("control", {"control": ProviderConfig("control", "declared-control", "declared-control")},
                    cache=CacheConfig(cache, tmp_path / "cache"), local_ollama=LocalOllamaConfig(enabled=False), execution=ExecutionPolicy(mode="research"))
    provider = ControlProvider()
    router = LLMRouter(cfg, providers={"control": provider})
    ledger = Budget(SessionLogWriter(tmp_path / "attempts.jsonl"), max_calls=calls, max_usd=Decimal(dollars))
    wrapper = RecordingRouter(router, ledger, provider="control", model="declared-control",
                              input_rate="1", output_rate="2", max_input=1000, controls=True)
    return wrapper, provider, ledger


def test_budget_checks_before_dispatch_and_logs_real_completion(tmp_path):
    router, provider, budget = audited(tmp_path, calls=1)
    c = router.chat([ChatMessage("user", "actual request")], tier=TaskTier.ADVANCED, max_tokens=20)
    assert c.content == "control response" and provider.calls == budget.attempts == 1
    with pytest.raises(ValueError):
        router.chat([ChatMessage("user", "second")], tier=TaskTier.ADVANCED, max_tokens=20)
    assert provider.calls == 1 and budget.spent > 0 and router.last_completion == c
    budget.writer.close()
    assert '"event": "completion"' in (tmp_path / "attempts.jsonl").read_text()


def test_spend_ceiling_blocks_before_network(tmp_path):
    router, provider, budget = audited(tmp_path, dollars="0.00001")
    with pytest.raises(ValueError):
        router.chat([ChatMessage("user", "request")], tier=TaskTier.ADVANCED, max_tokens=20)
    assert provider.calls == budget.attempts == 0


def test_failure_keeps_reservation_and_never_retries(tmp_path):
    router, provider, budget = audited(tmp_path)
    provider.fail = True
    with pytest.raises(RuntimeError):
        router.chat([ChatMessage("user", "request")], tier=TaskTier.ADVANCED, max_tokens=20)
    assert provider.calls == budget.attempts == 1 and budget.spent == Decimal("0.00104")
    budget.writer.close()


def test_cached_judge_is_not_a_new_provider_attempt(tmp_path):
    router, provider, budget = audited(tmp_path, cache=True)
    for _ in range(2):
        router.chat([ChatMessage("user", "same judgment")], tier=TaskTier.ADVANCED, max_tokens=20)
    assert provider.calls == budget.attempts == 1 and budget.cache_hits == 1
    budget.writer.close()


def test_controls_cannot_claim_a_live_provider(tmp_path):
    router, provider, budget = audited(tmp_path)
    router.controls = False
    with pytest.raises(ValueError):
        router.chat([ChatMessage("user", "request")], tier=TaskTier.ADVANCED, max_tokens=20)
    assert provider.calls == 0


def test_budget_carries_completed_and_unfinished_attempts_across_stages(tmp_path):
    router, provider, budget = audited(tmp_path, calls=2)
    router.chat([ChatMessage('user','first stage')],tier=TaskTier.ADVANCED,max_tokens=20)
    budget.reserve(Decimal('0.01'),{'provider':'control','model':'declared-control'})
    budget.writer.close()
    restored=Budget(SessionLogWriter(tmp_path/'attempts.jsonl'),max_calls=2,max_usd=Decimal('1'))
    assert restored.attempts==2 and restored.uncertain and restored.spent==budget.spent
    with pytest.raises(ValueError): restored.reserve(Decimal('0.001'),{})


def test_uncached_samples_dispatch_independent_requests(tmp_path):
    router, provider, budget = audited(tmp_path,calls=3)
    for _ in range(3): router.chat([ChatMessage('user','same probe')],tier=TaskTier.ADVANCED,max_tokens=20)
    assert provider.calls==budget.attempts==3 and budget.cache_hits==0
    budget.writer.close()


def test_unknown_usage_is_not_free_or_valid(tmp_path,monkeypatch):
    router,provider,budget=audited(tmp_path)
    original=provider.chat
    from dataclasses import replace
    monkeypatch.setattr(provider,'chat',lambda *a,**kw:replace(original(*a,**kw),prompt_tokens=0))
    with pytest.raises(ValueError):
        router.chat([ChatMessage('user','request')],tier=TaskTier.ADVANCED,max_tokens=20)
    assert budget.spent==Decimal('0.00104') and budget.uncertain and provider.calls==1
    budget.writer.close()


def test_wrong_actual_model_cannot_qualify(tmp_path,monkeypatch):
    router,provider,budget=audited(tmp_path)
    original=provider.chat
    from dataclasses import replace
    monkeypatch.setattr(provider,'chat',lambda *a,**kw:replace(original(*a,**kw),model_id='unexpected model'))
    with pytest.raises(ValueError):
        router.chat([ChatMessage('user','request')],tier=TaskTier.ADVANCED,max_tokens=20)
    assert provider.calls==budget.attempts==1
    budget.writer.close()


def test_actual_yaml_retry_setting_and_invalid_type(tmp_path):
    from psychology_base.llm import load_config
    p=tmp_path/'config.yaml'
    prefix='active_provider: openai\nproviders:\n  openai:\n    simple_model: a\n    advanced_model: b\n'
    p.write_text(prefix+'    max_retries: 0\n',encoding='utf-8')
    assert load_config(p).providers['openai'].max_retries==0
    p.write_text(prefix+'    max_retries: true\n',encoding='utf-8')
    with pytest.raises(ValueError): load_config(p)


def test_openai_retry_zero_is_forwarded_through_real_client_owner(monkeypatch):
    import sys, types
    from psychology_base.llm.providers.openai import OpenAIProvider
    captured=[]
    monkeypatch.setitem(sys.modules,'openai',types.SimpleNamespace(OpenAI=lambda **kw: captured.append(kw) or object()))
    OpenAIProvider(api_key='declared-test-key',max_retries=0)._ensure_client()
    assert captured==[{'api_key':'declared-test-key','max_retries':0}]
    captured.clear()
    OpenAIProvider(api_key='declared-test-key')._ensure_client()
    assert captured==[{'api_key':'declared-test-key'}]


def test_openai_owner_preserves_returned_model_and_completion_identity():
    from types import SimpleNamespace
    from psychology_base.llm.providers.openai import OpenAIProvider
    response=SimpleNamespace(id='declared-unit-completion',_request_id='declared-unit-request',
        model='returned-unit-model',system_fingerprint='unit-fingerprint',
        choices=[SimpleNamespace(message=SimpleNamespace(content='unit response'),finish_reason='stop')],
        usage=SimpleNamespace(prompt_tokens=3,completion_tokens=2))
    client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs:response)))
    completion=OpenAIProvider(_client=client,advanced_model='requested-unit-model').chat(
        [ChatMessage('user','unit request')],tier=TaskTier.ADVANCED)
    assert completion.model_id=='requested-unit-model'
    assert completion.metadata=={'completion_id':'declared-unit-completion','request_id':'declared-unit-request',
                                 'response_model':'returned-unit-model',
                                 'system_fingerprint':'unit-fingerprint'}


@pytest.mark.parametrize("value", [-1, True, "0", 0.5])
def test_retry_configuration_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        ProviderConfig("openai", "m", "m", max_retries=value)
