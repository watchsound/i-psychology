"""Single entry point for all LLM calls in the project.

Execution permission is orthogonal to SIMPLE/ADVANCED model slots. Local
execution permits installed loopback Ollama models and an explicit offline
pair scorer; paid execution
requires a named operation and durable shared allowance. Research execution
preserves historical routing and is protected by the evaluation owner's
separate contract. Both application factories select a serving policy.
Exact caches and concurrent batches remain in this single owner.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from contextlib import contextmanager
from contextvars import copy_context
import time
import uuid
from pathlib import Path
from typing import Optional, Sequence

from psychology_base.llm.cache import (
    LLMCache,
    chat_cache_key,
    embedding_cache_key,
)
from psychology_base.llm.config import LLMConfig, ProviderConfig
from psychology_base.llm.providers import (
    AlibabaTongyiProvider,
    AnthropicProvider,
    MockProvider,
    OllamaProvider,
    OpenAIProvider,
)
from psychology_base.llm.types import ChatMessage, Completion, LLMProvider, TaskTier, validate_embedding
from psychology_base.llm.providers.ollama import EMBEDDING_BATCH_SIZE
from psychology_base.llm.providers.llama_cpp import LlamaCppProvider
from psychology_base.llm.providers.local_pair import OnnxPairScorer, OnnxPairClassifier
from psychology_base.llm.execution import (ConversationBudget, ConversationLimits,
                                         ExecutionDeferred, Operation, current_operation,
                                         remaining_time, request_slot)
from psychology_base.llm.accounting import RuntimeBudget, request_input_bound, token_cost

_operation = current_operation


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoutingDecision:
    """Provider/model selection and the actual routing reason."""

    provider: str
    model_id: str
    tier: TaskTier
    reason: str


def _build_provider(cfg: ProviderConfig) -> LLMProvider:
    """Translate a config block into a concrete provider instance.

    Dispatch is name-based so the config file is the single source of
    truth. Unknown names raise; this is startup-time, failing loud is
    correct.
    """
    if cfg.name == "ollama":
        kwargs = dict(
            base_url=cfg.base_url or "http://localhost:11434",
            simple_model=cfg.simple_model,
            advanced_model=cfg.advanced_model,
        )
        if cfg.request_timeout_s is not None:
            kwargs["request_timeout_s"] = cfg.request_timeout_s
        if cfg.embedding_model is not None:
            kwargs["embedding_model"] = cfg.embedding_model
        if cfg.embedding_model_multilingual is not None:
            kwargs["embedding_model_multilingual"] = cfg.embedding_model_multilingual
        return OllamaProvider(**kwargs)
    if cfg.name == "openai":
        return OpenAIProvider(
            simple_model=cfg.simple_model,
            advanced_model=cfg.advanced_model,
            api_key=cfg.api_key(),
            base_url=cfg.base_url,
            max_retries=cfg.max_retries,
        )
    if cfg.name == "anthropic":
        return AnthropicProvider(
            simple_model=cfg.simple_model,
            advanced_model=cfg.advanced_model,
            api_key=cfg.api_key(),
            max_retries=cfg.max_retries,
        )
    if cfg.name == "tongyi":
        import os as _os
        kwargs = dict(
            simple_model=cfg.simple_model,
            advanced_model=cfg.advanced_model,
            api_key=cfg.api_key(),
        )
        if cfg.embedding_model is not None:
            kwargs["embedding_model"] = cfg.embedding_model
        if cfg.embedding_model_multilingual is not None:
            kwargs["embedding_model_multilingual"] = cfg.embedding_model_multilingual
        # Rerank is Phase 1.5-prime P3 only; defaults set by provider.
        # Env overrides exist because DashScope migrated reranker hosting
        # to maas.aliyuncs.com with WorkspaceId-scoped URLs in 2026, and
        # the legacy dashscope.aliyuncs.com endpoint may be sunset.
        rerank_model = _os.environ.get("DASHSCOPE_RERANK_MODEL")
        if rerank_model:
            kwargs["rerank_model"] = rerank_model
        rerank_url = _os.environ.get("DASHSCOPE_RERANK_URL")
        if rerank_url:
            kwargs["rerank_url"] = rerank_url
        return AlibabaTongyiProvider(**kwargs)
    raise ValueError(
        f"Unknown provider name: {cfg.name!r}. "
        f"Supported: ollama, openai, anthropic, tongyi."
    )


class LLMRouter:
    """Single entry point for chat + embeddings across providers.

    Builds only providers allowed by the execution policy. Research startup
    retains its historical probe; serving checks local capability on cache
    misses so an outage can recover without a new router or paid fallback.
    """

    def __init__(
        self,
        config: LLMConfig,
        *,
        providers: Optional[dict[str, LLMProvider]] = None,
        cache: Optional[LLMCache] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        local_reranker: Optional[OnnxPairScorer] = None,
        local_chat_provider: Optional[LlamaCppProvider] = None,
    ) -> None:
        self.config = config
        if local_reranker is not None and not isinstance(local_reranker, OnnxPairScorer):
            raise ValueError("Local rerank requires the offline pinned pair adapter")
        self.local_reranker = local_reranker
        self.execution = config.execution
        if local_chat_provider is not None:
            if (type(local_chat_provider) is not LlamaCppProvider or self.execution.mode != 'local'
                    or config.cache.enabled or cache is not None
                    or local_chat_provider.request_timeout_s > self.execution.request_timeout_s):
                raise ValueError('Native chat requires explicit local diagnostic injection without cache')
            providers = dict(providers or {})
            if 'llama_cpp' in providers:
                raise ValueError('Native chat injection must have one owner')
            providers['llama_cpp'] = local_chat_provider
        elif providers is not None and any(isinstance(p, LlamaCppProvider) for p in providers.values()):
            raise ValueError('Native chat requires the explicit diagnostic injection')
        self._diagnostic_chat = local_chat_provider
        self._paid_rates = dict(self.execution.rates)
        self.session_id = session_id
        self.user_id = user_id
        self.budget = None
        if self.execution.mode == "paid":
            if not session_id:
                raise ExecutionDeferred("Paid execution requires a session identifier")
            self.budget = RuntimeBudget(self.execution)
            self.budget.bind_session(session=session_id, user_id=user_id)

        # Allow tests to inject fully-constructed providers (e.g.
        # MockProvider). Falls back to building from config.
        if providers is None:
            providers = {}
            # Strict local startup never needs cloud credentials or a cloud client.
            if self.execution.mode != "local":
                active_cfg = config.get_provider(config.active_provider)
                if self.execution.mode == "paid":
                    active_cfg = replace(active_cfg, max_retries=0,
                                         request_timeout_s=self.execution.request_timeout_s)
                providers[active_cfg.name] = _build_provider(active_cfg)
            # Ollama provider — opt-in via config.
            if config.local_ollama.enabled and "ollama" in config.providers:
                ollama_cfg = config.get_provider("ollama")
                providers["ollama"] = _build_provider(ollama_cfg)
        self.providers = providers
        if self.execution.mode != "research" and isinstance(providers.get("ollama"), OllamaProvider):
            local = providers["ollama"]
            local.request_timeout_s = min(local.request_timeout_s, self.execution.request_timeout_s)
            local.model_metadata_timeout_s = min(local.model_metadata_timeout_s, self.execution.request_timeout_s)
        if self.execution.mode == "local" and isinstance(providers.get("ollama"), OllamaProvider):
            try:
                providers["ollama"].enforce_loopback()
            except ValueError as exc:
                raise ExecutionDeferred(str(exc)) from exc

        # Probe Ollama once at construction. If it's reachable AND the
        # simple_model is pulled, SIMPLE routes to it; otherwise fall
        # back to the active cloud provider's simple slot.
        self._ollama_ready = False
        ollama = self.providers.get("ollama")
        if (
            self.execution.mode == "research" and config.local_ollama.enabled
            and ollama is not None
            and hasattr(ollama, "probe")
        ):
            self._ollama_ready = ollama.probe(
                timeout_s=config.local_ollama.probe_timeout_s
            )
            if self._ollama_ready:
                if config.local_ollama.force_cloud_chat:
                    logger.info(
                        "Ollama reachable (embeddings active) but "
                        "force_cloud_chat=true; SIMPLE chat routes to %s",
                        config.active_provider,
                    )
                else:
                    logger.info(
                        "Ollama available; SIMPLE tier routes to local %s",
                        ollama.model_for(TaskTier.SIMPLE),
                    )
            else:
                logger.info(
                    "Ollama probe failed (server down or simple_model not pulled); "
                    "SIMPLE tier falls back to %s",
                    config.active_provider,
                )

        # Disk cache — SHA-keyed, atomic, sharded.
        if cache is None and config.cache.enabled:
            cache = LLMCache(cache_dir=config.cache.resolved_dir())
        self.cache = cache

    # ---------- routing ----------

    @contextmanager
    def operation(self, purpose, *, operation_id=None, timeout_s=None, inference_role=None):
        """One allowance and deadline shared by a turn/window and nested calls."""
        current = _operation.get()
        if inference_role not in (None, "response", "auxiliary"):
            raise ValueError("Inference role must be response or auxiliary")
        if not isinstance(purpose, str) or not purpose.strip():
            raise ValueError("Operation purpose must be nonempty")
        if timeout_s is None and self.execution.mode != "research":
            timeout_s = self.execution.operation_timeout_s
        if timeout_s is not None:
            import math
            if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or timeout_s <= 0:
                raise ValueError("Operation timeout must be finite and positive")
            if self.execution.mode != "research":
                timeout_s = min(timeout_s, self.execution.operation_timeout_s)
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        if current is not None:
            remaining_time()
            if current.deadline is not None:
                deadline = current.deadline if deadline is None else min(deadline, current.deadline)
            operation = replace(current, deadline=deadline)
        else:
            operation = Operation(operation_id or uuid.uuid4().hex, purpose, deadline)
        if inference_role is not None:
            operation = replace(operation, inference_role=inference_role)
        token = _operation.set(operation)
        try:
            yield operation.id
        finally:
            _operation.reset(token)

    @contextmanager
    def conversation_turn(self, *, limits=None):
        """Start before recall; nesting cannot reset or enlarge a turn allowance.

        Explicit research runners keep their historical contract outside serving.
        Custom responders may use this same scope around their whole agent turn.
        """
        if limits is not None and type(limits) is not ConversationLimits:
            raise ValueError("Conversation limits must be a ConversationLimits profile")
        current = _operation.get()
        active = current.conversation if current else None
        if active is not None:
            if limits is not None and limits != active.limits:
                raise ExecutionDeferred("Nested turns cannot change the conversation allowance")
            remaining_time()
            yield active
            return
        if self.execution.mode == "research":
            yield None
            return
        budget = ConversationBudget(limits or self.config.conversation)
        started = time.monotonic()
        with self.operation("conversation_turn"):
            operation_id = _operation.get().id
            token = _operation.set(replace(_operation.get(), conversation=budget, inference_role="auxiliary"))
            try:
                yield budget
            finally:
                budget.close()
                _operation.reset(token)
                logger.info("LLM conversation usage", extra={"conversation": dict(
                    budget.snapshot(), operation_id=operation_id, session_id=self.session_id,
                    latency_ms=(time.monotonic() - started) * 1000)})

    def _meter_turn(self, provider, model, kind, invoke, *, messages=(), max_tokens=0):
        """Reuse admission for ordinary dispatch and verified embedding chunks.

        This helper never acquires a transport slot or verifies a model twice.
        Paid reservation/settlement remain owned by the existing dispatch body.
        """
        operation = _operation.get()
        budget = operation.conversation if operation else None
        if budget is None:
            return invoke()
        if self.execution.mode == "research":
            raise ExecutionDeferred("Research dispatch cannot bypass a serving turn allowance")
        remaining_time()
        input_bound = request_input_bound(messages, response_schema=getattr(provider, "response_schema", None))
        name = budget.reserve(kind, operation.inference_role, input_bound, max_tokens)
        success, overrun = False, False
        try:
            result = invoke()
            if isinstance(result, Completion):
                if result.provider != provider.name or result.model_id != model:
                    raise ExecutionDeferred("Provider completion identity differs; turn bound retained")
                usage = (result.prompt_tokens, result.completion_tokens)
                if all(type(n) is int and n >= 0 for n in usage):
                    overrun = usage[0] > input_bound or usage[1] > max_tokens
            if overrun:
                raise ExecutionDeferred("Provider exceeded a turn request bound; further inference held")
            success = True
            return result
        finally:
            budget.finish(name, success=success, overrun=overrun)

    def _dispatch(self, provider, model, kind, invoke, *, messages=(), max_tokens=0):
        """Only provider-dispatch boundary for runtime permission/accounting."""
        policy = self.execution
        if isinstance(provider, LlamaCppProvider) and (policy.mode != 'local' or provider is not self._diagnostic_chat or kind != 'chat'):
            raise ExecutionDeferred('Native transport is restricted to the injected local chat diagnostic')
        if policy.mode == "research":
            return self._meter_turn(provider, model, kind, invoke, messages=messages, max_tokens=max_tokens)
        if _operation.get() is None:
            if policy.mode == "paid" and not isinstance(provider, (OllamaProvider, OnnxPairScorer, LlamaCppProvider)):
                raise ExecutionDeferred("Paid calls require a named operation")
            with self.operation(kind):
                return self._dispatch(provider, model, kind, invoke, messages=messages, max_tokens=max_tokens)
        remaining_time()
        def dispatch():
            with request_slot("local" if isinstance(provider, (OllamaProvider, OnnxPairScorer, LlamaCppProvider)) else "paid") as send:
                return self._dispatch_with_slot(provider, model, kind, invoke, send,
                                                 messages=messages, max_tokens=max_tokens)
        return self._meter_turn(provider, model, kind, dispatch, messages=messages, max_tokens=max_tokens)

    def _dispatch_with_slot(self, provider, model, kind, invoke, send, *, messages, max_tokens):
        policy = self.execution
        if isinstance(provider, (OllamaProvider, OnnxPairScorer, LlamaCppProvider)):
            if isinstance(provider, LlamaCppProvider) and (policy.mode != 'local' or provider is not self._diagnostic_chat or kind != 'chat'):
                raise ExecutionDeferred('Native transport is restricted to the injected local chat diagnostic')
            started = time.monotonic()
            try:
                def local_request():
                    if isinstance(provider, (OllamaProvider, LlamaCppProvider)):
                        provider.require_local_model(model, capability="completion" if kind == "chat" else "embedding")
                    remaining_time()
                    return invoke()
                result = send(local_request)
            except Exception as exc:
                raise ExecutionDeferred("Local model unavailable; input retained for a later attempt") from exc
            logger.info("LLM runtime dispatch", extra={"execution": dict(policy=policy.mode, provider=provider.name,
                        model=model, kind=kind, purpose=_operation.get().purpose,
                        latency_ms=(time.monotonic() - started) * 1000, paid_attempts=0)})
            return result
        if policy.mode == "local":
            raise ExecutionDeferred("Local execution refuses a remote provider")
        context = _operation.get()
        if context is None:
            raise ExecutionDeferred("Paid calls require a named operation")
        rate = self._paid_rates.get(f"{provider.name}/{kind}/{model}")
        if rate is None:
            raise ExecutionDeferred("No frozen rate/bounds for this provider, operation and model")
        input_bound = request_input_bound(messages, response_schema=getattr(provider, "response_schema", None))
        if (input_bound > rate.max_input_tokens or type(max_tokens) is not int
                or max_tokens < 0 or max_tokens > rate.max_output_tokens or (kind == "chat" and max_tokens == 0)):
            raise ExecutionDeferred("Request exceeds its paid input/output token bounds")
        self._bound_paid_provider(provider, kind)
        ceiling = token_cost(input_bound, max_tokens, rate.input_per_million, rate.output_per_million)
        dispatch = self.budget.reserve(ceiling, session=self.session_id, operation=context.id, purpose=context.purpose,
                                       provider=provider.name, model=model, kind=kind, user_id=self.user_id)
        started = time.monotonic()
        try:
            result = send(invoke)
        except Exception as exc:
            self.budget.settle(dispatch, latency_ms=(time.monotonic() - started) * 1000)
            raise ExecutionDeferred("Paid model request failed; its reservation is retained") from exc
        actual = None
        overrun = False
        usage = (None, None)
        identity_valid = True
        if isinstance(result, Completion):
            usage = (result.prompt_tokens, result.completion_tokens)
            identity_valid = result.provider == provider.name and result.model_id == model
            if (identity_valid
                    and all(type(n) is int for n in usage) and usage[0] > 0 and usage[1] >= 0):
                actual = token_cost(*usage, rate.input_per_million, rate.output_per_million)
                overrun = usage[0] > input_bound or usage[1] > max_tokens
            elif not all(type(n) is int for n in usage):
                usage = (None, None)
        # Embedding/rerank adapters do not expose usage: retain their full bound.
        if overrun and context.conversation is not None:
            context.conversation.hold()
        self.budget.settle(dispatch, actual=actual, input_tokens=usage[0], output_tokens=usage[1],
                           latency_ms=(time.monotonic() - started) * 1000, overrun=overrun)
        if not identity_valid:
            raise ExecutionDeferred("Provider completion identity differs; full reservation retained")
        return result

    def _bound_paid_provider(self, provider, kind):
        if not isinstance(provider, (OpenAIProvider, AnthropicProvider, AlibabaTongyiProvider)):
            raise ExecutionDeferred("Provider has no bounded runtime transport contract")
        if isinstance(provider, (OpenAIProvider, AnthropicProvider)):
            if kind != "chat":
                raise ExecutionDeferred("This provider has no bounded embedding/rerank adapter")
            if provider._client is not None and not getattr(provider, "_runtime_bounded", False):
                raise ExecutionDeferred("Paid runtime requires its own lazily constructed bounded client")
            if provider._client is not None and getattr(provider._client, "max_retries", None) != 0:
                raise ExecutionDeferred("Existing SDK client does not enforce zero hidden retries")
            provider.max_retries = 0
            provider.follow_redirects = False
            if isinstance(provider, OpenAIProvider):
                provider.total_completion_limit = True
        else:
            provider.bounded_transport = True
        provider.request_timeout_s = self.execution.request_timeout_s
        provider._runtime_bounded = True

    def _cache_hit(self, provider, model, kind):
        context = _operation.get()
        if context is not None and context.conversation is not None:
            context.conversation.cache_hit(kind, context.inference_role)
        if self.budget is not None:
            self.budget.record_cache_hit(session=self.session_id,
                purpose=_operation.get().purpose if _operation.get() else kind, provider=provider.name, model=model, kind=kind)

    def route(self, tier: TaskTier) -> tuple[LLMProvider, RoutingDecision]:
        """Decide which provider handles ``tier`` and return it + decision."""
        if self.execution.mode == "local":
            if self._diagnostic_chat is not None:
                provider = self._diagnostic_chat
                return provider, RoutingDecision(provider.name, provider.model_for(tier), tier, 'offline_local_diagnostic')
            provider = self.providers.get("ollama")
            if not isinstance(provider, OllamaProvider):
                raise ExecutionDeferred("Local Ollama provider is unavailable")
            return provider, RoutingDecision(provider.name, provider.model_for(tier), tier, "explicit_local")
        force_cloud_chat = self.config.local_ollama.force_cloud_chat
        if (
            tier == TaskTier.SIMPLE
            and self.execution.mode == "research"
            and self._ollama_ready
            and not force_cloud_chat
        ):
            ollama = self.providers["ollama"]
            return ollama, RoutingDecision(
                provider=ollama.name,
                model_id=ollama.model_for(tier),
                tier=tier,
                reason="ollama_available",
            )
        active = self.providers[self.config.active_provider]
        return active, RoutingDecision(
            provider=active.name,
            model_id=active.model_for(tier),
            tier=tier,
            reason=(
                "advanced_tier_uses_cloud" if tier == TaskTier.ADVANCED
                else "forced_cloud" if force_cloud_chat or self.execution.mode == "paid"
                else "ollama_unavailable_fallback_to_cloud"
            ),
        )

    # ---------- chat ----------

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        tier: TaskTier = TaskTier.SIMPLE,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        use_cache: bool = True,
    ) -> Completion:
        """Dispatch a chat completion.

        ``use_cache=True`` is the default because semantic-analysis calls
        over the same inputs are the project's dominant workload. Set
        ``False`` for exploratory one-offs or when the response should
        vary across identical inputs (temperature>0 and you actually
        want multiple samples)."""
        messages_list = list(messages)
        remaining_time()
        provider, decision = self.route(tier)
        if isinstance(provider, LlamaCppProvider) and use_cache is not False:
            raise ValueError('Native diagnostic does not accept cache reads or writes')

        if self.cache is not None and use_cache:
            key = chat_cache_key(
                provider=provider.name,
                model_id=decision.model_id,
                tier=tier,
                messages=messages_list,
                temperature=temperature,
                max_tokens=max_tokens,
                response_schema=getattr(provider, 'response_schema', None),
                think=getattr(provider, 'think', None),
            )
            cached = self.cache.get_chat(key)
            if cached is not None:
                self._cache_hit(provider, decision.model_id, "chat")
                return cached

        completion = self._dispatch(provider, decision.model_id, "chat", lambda: provider.chat(
            messages_list,
            tier=tier,
            max_tokens=max_tokens,
            temperature=temperature,
        ), messages=messages_list, max_tokens=max_tokens)

        if self.cache is not None and use_cache:
            self.cache.put_chat(key, completion)

        return completion

    def chat_batch(
        self,
        prompts: Sequence[Sequence[ChatMessage]],
        *,
        tier: TaskTier = TaskTier.SIMPLE,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        use_cache: bool = True,
        max_workers: int = 4,
    ) -> list[Completion]:
        with self.operation("chat_batch"):
            return self._chat_batch(prompts, tier=tier, max_tokens=max_tokens,
                                    temperature=temperature, use_cache=use_cache, max_workers=max_workers)

    def _chat_batch(self, prompts, *, tier, max_tokens, temperature, use_cache, max_workers):
        """Run many independent chat prompts concurrently.

        Semantic-analysis workloads batch naturally — summarizing a
        session's fragments, scoring a question's candidate recalls,
        classifying utterances by mentalistic object. This surface lets
        callers push them in one call rather than looping and paying the
        serial latency.

        ``max_workers`` kept small by default (4) — Ollama happily
        parallelizes on a single GPU up to this depth; cloud providers
        pipeline at this concurrency without hitting rate limits for
        typical research throughput.
        """
        if not prompts:
            return []
        prompts_list = [list(p) for p in prompts]

        def _one(msgs: list[ChatMessage]) -> Completion:
            return self.chat(
                msgs,
                tier=tier,
                max_tokens=max_tokens,
                temperature=temperature,
                use_cache=use_cache,
            )

        if max_workers <= 1 or len(prompts_list) == 1:
            return [_one(p) for p in prompts_list]

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            context = copy_context()
            return list(pool.map(lambda messages: context.copy().run(_one, messages), prompts_list))

    # ---------- embeddings ----------

    def _embedding_target(self, backend, *, probe=True):
        """Return an embedding vector. Routes to Ollama when available.

        Falls back to a provider error when Ollama is unreachable AND
        the active cloud provider isn't an embedding-capable one. (We
        don't silently invoke a cloud embedding endpoint — that would
        inflate spend on what the user set up to run locally.)

        ``local_ollama.force_cloud_embeddings=True`` overrides the
        local-first default and routes to the active cloud provider's
        ``embed()`` directly — the Phase 1.5 E1 v4 lever for swapping
        mxbai-embed-large → DashScope text-embedding-v3.

        ``backend`` lets callers request a specific backend regardless
        of config. Phase 1.5-prime P2 (embedder ensemble) uses this to
        issue parallel ``backend="local"`` + ``backend="cloud"`` calls
        and max-pool the cosines downstream. Values:
          - "auto"  (default): current config-driven routing.
          - "local": Ollama (hard-fails if unavailable).
          - "cloud": active provider's embed() (hard-fails if missing).
        """
        if backend not in ("auto", "local", "cloud"):
            raise ValueError("Embedding backend must be auto/local/cloud")
        if self.execution.mode == "local":
            if backend == "cloud":
                raise ExecutionDeferred("Local execution refuses cloud embeddings")
            backend = "local"
        elif self.execution.mode == "paid" and backend == "auto":
            backend = "cloud" if self.config.local_ollama.force_cloud_embeddings else "local"
        if backend == "local":
            provider = self.providers.get("ollama")
            if (provider is None or not hasattr(provider, "embed")
                    or (self.execution.mode != "research" and not isinstance(provider, OllamaProvider))
                    or (not self._ollama_ready and not isinstance(provider, OllamaProvider))):
                if self.execution.mode != "research":
                    raise ExecutionDeferred("Configured local embedding provider is unavailable")
                raise RuntimeError(
                    "embed(backend='local') requires Ollama reachable + an "
                    "embedding-capable local provider."
                )
        elif backend == "cloud":
            provider = self.providers[self.config.active_provider]
            if not hasattr(provider, "embed"):
                raise RuntimeError(
                    f"embed(backend='cloud') but active provider "
                    f"({self.config.active_provider}) does not implement embed()."
                )
        else:
            force_cloud = self.config.local_ollama.force_cloud_embeddings
            if force_cloud:
                provider = self.providers[self.config.active_provider]
                if not hasattr(provider, "embed"):
                    raise RuntimeError(
                        f"force_cloud_embeddings=True but active provider "
                        f"({self.config.active_provider}) does not implement embed()."
                    )
            else:
                provider = self.providers.get("ollama") if self._ollama_ready else None
                if provider is None or not hasattr(provider, "embed"):
                    # Also allow MockProvider / future embedding-capable cloud
                    # providers to serve via the active slot.
                    provider = self.providers[self.config.active_provider]
                    if not hasattr(provider, "embed"):
                        raise RuntimeError(
                            "No embedding-capable provider is available. Ollama is "
                            "unreachable or disabled, and the active cloud provider "
                            f"({self.config.active_provider}) does not implement embed()."
                        )

        # Prefer the provider's dedicated embedding_model when it has one;
        # fall back to the simple-tier chat model for providers that don't
        # distinguish (e.g. MockProvider in tests). This is the recall-
        # quality lever Phase 1.5 E1 v1 exposed.
        #
        # Phase 9+ i18n hook: when ``i18n.embedder_multilingual`` is true
        # AND the resolved provider exposes a non-null
        # ``embedding_model_multilingual``, prefer that slot. The default
        # path (flag OFF, slot None) is byte-identical to the pre-i18n
        # dispatch — ``multilingual`` only overrides when both conditions
        # hold, so providers whose default embedder is already strongly
        # multilingual (DashScope text-embedding-v3) need no second slot.
        default_model = (
            getattr(provider, "embedding_model", None)
            or provider.model_for(TaskTier.SIMPLE)
        )
        multilingual_slot = getattr(provider, "embedding_model_multilingual", None)
        if self.config.i18n.embedder_multilingual and multilingual_slot:
            model_id = multilingual_slot
        else:
            model_id = default_model

        if isinstance(provider, OllamaProvider) and provider.local_only:
            configured_model = (multilingual_slot if self.config.i18n.embedder_multilingual and multilingual_slot
                                else provider.embedding_model)
            if not configured_model or configured_model != model_id:
                raise ValueError("Pinned local embedding requires a dedicated embedding model")
        elif self.execution.mode == "research" and probe and backend == "local" and not self._ollama_ready:
            if not provider.probe(timeout_s=self.config.local_ollama.probe_timeout_s, model=model_id):
                raise RuntimeError("Configured local embedding model is unavailable")
        return provider, model_id

    def _embed_verified_batch(self, provider, model_id, texts, use_cache, expected_identity=None):
        if self.execution.mode == "research":
            if _operation.get() is not None and _operation.get().conversation is not None:
                raise ExecutionDeferred("Research dispatch cannot bypass a serving turn allowance")
            return self._embed_verified_batch_body(provider, model_id, texts, use_cache, expected_identity)
        with self.operation("embedding_batch"), request_slot() as send:
            return send(lambda: self._embed_verified_batch_body(provider, model_id, texts, use_cache, expected_identity))

    def _embed_verified_batch_body(self, provider, model_id, texts, use_cache, expected_identity=None):
        if any(not isinstance(text, str) for text in texts):
            raise ValueError("Embedding inputs must be text")
        results, missing, keys = {}, [], {}
        with provider.verified_embedding_batch(model_id) as (identity, encode):
            def encode_chunk(chunk):
                return self._meter_turn(provider, model_id, "embed", lambda: encode(chunk),
                                        messages=[ChatMessage("user", text) for text in chunk])
            if expected_identity is not None and identity != expected_identity:
                raise ValueError("Frozen embedding identity differs from the verified batch")
            if not use_cache:
                fresh = []
                for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
                    fresh.extend(encode_chunk(texts[start:start + EMBEDDING_BATCH_SIZE]))
                return fresh
            unique = list(dict.fromkeys(texts))
            for text in unique:
                key = embedding_cache_key(provider=provider.name, model_id=model_id,
                                          text=text, identity=identity)
                keys[text] = key
                cached = self.cache.get_embedding(key) if self.cache is not None and use_cache else None
                if cached is None:
                    missing.append(text)
                else:
                    results[text] = validate_embedding(cached, dimension=identity["dimension"])
                    self._cache_hit(provider, model_id, "embed")
            for start in range(0, len(missing), EMBEDDING_BATCH_SIZE):
                chunk = missing[start:start + EMBEDDING_BATCH_SIZE]
                results.update(zip(chunk, encode_chunk(chunk)))
        # Failed responses or identity changes cannot publish partial new entries.
        remaining_time()
        if self.cache is not None and use_cache:
            for text in missing:
                self.cache.put_embedding(keys[text], results[text], model_id=model_id)
        return [list(results[text]) for text in texts]

    def embed(self, text: str, *, use_cache: bool = True, backend: str = "auto") -> list[float]:
        """Embed through the configured target, retaining legacy unpinned routing."""
        remaining_time()
        provider, model_id = self._embedding_target(backend)
        if isinstance(provider, OllamaProvider) and provider.local_only:
            return self._embed_verified_batch(provider, model_id, [text], use_cache)[0]

        if self.cache is not None and use_cache:
            key = embedding_cache_key(
                provider=provider.name,
                model_id=model_id,
                text=text,
            )
            cached = self.cache.get_embedding(key)
            if cached is not None:
                self._cache_hit(provider, model_id, "embed")
                return cached

        # Only pass ``model=`` to providers that accept the kwarg — the
        # three real providers (Ollama / AlibabaTongyi / Mock) do; keeping
        # the no-kwarg call for providers that don't would be a wart, so
        # we standardize all three to accept it in this i18n slice.
        vector = self._dispatch(provider, model_id, "embed", lambda: provider.embed(text, model=model_id),
                                messages=[ChatMessage("user", text)])

        if self.cache is not None and use_cache:
            self.cache.put_embedding(key, vector, model_id=model_id)
        return vector

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        top_n: Optional[int] = None,
    ) -> list[tuple[int, float]]:
        """Rerank through an explicit offline adapter or the existing provider.

        No local adapter is selected by default. Attaching an experimental
        adapter permits measurement; it does not qualify automatic adoption.
        """
        if not documents:
            return []
        if self.local_reranker is not None:
            provider = self.local_reranker
            return self._dispatch(provider, provider.model_id, "rerank",
                                  lambda: provider.rerank(query, documents, top_n=top_n),
                                  messages=[ChatMessage("user", query + "\n" + document) for document in documents])
        if self.execution.mode == "local":
            raise ExecutionDeferred("A qualified local reranker is not configured")
        provider = self.providers[self.config.active_provider]
        if not hasattr(provider, "rerank"):
            raise RuntimeError(
                f"rerank() requires an active cloud provider that implements "
                f"rerank(). {self.config.active_provider} does not."
            )
        # Query is billed/processed alongside each candidate, not just once.
        messages = [ChatMessage("user", query + "\n" + document) for document in documents]
        model = getattr(provider, "rerank_model", provider.model_for(TaskTier.SIMPLE))
        return self._dispatch(provider, model, "rerank",
                              lambda: provider.rerank(query, list(documents), top_n=top_n), messages=messages)

    def classify_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[dict[str, float]]:
        """Explicit offline classification through the existing dispatch owner.

        No cloud provider or relevance-score substitution is permitted.
        """
        provider = self.local_reranker
        if not isinstance(provider, OnnxPairClassifier):
            raise ExecutionDeferred("An explicit local pair classifier is not configured")
        if not pairs:
            return []
        return self._dispatch(provider, provider.model_id, "classify_pairs",
                              lambda: provider.classify_pairs(pairs),
                              messages=[ChatMessage("user", first + "\n" + second) for first, second in pairs])

    def embed_batch(
        self,
        texts: Sequence[str],
        *,
        use_cache: bool = True,
        max_workers: int = 4,
        backend: str = "auto",
        expected_identity: Optional[dict] = None,
    ) -> list[list[float]]:
        with self.operation("embedding_batch"):
            return self._embed_batch(texts, use_cache=use_cache, max_workers=max_workers,
                                     backend=backend, expected_identity=expected_identity)

    def _embed_batch(self, texts, *, use_cache, max_workers, backend, expected_identity):
        if not texts:
            return []
        texts_list = list(texts)
        provider, model_id = self._embedding_target(backend, probe=False)
        if isinstance(provider, OllamaProvider) and provider.local_only:
            return self._embed_verified_batch(provider, model_id, texts_list, use_cache, expected_identity)
        if expected_identity is not None:
            raise ValueError("A frozen embedding identity requires pinned local transport")
        if max_workers <= 1 or len(texts_list) == 1:
            return [
                self.embed(t, use_cache=use_cache, backend=backend) for t in texts_list
            ]
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            context = copy_context()
            return list(
                pool.map(
                    lambda t: context.copy().run(self.embed, t, use_cache=use_cache, backend=backend),
                    texts_list,
                )
            )


# ---------- construction helpers ----------


def router_from_config(
    config_path: Optional[Path] = None,
    *,
    providers: Optional[dict[str, LLMProvider]] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> LLMRouter:
    """Build a router from a YAML config file (or env/default search)."""
    from psychology_base.llm.config import load_config

    config = load_config(config_path)
    return LLMRouter(config, providers=providers, session_id=session_id, user_id=user_id)


__all__ = [
    "LLMRouter",
    "RoutingDecision",
    "router_from_config",
]
