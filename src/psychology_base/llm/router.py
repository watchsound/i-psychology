"""Single entry point for all LLM calls in the project.

Pattern cribbed from the reference project's ``UnifiedLLMService``
(``ai-business/Stock-AI/.../services/llm/unified_service.py``), simplified
for sync + research-library use:

- probe Ollama on init when ``local_ollama.enabled``;
- route ``TaskTier.SIMPLE`` to Ollama when the probe passed, else to the
  active cloud provider's simple model;
- route ``TaskTier.ADVANCED`` to the active cloud provider's advanced
  model (never Ollama, because the only local models users realistically
  pull are the simple ones);
- cache responses on disk (SHA-keyed) by default — semantic-analysis
  calls over the same inputs during benchmark iteration never repeat;
- ``chat_batch`` runs independent prompts in parallel via a threadpool
  (Ollama can parallelize, cloud providers at least pipeline).

The router is cheap to construct. Hot paths should hold a singleton;
unit tests can build throwaway routers with ``MockProvider``.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
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
from psychology_base.llm.types import ChatMessage, Completion, LLMProvider, TaskTier


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoutingDecision:
    """Which provider+model handled a given call; stamped on each call for
    the on-disk ``llm_calls.jsonl`` debug log (future) and surfaced by
    tests to verify routing."""

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
        return OllamaProvider(**kwargs)
    if cfg.name == "openai":
        return OpenAIProvider(
            simple_model=cfg.simple_model,
            advanced_model=cfg.advanced_model,
            api_key=cfg.api_key(),
            base_url=cfg.base_url,
        )
    if cfg.name == "anthropic":
        return AnthropicProvider(
            simple_model=cfg.simple_model,
            advanced_model=cfg.advanced_model,
            api_key=cfg.api_key(),
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

    Constructor side effects (deliberate — single place, startup time):
      1. build the active cloud provider,
      2. if ``local_ollama.enabled``: build an Ollama provider and probe,
      3. open the on-disk cache dir.
    """

    def __init__(
        self,
        config: LLMConfig,
        *,
        providers: Optional[dict[str, LLMProvider]] = None,
        cache: Optional[LLMCache] = None,
    ) -> None:
        self.config = config

        # Allow tests to inject fully-constructed providers (e.g.
        # MockProvider). Falls back to building from config.
        if providers is None:
            providers = {}
            # Active cloud provider — must succeed.
            active_cfg = config.get_provider(config.active_provider)
            providers[active_cfg.name] = _build_provider(active_cfg)
            # Ollama provider — opt-in via config.
            if config.local_ollama.enabled and "ollama" in config.providers:
                ollama_cfg = config.get_provider("ollama")
                providers["ollama"] = _build_provider(ollama_cfg)
        self.providers = providers

        # Probe Ollama once at construction. If it's reachable AND the
        # simple_model is pulled, SIMPLE routes to it; otherwise fall
        # back to the active cloud provider's simple slot.
        self._ollama_ready = False
        ollama = self.providers.get("ollama")
        if (
            config.local_ollama.enabled
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

    def route(self, tier: TaskTier) -> tuple[LLMProvider, RoutingDecision]:
        """Decide which provider handles ``tier`` and return it + decision."""
        force_cloud_chat = self.config.local_ollama.force_cloud_chat
        if (
            tier == TaskTier.SIMPLE
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
        provider, decision = self.route(tier)

        if self.cache is not None and use_cache:
            key = chat_cache_key(
                provider=provider.name,
                model_id=decision.model_id,
                tier=tier,
                messages=messages_list,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            cached = self.cache.get_chat(key)
            if cached is not None:
                return cached

        completion = provider.chat(
            messages_list,
            tier=tier,
            max_tokens=max_tokens,
            temperature=temperature,
        )

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
            return list(pool.map(_one, prompts_list))

    # ---------- embeddings ----------

    def embed(
        self,
        text: str,
        *,
        use_cache: bool = True,
        backend: str = "auto",
    ) -> list[float]:
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
        if backend == "local":
            provider = self.providers.get("ollama") if self._ollama_ready else None
            if provider is None or not hasattr(provider, "embed"):
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
        model_id = (
            getattr(provider, "embedding_model", None)
            or provider.model_for(TaskTier.SIMPLE)
        )
        if self.cache is not None and use_cache:
            key = embedding_cache_key(
                provider=provider.name,
                model_id=model_id,
                text=text,
            )
            cached = self.cache.get_embedding(key)
            if cached is not None:
                return cached

        vector = provider.embed(text)

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
        """Pass-through to the active cloud provider's ``rerank`` method.

        Phase 1.5-prime P3 uses this to post-rerank the cosine top-K
        fragments through DashScope ``gte-rerank-v2``. Returns a list of
        ``(original_index, relevance_score)`` sorted by score desc.

        Hard-fails if the active provider doesn't implement rerank —
        reranking is a cloud-only feature in the Phase 1.5-prime scope.
        """
        provider = self.providers[self.config.active_provider]
        if not hasattr(provider, "rerank"):
            raise RuntimeError(
                f"rerank() requires an active cloud provider that implements "
                f"rerank(). {self.config.active_provider} does not."
            )
        return provider.rerank(query, list(documents), top_n=top_n)

    def embed_batch(
        self,
        texts: Sequence[str],
        *,
        use_cache: bool = True,
        max_workers: int = 4,
        backend: str = "auto",
    ) -> list[list[float]]:
        if not texts:
            return []
        texts_list = list(texts)
        if max_workers <= 1 or len(texts_list) == 1:
            return [
                self.embed(t, use_cache=use_cache, backend=backend) for t in texts_list
            ]
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return list(
                pool.map(
                    lambda t: self.embed(t, use_cache=use_cache, backend=backend),
                    texts_list,
                )
            )


# ---------- construction helpers ----------


def router_from_config(
    config_path: Optional[Path] = None,
    *,
    providers: Optional[dict[str, LLMProvider]] = None,
) -> LLMRouter:
    """Build a router from a YAML config file (or env/default search)."""
    from psychology_base.llm.config import load_config

    config = load_config(config_path)
    return LLMRouter(config, providers=providers)


__all__ = [
    "LLMRouter",
    "RoutingDecision",
    "router_from_config",
]
