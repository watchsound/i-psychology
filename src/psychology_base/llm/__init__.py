"""LLM provider abstraction for psychology-base.

The project uses LLMs in several places (RAG-mode benchmark reader, free-form
LoCoMo judging, future semantic analysis of memory fragments, Witness
divergence reasoning). All of them route through this package rather than
calling vendor SDKs directly, so a single config file controls provider +
model selection project-wide.

Two tiers:
  - SIMPLE: cheap / fast / default. Suitable for semantic classification,
    fragment summarization, judge-style yes/no calls. Prefers local Ollama
    when reachable (configurable), falls back to the configured cloud
    simple-model.
  - ADVANCED: higher-quality reasoning. Reserved for complicated planning,
    multi-step analysis, and tasks where a wrong answer invalidates
    downstream work.
"""

from psychology_base.llm.cache import LLMCache
from psychology_base.llm.config import (
    CacheConfig,
    LLMConfig,
    LocalOllamaConfig,
    ProviderConfig,
    default_config,
    load_config,
)
from psychology_base.llm.router import LLMRouter, RoutingDecision, router_from_config
from psychology_base.llm.types import (
    ChatMessage,
    Completion,
    LLMProvider,
    TaskTier,
)

__all__ = [
    "CacheConfig",
    "ChatMessage",
    "Completion",
    "LLMCache",
    "LLMConfig",
    "LLMProvider",
    "LLMRouter",
    "LocalOllamaConfig",
    "ProviderConfig",
    "RoutingDecision",
    "TaskTier",
    "default_config",
    "load_config",
    "router_from_config",
]
