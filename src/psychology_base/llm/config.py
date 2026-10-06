"""Config loader for the LLM router.

YAML file + env-var overrides for API keys. Loaded once at
``LLMRouter.__init__``; a changed config requires a new router.

Config shape (see ``configs/llm.yaml`` for the default)::

    active_provider: anthropic           # which cloud provider to use
    providers:
      anthropic:
        simple_model: claude-haiku-4-5-20251001
        advanced_model: claude-opus-4-7
        api_key_env: ANTHROPIC_API_KEY
      openai:
        simple_model: gpt-4o-mini
        advanced_model: gpt-4o
        api_key_env: OPENAI_API_KEY
      ollama:
        base_url: http://localhost:11434
        simple_model: qwen2.5:7b
        advanced_model: qwen2.5:32b
    local_ollama:
      enabled: true                      # auto-prefer local for SIMPLE
      probe_timeout_s: 1.0
      require_model_pulled: true         # fail probe if simple_model isn't pulled
    cache:
      enabled: true
      dir: null                          # null → default ~/.cache/psychology_base/llm
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


DEFAULT_CONFIG_PATHS: tuple[Path, ...] = (
    Path.cwd() / "configs" / "llm.yaml",
    Path.home() / ".config" / "psychology_base" / "llm.yaml",
)


@dataclass(frozen=True)
class ProviderConfig:
    """Per-provider slot config.

    ``base_url`` is only meaningful for Ollama (and future self-hosted
    OpenAI-compatible providers). Cloud providers ignore it.
    """

    name: str
    simple_model: str
    advanced_model: str
    api_key_env: Optional[str] = None
    base_url: Optional[str] = None
    request_timeout_s: Optional[float] = None  # Ollama only; None → provider default
    # Dedicated embedding model (Ollama only). Chat-tuned LLMs like
    # qwen2.5:7b expose /api/embeddings but their vectors are not
    # retrieval-tuned — Phase 1.5 E1 proved them degenerate (cos(cue,
    # target) < cos(cue, distractor)). Set this to a true embedder like
    # bge-m3, mxbai-embed-large, or bge-small-zh-v1.5. None → fall back
    # to simple_model (preserved for pre-Phase-1.5 config files).
    embedding_model: Optional[str] = None

    def api_key(self) -> Optional[str]:
        """Resolve the API key from env. Returns ``None`` for providers
        that don't need one (Ollama) or when the env var is unset."""
        if not self.api_key_env:
            return None
        return os.environ.get(self.api_key_env)


@dataclass(frozen=True)
class LocalOllamaConfig:
    enabled: bool = True
    probe_timeout_s: float = 1.0
    require_model_pulled: bool = True
    # When true, SIMPLE-tier chat is forced to the active cloud provider
    # even if the Ollama probe succeeds. Embeddings continue to use
    # Ollama (if ``enabled`` + probe pass). Lets us run a cloud-chat
    # + local-embed hybrid without disabling the Ollama provider wholesale.
    force_cloud_chat: bool = False
    # When true, embeddings also route to the active cloud provider
    # (which must implement ``embed()``). Phase 1.5 E1 v3 showed
    # mxbai-embed-large inverted "origin Sweden" vs "experience transition"
    # on LoCoMo cues — the swap to DashScope text-embedding-v3 is the
    # Bucket-A fix. Independent of ``force_cloud_chat`` so a run can
    # keep local qwen-turbo for chat but still use cloud embeddings.
    force_cloud_embeddings: bool = False


@dataclass(frozen=True)
class CacheConfig:
    enabled: bool = True
    dir: Optional[Path] = None  # None → default under ~/.cache/

    def resolved_dir(self) -> Path:
        if self.dir is not None:
            return self.dir
        env = os.environ.get("PSYCHOLOGY_BASE_LLM_CACHE_DIR")
        if env:
            return Path(env)
        return Path.home() / ".cache" / "psychology_base" / "llm"


@dataclass(frozen=True)
class LLMConfig:
    active_provider: str
    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    local_ollama: LocalOllamaConfig = field(default_factory=LocalOllamaConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)

    def get_provider(self, name: str) -> ProviderConfig:
        if name not in self.providers:
            raise KeyError(
                f"Provider {name!r} not configured. Available: "
                f"{sorted(self.providers)}"
            )
        return self.providers[name]


def _parse_provider(name: str, raw: dict[str, Any]) -> ProviderConfig:
    try:
        timeout = raw.get("request_timeout_s")
        return ProviderConfig(
            name=name,
            simple_model=raw["simple_model"],
            advanced_model=raw["advanced_model"],
            api_key_env=raw.get("api_key_env"),
            base_url=raw.get("base_url"),
            request_timeout_s=float(timeout) if timeout is not None else None,
            embedding_model=raw.get("embedding_model"),
        )
    except KeyError as exc:
        raise ValueError(
            f"Provider {name!r} missing required field: {exc.args[0]!r}. "
            f"Each provider must declare simple_model + advanced_model."
        ) from exc


def _parse_config(raw: dict[str, Any]) -> LLMConfig:
    if "active_provider" not in raw:
        raise ValueError("LLM config missing required field: 'active_provider'.")

    providers_raw = raw.get("providers", {})
    if not isinstance(providers_raw, dict) or not providers_raw:
        raise ValueError("LLM config 'providers' must be a non-empty mapping.")

    providers = {
        name: _parse_provider(name, cfg)
        for name, cfg in providers_raw.items()
    }

    ollama_raw = raw.get("local_ollama") or {}
    local_ollama = LocalOllamaConfig(
        enabled=bool(ollama_raw.get("enabled", True)),
        probe_timeout_s=float(ollama_raw.get("probe_timeout_s", 1.0)),
        require_model_pulled=bool(ollama_raw.get("require_model_pulled", True)),
        force_cloud_chat=bool(ollama_raw.get("force_cloud_chat", False)),
        force_cloud_embeddings=bool(ollama_raw.get("force_cloud_embeddings", False)),
    )

    cache_raw = raw.get("cache") or {}
    cache_dir_raw = cache_raw.get("dir")
    cache = CacheConfig(
        enabled=bool(cache_raw.get("enabled", True)),
        dir=Path(cache_dir_raw) if cache_dir_raw else None,
    )

    active = raw["active_provider"]
    if active not in providers:
        raise ValueError(
            f"active_provider={active!r} but no such provider is configured. "
            f"Configured: {sorted(providers)}"
        )

    return LLMConfig(
        active_provider=active,
        providers=providers,
        local_ollama=local_ollama,
        cache=cache,
    )


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader — populates ``os.environ`` from ``KEY=value``
    lines, skipping blanks and ``#`` comments. Does NOT overwrite an
    existing env var (so an explicit shell export always wins).

    Pattern deliberately shallow: no quoted-string parsing, no var
    interpolation, no multi-line values. The ``.env.example`` shape is
    bare ``KEY=value`` and that is all this handles.
    """
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _ensure_dotenv_loaded() -> None:
    """Load ``./.env`` once per process on config access. Idempotent."""
    marker = "_PSYCHOLOGY_BASE_DOTENV_LOADED"
    if os.environ.get(marker) == "1":
        return
    _load_dotenv(Path.cwd() / ".env")
    os.environ[marker] = "1"


def load_config(path: Optional[Path] = None) -> LLMConfig:
    """Load LLM config from ``path``, or search default locations.

    Default search order:
      1. ``$PSYCHOLOGY_BASE_LLM_CONFIG`` env var if set
      2. ``./configs/llm.yaml`` (project local)
      3. ``~/.config/psychology_base/llm.yaml`` (user global)

    Side effect: loads ``./.env`` into ``os.environ`` (shallow parser,
    doesn't overwrite existing vars) so ``ProviderConfig.api_key()``
    resolves without the user having to export every key per shell.

    Raises ``FileNotFoundError`` if nothing is found. Callers that want
    a built-in default should catch that and call ``default_config()``.
    """
    _ensure_dotenv_loaded()
    if path is not None:
        return _parse_config(yaml.safe_load(path.read_text(encoding="utf-8")) or {})

    env_path = os.environ.get("PSYCHOLOGY_BASE_LLM_CONFIG")
    if env_path:
        return _parse_config(
            yaml.safe_load(Path(env_path).read_text(encoding="utf-8")) or {}
        )

    for candidate in DEFAULT_CONFIG_PATHS:
        if candidate.exists():
            return _parse_config(
                yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
            )

    raise FileNotFoundError(
        f"No LLM config found. Looked in: PSYCHOLOGY_BASE_LLM_CONFIG, "
        f"{', '.join(str(p) for p in DEFAULT_CONFIG_PATHS)}. "
        f"Call default_config() for an in-memory fallback or create "
        f"configs/llm.yaml."
    )


def default_config() -> LLMConfig:
    """In-memory fallback config — Ollama-only, no cloud providers.

    Use this when no YAML file exists and the caller wants a working
    router against local models only. Mostly for tests + onboarding.
    """
    return LLMConfig(
        active_provider="ollama",
        providers={
            "ollama": ProviderConfig(
                name="ollama",
                simple_model="qwen2.5:7b",
                advanced_model="qwen2.5:32b",
                base_url="http://localhost:11434",
            ),
        },
        local_ollama=LocalOllamaConfig(enabled=True),
        cache=CacheConfig(enabled=True),
    )
