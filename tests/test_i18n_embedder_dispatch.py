"""Verification contract for the multilingual embedder dispatch.

Phase 9+ i18n foundation shipped ``i18n.embedder_multilingual`` as a
reserved knob. This slice wires it live into ``LLMRouter.embed``.

The core invariants are:
  * Default (flag OFF): the resolved embedding model is
    byte-identical to the pre-i18n dispatch — ``provider.embedding_model``
    (or ``simple_model`` fallback). This locks the English pipeline
    against drift across every migrated gate (Phase 1.5 Bucket A F1,
    Phase 2 M3 affect retrieval, Phase 7 B4, etc.).
  * Flag ON + slot populated: the resolved model switches to
    ``provider.embedding_model_multilingual``.
  * Flag ON + slot None: default model is kept (DashScope v3 scenario —
    its default is already multilingual; no second slot needed).
  * Cache keys segregate on model id, so flipping the flag mid-run
    does NOT serve a cached English-model vector for a Chinese query.
  * YAML config round-trip: ``i18n:`` section parses into
    ``LLMConfig.i18n`` with the three documented fields.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from psychology_base.llm import (
    I18nConfig,
    LLMCache,
    LLMConfig,
    LLMRouter,
    LocalOllamaConfig,
    ProviderConfig,
    load_config,
)
from psychology_base.llm.providers import MockProvider


def _build_config(
    *,
    embedder_multilingual: bool,
    multilingual_slot: str | None,
) -> LLMConfig:
    return LLMConfig(
        active_provider="mock",
        providers={
            "mock": ProviderConfig(
                name="mock",
                simple_model="mock-simple",
                advanced_model="mock-advanced",
                embedding_model="mxbai-embed-large",
                embedding_model_multilingual=multilingual_slot,
            ),
        },
        local_ollama=LocalOllamaConfig(enabled=False),
        i18n=I18nConfig(embedder_multilingual=embedder_multilingual),
    )


def test_i18n_default_is_english_byte_identical(tmp_path: Path) -> None:
    """Flag OFF + slot None reproduces the pre-i18n embed path exactly.

    The resolved model id (consulted by the cache key) must be the
    provider's ``embedding_model``, and the vector must match what the
    MockProvider emits for text alone — this is the regression lock
    against drifting Phase 1.5 / Phase 2 / Phase 7 numbers.
    """
    cfg = _build_config(embedder_multilingual=False, multilingual_slot=None)
    mock = MockProvider(
        name="mock",
        embedding_model="mxbai-embed-large",
        embedding_model_multilingual="bge-m3",
    )
    router = LLMRouter(cfg, providers={"mock": mock}, cache=None)

    v = router.embed("hello world")

    reference = MockProvider(name="mock").embed("hello world", model="mxbai-embed-large")
    assert v == reference


def test_i18n_flag_off_ignores_populated_slot(tmp_path: Path) -> None:
    """Even when the multilingual slot is set, OFF must not route to it.

    This is the discipline: a config change that adds a model name but
    leaves the flag OFF cannot shift the English pipeline. Users have
    to flip the flag explicitly.
    """
    cfg = _build_config(embedder_multilingual=False, multilingual_slot="bge-m3")
    mock = MockProvider(
        name="mock",
        embedding_model="mxbai-embed-large",
        embedding_model_multilingual="bge-m3",
    )
    router = LLMRouter(cfg, providers={"mock": mock}, cache=None)

    v = router.embed("hello world")
    reference_default = MockProvider(name="mock").embed(
        "hello world", model="mxbai-embed-large"
    )
    reference_multi = MockProvider(name="mock").embed(
        "hello world", model="bge-m3"
    )
    assert v == reference_default
    assert v != reference_multi


def test_i18n_flag_on_routes_to_multilingual_slot(tmp_path: Path) -> None:
    cfg = _build_config(embedder_multilingual=True, multilingual_slot="bge-m3")
    mock = MockProvider(
        name="mock",
        embedding_model="mxbai-embed-large",
        embedding_model_multilingual="bge-m3",
    )
    router = LLMRouter(cfg, providers={"mock": mock}, cache=None)

    v = router.embed("你好,世界")

    reference_multi = MockProvider(name="mock").embed("你好,世界", model="bge-m3")
    reference_default = MockProvider(name="mock").embed(
        "你好,世界", model="mxbai-embed-large"
    )
    assert v == reference_multi
    assert v != reference_default


def test_i18n_flag_on_but_slot_none_keeps_default(tmp_path: Path) -> None:
    """DashScope-v3 shape: flag ON, but provider has no second slot.

    v3 is already multilingual, so there is no benefit to defining a
    separate slot. The flag MUST not degrade the request to the
    simple-tier chat model fallback — it must stay on the default
    embedding_model.
    """
    cfg = _build_config(embedder_multilingual=True, multilingual_slot=None)
    mock = MockProvider(
        name="mock",
        embedding_model="text-embedding-v3",
        embedding_model_multilingual=None,
    )
    router = LLMRouter(cfg, providers={"mock": mock}, cache=None)

    v = router.embed("你好")
    reference = MockProvider(name="mock").embed("你好", model="text-embedding-v3")
    assert v == reference


def test_cache_segregates_on_resolved_model(tmp_path: Path) -> None:
    """Same text under different flag states must NOT share a cache slot.

    If the cache keyed only on text, a run that embedded "hello" with
    the English model, then flipped the flag and re-embedded, would
    serve the stale English vector for Chinese context. The cache key
    carries ``model_id``, so each flag state owns its own slot.
    """
    cache_dir = tmp_path / "cache"
    cfg_off = _build_config(embedder_multilingual=False, multilingual_slot="bge-m3")
    cfg_on = _build_config(embedder_multilingual=True, multilingual_slot="bge-m3")
    mock = MockProvider(
        name="mock",
        embedding_model="mxbai-embed-large",
        embedding_model_multilingual="bge-m3",
    )

    router_off = LLMRouter(cfg_off, providers={"mock": mock}, cache=LLMCache(cache_dir=cache_dir))
    v_off = router_off.embed("hello")

    router_on = LLMRouter(cfg_on, providers={"mock": mock}, cache=LLMCache(cache_dir=cache_dir))
    v_on = router_on.embed("hello")

    assert v_off != v_on, (
        "flipping the multilingual flag must select a different model, "
        "and the cache must not conflate them"
    )
    # Both calls hit the provider (no cache carry-over between models).
    assert mock.embed_calls == ["hello", "hello"]


def test_i18n_section_round_trips(tmp_path: Path) -> None:
    data = {
        "active_provider": "mock",
        "providers": {
            "mock": {
                "simple_model": "mock-simple",
                "advanced_model": "mock-advanced",
                "embedding_model": "mxbai-embed-large",
                "embedding_model_multilingual": "bge-m3",
            },
        },
        "i18n": {
            "mode": "auto",
            "tokenizer": "unicode",
            "embedder_multilingual": True,
        },
    }
    p = tmp_path / "llm.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    cfg = load_config(p)

    assert cfg.i18n.mode == "auto"
    assert cfg.i18n.tokenizer == "unicode"
    assert cfg.i18n.embedder_multilingual is True
    assert cfg.providers["mock"].embedding_model_multilingual == "bge-m3"


def test_i18n_section_missing_defaults_safe(tmp_path: Path) -> None:
    """A pre-i18n YAML (no ``i18n:`` section) must still load with
    defaults that preserve the English pipeline (flag OFF)."""
    data = {
        "active_provider": "mock",
        "providers": {
            "mock": {
                "simple_model": "mock-simple",
                "advanced_model": "mock-advanced",
            },
        },
    }
    p = tmp_path / "llm.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    cfg = load_config(p)

    assert cfg.i18n.mode == "en"
    assert cfg.i18n.tokenizer == "unicode"
    assert cfg.i18n.embedder_multilingual is False
    assert cfg.providers["mock"].embedding_model_multilingual is None
