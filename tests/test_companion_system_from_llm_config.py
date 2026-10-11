"""Phase 9+ i18n foundation — ``CompanionSystem.from_llm_config`` factory.

The factory maps ``cfg.i18n.tokenizer`` → ``cjk_tokenizer`` so that
flipping ``i18n.tokenizer: jieba`` in ``configs/llm.yaml`` reaches the
live tutor / companion / argilla / pam-restore surfaces without each
entry point having to re-plumb the knob manually.

Benchmarks and FANToM / BIG5 / B4 scorers continue to construct
``CompanionSystem()`` directly — the factory is NOT the default
construction path, by design (English-forcing determinism).
"""

from __future__ import annotations

from psychology_base.companion.system import CompanionSystem
from psychology_base.llm.config import (
    I18nConfig,
    LLMConfig,
    ProviderConfig,
)


def _cfg_with_tokenizer(tokenizer: str) -> LLMConfig:
    """Minimal LLMConfig satisfying the dataclass contract.

    Only ``i18n.tokenizer`` is consulted by the factory; the provider
    slot is required to construct a valid ``LLMConfig`` but is unused
    here. Mirrors the shape ``load_config()`` would yield from a YAML
    with the matching ``i18n.tokenizer`` value.
    """
    return LLMConfig(
        active_provider="dummy",
        providers={
            "dummy": ProviderConfig(
                name="dummy",
                simple_model="unused",
                advanced_model="unused",
            )
        },
        i18n=I18nConfig(tokenizer=tokenizer),
    )


def test_factory_reads_tokenizer_from_llm_config_jieba() -> None:
    """``i18n.tokenizer=jieba`` flows into ``cjk_tokenizer``."""
    cfg = _cfg_with_tokenizer("jieba")
    system = CompanionSystem.from_llm_config(cfg)
    assert system.cjk_tokenizer == "jieba"


def test_factory_reads_tokenizer_from_llm_config_unicode_default() -> None:
    """Default YAML (``tokenizer: unicode``) stays byte-identical to
    direct ``CompanionSystem()`` construction."""
    cfg = _cfg_with_tokenizer("unicode")
    factory_system = CompanionSystem.from_llm_config(cfg)
    direct_system = CompanionSystem()
    assert factory_system.cjk_tokenizer == direct_system.cjk_tokenizer == "unicode"


def test_factory_explicit_override_wins_over_config() -> None:
    """A caller passing ``cjk_tokenizer=...`` explicitly overrides the
    YAML. This path is how an English-forcing benchmark that happens
    to load the shared YAML (e.g. for provider slots) can still pin
    the legacy "unicode" path."""
    cfg = _cfg_with_tokenizer("jieba")
    system = CompanionSystem.from_llm_config(cfg, cjk_tokenizer="unicode")
    assert system.cjk_tokenizer == "unicode"


def test_factory_passes_through_other_kwargs() -> None:
    """Non-i18n kwargs flow through untouched — the factory only owns
    the tokenizer mapping, not the full constructor surface."""
    from psychology_base.companion.aging import EbbinghausPolicy

    cfg = _cfg_with_tokenizer("unicode")
    policy = EbbinghausPolicy()
    system = CompanionSystem.from_llm_config(cfg, aging_policy=policy)
    assert system.aging_policy is policy
    assert system.cjk_tokenizer == "unicode"


def test_factory_default_cjk_tokenizer_is_unicode_when_cfg_omits() -> None:
    """``I18nConfig()`` default ``tokenizer="unicode"`` means a config
    constructed without any explicit i18n block still yields the
    legacy English-forcing path. Regression lock on the default."""
    cfg = LLMConfig(
        active_provider="dummy",
        providers={
            "dummy": ProviderConfig(
                name="dummy",
                simple_model="unused",
                advanced_model="unused",
            )
        },
    )
    system = CompanionSystem.from_llm_config(cfg)
    assert system.cjk_tokenizer == "unicode"
