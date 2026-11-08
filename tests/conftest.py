"""Shared text controls; lifecycle and assertions stay in their existing suites."""

import json
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def chinese_cases():
    path = Path(__file__).resolve().parents[1] / "benchmarks/fixtures/chinese_tokenization/cases.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def unavailable_jieba(monkeypatch):
    """Negative-path injection; positive cases always use the installed backend."""
    from dataclasses import replace
    import psychology_base.i18n as i18n
    import psychology_base.llm.config as config

    cfg = replace(config.load_config(), i18n=config.I18nConfig(tokenizer="jieba"))
    monkeypatch.setattr(config, "load_config", lambda: cfg)
    monkeypatch.setattr(i18n, "_load_jieba", lambda: (None, {}, "not installed"))
