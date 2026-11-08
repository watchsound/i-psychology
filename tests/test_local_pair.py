"""Offline adapter safety checks; inert units are not model-quality evidence."""
from dataclasses import replace
import hashlib
from pathlib import Path
from threading import Event, Lock
from types import SimpleNamespace

import numpy as np
import pytest

from psychology_base.llm.config import default_config
from psychology_base.llm.execution import ExecutionDeferred
from psychology_base.llm.providers.local_pair import OnnxPairScorer
from psychology_base.llm.router import LLMRouter


def inert_scorer(logits=None):
    """Test transport, validation and cancellation without claiming accuracy."""
    scorer = object.__new__(OnnxPairScorer)
    calls = []
    def encode(pairs):
        calls.append(pairs)
        return [SimpleNamespace(ids=list(range(len(d))), attention_mask=[1]*len(d),
                                type_ids=[0]*len(d)) for q, d in pairs]
    scorer._tokenizer = SimpleNamespace(encode_batch=encode)
    scorer._session = SimpleNamespace(run=lambda _, features:
        [np.zeros((len(features["input_ids"]), 1)) if logits is None else np.asarray(logits)])
    scorer._inputs = {"input_ids", "attention_mask"}
    scorer._lock, scorer.max_length, scorer.batch_size = Lock(), 512, 16
    scorer.model_id = "offline-unit"
    return scorer, calls


def router(scorer, mode="local", ledger=None):
    cfg = default_config()
    cfg = replace(cfg, cache=replace(cfg.cache, enabled=False),
                  execution=replace(cfg.execution, mode=mode))
    if mode == "paid":
        from psychology_base.llm.execution import PaidRate
        cfg = replace(cfg, execution=replace(cfg.execution, ledger=ledger,
            budget_id="inert-paid-policy", max_calls=1, max_usd="0.01",
            session_max_calls=1, session_max_usd="0.01", operation_max_calls=1,
            operation_max_usd="0.01", rates={"unused/chat/unit": PaidRate("1", "1", 100, 100)}))
    return LLMRouter(cfg, providers={}, local_reranker=scorer,
                     session_id="local-pair-test" if mode == "paid" else None)


def test_local_artifact_replacement_and_missing_pin_refused(tmp_path):
    path = tmp_path / "model"
    path.write_bytes(b"original")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert OnnxPairScorer._verified_bytes(path, expected, 8) == b"original"
    path.write_bytes(b"replacement")
    with pytest.raises(ValueError, match="pin or size"):
        OnnxPairScorer._verified_bytes(path, expected, 20)
    with pytest.raises(ValueError, match="pin or size"):
        OnnxPairScorer._verified_bytes(path, hashlib.sha256(path.read_bytes()).hexdigest(), 3)
    with pytest.raises(ValueError, match="pin is required"):
        OnnxPairScorer._verified_bytes(path, "", 20)


@pytest.mark.parametrize("field,value", [("max_length", 513), ("batch_size", 0),
    ("threads", 9), ("threads", True)])
def test_resource_bounds_before_loading(field, value):
    with pytest.raises(ValueError, match="resource bound"):
        OnnxPairScorer(Path("missing"), Path("missing"), model_sha256="0"*64,
                       tokenizer_sha256="0"*64, **{field: value})


@pytest.mark.parametrize("query,documents,top_n", [("", ["a"], None), ("q", [""], None),
    ("q", ["a"]*501, None), ("q", ["a"], False), ("q", ["a"], 0),
    ("q", ["x"*32_768], None)])
def test_input_bounds_refuse_before_model_inference(query, documents, top_n):
    scorer, calls = inert_scorer()
    with pytest.raises(ValueError):
        scorer.rerank(query, documents, top_n=top_n)
    assert calls == []


def test_no_silent_token_truncation():
    scorer, _ = inert_scorer()
    with pytest.raises(ValueError, match="silent truncation refused"):
        scorer.rerank("q", ["x"*513])


@pytest.mark.parametrize("logits", [[[float("nan")]], [[float("inf")]], [[1, 2]], [1], [[1], [2]]])
def test_invalid_model_output_never_published(logits):
    scorer, _ = inert_scorer(logits)
    with pytest.raises(ValueError, match="invalid relevance logits"):
        scorer.rerank("q", ["a"])


def test_extreme_logits_stable_sorted_and_top_n():
    scorer, calls = inert_scorer([[-10_000], [10_000], [0]])
    assert scorer.rerank("q", ["a", "b", "c"]) == [(1, 1.0), (2, 0.5), (0, 0.0)]
    assert scorer.rerank("q", ["a", "b", "c"], top_n=1) == [(1, 1.0)]
    assert scorer.rerank("q", []) == []
    assert len(calls) == 2


@pytest.mark.parametrize("mode", ["local", "paid", "research"])
def test_explicit_adapter_uses_existing_rerank_without_remote_provider(mode, tmp_path):
    scorer, calls = inert_scorer([[2]])
    instance = router(scorer, mode, tmp_path / "budget.sqlite")
    assert instance.rerank("query", ["a"])[0][0] == 0
    assert calls == [[("query", "a")]]
    if instance.budget:
        assert instance.budget.summary()["provider_attempts"] == 0


def test_default_remains_unconfigured_and_foreign_adapter_refused():
    cfg = default_config()
    with pytest.raises(ExecutionDeferred, match="not configured"):
        LLMRouter(cfg, providers={}).rerank("q", ["a"])
    with pytest.raises(ValueError, match="offline pinned"):
        LLMRouter(cfg, providers={}, local_reranker=object())


def test_late_scoring_result_discarded_and_slot_released():
    scorer, _ = inert_scorer()
    entered, release, ended = Event(), Event(), Event()
    def blocked(*args):
        entered.set()
        try:
            release.wait(10)
            return [np.zeros((1, 1))]
        finally:
            ended.set()
    scorer._session.run = blocked
    instance = router(scorer)
    try:
        with pytest.raises(ExecutionDeferred):
            with instance.operation("test-pair-timeout", timeout_s=1):
                instance.rerank("q", ["a"])
        assert entered.is_set()
    finally:
        release.set()
        assert ended.wait(5)
    assert instance.rerank("q", ["a"]) == [(0, 0.5)]


def test_busy_scorer_obeys_shared_deadline():
    scorer, calls = inert_scorer()
    instance = router(scorer)
    scorer._lock.acquire()
    try:
        with pytest.raises(ExecutionDeferred):
            with instance.operation("test-pair-busy", timeout_s=0.05):
                instance.rerank("q", ["a"])
        assert not calls
    finally:
        scorer._lock.release()


def test_real_offline_model_unicode_and_batch_contract():
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")
    import json
    path = Path(__file__).resolve().parents[1] / ".cache/models/mmarco-minilm-int8-v1"
    if not (path / "manifest.json").exists():
        pytest.skip("Optional locally downloaded pinned model absent")
    files = json.loads((path / "manifest.json").read_text())["files"]
    scorer = OnnxPairScorer(path / "onnx/model_quint8_avx2.onnx", path / "tokenizer.json",
        model_sha256=files["onnx/model_quint8_avx2.onnx"]["sha256"],
        tokenizer_sha256=files["tokenizer.json"]["sha256"], batch_size=1)
    instance = router(scorer)
    documents = ["I enjoy tea.", "考试让我焦虑。", "The library closes at six."]
    single_batches = dict(instance.rerank("测验引发恐慌", documents))
    scorer.batch_size = 16
    joint = dict(instance.rerank("测验引发恐慌", documents))
    assert joint.keys() == single_batches.keys() == {0, 1, 2}
    # Dynamic quantization may depend on batch padding. Expose numerical drift
    # rather than requiring bit identity across a different batching policy.
    assert max(joint, key=joint.get) == max(single_batches, key=single_batches.get) == 1
    assert all(0 <= value <= 1 for value in joint.values())
    assert instance.rerank("测验引发恐慌", documents, top_n=1)[0][0] == 1
