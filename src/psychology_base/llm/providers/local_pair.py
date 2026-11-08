"""Optional offline cross-encoder, loaded from explicitly pinned local bytes.

No Hub client, remote code, download, chat fallback or confidence-based paid
escalation. Scores are sigmoid-transformed relevance logits, not calibrated
probabilities of truth, entailment or sufficient evidence.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from threading import Lock

import numpy as np

from psychology_base.llm.execution import ExecutionDeferred, remaining_time


def pair_tokenizer(tokenizer_bytes):
    """The same pinned-tokenizer preprocessing for serving and offline alignment."""
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_str(tokenizer_bytes.decode('utf-8'))
    if tokenizer.token_to_id('<pad>') != 1:
        raise ValueError('Local pair adapter requires the pinned XLM-R padding contract')
    tokenizer.no_truncation()
    tokenizer.enable_padding(pad_id=1, pad_token='<pad>')
    return tokenizer


def span_token_mask(encoding, text_pair, span):
    """Align a formatter-owned character span using this backend's encoding.

    A boundary token may also encode JSON punctuation: include it only when it
    overlaps actual focus content. Tokens wholly outside the span, another
    sequence, special tokens and padding never participate. No retokenization.
    """
    if (not isinstance(text_pair, (tuple, list)) or len(text_pair) != 2
            or any(not isinstance(text, str) for text in text_pair)
            or not isinstance(span, dict) or set(span) != {'version', 'sequence', 'start', 'end'}
            or type(span['version']) is not int or span['version'] != 1
            or type(span['sequence']) is not int or span['sequence'] != 0
            or any(type(span[k]) is not int for k in ('start', 'end'))
            or not 0 <= span['start'] < span['end'] <= len(text_pair[0])):
        raise ValueError('Invalid focus span contract')
    start, end = span['start'], span['end']
    mask = [int(sequence == 0 and attention == 1 and special == 0
                and right > start and left < end and right > left)
            for sequence, attention, special, (left, right) in zip(
                encoding.sequence_ids, encoding.attention_mask,
                encoding.special_tokens_mask, encoding.offsets)]
    if not any(mask):
        raise ValueError('Focus span has no untruncated content tokens')
    return np.asarray(mask, dtype=np.int64)


class OnnxPairScorer:
    name = "local_pair"
    MODEL_BYTES_LIMIT = 125_000_000

    def __init__(self, model_path: Path, tokenizer_path: Path, *,
                 model_sha256: str, tokenizer_sha256: str,
                 max_length: int = 512, batch_size: int = 16, threads: int = 2):
        for value, limit in ((max_length, 512), (batch_size, 64), (threads, 8)):
            if type(value) is not int or not 0 < value <= limit:
                raise ValueError("Invalid local scorer resource bound")
        # Both artifacts are consumed from the same bytes we verify. Passing a
        # byte model also prevents ONNX external-data paths from being loaded.
        model = self._verified_bytes(model_path, model_sha256, self.MODEL_BYTES_LIMIT)
        tokenizer = self._verified_bytes(tokenizer_path, tokenizer_sha256, 25_000_000)
        try:
            import onnxruntime as ort
            self._tokenizer = pair_tokenizer(tokenizer)
        except ImportError as exc:
            raise ExecutionDeferred("Install the optional local-rerank dependencies") from exc
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(model, sess_options=options,
                                             providers=["CPUExecutionProvider"])
        self._inputs = {item.name for item in self._session.get_inputs()}
        if not {"input_ids", "attention_mask"} <= self._inputs <= {
                "input_ids", "attention_mask", "token_type_ids"}:
            raise ValueError("Unsupported local pair model input contract")
        self.max_length, self.batch_size = max_length, batch_size
        self.model_id = "sha256:" + model_sha256
        self.tokenizer_sha256 = tokenizer_sha256
        self._lock = Lock()

    @staticmethod
    def _verified_bytes(path, expected, size_limit):
        if (not isinstance(expected, str) or len(expected) != 64
                or any(c not in "0123456789abcdef" for c in expected)):
            raise ValueError("An explicit lowercase SHA-256 artifact pin is required")
        with Path(path).open("rb") as handle:
            data = handle.read(size_limit + 1)
        if len(data) > size_limit or hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("Local scorer artifact differs from its pin or size bound")
        return data

    def rerank(self, query: str, documents, *, top_n=None):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("A nonempty query is required")
        documents = list(documents)
        if len(documents) > 500 or any(not isinstance(d, str) or not d.strip() for d in documents):
            raise ValueError("Local rerank requires at most 500 nonempty documents")
        if top_n is not None and (type(top_n) is not int or top_n <= 0):
            raise ValueError("top_n must be a positive integer")
        if not documents:
            return []
        if any(len(query) + len(d) > 32_768 for d in documents):
            raise ValueError("Local pair text exceeds its preprocessing bound")
        scores = self._pair_logits([(query, d) for d in documents], 1)[:, 0]
        values = np.exp(-np.logaddexp(0, -scores))
        ranked = sorted(enumerate(float(value) for value in values), key=lambda item: (-item[1], item[0]))
        return ranked if top_n is None else ranked[:top_n]

    def _pair_logits(self, pairs, columns):
        """One tokenization, batching, deadline and native-inference owner."""
        timeout = remaining_time()
        acquired = self._lock.acquire() if timeout is None else self._lock.acquire(timeout=timeout)
        if not acquired:
            raise ExecutionDeferred("Local scorer busy until operation deadline")
        try:
            batches = []
            for start in range(0, len(pairs), self.batch_size):
                remaining_time()
                encodings = self._tokenizer.encode_batch(
                    pairs[start:start + self.batch_size])
                if any(len(e.ids) > self.max_length for e in encodings):
                    raise ValueError("Local pair exceeds token limit; silent truncation refused")
                features = {"input_ids": np.asarray([e.ids for e in encodings], dtype=np.int64),
                            "attention_mask": np.asarray([e.attention_mask for e in encodings], dtype=np.int64)}
                if "token_type_ids" in self._inputs:
                    features["token_type_ids"] = np.asarray([e.type_ids for e in encodings], dtype=np.int64)
                logits = np.asarray(self._session.run(None, features)[0], dtype=np.float64)
                if logits.shape != (len(encodings), columns) or not np.isfinite(logits).all():
                    kind = "relevance" if columns == 1 else "classification"
                    raise ValueError(f"Local scorer returned invalid {kind} logits")
                batches.append(logits)
            remaining_time()
            return np.concatenate(batches, axis=0)
        finally:
            self._lock.release()


class OnnxPairClassifier(OnnxPairScorer):
    """Explicit NLI or task-trained membership labels; scores are uncalibrated.

    Shares the pinned offline loader and execution path with relevance scoring.
    It cannot be used as a reranker or selected by a default application factory.
    """
    MODEL_BYTES_LIMIT = 450_000_000

    def __init__(self, *args, label_names, **kwargs):
        if (not isinstance(label_names, (tuple, list)) or len(label_names) != 3
                or any(type(label) is not str for label in label_names)
                or set(label_names) not in ({"entailment", "neutral", "contradiction"},
                                            {"match", "uncertain", "nonmatch"})):
            raise ValueError("Explicit NLI or membership label order is required")
        self.label_names = tuple(label_names)
        super().__init__(*args, **kwargs)
        labels = "\0".join(self.label_names).encode("utf-8")
        self.model_id += ":labels:" + hashlib.sha256(labels).hexdigest()

    def rerank(self, *args, **kwargs):
        raise ValueError("Pair classification cannot be reinterpreted as relevance ranking")

    def classify_pairs(self, pairs):
        pairs = list(pairs)
        if len(pairs) > 500 or any(
                not isinstance(pair, (tuple, list)) or len(pair) != 2
                or any(not isinstance(text, str) or not text.strip() for text in pair)
                or sum(len(text) for text in pair) > 32_768 for pair in pairs):
            raise ValueError("Local classification requires at most 500 bounded nonempty text pairs")
        if not pairs:
            return []
        logits = self._pair_logits(pairs, 3)
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        values = exp / exp.sum(axis=1, keepdims=True)
        return [dict(zip(self.label_names, map(float, row))) for row in values]
