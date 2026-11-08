"""Classifier transport/contract tests. Inert logits are not accuracy evidence."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from psychology_base.applications.reporting_order import reporting_descriptor_constraints
from psychology_base.llm.execution import ExecutionDeferred
from psychology_base.llm.providers.local_pair import OnnxPairClassifier
from tests.test_local_pair import inert_scorer, router


def inert_classifier(logits):
    old, calls = inert_scorer()
    scorer = object.__new__(OnnxPairClassifier)
    scorer.__dict__.update(old.__dict__)
    scorer.label_names = ('entailment', 'neutral', 'contradiction')
    scorer._session = SimpleNamespace(run=lambda _, features: [np.asarray(logits)])
    return scorer, calls


@pytest.mark.parametrize('labels', [None, ['entailment'], ['neutral']*3,
    ['entailment', 'neutral', 'other'], ['entailment', 'neutral', []]])
def test_label_contract_refused_before_artifact_loading(labels):
    with pytest.raises(ValueError, match='label order'):
        OnnxPairClassifier(Path('absent'), Path('absent'), label_names=labels,
                           model_sha256='0'*64, tokenizer_sha256='0'*64)


def test_softmax_is_stable_and_respects_explicit_label_order():
    scorer, calls = inert_classifier([[10000, 0, -10000], [0, 0, 0]])
    result = scorer.classify_pairs([('事实', '描述'), ('fact', 'ok')])
    assert result[0] == dict(entailment=1, neutral=0, contradiction=0)
    assert list(result[1].values()) == pytest.approx([1/3]*3)
    scorer.label_names = ('contradiction', 'entailment', 'neutral')
    assert scorer.classify_pairs([('a', 'b'), ('c', 'd')])[0]['contradiction'] == 1
    assert len(calls) == 2
    assert scorer.classify_pairs([]) == []
    with pytest.raises(ValueError, match='relevance'):
        scorer.rerank('q', ['d'])


def test_membership_labels_have_explicit_identity_and_share_existing_transport(monkeypatch):
    monkeypatch.setattr('psychology_base.llm.providers.local_pair.OnnxPairScorer.__init__',
                        lambda self, *a, **kw: setattr(self, 'model_id', 'sha256:actual-trained-model'))
    first = OnnxPairClassifier(label_names=('match', 'uncertain', 'nonmatch'))
    second = OnnxPairClassifier(label_names=('nonmatch', 'uncertain', 'match'))
    nli = OnnxPairClassifier(label_names=('entailment', 'neutral', 'contradiction'))
    assert len({first.model_id, second.model_id, nli.model_id}) == 3
    scorer, _ = inert_classifier([[0, 0, 10]])
    scorer.label_names = first.label_names
    assert router(scorer).classify_pairs([('utterance', 'description')])[0]['nonmatch'] > .99


@pytest.mark.parametrize('labels', [('match', 'neutral', 'nonmatch'), ('match', 'match', 'nonmatch')])
def test_mixed_label_families_refused(labels):
    with pytest.raises(ValueError, match='label order'):
        OnnxPairClassifier(label_names=labels)


@pytest.mark.parametrize('logits', [[[0]], [[0, 0]], [[0, 0, 0, 0]],
    [[0, float('nan'), 0]], [[0, 0, float('inf')]], [0, 0, 0]])
def test_invalid_shape_and_nonfinite_outputs_refused(logits):
    scorer, _ = inert_classifier(logits)
    with pytest.raises(ValueError, match='invalid classification logits'):
        scorer.classify_pairs([('a', 'b')])


@pytest.mark.parametrize('pairs', [[('', 'b')], [('a',)], ['ab'], [(True, 'b')],
    [('a', 'x'*32768)], [('a', 'b')]*501])
def test_pair_bounds_before_tokenization(pairs):
    scorer, calls = inert_classifier([[0, 0, 0]])
    with pytest.raises(ValueError):
        scorer.classify_pairs(pairs)
    assert calls == []


def test_token_overflow_and_batching_are_shared():
    scorer, _ = inert_classifier([[0, 0, 0]])
    with pytest.raises(ValueError, match='silent truncation'):
        scorer.classify_pairs([('a', 'b'*513)])
    scorer.batch_size = 1
    assert len(scorer.classify_pairs([('a', 'b'), ('c', 'd')])) == 2


@pytest.mark.parametrize('mode', ['local', 'paid', 'research'])
def test_router_uses_existing_local_dispatch_and_never_cloud(mode, tmp_path):
    scorer, calls = inert_classifier([[0, 0, 0]])
    instance = router(scorer, mode, tmp_path/'budget.sqlite')
    assert instance.classify_pairs([('a', 'b')])[0]['neutral'] == pytest.approx(1/3)
    assert calls == [[('a', 'b')]]
    if instance.budget:
        assert instance.budget.summary()['provider_attempts'] == 0
    old, _ = inert_scorer()
    with pytest.raises(ExecutionDeferred, match='not configured'):
        router(old, mode, tmp_path/'budget2.sqlite').classify_pairs([('a', 'b')])


def test_busy_classifier_uses_shared_deadline():
    scorer, calls = inert_classifier([[0, 0, 0]])
    instance = router(scorer)
    scorer._lock.acquire()
    try:
        with pytest.raises(ExecutionDeferred):
            with instance.operation('classification-busy', timeout_s=.05):
                instance.classify_pairs([('a', 'b')])
        assert calls == []
    finally:
        scorer._lock.release()


@pytest.mark.parametrize('text,expected', [
    ("the learner's first calculation", ('user', True, 'calculation')),
    ("The Teacher's opening question", ('tutor', True, 'question')),
    ('我的最初计算', ('user', True, '计算')),
    ('交代其职业', (None, False, '交代其职业')),
    ('your initial question', ('companion', True, 'question')),
    ('firsthand account', (None, False, 'firsthand account'))])
def test_existing_descriptor_constraints_have_one_owner(text, expected):
    assert reporting_descriptor_constraints(text, 'companion') == expected
