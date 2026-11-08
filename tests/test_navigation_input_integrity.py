"""Content gates for authored diagnostics; no model-quality stand-ins."""
from types import SimpleNamespace

import pytest

from psychology_base.eval.benchmarks import application_context as study
from psychology_base.llm.providers.ollama import OllamaProvider


def controls():
    return study.read(study.FIXTURES / 'memory-navigation-controls-v2.json')['controls']


def test_corrected_controls_have_meaningful_distinct_zero_overlap_inputs():
    study.validate_navigation_controls(controls())


@pytest.mark.parametrize('field', ['query', 'source_text', 'distractor'])
@pytest.mark.parametrize('damage', ['???????', '\ufffd broken', '    ', '12345'])
def test_damaged_control_rejects_before_qualification(field, damage):
    cases = controls()
    if field == 'distractor':
        cases[2]['distractors'][1] = damage
    else:
        cases[2][field] = damage
    with pytest.raises(ValueError):
        study.validate_navigation_controls(cases)


def test_declared_chinese_cannot_be_english_text():
    cases = controls()
    cases[2]['query'] = 'Meaningful English, wrong language'
    with pytest.raises(ValueError, match='language'):
        study.validate_navigation_controls(cases)


def test_competitor_cannot_duplicate_the_target_after_unicode_normalization():
    cases = controls()
    cases[0]['distractors'][0] = cases[0]['source_text'].upper() + '  '
    with pytest.raises(ValueError, match='distinct'):
        study.validate_navigation_controls(cases)


def test_v1_is_preserved_but_cannot_start_another_invalid_run(tmp_path, monkeypatch):
    plan = study.FIXTURES / 'memory-navigation-local-diagnostic-v1.json'
    monkeypatch.setattr(OllamaProvider, '_open', lambda *a, **k: pytest.fail('Invalid text reached transport'))
    with pytest.raises(ValueError):
        study.navigation_diagnostic(SimpleNamespace(navigation_plan=plan, output=tmp_path/'invalid.json'))
    assert not (tmp_path/'invalid').exists()
    assert study.read(plan)['authored_controls'][2]['source_text'] == '?' * 10


def test_discrimination_cannot_use_missing_cached_vectors(tmp_path):
    from psychology_base.llm.cache import LLMCache
    plan={'authored_controls':controls(),'query_prefix':'Query:',
          'model_identity':{'model':'arithmetic-only','dimension':2},
          'minimum_unrelated_margin':.05,
          'authored_control_design':{'unrelated_absent_indices':[1,2,3,4,6,7]}}
    with pytest.raises(ValueError,match='actual cached'):
        study.navigation_discrimination(plan,LLMCache(tmp_path))


def test_tied_vectors_cannot_pass_discrimination(tmp_path):
    from psychology_base.llm.cache import LLMCache, embedding_cache_key
    plan={'authored_controls':controls(),'query_prefix':'Query:',
          'model_identity':{'model':'arithmetic-only','dimension':2},
          'minimum_unrelated_margin':.05,
          'authored_control_design':{'unrelated_absent_indices':[1,2,3,4,6,7]}}
    cache=LLMCache(tmp_path)
    for control in controls():
        for text in [plan['query_prefix']+control['query'],control['source_text'],*control['distractors']]:
            key=embedding_cache_key(provider='ollama',model_id='arithmetic-only',text=text,identity=plan['model_identity'])
            cache.put_embedding(key,[.2,.8],model_id='arithmetic-only')
    result=study.navigation_discrimination(plan,cache)
    assert not result['passed']
    assert all(row['target_rank']==9 and not row['passed'] for row in result['records'])
    assert result['distinct_input_vector_collisions']
