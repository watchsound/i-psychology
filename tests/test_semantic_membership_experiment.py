"""Structural grading checks use inert authored observations, not model evidence."""
import importlib.util
from pathlib import Path

PATH = Path(__file__).parents[1]/'docs/experiments/phase-9-memory-navigation/run-semantic-membership-v1.py'
spec = importlib.util.spec_from_file_location('semantic_experiment_tests', PATH)
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def sample(score):
    cases=[dict(id=lang,language=lang) for lang in ('en','zh')]
    refs={lang:dict(matches=[[0],[1]],uncertain=[[1],[]]) for lang in ('en','zh')}
    rows=[dict(id=lang,order=0,pairs=[dict(slot=slot,record=record,constraint=None,scores=score)
          for slot in range(2) for record in range(2)]) for lang in ('en','zh')]
    return rows,cases,refs


def test_unknown_everywhere_does_not_satisfy_coverage():
    result=experiment.grade(*sample(dict(entailment=0,neutral=1,contradiction=0)), 'nli', (.8,.8,.1))
    assert result['total']['unsafe_inclusions']==result['total']['unsafe_exclusions']==0
    assert result['total']['positive_accepted']==0
    assert all(v['exact_case_coverage']==v['positive_coverage']==0 for v in result['languages'].values())


def test_exclusions_on_matches_and_ambiguity_are_both_unsafe():
    result=experiment.grade(*sample(dict(entailment=0,neutral=0,contradiction=1)), 'nli', (.8,.8,.1))
    assert result['total']['unsafe_exclusions']==6
    assert result['total']['unsafe_inclusions']==0


def test_inclusions_on_nonmatches_and_ambiguity_are_both_unsafe():
    result=experiment.grade(*sample(dict(entailment=1,neutral=0,contradiction=0)), 'nli', (.8,.8,.1))
    assert result['total']['unsafe_inclusions']==4
    assert result['total']['positive_accepted']==4


def test_calibration_prefers_safe_abstention_over_unsafe_coverage():
    rows,cases,refs=sample(dict(entailment=.9,neutral=.05,contradiction=.05))
    chosen,result=experiment.choose(rows,cases,refs,'nli',[(.8,.8,.1),(.99,.99,.1)])
    assert chosen==(.99,.99,.1)
    assert result['total']['positive_accepted']==0
