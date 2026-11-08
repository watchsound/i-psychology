"""Prospective contract checks, using authored text and inert scores only."""
import pytest

from psychology_base.applications.measurement.semantic_membership import membership_inputs, membership_decision


def case():
    return dict(language='en', descriptions=["my first calculation", "the teacher's question"], records=[
        dict(id=7, speaker='user', text='I multiplied six by four.'),
        dict(id=2, speaker='tutor', text='How did you calculate it?'),
        dict(id=9, speaker=None, text='I added six and four.')])


def test_every_description_scores_every_candidate_without_positions_or_labels():
    rows = membership_inputs(case())
    assert len(rows) == 6
    assert [r['record'] for r in rows] == [7, 2, 9]*2
    assert [r['constraint'] for r in rows] == [None, 'nonmatch', 'uncertain', 'nonmatch', None, 'uncertain']
    assert all('first' not in r['pair'][1] for r in rows)
    assert rows[0]['pair'] == ('Outer speaker: user\nUtterance: I multiplied six by four.',
                             'This utterance matches the description: calculation.')
    changed = case()
    changed['records'].reverse()
    assert {tuple(r['pair']) for r in rows} == {tuple(r['pair']) for r in membership_inputs(changed)}


def test_bounded_pool_refuses_instead_of_clipping():
    sample = case()
    sample['records'] *= 12
    with pytest.raises(ValueError, match='candidate pool'):
        membership_inputs(sample)
    sample = case()
    sample['records'][1]['id'] = 7
    with pytest.raises(ValueError):
        membership_inputs(sample)


def test_uncertainty_is_not_an_exclusion_and_roles_cannot_be_overridden():
    neutral = dict(entailment=.02, neutral=.96, contradiction=.02)
    assert membership_decision(neutral, 'nli', (.8, .8, .1)) == 'uncertain'
    yes = dict(entailment=.98, neutral=.01, contradiction=.01)
    assert membership_decision(yes, 'nli', (.8, .8, .1)) == 'match'
    assert membership_decision(yes, 'nli', (.8, .8, .1), 'nonmatch') == 'nonmatch'
    assert membership_decision(yes, 'nli', (.8, .8, .1), 'uncertain') == 'uncertain'
    assert membership_decision(dict(entailment=.01, neutral=.01, contradiction=.98), 'nli', (.8,.8,.1)) == 'nonmatch'
    assert membership_decision(dict(entailment=.5, neutral=.49, contradiction=.01), 'nli', (.4,.8,.1)) == 'uncertain'
    assert membership_decision({'relevance': .3}, 'relevance', (.8,.2,0)) == 'uncertain'


def test_trained_membership_labels_do_not_reinterpret_generic_nli_neutral():
    scores = dict(match=.01, uncertain=.01, nonmatch=.98)
    assert membership_decision(scores, 'membership', (.8, .8, .1)) == 'nonmatch'
    assert membership_decision(dict(match=.98, uncertain=.01, nonmatch=.01), 'membership', (.8,.8,.1)) == 'match'
    assert membership_decision(dict(match=.01, uncertain=.98, nonmatch=.01), 'membership', (.8,.8,.1)) == 'uncertain'
    assert membership_decision(scores, 'membership', (.8,.8,.1), 'uncertain') == 'uncertain'
    with pytest.raises(KeyError):
        membership_decision(dict(entailment=.01, neutral=.98, contradiction=.01), 'membership', (.8,.8,.1))
