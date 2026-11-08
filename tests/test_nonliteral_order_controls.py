"""Reference-assisted software checks; these do not establish model recovery."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from psychology_base.applications.reporting_order import bind_reporting_order

ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location('nonliteral_order_review',ROOT/'docs/experiments/phase-9-memory-navigation/prepare-nonliteral-review-v1.py')
review=importlib.util.module_from_spec(spec);spec.loader.exec_module(review)
DATA=json.loads(review.FIXTURE.read_text(encoding='utf-8'))


def packet(family,language,order,reverse=False):
    definition=next(f for f in DATA['families'] if f['family']=='recovery-'+family)
    cases,refs,_=review.build_authored(dict(families=[definition]))
    case=deepcopy(next(c for c in cases if c['language']==language))
    ref=next(r for r in refs if r['id']==case['id'])
    for record in case['records']:record['source_order']['position']=order.index(record['id'])
    if reverse:case['records'].reverse()
    return case,ref,definition


def block(case):
    return '\n'.join(json.dumps(dict(speaker=r['speaker'],text=r['text'],session='authored-software-control',
        source_order=r['source_order']),ensure_ascii=False) for r in case['records'])


class ReferenceProposal:
    identity='authored-reference-assisted-software-test-only'

    def __init__(self,case,reference):
        self.coordinates={r['id']:(r['source_order']['group'],r['source_order']['position']) for r in case['records']}
        self.reference=reference
        self.calls=0

    def __call__(self,application,query,descriptions,records):
        self.calls+=1
        assert not any(j['label']=='uncertain' for j in self.reference['judgments'])
        return dict(complete=True,targets=[[dict(record=r['id'],quote=r['text']) for r in records
            if any(j['slot']==slot and j['label']=='match'
                   and self.coordinates.get(j['record'])==(r['source_order']['group'],r['source_order']['position'])
                   for j in self.reference['judgments'])] for slot in (0,1)])


def query(case,direction):
    a,b=case['descriptions']
    return (f'Which came {direction} in the conversation: {a} or {b}?' if case['language']=='en'
        else f'原对话中，{a}和{b}，哪一句出现得'+('较早' if direction=='earlier' else '较晚')+'？')


@pytest.mark.parametrize('family',['encouragement','invitation'])
@pytest.mark.parametrize('language',['en','zh'])
@pytest.mark.parametrize('application',['tutor','companion'])
@pytest.mark.parametrize('reverse',[False,True])
@pytest.mark.parametrize('direction',['earlier','later'])
@pytest.mark.parametrize('layout',['base','swapped','shifted'])
def test_source_order_and_direction_change_winner_without_semantic_aliases(family,language,application,reverse,direction,layout):
    definition=next(f for f in DATA['families'] if f['family']=='recovery-'+family)
    specification=definition['ordering']
    order=list(range(6)) if layout=='base' else specification[layout+'_order']
    case,ref,_=packet(family,language,order,reverse)
    earlier=specification['base_earlier_slot'] if layout=='base' else specification[layout+'_earlier_slot']
    proposer=ReferenceProposal(case,ref)
    binding=bind_reporting_order(application,query(case,direction),block(case),proposer=proposer)
    assert binding.selected is not None and binding.resolution.query_target==(earlier if direction=='earlier' else 3-earlier)
    if direction=='earlier':
        winner=1 if layout=='base' else specification[layout+'_winner']
        assert binding.selected.text==next(r['text'] for r in case['records'] if r['id']==winner)
        assert binding.selected.source_position>0
    assert proposer.calls==1


@pytest.mark.parametrize('damage',['missing-target','missing-prefix','different-group','conflict','unknown-actor'])
def test_incomplete_or_incomparable_evidence_cannot_become_a_comparison(damage):
    case,ref,_=packet('encouragement','en',list(range(6)))
    if damage=='missing-target':case['records']=[r for r in case['records'] if r['id']!=2]
    elif damage=='missing-prefix':case['records']=[r for r in case['records'] if r['id']!=0]
    elif damage=='different-group':case['records'][2]['source_order']['group']='f'*32
    elif damage=='conflict':case['records'][0]['source_order']['position']=1
    elif damage=='unknown-actor':case['records'][2]['speaker']=None
    assert bind_reporting_order('tutor',query(case,'earlier'),block(case),proposer=ReferenceProposal(case,ref)).selected is None
