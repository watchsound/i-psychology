"""Measurement-only description membership inputs and conservative decisions.

This is not installed in application factories. References never enter pair
construction. The existing binder still owns chronology and source validation.
"""
from __future__ import annotations

import json
import hashlib
import math

from psychology_base.applications.reporting_order import reporting_descriptor_constraints


def _prior_context(records):
    from psychology_base.companion.turn_provenance import source_conflicts

    if source_conflicts(r.get('source_order') for r in records):
        raise ValueError('Context membership requires unambiguous source coordinates')
    result = {}
    for focus in records:
        coordinate = focus.get('source_order')
        preceding = [] if coordinate is None else [r for r in records
            if r.get('source_order') is not None
            and r['source_order']['group'] == coordinate['group']
            and r['source_order']['position'] < coordinate['position']]
        context = []
        expected = coordinate['position'] - 1 if coordinate is not None else -1
        for record in sorted(preceding, key=lambda r: r['source_order']['position'], reverse=True):
            if len(context) == 2 or record['source_order']['position'] != expected:
                break
            context.append(record)
            expected -= 1
        result[focus['id']] = list(reversed(context))
    return result


def membership_inputs(case, *, application='tutor', max_records=32, format_version=1,
                      include_focus_span=False):
    if type(format_version) is not int or format_version not in (1, 2, 3, 4):
        raise ValueError('Unsupported membership input format')
    if type(include_focus_span) is not bool or (include_focus_span and format_version not in (3, 4)):
        raise ValueError('Focus spans require a contextual membership input')
    records = case['records']
    descriptions = case['descriptions']
    if (application not in ('tutor', 'companion') or case['language'] not in ('en', 'zh')
            or type(max_records) is not int or not 0 < max_records <= 32
            or not 0 < len(records) <= max_records or len(descriptions) != 2
            or any(not isinstance(d, str) or not d.strip() for d in descriptions)
            or any(type(r['id']) is not int or not isinstance(r['text'], str) or not r['text'].strip() for r in records)
            or len({r['id'] for r in records}) != len(records)):
        raise ValueError('Incomplete or unbounded membership candidate pool')
    prior = _prior_context(records) if format_version in (3, 4) else None
    rows = []
    for slot, description in enumerate(descriptions):
        actor, initial, predicate = reporting_descriptor_constraints(description, application,
                                                                     predicate_only=format_version == 4)
        if not predicate:
            raise ValueError('No semantic predicate remains after structural constraints')
        for record in records:
            speaker = record['speaker']
            if speaker not in (None, 'user', 'tutor', 'companion'):
                raise ValueError('Unknown outer-speaker encoding')
            constraint = ('uncertain' if speaker is None else 'nonmatch') if actor is not None and speaker != actor else None
            hypothesis = (f'这句话符合以下描述：{predicate}。' if case['language'] == 'zh'
                          else f'This utterance matches the description: {predicate}.')
            if format_version in (2, 3, 4):
                hypothesis = (f'Description: {description}\nRequested outer speaker: {actor or "any"}\n'
                              'Task: utterance-act membership; occurrence order is handled separately.')
            premise = f"Outer speaker: {speaker or 'unknown'}\nUtterance: {record['text']}"
            if format_version in (3, 4):
                premise = json.dumps(dict(
                    context_scope='Up to two contiguous preceding supplied turns in the same source; earlier context may be missing.',
                    prior_utterances=[dict(speaker=r['speaker'] or 'unknown', text=r['text'])
                                      for r in prior[record['id']]],
                    focus_utterance=dict(speaker=speaker or 'unknown', text=record['text'])),
                    ensure_ascii=False, separators=(',', ':'))
                hypothesis = (f'Description: {predicate if format_version == 4 else description}\nRequested outer speaker: {actor or "any"}\n'
                              'Task: focus utterance-act membership using only supplied prior context; '
                              'occurrence order is handled separately.')
            row = dict(slot=slot, record=record['id'], constraint=constraint,
                initial=initial, pair=(premise, hypothesis),
                relevance_query=predicate, relevance_document=premise)
            if format_version in (3, 4):
                row['context_record_ids'] = tuple(r['id'] for r in prior[record['id']])
            if format_version == 4:
                row['input_format_version'] = 4
            if include_focus_span:
                # The canonical serializer puts the focus text last, followed by
                # exactly two object closers. Locate its encoded content by
                # construction, never by searching for possibly repeated text.
                encoded = json.dumps(record['text'], ensure_ascii=False)
                if not premise.endswith(encoded + '}}'):
                    raise ValueError('Canonical focus serialization changed')
                end = len(premise) - 3
                start = end - (len(encoded) - 2)
                row['focus_span'] = dict(version=1, sequence=0, start=start, end=end)
            rows.append(row)
    return rows


MEMBERSHIP_REASONS = {
    'match': frozenset({'direct_act'}),
    'nonmatch': frozenset({'different_act', 'different_participant', 'different_relation',
                          'negated_or_hypothetical', 'reported_not_endorsed', 'wrong_outer_actor'}),
    'uncertain': frozenset({'missing_reference', 'ambiguous_scope', 'unknown_outer_actor'}),
}


def reviewed_membership_rows(case, reference, *, application='tutor'):
    """Validate complete reviewed annotations and join to label-free inputs.

    This verifies structure and quoted bytes, not independent semantic accuracy.
    Code-constrained rows stay in case references but are not training rows.
    """
    version = reference.get('contract_version') if isinstance(reference, dict) else None
    if (not isinstance(reference, dict) or reference.get('id') != case['id']
            or type(version) is not int or version not in (2, 3, 4)
            or reference.get('reviewed') is not True
            or not isinstance(reference.get('reviewer'), str) or not reference['reviewer'].strip()
            or type(reference.get('judgments')) is not list):
        raise ValueError('A versioned, explicitly reviewed membership reference is required')
    rows = membership_inputs(case, application=application, format_version=version)
    expected = {(r['slot'], r['record']): r for r in rows}
    judgments = {}
    texts = {r['id']: r['text'] for r in case['records']}
    for judgment in reference['judgments']:
        if not isinstance(judgment, dict):
            raise ValueError('Membership judgments must be objects')
        slot, record = judgment.get('slot'), judgment.get('record')
        if type(slot) is not int or type(record) is not int or (slot, record) not in expected:
            raise ValueError('Reference points outside the candidate pool')
        key = (slot, record)
        if key in judgments:
            raise ValueError('Duplicate membership judgment')
        label, reason = judgment.get('label'), judgment.get('reason')
        if not isinstance(label, str) or not isinstance(reason, str) or reason not in MEMBERSHIP_REASONS.get(label, ()):
            raise ValueError('Invalid membership label/reason combination')
        constraint = expected[key]['constraint']
        if constraint is not None and label != constraint:
            raise ValueError('Reference contradicts the existing actor constraint')
        if (reason == 'unknown_outer_actor' and constraint != 'uncertain'
                or reason == 'wrong_outer_actor' and constraint != 'nonmatch'):
            raise ValueError('Actor reason lacks its structural constraint')
        rationale, evidence = judgment.get('rationale'), judgment.get('evidence')
        if not isinstance(rationale, str) or not rationale.strip() or not isinstance(evidence, list) or not evidence:
            raise ValueError('Every label requires a rationale and original evidence span')
        for span in evidence:
            if not isinstance(span, dict):
                raise ValueError('Evidence spans must be objects')
            start, end, quote = span.get('start'), span.get('end'), span.get('quote')
            if (type(start) is not int or type(end) is not int
                    or not 0 <= start < end <= len(texts[record]) or texts[record][start:end] != quote):
                raise ValueError('Evidence does not match the original utterance')
        if version in (3, 4):
            context_evidence = judgment.get('context_evidence')
            if type(context_evidence) is not list:
                raise ValueError('Context references require explicit supporting evidence lists')
            for span in context_evidence:
                if not isinstance(span, dict):
                    raise ValueError('Context evidence spans must be objects')
                support, start, end, quote = (span.get(k) for k in ('record', 'start', 'end', 'quote'))
                if (type(support) is not int or support not in expected[key]['context_record_ids']
                        or type(start) is not int or type(end) is not int
                        or not 0 <= start < end <= len(texts[support])
                        or texts[support][start:end] != quote):
                    raise ValueError('Supporting evidence is absent from the supplied prior context')
        judgments[key] = judgment
    if set(judgments) != set(expected):
        raise ValueError('Incomplete membership annotations; candidates cannot be silently omitted')
    return [dict(row, label=judgments[row['slot'], row['record']]['label'],
                 semantic_training=row['constraint'] is None) for row in rows]


def membership_decision(scores, arm, thresholds, constraint=None):
    if constraint is not None:
        if constraint not in ('nonmatch', 'uncertain'):
            raise ValueError('Invalid structural membership constraint')
        return constraint
    yes, no, margin = thresholds
    if arm in ('nli', 'membership'):
        keys = (('entailment', 'neutral', 'contradiction') if arm == 'nli'
                else ('match', 'uncertain', 'nonmatch'))
        entail, neutral, contradict = (scores[k] for k in keys)
        if entail >= yes and entail-max(neutral, contradict) >= margin:
            return 'match'
        if contradict >= no and contradict-max(neutral, entail) >= margin:
            return 'nonmatch'
    elif arm == 'relevance':
        score = scores['relevance']
        if score >= yes:
            return 'match'
        if score <= no:
            return 'nonmatch'
    else:
        raise ValueError('Unknown membership diagnostic arm')
    return 'uncertain'


class LocalMembershipAssessor:
    """Optional local diagnostic adapter; no factory installs this by default."""

    def __init__(self,router,*,format_version=3,thresholds=(.7,.7,.2)):
        if (type(format_version) is not int or format_version not in (2,3,4)
                or len(thresholds)!=3 or any(type(v) not in (int,float) or not math.isfinite(v)
                    or not 0<=v<=1 for v in thresholds)):
            raise ValueError('Explicit membership input and decision bounds are required')
        self.router=router;self.format_version=format_version;self.thresholds=tuple(thresholds)
        self._model=getattr(getattr(router,'local_reranker',None),'model_id',None)
        self._tokenizer=getattr(getattr(router,'local_reranker',None),'tokenizer_sha256',None)
        self._configuration=(format_version,self.thresholds)
        if type(self._model) is not str or not self._model:
            raise ValueError('An explicitly pinned local membership model is required')
        self._check()
        self.identity=hashlib.sha256(json.dumps([self._model,self._tokenizer,format_version,self.thresholds]).encode()).hexdigest()

    def _check(self):
        from psychology_base.llm.execution import ExecutionDeferred
        if (getattr(getattr(self.router,'execution',None),'mode',None)!='local'
                or getattr(getattr(self.router,'local_reranker',None),'model_id',None)!=self._model
                or getattr(getattr(self.router,'local_reranker',None),'tokenizer_sha256',None)!=self._tokenizer
                or (self.format_version,self.thresholds)!=self._configuration):
            raise ExecutionDeferred('Membership requires an unchanged strict-local classifier')

    def __call__(self,rows):
        self._check()
        if (self.format_version == 4 or any(r.get('input_format_version') == 4 for r in rows)) and any(
                type(r.get('input_format_version')) is not int
                or r.get('input_format_version') != self.format_version for r in rows):
            raise ValueError('Membership rows do not match the pinned input format')
        selected=[(i,r) for i,r in enumerate(rows) if r['constraint'] is None]
        labels=[r['constraint'] for r in rows]
        if selected:
            with self.router.operation('reporting-membership',timeout_s=60):
                scores=self.router.classify_pairs([r['pair'] for _,r in selected])
            self._check()
            if not isinstance(scores,list) or len(scores)!=len(selected):
                raise ValueError('Local membership returned an incomplete score batch')
            for (i,row),score in zip(selected,scores):
                if (type(score) is not dict or set(score)!={'match','nonmatch','uncertain'}
                        or any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in score.values())
                        or abs(sum(score.values())-1)>1e-5):
                    raise ValueError('Invalid local membership scores')
                labels[i]=membership_decision(score,'membership',self.thresholds)
        return labels
