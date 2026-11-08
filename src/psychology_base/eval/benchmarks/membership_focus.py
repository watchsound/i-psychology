"""Authored contrast construction and grading, separate from model dispatch."""
from copy import deepcopy
from itertools import product

from psychology_base.applications.measurement.semantic_membership import (
    membership_inputs, reviewed_membership_rows,
)
from psychology_base.eval.benchmarks.evaluation_support import complete_keys


def build_cases(bank):
    """Expand explicit authored references; never infer labels from model output."""
    cases, references = {}, {}
    for family in bank['families']:
        for language, history in product(('en', 'zh'), family['histories']):
            key = f"{family['id']}-{language}-{history['id']}"
            texts = history[language] + [family['focus'][language]]
            labels = history['labels'] + [history.get('focus_labels', family.get('focus_labels'))]
            rationales = history['rationales'] + [history.get('focus_rationale', family.get('focus_rationale'))]
            ids = list(range(len(texts)-1)) + [2]
            case, reference = _case(key, family, language, texts, ids, labels, rationales,
                                    history.get('focus_support', []))
            cases[key], references[key] = case, reference
    for control, language in product(bank['controls'], ('en', 'zh')):
        family = next(f for f in bank['families'] if f['id'] == control['family'])
        key = f"{control['id']}-{language}"
        cases[key], references[key] = _case(key, family, language, [control[language]], [2],
                                            [control['labels']], [control['rationale']], [])
    complete_keys(cases, set(references))
    if len(cases) != 20:
        raise ValueError('Authored contrast bank must contain twenty cases')
    return cases, references


def _case(key, family, language, texts, ids, labels, rationales, support):
    case = dict(id=key, family=family['id'], language=language, application=family['application'],
                split='regression', origin='authored', focus_id=2,
                description_ids=family['description_ids'], descriptions=family['descriptions'][language],
                records=[dict(id=i, text=text, speaker='user' if i == 2 else family['application'],
                              source_order=dict(group='f'*32, position=i)) for i, text in zip(ids, texts)])
    judgments = []
    for record, vector, rationale in zip(case['records'], labels, rationales):
        if len(vector) != 2 or not rationale:
            raise ValueError('Missing authored label or rationale')
        for slot, label in enumerate(vector):
            reason = dict(match='direct_act', nonmatch='different_act', uncertain='ambiguous_scope')[label]
            # A resolved referent mismatch concerns participants, not act type.
            if family['id'] == 'referent' and record['id'] == 2 and label == 'nonmatch':
                reason = 'different_participant'
            evidence = dict(start=0, end=len(record['text']), quote=record['text'])
            context = [dict(record=i, start=0, end=len(texts[ids.index(i)]), quote=texts[ids.index(i)])
                       for i in support] if record['id'] == 2 else []
            judgments.append(dict(slot=slot, record=record['id'], label=label, reason=reason,
                                  rationale=rationale, evidence=[evidence], context_evidence=context))
    reference = dict(id=key, contract_version=3, reviewed=True,
                     reviewer='Implementing agent, authored pre-inference review; not independent gold',
                     judgments=judgments)
    reviewed_membership_rows(case, reference, application=case['application'])
    return case, reference


def condition(case, *, reverse=False, profile='baseline-v1', repeat=False):
    order = [1, 0] if reverse else [0, 1]
    return dict(id=f"{case['id']}/{'reverse' if reverse else 'forward'}/{profile}" + ('/repeat' if repeat else ''),
                case=case['id'], language=case['language'], description_order=order,
                description_ids=[case['description_ids'][i] for i in order],
                prompt_profile=profile, repeat=repeat)


def build_schedule(cases):
    entries = []
    for case in cases.values():
        for reverse in ((False,) if case['id'].startswith(('both-', 'neither-')) else (False, True)):
            entries.append(condition(case, reverse=reverse))
    for key in ('calculation-en-act', 'occupation-zh-act'):
        for reverse in (False, True):
            entries.append(condition(cases[key], reverse=reverse, profile='example-reversed-v1'))
    for key in ('calculation-en-neutral', 'occupation-zh-act'):
        entries.append(condition(cases[key], repeat=True))
    first = [condition(cases[key], reverse=rev) for key, rev in (
        ('calculation-en-neutral', False), ('calculation-en-act', False),
        ('occupation-zh-act', False), ('occupation-zh-act', True),
        ('referent-zh-ambiguous', False), ('both-en', False), ('neither-zh', False),
        ('referent-en-unique', False))]
    # Finish example comparators and fixed repeats early, before broader coverage.
    early = [condition(cases['calculation-en-act'], reverse=True)]
    early += [e for e in entries if e['prompt_profile'] != 'baseline-v1' or e['repeat']]
    chosen = first + early
    selected = {e['id'] for e in chosen}
    remaining = {lang: [e for e in entries if e['language'] == lang and e['id'] not in selected]
                 for lang in ('en', 'zh')}
    while any(remaining.values()):
        for lang in ('en', 'zh'):
            if remaining[lang]:
                chosen.append(remaining[lang].pop(0))
    complete_keys([e['id'] for e in chosen], {e['id'] for e in entries})
    if len(chosen) != 42 or any(sum(e['language'] == lang for e in chosen) != 21 for lang in ('en', 'zh')):
        raise ValueError('Frozen 42-call bilingual schedule changed')
    return chosen


def focus_rows(case, entry, *, format_version=3, allow_constraints=False):
    if type(format_version) is not int or format_version not in (3, 4):
        raise ValueError('Focus diagnostic requires contextual membership format 3 or 4')
    changed = deepcopy(case)
    order = entry['description_order']
    if order not in ([0, 1], [1, 0]) or entry['description_ids'] != [case['description_ids'][i] for i in order]:
        raise ValueError('Description permutation does not match semantic IDs')
    changed['descriptions'] = [case['descriptions'][i] for i in order]
    rows = [r for r in membership_inputs(changed, application=case['application'], format_version=format_version)
            if r['record'] == case['focus_id']]
    if (type(allow_constraints) is not bool or len(rows) != 2
            or (not allow_constraints and any(r['constraint'] is not None for r in rows))
            or all(r['constraint'] is not None for r in rows)):
        raise ValueError('Diagnostic requires two active canonical focus rows')
    return rows


def normalize(entry, labels):
    if (type(labels) is not list or len(labels) != 2
            or any(label not in ('match', 'nonmatch', 'uncertain') for label in labels)
            or len(set(entry['description_ids'])) != 2):
        raise ValueError('Incomplete categorical prediction or semantic IDs')
    return dict(zip(entry['description_ids'], labels))


def grade(cases, references, schedule, observations, *, contrast_pairs=None):
    """Coverage includes missing conditions; only complete contrasts are compared."""
    expected = {e['id']: e for e in schedule}
    if len(expected) != len(schedule):
        raise ValueError('Duplicate scheduled condition')
    observed = {}
    for observation in observations:
        key = observation['id']
        if key not in expected or key in observed:
            raise ValueError('Unexpected or duplicate observation')
        observed[key] = observation
    records, predictions = [], {}
    for entry in schedule:
        case = cases[entry['case']]
        ref_rows = reviewed_membership_rows(case, references[case['id']], application=case['application'])
        truth = {case['description_ids'][r['slot']]: r['label'] for r in ref_rows if r['record'] == case['focus_id']}
        obs = observed.get(entry['id'], {})
        pred = normalize(entry, obs['labels']) if 'labels' in obs else None
        if pred is not None:
            predictions[entry['id']] = pred
        records.append(dict(id=entry['id'], language=entry['language'], expected=truth,
                            predicted=pred, correct=pred == truth if pred is not None else False,
                            status='completed' if pred is not None else ('failed' if obs else 'missing')))
    contrasts = []

    def compare(kind, left, right, relation):
        a, b = predictions.get(left['id']), predictions.get(right['id'])
        contrasts.append(dict(kind=kind, left=left['id'], right=right['id'],
                              complete=a is not None and b is not None,
                              same=a == b if a is not None and b is not None else None,
                              expected_relation=relation))

    if contrast_pairs is not None:
        if type(contrast_pairs) is not list:
            raise ValueError('Explicit contrasts must be a list')
        truths = {r['id']: r['expected'] for r in records}
        seen_pairs = set()
        for pair in contrast_pairs:
            if (type(pair) is not dict or set(pair) != {'kind', 'left', 'right', 'relation'}
                    or any(type(pair.get(k)) is not str or not pair[k] for k in pair)
                    or pair['left'] not in expected or pair['right'] not in expected
                    or pair['left'] == pair['right'] or pair['relation'] not in ('same', 'different')):
                raise ValueError('Invalid explicit contrast')
            key = pair['kind'], frozenset((pair['left'], pair['right']))
            if key in seen_pairs:
                raise ValueError('Duplicate explicit contrast')
            seen_pairs.add(key)
            relation = 'same' if truths[pair['left']] == truths[pair['right']] else 'different'
            if pair['relation'] != relation:
                raise ValueError('Contrast contradicts reviewed references')
            compare(pair['kind'], expected[pair['left']], expected[pair['right']], pair['relation'])
    else:
        for case in cases.values():
            if not case['id'].startswith(('both-', 'neither-')):
                compare('description_order', condition(case), condition(case, reverse=True), 'same')
        for family, language, reverse in product(('calculation', 'occupation'), ('en', 'zh'), (False, True)):
            a, b, c = [cases[f'{family}-{language}-{h}'] for h in ('empty', 'neutral', 'act')]
            compare('neutral_context', condition(a, reverse=reverse), condition(b, reverse=reverse), 'same')
            compare('preceding_act', condition(b, reverse=reverse), condition(c, reverse=reverse), 'same')
        for language, reverse in product(('en', 'zh'), (False, True)):
            compare('reference_context', condition(cases[f'referent-{language}-unique'], reverse=reverse),
                    condition(cases[f'referent-{language}-ambiguous'], reverse=reverse), 'different')
        for key, reverse in product(('calculation-en-act', 'occupation-zh-act'), (False, True)):
            compare('response_example', condition(cases[key], reverse=reverse),
                    condition(cases[key], reverse=reverse, profile='example-reversed-v1'), 'same')
        for key in ('calculation-en-neutral', 'occupation-zh-act'):
            compare('repeat', condition(cases[key]), condition(cases[key], repeat=True), 'same')
    summaries = {}
    for language in ('en', 'zh', 'all'):
        selected = [r for r in records if language == 'all' or r['language'] == language]
        done = [r for r in selected if r['predicted'] is not None]
        pairs = [(r['expected'][k], v) for r in done for k, v in r['predicted'].items()]
        summaries[language] = dict(planned=len(selected), completed=len(done),
            correct_conditions=sum(r['correct'] for r in selected),
            correct_pairs=sum(a == b for a, b in pairs), measured_pairs=len(pairs),
            planned_pairs=2*len(selected), unsafe_inclusions=sum(b == 'match' and a != b for a, b in pairs),
            unsafe_exclusions=sum(b == 'nonmatch' and a != b for a, b in pairs),
            uncertain_expected=sum(a == 'uncertain' for a, b in pairs),
            uncertain_preserved=sum(a == b == 'uncertain' for a, b in pairs),
            positive_expected=sum(a == 'match' for a, b in pairs),
            positives_recovered=sum(a == b == 'match' for a, b in pairs),
            failed=sum(r['status'] == 'failed' for r in selected), missing=sum(r['status'] == 'missing' for r in selected))
    return dict(qualified=False, scope='Authored exposed focus diagnostic, not full-pool/replay qualification',
                summary=summaries, records=records, contrasts=contrasts)
