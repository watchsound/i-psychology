"""Pre-inference reference construction and grading, never runtime inference.

Authored quote equivalences are review labels. None enter model payloads or
the production reference validator. This module does not execute a model.
"""
from copy import deepcopy
import hashlib
import json

from psychology_base.applications.reporting_order import _phrase_targets
from psychology_base.applications.measurement.reference_readings import reference_turns, decode_reference_readings
from psychology_base.eval.benchmarks.membership_focus import condition, focus_rows


def rows_for(case, entry):
    if (case['id'] != entry['case'] or entry['version'] != 4
            or entry['prompt_profile'] != 'baseline-v1'
            or type(entry['descriptions_per_call']) is not int or entry['descriptions_per_call'] != 2):
        raise ValueError('Extraction requires the frozen joint format-4 focus')
    return focus_rows(case, entry, format_version=4)


def _quote_spans(turns, quote):
    return [(t, span.start, span.end) for t, turn in enumerate(turns)
            for span in _phrase_targets([dict(turn, id=t)], quote)]


def reviewed_reference(case, entry, declared, inherited=None):
    rows = rows_for(case, entry)
    focus = json.loads(rows[0]['pair'][0])
    turns = reference_turns(focus)
    names = {}
    for entity, quotes in declared['names'].items():
        anchors = sorted(set(span for quote in quotes for span in _quote_spans(turns, quote)))
        if not anchors:
            raise ValueError('Declared entity lacks a supplied source anchor')
        names[entity] = [list(span) for span in anchors]
    mentions = []
    for quote, occurrence, kind in declared['focus']:
        spans = _phrase_targets([dict(turns[-1], id=len(turns)-1)], quote)
        if type(occurrence) is not int or not 0 <= occurrence < len(spans):
            raise ValueError('Declared focus mention is absent')
        span = spans[occurrence]
        mentions.append(dict(turn=len(turns)-1, start=span.start, end=span.end, quote=quote, kind=kind))
    if len({(m['turn'], m['start'], m['end']) for m in mentions}) != len(mentions):
        raise ValueError('Repeated reviewed focus mention')
    allowed = set(names) | {'speaker:'+turn['speaker'] for turn in turns if turn['speaker'] != 'unknown'}
    readings = declared['readings']
    if (not readings or len({tuple(r) for r in readings}) != len(readings)
            or any(len(r) != len(mentions) or any(v is not None and v not in allowed for v in r) for r in readings)
            or not declared['rationale']):
        raise ValueError('Incomplete declared reference readings')
    return dict(id=case['id'], reviewed=True, reviewer='Implementer-authored pre-inference review; not independent gold',
        names=names, focus=mentions, readings=deepcopy(readings), coverage=declared['coverage'],
        rationale=declared['rationale'], inherited_membership_reference=deepcopy(inherited),
        context_sha256=hashlib.sha256(rows[0]['pair'][0].encode()).hexdigest())


def authored_schema_example(reference, focus):
    """A disclosed software oracle for schema preparation, never inference."""
    turns = reference_turns(focus)
    mentions, by_span = [], {}
    def add(turn, start, end, kind):
        key = (turn,start,end)
        if key in by_span:
            return by_span[key]
        quote = turns[turn]['text'][start:end]
        spans = _phrase_targets([dict(turns[turn],id=turn)],quote)
        occurrence = next(i for i,s in enumerate(spans) if (s.start,s.end)==(start,end))
        by_span[key] = len(mentions)
        mentions.append(dict(turn=turn,quote=quote,occurrence=occurrence,kind=kind))
        return by_span[key]
    entities, entity_ids = [], {}
    for key, spans in reference['names'].items():
        entity_ids[key] = len(entities)
        entities.append(dict(mentions=[add(*span,'named') for span in spans]))
    for key in sorted({v for row in reference['readings'] for v in row if v is not None and v.startswith('speaker:')}):
        speaker = key.split(':',1)[1]
        entity_ids[key] = len(entities)
        entities.append(dict(speaker=speaker,turn=next(i for i,t in enumerate(turns) if t['speaker']==speaker)))
    refs = [add(m['turn'],m['start'],m['end'],m['kind']) for m in reference['focus']]
    raw = dict(mentions=mentions,entities=entities,focus_references=refs,
        readings=[[entity_ids[v] if v is not None else None for v in row] for row in reference['readings']],
        coverage=reference['coverage'])
    return raw, decode_reference_readings(json.dumps(raw,ensure_ascii=False),focus)


def authored_selection_example(reference, focus):
    """Reviewed-answer software oracle, never a model-output conversion or score."""
    from psychology_base.applications.measurement.reference_readings import (
        reference_source_units, decode_reference_selections,
    )
    units = reference_source_units(focus)
    starts = {(u.record_id,u.start):i for i,u in enumerate(units)}
    ends = {(u.record_id,u.end):i for i,u in enumerate(units)}
    def select(turn,start,end):
        if (turn,start) not in starts or (turn,end) not in ends:
            raise ValueError('Reviewed oracle span is not representable by source units')
        return [starts[turn,start],ends[turn,end]]
    worlds = []
    for reading in reference['readings']:
        row = []
        for mention,person in zip(reference['focus'],reading,strict=True):
            target = None
            if person is not None:
                target = (dict(speaker=person.split(':',1)[1]) if person.startswith('speaker:')
                          else dict(name=select(*reference['names'][person][0])))
            row.append(dict(span=select(mention['turn'],mention['start'],mention['end']),
                kind=mention['kind'],target=target))
        worlds.append(row)
    raw = dict(coverage=reference['coverage'],readings=worlds)
    return raw, decode_reference_selections(json.dumps(raw,ensure_ascii=False),focus)


def authored_interpretation_example(declared, focus):
    """Compile exposed typed software references, never construct predictions."""
    from psychology_base.applications.measurement.reference_readings import (
        INTERPRETATION_SCHEMA, reference_source_units, interpretation_candidate_view, decode_interpretation_selections,
    )
    units, turns = reference_source_units(focus), reference_turns(focus)
    def phrase(turn, quote):
        choices = _phrase_targets([dict(turns[turn],id=turn)],quote)
        if len(choices) != 1:
            raise ValueError('Authored typed reference requires an unambiguous source quotation')
        span = choices[0]
        return [next(i for i,u in enumerate(units) if u.record_id==turn and u.start==span.start),
                next(i for i,u in enumerate(units) if u.record_id==turn and u.end==span.end)]
    turn = len(turns)-1
    clause = phrase(turn,declared['clause'])
    readings = []
    for name in declared['targets']:
        anchor = None if name is None else dict(span=phrase(0,name),type=declared['type'])
        subject = dict(speaker='user') if declared['type']=='object' else anchor
        attribution = anchor if declared.get('quoted') else dict(speaker='user')
        bindings = [dict(span=phrase(turn,declared['subject']),
            kind='deictic' if declared['type']=='object' or declared.get('quoted') else 'anaphoric',
            role='agent',event=clause,target=subject)]
        if declared['type']=='object':
            bindings.append(dict(span=phrase(turn,declared.get('object',declared['clause'])),
                kind='anaphoric' if 'object' in declared else 'implicit',role='patient',event=clause,target=anchor))
        events = [dict(span=clause,predicate=phrase(turn,declared['predicate']),polarity=declared['polarity'],
            modality=declared['modality'],attribution=attribution)]
        if declared.get('quoted'):
            outer = phrase(turn,declared['focus'])
            bindings.extend([
                dict(span=phrase(turn,name),kind='named',role='agent',event=outer,target=anchor),
                dict(span=clause,kind='clausal',role='patient',event=outer,target=dict(span=clause,type='event'))])
            events.append(dict(span=outer,predicate=phrase(turn,declared['reporting_predicate']),polarity='affirmed',
                modality='asserted',attribution=dict(speaker='user')))
        readings.append(dict(bindings=bindings,events=events))
    raw = dict(schema=INTERPRETATION_SCHEMA,source_identity=interpretation_candidate_view(focus)['source_identity'],
        coverage=declared['coverage'],readings=readings)
    return raw, decode_interpretation_selections(json.dumps(raw,ensure_ascii=False),focus)


def compare_interpretation_analysis(analysis, reference, focus):
    """Compare complete joint proposals against explicit reviewed equivalences.

    Equivalence classes are pre-inference reference annotations, not runtime
    aliases. Never collapse independent bindings into Cartesian combinations.
    """
    from psychology_base.applications.measurement.reference_readings import (
        decode_interpretation_selections, reference_source_units, _selected_span,
    )
    actual = decode_interpretation_selections(json.dumps(analysis,ensure_ascii=False),focus)
    expected = decode_interpretation_selections(json.dumps(reference['expected'],ensure_ascii=False),focus)
    aliases = {}
    turns, units = reference_turns(focus), reference_source_units(focus)
    for number,group in enumerate(reference.get('span_equivalences',[])):
        if type(group) is not list or len(group)<2:
            raise ValueError('Reviewed source equivalence requires distinct alternatives')
        for span in group:
            _selected_span(span,units,turns)
            key = tuple(span)
            if key in aliases:
                raise ValueError('Reviewed source equivalence classes overlap')
            aliases[key] = number
    def canonical(value):
        if type(value) is dict:
            return {key:({'reviewed_span_class':aliases[tuple(item)]}
                    if key in ('span','predicate','event') and type(item) is list and tuple(item) in aliases
                    else canonical(item)) for key,item in value.items()}
        if type(value) is list:
            return [canonical(item) for item in value]
        return value
    def worlds(value):
        return {json.dumps(dict(bindings=sorted(canonical(row['bindings']),key=lambda b:json.dumps(b,sort_keys=True)),
            events=sorted(canonical(row['events']),key=lambda e:json.dumps(e,sort_keys=True))),sort_keys=True)
            for row in value['readings']}
    observed, wanted = worlds(actual), worlds(expected)
    same_coverage = actual['coverage']==expected['coverage']
    return dict(correct=same_coverage and observed==wanted,coverage_correct=same_coverage,
        expected_readings=len(wanted),observed_readings=len(observed),
        missing_readings=len(wanted-observed),extra_readings=len(observed-wanted))


def grade_interpretation_packet(cases, references, schedule, observations):
    """Full-denominator exposed development grading; never a qualification claim."""
    from psychology_base.applications.measurement.semantic_membership import membership_inputs
    planned = {e['id']:e for e in schedule}
    observed = {o['id']:o for o in observations}
    if (len(planned)!=len(schedule) or len(observed)!=len(observations)
            or not set(observed)<=set(planned)):
        raise ValueError('Duplicate or unscheduled interpretation observation')
    records=[]
    for entry in schedule:
        case=cases[entry['case']]
        rows=membership_inputs(case,application=case['application'],format_version=4)
        focus=json.loads(next(r for r in rows if r['record']==case['focus_id'])['pair'][0])
        obs=observed.get(entry['id'])
        record=dict(id=entry['id'],language=case['language'],attempted=obs is not None,
            completed=obs is not None and 'analysis' in obs,correct=False)
        if record['completed']:
            record.update(compare_interpretation_analysis(obs['analysis'],references[entry['case']],focus))
        records.append(record)
    summary=dict(planned=len(records),attempted=sum(r['attempted'] for r in records),
        completed=sum(r['completed'] for r in records),correct=sum(r['correct'] for r in records),
        not_attempted=sum(not r['attempted'] for r in records))
    return dict(records=records,summary=dict(all=summary),qualified=False,
        candidate_screen_passed=bool(records) and all(r['correct'] for r in records))


def build_packet(parent, inherited, bank, descriptions):
    if len(parent['cases']) != 16 or set(parent['cases']) != set(bank['preserved']):
        raise ValueError('All sixteen previous scenarios must be preserved')
    cases = deepcopy(parent['cases'])
    declarations = deepcopy(bank['preserved'])
    for authored in bank['new_cases']:
        key, language = authored['id'], authored['language']
        if key in cases:
            raise ValueError('Duplicate extraction scenario')
        texts = authored['prior'] + [authored['text']]
        if language == 'zh' and any(not any('\u3400' <= ch <= '\u9fff' for ch in text) for text in texts):
            raise ValueError('Authored Chinese extraction control lost its source script')
        group = hashlib.sha256(('reference-extraction/'+key).encode()).hexdigest()[:32]
        cases[key] = dict(id=key, family=authored['family'], language=language, application='companion',
            split='regression', origin='authored', focus_id=len(texts)-1,
            description_ids=deepcopy(descriptions['description_ids']), descriptions=deepcopy(descriptions['descriptions'][language]),
            records=[dict(id=i, text=text, speaker='user' if i==len(texts)-1 else 'companion',
                          source_order=dict(group=group, position=i)) for i, text in enumerate(texts)])
        declarations[key] = deepcopy(authored['reference'])
    order=['referent-zh-ambiguous','referent-en-unique','reference-correlated-zh','reference-quotation-en',
        'original-calculation/with-prior','reference-missing-antecedent-zh','reference-alias-en','reference-implicit-zh',
        'referent-zh-unique','reference-correlated-en','reference-direct-name-zh','reference-quotation-zh',
        'original-reflection/with-prior','reference-unique-other-en','reference-implicit-en','reference-alias-zh',
        'both-en','reference-incompatible-act-zh','referent-en-ambiguous','reference-direct-name-en',
        'denial-zh-act','reference-unique-other-zh','reference-incompatible-act-en','reference-missing-antecedent-en']
    if len(cases) != 24 or set(order) != set(cases):
        raise ValueError('Prospective twenty-four-scenario scope changed')
    schedule = [dict(condition(cases[key]), version=4, descriptions_per_call=2) for key in order]
    schedule += [dict(condition(cases[key], repeat=True), version=4, descriptions_per_call=2)
                 for key in ('referent-en-unique','referent-zh-ambiguous','referent-zh-unique','referent-en-ambiguous')]
    refs = {e['case']: reviewed_reference(cases[e['case']],e,declarations[e['case']],inherited.get(e['case']))
            for e in schedule if not e['repeat']}
    return dict(cases=cases, schedule=schedule), refs


def compare_analysis(analysis, reference):
    """Normalize only through frozen reviewed source equivalences, after inference."""
    focus = reference['focus']
    expected_keys = [(m['turn'],m['start'],m['end']) for m in focus]
    mentions = analysis['mentions']
    keys = [(m['turn'],m['start'],m['end']) for m in mentions]
    actual_keys = [keys[i] for i in analysis['focus_references']]
    missing = set(expected_keys)-set(actual_keys)
    extra = set(actual_keys)-set(expected_keys)
    kinds = {key: mentions[i]['kind'] for i,key in zip(analysis['focus_references'],actual_keys)}
    kind_errors = sum(kinds.get(key) != m['kind'] for key,m in zip(expected_keys,focus) if key in kinds)
    names = {tuple(span): entity for entity,spans in reference['names'].items() for span in spans}
    entities, errors = [], []
    for entity in analysis['entities']:
        if 'speaker' in entity:
            entities.append('speaker:'+entity['speaker'])
        else:
            meanings = {names.get(keys[i]) for i in entity['mentions']}
            if len(meanings) != 1 or None in meanings:
                errors.append('invented_or_conflicting_named_entity')
                entities.append('unreviewed:'+str(len(entities)))
            else:
                entities.append(next(iter(meanings)))
    if len(set(entities)) != len(entities):
        errors.append('duplicate_semantic_entity')
    expected = {tuple(row) for row in reference['readings']}
    predicted = set()
    comparable = not missing and not extra
    if comparable:
        positions = [actual_keys.index(key) for key in expected_keys]
        for row in analysis['readings']:
            predicted.add(tuple(entities[row[p]] if row[p] is not None else None for p in positions))
    omitted, added = (expected-predicted, predicted-expected) if comparable else (None,None)
    correct = (not missing and not extra and not kind_errors and not errors and not omitted and not added
               and analysis['coverage'] == reference['coverage'] and analysis['coverage'] != 'unsupported')
    return dict(correct=correct, missing_focus_mentions=len(missing), extra_focus_mentions=len(extra),
        mention_kind_errors=kind_errors, entity_errors=errors, readings_comparable=comparable,
        omitted_readings=len(omitted) if comparable else None, extra_readings=len(added) if comparable else None,
        expected_coverage=reference['coverage'], predicted_coverage=analysis['coverage'],
        focus_signature=sorted([list(key)+[kinds[key]] for key in actual_keys]),
        normalized_readings=sorted([list(row) for row in predicted],key=lambda r:json.dumps(r)) if comparable else None,
        expected_readings=sorted([list(row) for row in expected],key=lambda r:json.dumps(r)))


def grade(cases, refs, schedule, observations, *, expected_conditions=28, expected_repeats=4):
    if (type(expected_conditions) is not int or expected_conditions <= 0
            or type(expected_repeats) is not int or not 0 <= expected_repeats < expected_conditions):
        raise ValueError('Invalid prospective extraction screen size')
    entries = {e['id']:e for e in schedule}
    if len(entries) != len(schedule) or set(cases) != set(refs):
        raise ValueError('Incomplete extraction scope')
    observed = {}
    for observation in observations:
        if observation['id'] not in entries or observation['id'] in observed:
            raise ValueError('Unexpected or duplicate extraction observation')
        observed[observation['id']] = observation
    records = []
    for entry in schedule:
        ref = refs[entry['case']]
        rows = rows_for(cases[entry['case']],entry)
        if hashlib.sha256(rows[0]['pair'][0].encode()).hexdigest() != ref['context_sha256']:
            raise ValueError('Extraction reference context changed')
        obs = observed.get(entry['id'])
        record = dict(id=entry['id'],case=entry['case'],language=entry['language'],
            repeat=entry['repeat'],correct=False,status='missing' if obs is None else 'failed')
        if obs is not None and 'analysis' in obs:
            record.update(compare_analysis(obs['analysis'],ref),status='completed')
        records.append(record)
    summaries = {}
    for language in ('en','zh','all'):
        selected = [r for r in records if language=='all' or r['language']==language]
        done = [r for r in selected if r['status']=='completed']
        summaries[language] = dict(planned=len(selected),completed=len(done),correct_conditions=sum(r['correct'] for r in selected),
            failed=sum(r['status']=='failed' for r in selected),missing=sum(r['status']=='missing' for r in selected),
            missing_focus_mentions=sum(r['missing_focus_mentions'] for r in done),
            extra_focus_mentions=sum(r['extra_focus_mentions'] for r in done),
            mention_kind_errors=sum(r['mention_kind_errors'] for r in done),entity_errors=sum(len(r['entity_errors']) for r in done),
            unaligned_reading_conditions=sum(not r['readings_comparable'] for r in done),
            omitted_readings=sum(r['omitted_readings'] for r in done if r['readings_comparable']),
            extra_readings=sum(r['extra_readings'] for r in done if r['readings_comparable']),
            coverage_errors=sum(r['expected_coverage']!=r['predicted_coverage'] for r in done),
            planned_readings=sum(len(refs[r['case']]['readings']) for r in selected))
    repeats = []
    for repeated in [r for r in records if r['repeat']]:
        original = next(r for r in records if r['case']==repeated['case'] and not r['repeat'])
        complete = original['status']==repeated['status']=='completed'
        fields = ('normalized_readings','predicted_coverage','focus_signature','mention_kind_errors','entity_errors')
        repeats.append(dict(case=repeated['case'],complete=complete,
            same=all(original[k]==repeated[k] for k in fields) if complete else None))
    return dict(qualified=False,scope='Exposed/authored extraction diagnostic, not act accuracy or recall qualification',
        summary=summaries,records=records,repeats=repeats,
        candidate_screen_passed=(len(records)==expected_conditions and all(r['correct'] for r in records)
                                 and len(repeats)==expected_repeats and all(r['same'] is True for r in repeats)))
