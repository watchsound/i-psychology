"""Source-bound reference proposals and conditional-label reduction.

Structural validity never establishes semantic binding or coverage accuracy.
This module has no model, retrieval, chronology or persistence workflow.
"""
from copy import deepcopy
import hashlib
import json
from unicodedata import category, name

from psychology_base.applications.measurement.memory_grounding import _unique_json_object
from psychology_base.applications.reporting_order import _phrase_targets, _target

SCHEMA = 'reference-readings-v1'
MAX_MENTIONS, MAX_ENTITIES, MAX_REFERENCES, MAX_READINGS = 16, 8, 8, 8
KINDS = frozenset(('named', 'anaphoric', 'deictic', 'implicit'))
MAX_PRIOR_TURNS = 2
PROJECTION_VERSION = 'reference-turn-ids-v1'


def reference_turns(focus):
    """Validate the existing canonical window without adding outside evidence."""
    if (type(focus) is not dict or set(focus) != {'context_scope', 'prior_utterances', 'focus_utterance'}
            or type(focus['context_scope']) is not str or not focus['context_scope'].strip()
            or type(focus['prior_utterances']) is not list or len(focus['prior_utterances']) > MAX_PRIOR_TURNS):
        raise ValueError('Reference extraction requires a canonical bounded context window')
    turns = focus['prior_utterances'] + [focus['focus_utterance']]
    for turn in turns:
        if (type(turn) is not dict or set(turn) != {'speaker', 'text'}
                or type(turn['speaker']) is not str or turn['speaker'] not in ('user', 'tutor', 'companion', 'unknown')
                or type(turn['text']) is not str or not turn['text'].strip()):
            raise ValueError('Invalid reference context turn')
    return turns


def reference_projection(focus):
    """Code-owned local IDs over exactly the existing canonical source window."""
    turns = reference_turns(focus)
    return dict(schema=PROJECTION_VERSION, context_scope=focus['context_scope'],
        turns=[dict(id=i, speaker=t['speaker'], text=t['text']) for i, t in enumerate(turns)],
        focus_turn=len(turns)-1)


def reference_response_schema():
    """Shape/bounds only; no reviewed names, referents or semantic answers.

    Turn IDs use the protocol's maximum window. Actual source membership,
    cross-field indices and alternative coverage still require the validator.
    Each caller receives a fresh tree, preventing shared schema mutation.
    """
    def obj(properties):
        return dict(type='object', properties=properties, required=list(properties), additionalProperties=False)
    def index(maximum):
        return dict(type='integer', minimum=0, maximum=maximum)
    def array(items, maximum, minimum=0):
        return dict(type='array', items=items, minItems=minimum, maxItems=maximum)
    mention = obj(dict(turn=index(MAX_PRIOR_TURNS), quote=dict(type='string', minLength=1),
        occurrence=dict(type='integer', minimum=0), kind=dict(type='string', enum=sorted(KINDS))))
    entity = dict(anyOf=[obj(dict(mentions=array(index(MAX_MENTIONS-1), MAX_MENTIONS, 1))),
        obj(dict(speaker=dict(type='string', enum=['user', 'tutor', 'companion']), turn=index(MAX_PRIOR_TURNS)))])
    target = dict(anyOf=[index(MAX_ENTITIES-1), dict(type='null')])
    return obj(dict(mentions=array(mention, MAX_MENTIONS), entities=array(entity, MAX_ENTITIES),
        focus_references=array(index(MAX_MENTIONS-1), MAX_REFERENCES),
        readings=array(array(target, MAX_REFERENCES), MAX_READINGS, 1),
        coverage=dict(type='string', enum=['enumerated', 'open', 'unsupported'])))


def decode_reference_readings(content, focus):
    """Resolve exact quotations to source offsets; preserve all proposed readings."""
    turns = reference_turns(focus)
    value = json.loads(content, object_pairs_hook=_unique_json_object)
    if (type(value) is not dict or set(value) != {'mentions', 'entities', 'focus_references', 'readings', 'coverage'}
            or type(value['mentions']) is not list or len(value['mentions']) > MAX_MENTIONS
            or type(value['entities']) is not list or len(value['entities']) > MAX_ENTITIES
            or type(value['focus_references']) is not list or len(value['focus_references']) > MAX_REFERENCES
            or type(value['readings']) is not list or not 1 <= len(value['readings']) <= MAX_READINGS
            or type(value['coverage']) is not str or value['coverage'] not in ('enumerated', 'open', 'unsupported')):
        raise ValueError('Malformed or unbounded reference envelope')
    mentions, seen_spans = [], set()
    for mention in value['mentions']:
        if (type(mention) is not dict or set(mention) != {'turn', 'quote', 'occurrence', 'kind'}
                or type(mention['turn']) is not int or not 0 <= mention['turn'] < len(turns)
                or type(mention['occurrence']) is not int or mention['occurrence'] < 0
                or type(mention['quote']) is not str or not mention['quote'].strip()
                or type(mention['kind']) is not str or mention['kind'] not in KINDS):
            raise ValueError('Invalid reference mention')
        turn = turns[mention['turn']]
        spans = _phrase_targets([dict(id=mention['turn'], text=turn['text'], speaker=turn['speaker'])], mention['quote'])
        if mention['occurrence'] >= len(spans):
            raise ValueError('Reference quotation/occurrence is absent from its source turn')
        span = spans[mention['occurrence']]
        key = (mention['turn'], span.start, span.end)
        if key in seen_spans:
            raise ValueError('Duplicate source reference mention')
        seen_spans.add(key)
        mentions.append(dict(mention, start=span.start, end=span.end))
    used, entity_signatures = {}, set()
    for index, entity in enumerate(value['entities']):
        if type(entity) is not dict:
            raise ValueError('Reference entities must be objects')
        if set(entity) == {'mentions'}:
            members = entity['mentions']
            if (type(members) is not list or not members or len(members) > MAX_MENTIONS
                    or any(type(i) is not int or not 0 <= i < len(mentions) for i in members)
                    or len(set(members)) != len(members)
                    or any(mentions[i]['kind'] != 'named' for i in members)):
                raise ValueError('Entity clusters contain only actual names/aliases; other references belong in readings')
            for member in members:
                if member in used:
                    raise ValueError('A mention cannot anchor two different entities')
                used[member] = index
            signature = ('mentions', tuple(sorted(members)))
        elif set(entity) == {'speaker', 'turn'}:
            if (type(entity['turn']) is not int or not 0 <= entity['turn'] < len(turns)
                    or type(entity['speaker']) is not str or entity['speaker'] not in ('user', 'tutor', 'companion')
                    or turns[entity['turn']]['speaker'] != entity['speaker']):
                raise ValueError('Role anchor is not a known supplied outer speaker')
            signature = ('speaker', entity['speaker'])
        else:
            raise ValueError('Invalid entity anchor shape')
        if signature in entity_signatures:
            raise ValueError('Repeated reference entity')
        entity_signatures.add(signature)
    refs = value['focus_references']
    if (any(type(i) is not int or not 0 <= i < len(mentions) for i in refs)
            or len(set(refs)) != len(refs)
            or set(refs) != {i for i, mention in enumerate(mentions) if mention['turn'] == len(turns)-1}):
        raise ValueError('Every supplied focus mention must be assigned exactly once')
    signatures, unresolved = set(), False
    for reading in value['readings']:
        if (type(reading) is not list or len(reading) != len(refs)
                or any(target is not None and (type(target) is not int or not 0 <= target < len(value['entities']))
                       for target in reading)):
            raise ValueError('Incomplete or dangling joint reference reading')
        key = tuple(reading)
        if key in signatures:
            raise ValueError('Duplicate reference reading')
        signatures.add(key)
        unresolved |= any(target is None for target in reading)
        if any(ref in used and used[ref] != target for ref, target in zip(refs, reading)):
            raise ValueError('Reading contradicts its own anchored focus mention')
    if (value['coverage'] == 'enumerated' and unresolved
            or value['coverage'] == 'open' and not unresolved):
        raise ValueError('Coverage claim contradicts unresolved assignments')
    result = deepcopy(value)
    result.update(schema=SCHEMA, mentions=mentions)
    return result


def reduce_reading_labels(labels, *, expected_readings):
    """Agreement conditional on a complete, correctly aligned reading table."""
    if (type(expected_readings) is not int or not 1 <= expected_readings <= MAX_READINGS
            or type(labels) is not list or len(labels) != expected_readings
            or any(type(row) is not list or len(row) != 2
                   or any(type(v) is not str or v not in ('match', 'nonmatch', 'uncertain') for v in row)
                   for row in labels)):
        raise ValueError('A complete nonempty reading-by-description matrix is required')
    result = []
    for column in zip(*labels):
        values = set(column)
        result.append(next(iter(values)) if len(values) == 1 and 'uncertain' not in values else 'uncertain')
    return result


SELECTION_SCHEMA = 'reference-source-selections-v1'
MAX_SOURCE_UNITS, MAX_SOURCE_BYTES = 256, 4096


def _source_intervals(text):
    """Lossless source coordinates, independent of retrieval word segmentation.

    Letter/number/mark runs stay together. Each Han character and punctuation
    has its own selectable boundary. Whitespace remains in original source
    slices between units; no normalization, NER or semantic filtering occurs.
    """
    start = None
    for offset, char in enumerate(text):
        han = name(char, '').startswith(('CJK UNIFIED IDEOGRAPH', 'CJK COMPATIBILITY IDEOGRAPH'))
        run = not han and (category(char)[0] in ('L', 'N', 'M') or char == '_')
        if run:
            if start is None:
                start = offset
            continue
        if start is not None:
            yield start, offset
            start = None
        if not char.isspace():
            yield offset, offset+1
    if start is not None:
        yield start, len(text)


def reference_source_units(focus):
    """One bounded, source-only coordinate registry of existing ReportingTargets.

    Unit IDs are tuple positions, scoped to this canonical window. Overflow
    rejects the whole view; it never returns a truncated candidate prefix.
    These units are not a replacement for i18n retrieval tokens.
    """
    turns = reference_turns(focus)
    if sum(len(t['text']) for t in turns) > MAX_SOURCE_BYTES:
        raise ValueError('Reference source exceeds the byte bound')
    if sum(len(t['text'].encode('utf-8')) for t in turns) > MAX_SOURCE_BYTES:
        raise ValueError('Reference source exceeds the byte bound')
    units = []
    for turn, record in enumerate(turns):
        source = dict(record, id=turn)
        for start, end in _source_intervals(record['text']):
            if len(units) == MAX_SOURCE_UNITS:
                raise ValueError('Reference source exceeds the unit bound')
            units.append(_target(source, start, end))
    return tuple(units)


def reference_candidate_view(focus):
    """Render selectable source IDs, never proposed names, aliases or labels."""
    turns, units = reference_turns(focus), reference_source_units(focus)
    identity = hashlib.sha256(json.dumps(focus, sort_keys=True, ensure_ascii=False,
        separators=(',', ':')).encode('utf-8')).hexdigest()
    return dict(schema=SELECTION_SCHEMA, source_identity=identity, context_scope=focus['context_scope'],
        focus_turn=len(turns)-1, speakers=sorted({t['speaker'] for t in turns if t['speaker'] != 'unknown'}),
        turns=[dict(id=i, speaker=t['speaker'], text=t['text'],
            units=[[j,u.text] for j,u in enumerate(units) if u.record_id == i]) for i,t in enumerate(turns)])


def reference_selection_schema(focus):
    """Request-specific source ID bounds; semantic choices remain model supplied."""
    units = reference_source_units(focus)
    roles = sorted({t['speaker'] for t in reference_turns(focus) if t['speaker'] != 'unknown'})
    def obj(properties):
        return dict(type='object', properties=properties, required=list(properties), additionalProperties=False)
    span = dict(type='array', minItems=2, maxItems=2,
        items=dict(type='integer', minimum=0, maximum=len(units)-1))
    focus_span = deepcopy(span)
    focus_span['items']['minimum'] = next(i for i,u in enumerate(units) if u.record_id == len(reference_turns(focus))-1)
    targets = [dict(type='null'), obj(dict(name=deepcopy(span)))]
    if roles:
        targets.append(obj(dict(speaker=dict(type='string', enum=roles))))
    reference = obj(dict(span=focus_span, kind=dict(type='string', enum=sorted(KINDS)), target=dict(anyOf=targets)))
    return obj(dict(coverage=dict(type='string',enum=['enumerated','open','unsupported']),
        readings=dict(type='array', minItems=1, maxItems=MAX_READINGS,
        items=dict(type='array', maxItems=MAX_REFERENCES, items=reference))))


def _selected_span(selection, units, turns):
    if (type(selection) is not list or len(selection) != 2
            or any(type(i) is not int or not 0 <= i < len(units) for i in selection)):
        raise ValueError('Invalid selected source IDs')
    first, last = (units[i] for i in selection)
    if selection[0] > selection[1] or first.record_id != last.record_id:
        raise ValueError('Selected source interval crosses turns or reverses order')
    turn, start, end = first.record_id, first.start, last.end
    source = dict(turns[turn], id=turn)
    quote = source['text'][start:end]
    matches = _phrase_targets([source], quote)
    occurrence = next((i for i,s in enumerate(matches) if (s.start,s.end) == (start,end)), None)
    if occurrence is None:
        raise ValueError('Selected source interval violates the existing literal boundary contract')
    return (turn,start,end), dict(turn=turn,quote=quote,occurrence=occurrence)


def decode_reference_selections(content, focus):
    """Translate source choices to the existing graph without semantic repair.

    Each reading contains span/kind/target objects, not model-created graph
    indices. All readings must supply the same focus mentions/kinds. Name
    targets assert one representative source anchor; code never merges aliases
    or maps first-person references to an outer speaker on its own.
    """
    value = json.loads(content, object_pairs_hook=_unique_json_object)
    if (type(value) is not dict or set(value) != {'coverage','readings'}
            or type(value['coverage']) is not str or value['coverage'] not in ('enumerated','open','unsupported')
            or type(value['readings']) is not list
            or not 1 <= len(value['readings']) <= MAX_READINGS):
        raise ValueError('Malformed or unbounded source-selection envelope')
    turns, units = reference_turns(focus), reference_source_units(focus)
    roles = {t['speaker']:next(i for i,s in enumerate(turns) if s['speaker'] == t['speaker'])
             for t in turns if t['speaker'] != 'unknown'}
    mentions, kinds, targets, worlds = {}, {}, set(), []
    focus_kinds = None
    def add_span(selection, kind):
        key, mention = _selected_span(selection, units, turns)
        if key in kinds and kinds[key] != kind:
            raise ValueError('Selected source span has inconsistent mention kinds')
        mentions[key], kinds[key] = mention, kind
        return key
    for reading in value['readings']:
        if type(reading) is not list or len(reading) > MAX_REFERENCES:
            raise ValueError('Invalid selected reading')
        current, world = {}, {}
        for reference in reading:
            if (type(reference) is not dict or set(reference) != {'span','kind','target'}
                    or type(reference['kind']) is not str or reference['kind'] not in KINDS):
                raise ValueError('Invalid selected focus reference')
            key = add_span(reference['span'],reference['kind'])
            if key[0] != len(turns)-1 or key in current:
                raise ValueError('Selected focus reference is prior or repeated')
            current[key] = reference['kind']
            target = reference['target']
            if target is None:
                anchor = None
            elif type(target) is dict and set(target) == {'name'}:
                anchor = ('name',) + add_span(target['name'],'named')
            elif (type(target) is dict and set(target) == {'speaker'}
                    and type(target['speaker']) is str and target['speaker'] in roles):
                anchor = ('speaker',target['speaker'])
            else:
                raise ValueError('Selected referent is not a supplied name span, known speaker or null')
            if anchor is not None:
                targets.add(anchor)
            world[key] = anchor
        if focus_kinds is None:
            focus_kinds = current
        elif current != focus_kinds:
            raise ValueError('Selected readings disagree about focus mention coverage or kinds')
        worlds.append(world)
    if len(mentions) > MAX_MENTIONS or len(targets) > MAX_ENTITIES:
        raise ValueError('Selected graph exceeds existing mention/entity bounds')
    order = sorted(mentions)
    mention_ids = {key:i for i,key in enumerate(order)}
    target_order = sorted(targets)
    target_ids = {key:i for i,key in enumerate(target_order)}
    entities = [dict(mentions=[mention_ids[key[1:]]]) if key[0] == 'name'
        else dict(speaker=key[1],turn=roles[key[1]]) for key in target_order]
    focus_order = sorted(focus_kinds)
    readings = [[target_ids[world[key]] if world[key] is not None else None for key in focus_order] for world in worlds]
    graph = dict(mentions=[dict(mentions[key],kind=kinds[key]) for key in order],entities=entities,
        focus_references=[mention_ids[key] for key in focus_order],readings=readings,coverage=value['coverage'])
    return decode_reference_readings(json.dumps(graph,ensure_ascii=False),focus)


# Opt-in typed source selections. The v1 registry and decoder above stay exact.
INTERPRETATION_SCHEMA = 'reference-source-selections-v2'
CONDITIONAL_SCHEMA = 'reading-membership-v2'
INTERPRETATION_KINDS = KINDS | {'nominal', 'clausal'}
INTERPRETATION_ROLES = frozenset(('referent', 'agent', 'patient', 'recipient'))
INTERPRETATION_VALIDATION_REVISION = 'full-implicit-clause-v1'
MAX_EVENTS = 8


def interpretation_candidate_view(focus):
    """The existing lossless source registry, with an explicit new protocol."""
    view = reference_candidate_view(focus)
    view['schema'] = INTERPRETATION_SCHEMA
    return view


def decode_interpretation_response(content, focus):
    """Attach code-owned protocol/source identity to semantic choices only."""
    value = json.loads(content, object_pairs_hook=_unique_json_object)
    if type(value) is not dict or set(value) != {'coverage', 'readings'}:
        raise ValueError('Interpretation response must contain only semantic choices')
    view = interpretation_candidate_view(focus)
    envelope = dict(value, schema=INTERPRETATION_SCHEMA, source_identity=view['source_identity'])
    return decode_interpretation_selections(json.dumps(envelope,ensure_ascii=False),focus)


def decode_interpretation_selections(content, focus):
    """Validate typed source choices, never infer or repair their semantics.

    Role/event coordinates distinguish omitted slots sharing a clause. All
    readings preserve the same source-stable frames and slots. A valid envelope
    does not establish that the model supplied every required frame or reading.
    """
    value = json.loads(content, object_pairs_hook=_unique_json_object)
    view = interpretation_candidate_view(focus)
    if (type(value) is not dict
            or set(value) != {'schema', 'source_identity', 'coverage', 'readings'}
            or value['schema'] != INTERPRETATION_SCHEMA
            or value['source_identity'] != view['source_identity']
            or type(value['coverage']) is not str
            or value['coverage'] not in ('enumerated', 'open', 'unsupported')
            or type(value['readings']) is not list
            or not 1 <= len(value['readings']) <= MAX_READINGS):
        raise ValueError('Malformed or mismatched interpretation envelope')
    turns, units = reference_turns(focus), reference_source_units(focus)
    roles = {t['speaker'] for t in turns if t['speaker'] != 'unknown'}
    spans, targets, worlds = set(), set(), set()
    expected_shape = None
    unresolved = False

    def source_span(selection, *, focus_only=False):
        key, _ = _selected_span(selection, units, turns)
        if focus_only and key[0] != len(turns)-1:
            raise ValueError('Interpretation focus anchor is outside focus')
        spans.add(key)
        return key

    def target(anchor, *, attribution=False):
        nonlocal unresolved
        if anchor is None:
            unresolved = True
            return
        if type(anchor) is dict and set(anchor) == {'speaker'}:
            if type(anchor['speaker']) is not str or anchor['speaker'] not in roles:
                raise ValueError('Interpretation role is not a supplied known speaker')
            targets.add(('speaker', anchor['speaker']))
        elif type(anchor) is dict and set(anchor) == {'span', 'type'}:
            if (type(anchor['type']) is not str or anchor['type'] not in ('person', 'object', 'event')
                    or attribution and anchor['type'] != 'person'):
                raise ValueError('Invalid typed interpretation target')
            targets.add((anchor['type'], *source_span(anchor['span'])))
        else:
            raise ValueError('Invalid interpretation target anchor')

    def contained(inner, outer):
        return inner[0] == outer[0] and outer[1] <= inner[1] < inner[2] <= outer[2]

    for reading in value['readings']:
        if (type(reading) is not dict or set(reading) != {'bindings', 'events'}
                or type(reading['bindings']) is not list or len(reading['bindings']) > MAX_REFERENCES
                or type(reading['events']) is not list or len(reading['events']) > MAX_EVENTS):
            raise ValueError('Malformed or unbounded interpretation reading')
        frames = {}
        for event in reading['events']:
            if (type(event) is not dict
                    or set(event) != {'span', 'predicate', 'polarity', 'modality', 'attribution'}
                    or type(event['polarity']) is not str
                    or event['polarity'] not in ('affirmed', 'negated', 'unknown')
                    or type(event['modality']) is not str
                    or event['modality'] not in ('asserted', 'intended', 'hypothetical', 'unknown')):
                raise ValueError('Invalid interpretation event')
            key = source_span(event['span'], focus_only=True)
            predicate = source_span(event['predicate'], focus_only=True)
            if key in frames or not contained(predicate, key):
                raise ValueError('Repeated event or predicate outside its event')
            frames[key] = predicate
            unresolved |= event['polarity'] == 'unknown' or event['modality'] == 'unknown'
            target(event['attribution'], attribution=True)
        slots = {}
        for binding in reading['bindings']:
            if (type(binding) is not dict
                    or set(binding) != {'span', 'kind', 'role', 'event', 'target'}
                    or type(binding['kind']) is not str or binding['kind'] not in INTERPRETATION_KINDS
                    or type(binding['role']) is not str or binding['role'] not in INTERPRETATION_ROLES):
                raise ValueError('Invalid interpretation binding')
            key = source_span(binding['span'], focus_only=True)
            event = None if binding['event'] is None else source_span(binding['event'], focus_only=True)
            if ((event is None) != (binding['role'] == 'referent')
                    or event is not None and (event not in frames or not contained(key, event))):
                raise ValueError('Interpretation binding has an orphan or incompatible event')
            if binding['kind'] == 'implicit' and event is not None:
                import unicodedata
                text = turns[event[0]]['text']
                omitted = text[event[1]:key[1]] + text[key[2]:event[2]]
                if any(not c.isspace() and not unicodedata.category(c).startswith('P') for c in omitted):
                    raise ValueError('Implicit interpretation slot must select its full event clause')
            slot = (key, binding['role'], event)
            if slot in slots:
                raise ValueError('Repeated interpretation slot')
            slots[slot] = binding['kind']
            target(binding['target'])
        shape = (slots, frames)
        if expected_shape is not None and shape != expected_shape:
            raise ValueError('Interpretation readings disagree about source frames or slots')
        expected_shape = shape
        # Ordering is not a semantic choice; do not accept a reordered duplicate.
        signature = json.dumps(dict(
            bindings=sorted(reading['bindings'], key=lambda b: json.dumps(b, sort_keys=True)),
            events=sorted(reading['events'], key=lambda e: json.dumps(e, sort_keys=True))), sort_keys=True)
        if signature in worlds:
            raise ValueError('Duplicate interpretation reading')
        worlds.add(signature)
    if len(spans) > MAX_MENTIONS or len(targets) > MAX_ENTITIES:
        raise ValueError('Interpretation exceeds existing source or target bounds')
    if (value['coverage'] == 'enumerated' and unresolved
            or value['coverage'] == 'open' and not unresolved):
        raise ValueError('Interpretation coverage contradicts explicit unknowns')
    return deepcopy(value)


def interpretation_membership_payload(focus, interpretation, descriptions):
    """Bind a complete conditional task to its exact source, readings and goals."""
    analysis = decode_interpretation_selections(json.dumps(interpretation, ensure_ascii=False), focus)
    if analysis['coverage'] == 'unsupported':
        raise ValueError('Unsupported interpretation cannot supply conditional membership')
    if (type(descriptions) not in (list, tuple) or len(descriptions) != 2
            or any(type(d) is not str or not d.strip() for d in descriptions)):
        raise ValueError('Conditional membership requires exactly two descriptions')
    payload = dict(schema=CONDITIONAL_SCHEMA, source=interpretation_candidate_view(focus),
                   interpretation=analysis, descriptions=list(descriptions))
    identity = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
        separators=(',', ':')).encode('utf-8')).hexdigest()
    return dict(payload, request_identity=identity)


def interpretation_reuse_key(focus, *, scope, provenance, configuration, interpretation=None, descriptions=None):
    """Exact source/provenance identity, separate from conditional query labels.

    Provenance is supplied by the existing occurrence owner, never by a model.
    Temporary renderer IDs are excluded; stable occurrence IDs are retained.
    """
    if (type(scope) not in (tuple, list) or len(scope) != 2 or scope[0] not in ('tutor', 'companion')
            or type(scope[1]) is not str or not scope[1].strip() or scope[1] != scope[1].strip()
            or len(scope[1].encode('utf-8')) > 256
            or type(configuration) is not str or len(configuration) != 64
            or any(c not in '0123456789abcdef' for c in configuration)
            or type(provenance) is not dict or set(provenance) != {'focus', 'prior'}
            or type(provenance['prior']) not in (tuple, list)):
        raise ValueError('Interpretation reuse requires bounded user/application provenance')
    source = interpretation_candidate_view(focus)
    turns = reference_turns(focus)
    entries = [*provenance['prior'], provenance['focus']]
    if len(entries) != len(turns):
        raise ValueError('Reuse provenance does not cover the exact source window')
    ids = set()
    for entry, turn in zip(entries, turns):
        if (type(entry) is not dict or set(entry) != {'occurrence_id', 'record'}
                or type(entry['occurrence_id']) is not str or not entry['occurrence_id']
                or len(entry['occurrence_id'].encode('utf-8')) > 256
                or entry['occurrence_id'] in ids or type(entry['record']) is not dict
                or set(entry['record']) != {'speaker', 'session', 'text', 'source_order'}
                or entry['record']['text'] != turn['text']
                or (entry['record']['speaker'] or 'unknown') != turn['speaker']):
            raise ValueError('Reuse provenance differs from the actual source window')
        ids.add(entry['occurrence_id'])
    if len(json.dumps(provenance, ensure_ascii=False, allow_nan=False).encode('utf-8')) > 16384:
        raise ValueError('Reuse provenance exceeds its byte bound')
    value = dict(version='bounded-interpretation-reuse-v1', scope=list(scope),
                 configuration=configuration, source=source, provenance=provenance)
    if interpretation is not None or descriptions is not None:
        value['conditional'] = interpretation_membership_payload(focus, interpretation, descriptions)
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def reduce_interpretation_labels(content, focus, interpretation, descriptions, *, request_identity):
    """Reuse v1 agreement, including an explicit unresolved-possibility judgment."""
    payload = interpretation_membership_payload(focus, interpretation, descriptions)
    value = json.loads(content, object_pairs_hook=_unique_json_object)
    if (type(value) is not dict or set(value) != {'labels', 'open_labels'}
            or request_identity != payload['request_identity']):
        raise ValueError('Conditional labels do not belong to this exact interpretation task')
    labels = reduce_reading_labels(value['labels'], expected_readings=len(interpretation['readings']))
    if interpretation['coverage'] == 'open':
        # A table with eight explicit readings still reserves unknown possibilities.
        return reduce_reading_labels([labels, value['open_labels']], expected_readings=2)
    if value['open_labels'] is not None:
        raise ValueError('Enumerated coverage cannot attach an unknown-possibility row')
    return labels
