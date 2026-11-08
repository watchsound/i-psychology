"""Offline provenance and export checks over the canonical membership contract.

No annotation is inferred here. Coordinates verify supplied source occurrences;
the existing formatter alone selects context, and the existing reviewer validator
alone validates labels. Historical packets are never modified.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from psychology_base.applications.measurement.semantic_membership import (
    MEMBERSHIP_REASONS, membership_inputs, reviewed_membership_rows,
)
from psychology_base.companion.turn_provenance import source_conflicts


EXPOSURES = frozenset({
    'train', 'tuning', 'regression', 'diagnostic', 'reserved', 'unclassified_prior_use',
})
PARTITIONS = frozenset({'train', 'tuning', 'qualification', 'regression', 'reserved'})
REVIEW_STATES = frozenset({
    'draft', 'projected_only', 'implementer_reviewed', 'independently_reviewed',
    'adjudicated', 'unresolved', 'invalid',
})


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def pinned_json(path: Path, expected_sha256: str):
    """Read a local snapshot only after verifying its exact bytes."""
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError('Pinned source changed: ' + str(path))
    text = data.decode('utf-8-sig')
    return ([json.loads(line) for line in text.splitlines()]
            if path.suffix == '.jsonl' else json.loads(text))


def _coordinates(case, turns, positions, source_key):
    # An explicit native-position map preserves publisher gaps. Existing dense
    # importers retain their exact enumerate-based coordinate convention.
    source_turns = turns if isinstance(turns, dict) else dict(enumerate(turns))
    if any(type(p) is not int or p < 0 for p in source_turns):
        raise ValueError('Native source positions must be nonnegative integers')
    records = case['records']
    ids = [r['id'] for r in records]
    if (any(type(i) is not int for i in ids) or len(set(ids)) != len(ids)
            or set(positions) != set(ids)
            or any(type(i) is not int or type(p) is not int or p not in source_turns
                   for i, p in positions.items())
            or len(set(positions.values())) != len(ids)):
        raise ValueError('Every supplied occurrence needs a distinct verified position')
    result = deepcopy(case)
    group = _digest(['membership-source-coordinate-v1', source_key])[:32]
    for record in result['records']:
        position = positions[record['id']]
        original = source_turns[position]
        if (record['text'], record['speaker']) != (original['text'], original['speaker']):
            raise ValueError('Source occurrence text or outer speaker changed')
        coordinate = dict(group=group, position=position)
        if record.get('source_order') is not None and record['source_order'] != coordinate:
            raise ValueError('Existing coordinates conflict with recovered source')
        record['source_order'] = coordinate
    if source_conflicts(r['source_order'] for r in result['records']):
        raise ValueError('Conflicting recovered source coordinates')
    return result


def recover_public_coordinates(case, *, corpus_path, corpus_sha256, corpus_kind, imported):
    """Recover the first-six import from a separately pinned historical draft.

    Record IDs are interpreted as positions only because that importer explicitly
    created IDs 0..5 from the first six messages. Case-array order is irrelevant.
    KdConv intentionally supplies no application outer-speaker identities.
    """
    if (case.get('origin') != 'public-source' or corpus_kind not in ('kdconv', 'mathdial')
            or any(case.get(k) != imported.get(k) for k in ('id', 'source', 'row', 'identity', 'language'))
            or [r['id'] for r in imported['records']] != list(range(6))
            or type(case.get('row')) is not int or case['row'] < 0):
        raise ValueError('A verified first-six import declaration is required')
    raw = pinned_json(Path(corpus_path), corpus_sha256)
    if case['row'] >= len(raw):
        raise ValueError('Source row is outside pinned corpus')
    source = raw[case['row']]
    identity = source['name'] if corpus_kind == 'kdconv' else source['conversation_id']
    if identity != case['identity']:
        raise ValueError('Source identity changed')
    if corpus_kind == 'kdconv':
        turns = [dict(text=t['message'], speaker=None) for t in source['messages']]
    else:
        if any(t['role'] not in ('student', 'teacher') for t in source['turns']):
            raise ValueError('Unknown publisher outer speaker')
        turns = [dict(text=t['utterance'], speaker='user' if t['role'] == 'student' else 'tutor')
                 for t in source['turns']]
    if (len(turns) < 6 or any((r['text'], r['speaker']) != (turns[i]['text'], turns[i]['speaker'])
                              for i, r in enumerate(imported['records']))):
        raise ValueError('First-six importer no longer matches pinned transcript')
    if {r['id'] for r in case['records']} != set(range(6)):
        raise ValueError('Recovered seed must retain all six imported occurrences')
    return _coordinates(case, turns, {i: i for i in range(6)},
                        [corpus_kind, case['source'], corpus_sha256, case['row'], identity])


def recover_authored_coordinates(case, *, declaration_path, declaration_sha256):
    """Require an explicit frozen chronology, never an old contrast-array order."""
    declaration = pinned_json(Path(declaration_path), declaration_sha256)
    if declaration.get('schema') != 'authored-membership-sequences-v1' or case.get('origin') != 'authored':
        raise ValueError('An explicit authored chronology declaration is required')
    sequences = [s for s in declaration['sequences'] if s['id'] == case['id']]
    if len(sequences) != 1:
        raise ValueError('Missing or duplicate authored chronology')
    sequence = sequences[0]
    if (sequence.get('chronology') != 'authored' or sequence.get('language') != case['language']
            or not isinstance(sequence.get('author'), str) or not sequence['author'].strip()):
        raise ValueError('Authored chronology must identify its author and language')
    turns = sequence['records']
    ids = [r['id'] for r in turns]
    if any(type(i) is not int for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('Duplicate or invalid authored source occurrence')
    return _coordinates(case, turns, {r['id']: i for i, r in enumerate(turns)},
                        ['authored', declaration_sha256, sequence['id']])


def build_source_ledger(entries):
    """Transitive source/translation/derivation components, never entity guesses.

    source_keys are namespaced, explicitly established source/problem/family
    identities. Shared corpus, actor name or language is not a component edge.
    Unknown historical use fails closed rather than being declared fresh.
    """
    entries = deepcopy(entries)
    by_id = {}
    for entry in entries:
        key = entry.get('id')
        if not isinstance(key, str) or not key or key in by_id:
            raise ValueError('Duplicate or missing source entry ID')
        keys, parents, exposure = (entry.get(k) for k in ('source_keys', 'parents', 'exposure'))
        if (type(keys) is not list or not keys or any(not isinstance(k, str) or not k for k in keys)
                or type(parents) is not list or any(not isinstance(p, str) for p in parents)
                or type(exposure) is not list or any(not isinstance(e, str) or e not in EXPOSURES for e in exposure)
                or type(entry.get('qualification_only', False)) is not bool):
            raise ValueError('Explicit source keys, parents and exposure history are required')
        by_id[key] = entry
    groups = {key: key for key in by_id}

    def root(key):
        while groups[key] != key:
            groups[key] = groups[groups[key]]
            key = groups[key]
        return key

    def join(a, b):
        a, b = root(a), root(b)
        groups[max(a, b)] = min(a, b)

    owners = {}
    for key, entry in by_id.items():
        for parent in entry['parents']:
            if parent not in by_id or parent == key:
                raise ValueError('Missing or self-referential derivation parent')
            join(key, parent)
        for source in entry['source_keys']:
            if source in owners:
                join(key, owners[source])
            owners[source] = key
    # A cycle cannot establish a source origin even though union-find accepts it.
    visiting, done = set(), set()

    def visit(key):
        if key in visiting:
            raise ValueError('Cyclic source derivation')
        if key not in done:
            visiting.add(key)
            for parent in by_id[key]['parents']:
                visit(parent)
            visiting.remove(key)
            done.add(key)

    for key in by_id:
        visit(key)
    components = {}
    for key in sorted(by_id):
        components.setdefault(root(key), []).append(key)
    result = []
    for members in components.values():
        exposure = {e for key in members for e in by_id[key]['exposure']}
        allowed = set(PARTITIONS)
        if exposure:
            allowed &= {'train', 'regression'}
        if 'tuning' in exposure:
            allowed = {'tuning', 'regression'}
        if 'reserved' in exposure:
            allowed = {'reserved', 'regression'} if exposure == {'reserved'} else {'regression'}
        if exposure & {'regression', 'diagnostic', 'unclassified_prior_use'}:
            allowed = {'regression'}
        # An unreviewed qualification candidate can be locked before it has
        # labels, without weakening the historical reserved-source restriction.
        if any(by_id[key].get('qualification_only', False) for key in members):
            allowed &= {'qualification', 'regression'}
        result.append(dict(id=_digest(members), members=members, exposure=sorted(exposure),
                           allowed_partitions=sorted(allowed)))
    return dict(schema='membership-source-ledger-v1', entries=sorted(entries, key=lambda e: e['id']),
                components=sorted(result, key=lambda c: c['id']))


def validate_membership_partitions(cases, ledger):
    """Check source and exact-input separation before any labels are admitted.

    Pending qualification cases must take part in this check even though the
    review/export gate correctly refuses their unqualified references. This
    establishes partition integrity only, never semantic or review readiness.
    """
    if ledger != build_source_ledger(ledger['entries']):
        raise ValueError('Source ledger is not its canonical component projection')
    if len({c['id'] for c in cases}) != len(cases):
        raise ValueError('Every unique case requires its own source identity')
    components = {key: component for component in ledger['components'] for key in component['members']}
    entries = {entry['id']: entry for entry in ledger['entries']}
    selected, used_components, payload_splits = {}, {}, {}
    for case in cases:
        component = components.get(case.get('source_entry'))
        split = case.get('split')
        if (component is None or split not in component['allowed_partitions']
                or entries[case['source_entry']].get('case_id') != case['id']):
            raise ValueError('Source exposure forbids this partition')
        if used_components.setdefault(component['id'], split) != split:
            raise ValueError('Source/translation/derivation component crosses partitions')
        for row in membership_inputs(case, application=case.get('application'), format_version=4):
            digest = _digest(row['pair'])
            if payload_splits.setdefault(digest, split) != split:
                raise ValueError('Identical model input crosses partitions')
        selected[case['id']] = component
    return selected


def prepare_membership_bank(cases, references, annotations, ledger):
    """Compose format-4 reviews, component admission and exact-input deduplication.

    Only complete, explicitly reviewed references reach this export boundary.
    Membership loss masks remain separate from any future reference loss masks.
    No review states or labels are manufactured from historical contract 2.
    """
    components = validate_membership_partitions(cases, ledger)
    by_ref = {r['id']: r for r in references}
    by_review = {r['id']: r for r in annotations}
    ids = {c['id'] for c in cases}
    if (len(ids) != len(cases) or len(by_ref) != len(references) or len(by_review) != len(annotations)
            or set(by_ref) != ids or set(by_review) != ids):
        raise ValueError('Every unique case requires its own complete reference and review metadata')
    payloads, complete, semantic = {}, [], []
    for case in cases:
        component = components[case['id']]
        split = case.get('split')
        application = case.get('application')
        ref, review = by_ref[case['id']], by_review[case['id']]
        if (ref.get('contract_version') != 4 or review.get('state') not in REVIEW_STATES
                or type(review.get('membership_loss')) is not bool
                or type(review.get('reviewers')) is not list
                or any(not isinstance(r, str) or not r.strip() for r in review['reviewers'])):
            raise ValueError('Explicit format-4 review and membership loss mask required')
        if review['membership_loss'] and (
                review['state'] not in ('implementer_reviewed', 'independently_reviewed', 'adjudicated')
                or ref.get('reviewer') not in review['reviewers']):
            raise ValueError('Unresolved or projected supervision cannot contribute membership loss')
        # An independent review can establish a disagreement without resolving
        # it. Retain that qualification reference only as a complete masked
        # record; the semantic export below still requires membership_loss.
        dispute = review.get('dispute')
        reviewed_hold = (split == 'qualification' and review['state'] == 'unresolved'
            and review['membership_loss'] is False and type(dispute) is dict
            and isinstance(dispute.get('reason'), str) and bool(dispute['reason'].strip())
            and ref.get('reviewer') in review['reviewers'])
        if (review['state'] == 'independently_reviewed' or split == 'qualification') and (
                (review['state'] not in ('independently_reviewed', 'adjudicated') and not reviewed_hold)
                or len(set(review['reviewers'])) < 2 or review.get('independent_review') is not True):
            raise ValueError('Independent review is not established by adjudication alone')
        rows = reviewed_membership_rows(case, ref, application=application)
        pairs = [dict(slot=r['slot'], record=r['record'], input_sha256=_digest(r['pair'])) for r in rows]
        if review.get('pairs') != pairs:
            raise ValueError('Review does not pin the exact supplied format-4 windows')
        for row in rows:
            digest = _digest(row['pair'])
            if digest in payloads:
                previous = payloads[digest]
                if previous['label'] != row['label']:
                    raise ValueError('Identical model input has conflicting labels')
            else:
                previous = None
            item = dict(id=f"{case['id']}/{row['slot']}/{row['record']}", case=case['id'],
                        component=component['id'], split=split, language=case['language'], application=application,
                        input=row['pair'], input_sha256=digest, label=row['label'],
                        constraint=row['constraint'], semantic_training=row['semantic_training'],
                        membership_loss=review['membership_loss'],
                        duplicate_of=previous['id'] if previous else None)
            complete.append(item)
            if (row['semantic_training'] and review['membership_loss']
                    and (previous is None or not previous['eligible'])):
                semantic.append(item)
                payloads[digest] = dict(id=item['id'], split=split, label=row['label'], eligible=True)
            elif previous is None:
                payloads[digest] = dict(id=item['id'], split=split, label=row['label'], eligible=False)
    return semantic, complete


def review_projection(case):
    """Unlabeled format-4 task identity; labels cannot select the context."""
    rows = membership_inputs(case, application=case['application'], format_version=4)
    return [dict(slot=r['slot'], record=r['record'], input_sha256=_digest(r['pair']),
                 context_record_ids=list(r['context_record_ids']), input=r['pair']) for r in rows]


INTERPRETATION_EXPORT_SCHEMA = 'source-interpretation-supervision-v1'


def interpretation_review_projection(case):
    """Source-only tasks through the same formatter, prompt and unit registry.

    An existing explicit focus_id selects one occurrence; otherwise every
    occurrence requires review. Descriptions never enter these model messages.
    Source occurrence/protocol pins are review metadata, not additional input.
    """
    from psychology_base.applications.measurement.local_query_binding import reference_messages
    from psychology_base.applications.measurement import reference_readings as protocol

    rows = membership_inputs(case, application=case['application'], format_version=4)
    if 'focus_id' in case and (type(case['focus_id']) is not int
            or case['focus_id'] not in {r['record'] for r in rows}):
        raise ValueError('Interpretation focus must name an actual source occurrence')
    records = {r['id']: r for r in case['records']}
    tasks = []
    for row in rows:
        if row['slot'] != 0 or 'focus_id' in case and row['record'] != case['focus_id']:
            continue
        focus = json.loads(row['pair'][0])
        messages = [dict(role=m.role, content=m.content) for m in reference_messages(focus, interpretation=True)]
        contract = dict(schema=INTERPRETATION_EXPORT_SCHEMA, interpretation_schema=protocol.INTERPRETATION_SCHEMA,
            validation_revision=protocol.INTERPRETATION_VALIDATION_REVISION, prompt=messages[0]['content'],
            max_readings=protocol.MAX_READINGS, max_bindings=protocol.MAX_REFERENCES,
            max_events=protocol.MAX_EVENTS, max_spans=protocol.MAX_MENTIONS, max_targets=protocol.MAX_ENTITIES,
            max_prior_turns=protocol.MAX_PRIOR_TURNS)
        prior = list(row['context_record_ids'])
        occurrence = dict(case=case['id'], source_entry=case.get('source_entry'),
            records=[dict(id=rid, source_order=records[rid].get('source_order')) for rid in prior+[row['record']]])
        tasks.append(dict(record=row['record'], context_record_ids=prior,
            input_sha256=_digest(messages), source_identity=json.loads(messages[1]['content'])['source_identity'],
            source_occurrence_sha256=_digest(occurrence), protocol_identity=_digest(contract), messages=messages))
    return tasks


def _interpretation_target_key(analysis):
    """Ignore list order only; never infer aliases or add missing readings."""
    readings = [dict(bindings=sorted(r['bindings'], key=_digest), events=sorted(r['events'], key=_digest))
                for r in analysis['readings']]
    return _digest(dict(coverage=analysis['coverage'], readings=sorted(readings, key=_digest)))


def prepare_interpretation_bank(cases, references, annotations, ledger):
    """Validate full source-only supervision, including masked/pending tasks.

    references and annotations are per (case, record), independent of primary
    membership labels. Review claims are metadata checks, not authentication or
    proof of semantic completeness. Only explicit reviewed/admitted targets can
    contribute loss. Qualification targets are kept separate from SFT exports.
    """
    from psychology_base.applications.measurement.reference_readings import decode_interpretation_selections

    components = validate_membership_partitions(cases, ledger)
    projected = {c['id']: interpretation_review_projection(c) for c in cases}
    tasks = {(key, t['record']): t for key, values in projected.items() for t in values}
    def indexed(values):
        if type(values) is not list or any(type(v) is not dict or type(v.get('case')) is not str
                or type(v.get('record')) is not int for v in values):
            raise ValueError('Interpretation references/reviews must identify source occurrences')
        result = {(v['case'], v['record']): v for v in values}
        if len(result) != len(values) or set(result) != set(tasks):
            raise ValueError('Every interpretation occurrence needs exactly one reference and review')
        return result
    by_ref, by_review = indexed(references), indexed(annotations)
    payloads, eligible, complete = {}, [], []
    for case in cases:
        focus_rows = membership_inputs(case, application=case['application'], format_version=4)
        focuses = {r['record']: json.loads(r['pair'][0]) for r in focus_rows if r['slot'] == 0}
        for task in projected[case['id']]:
            key = (case['id'], task['record'])
            ref, review = by_ref[key], by_review[key]
            loss, state = review.get('reference_loss'), review.get('state')
            reviewers, independent = review.get('reviewers'), review.get('independent_review')
            admission = review.get('source_admission')
            if (type(loss) is not bool or type(state) is not str or state not in REVIEW_STATES
                    or type(reviewers) is not list or any(type(r) is not str or not r.strip() for r in reviewers)
                    or len(set(reviewers)) != len(reviewers) or type(independent) is not bool
                    or type(admission) is not dict or type(admission.get('admitted')) is not bool
                    or any(type(admission.get(k)) is not str or not admission[k].strip()
                           for k in ('source_basis', 'attribution_status'))):
                raise ValueError('Explicit interpretation review, loss mask and source admission required')
            for field in ('input_sha256', 'source_occurrence_sha256', 'protocol_identity'):
                if review.get(field) != task[field]:
                    raise ValueError('Interpretation review has stale source, input or protocol pins')
            analysis = ref.get('analysis')
            if 'analysis' not in ref or review.get('analysis_sha256') != _digest(analysis):
                raise ValueError('Interpretation review does not pin the exact target')
            if analysis is not None:
                # Reconstruct exactly the formatter-owned focus, not annotated context.
                analysis = decode_interpretation_selections(
                    json.dumps(analysis, ensure_ascii=False), focuses[task['record']])
            if independent or state == 'independently_reviewed' or loss and case['split'] == 'qualification':
                if not independent or len(reviewers) < 2:
                    raise ValueError('Independent interpretation review requires distinct reviewer identities')
            if loss and (state not in ('implementer_reviewed', 'independently_reviewed', 'adjudicated')
                    or analysis is None or type(ref.get('reviewer')) is not str or ref['reviewer'] not in reviewers
                    or not admission['admitted'] or case['split'] == 'reserved'
                    or case['split'] == 'qualification' and state not in ('independently_reviewed', 'adjudicated')
                    or any(type(review.get(k)) is not str or not review[k].strip()
                           for k in ('rationale', 'coverage_basis'))):
                raise ValueError('Unreviewed, unadmitted or reserved interpretation cannot contribute loss')
            digest = task['input_sha256']
            previous = payloads.get(digest)
            target_key = _interpretation_target_key(analysis) if analysis is not None else None
            if previous is not None:
                if previous['split'] != case['split']:
                    raise ValueError('Identical source-only input crosses partitions')
                if target_key is not None and previous['target'] is not None and target_key != previous['target']:
                    raise ValueError('Identical source-only input has conflicting interpretation targets')
            else:
                previous = dict(split=case['split'], target=None, first_id=None, eligible_id=None)
                payloads[digest] = previous
            item = dict(task, id=f"{case['id']}/{task['record']}", case=case['id'],
                schema=INTERPRETATION_EXPORT_SCHEMA, component=components[case['id']]['id'],
                split=case['split'], application=case['application'], language=case['language'],
                analysis=analysis, review=deepcopy(review), reference_loss=loss,
                duplicate_of=previous['eligible_id'] or previous['first_id'])
            if target_key is not None:
                previous['target'] = target_key
            if loss and previous['eligible_id'] is None:
                eligible.append(deepcopy(item))
                previous['eligible_id'] = item['id']
            if previous['first_id'] is None:
                previous['first_id'] = item['id']
            complete.append(item)
    return eligible, complete


def interpretation_sft_records(cases, references, annotations, ledger, *, split):
    """Trainer-facing messages only; validate the whole bank before filtering.

    Qualification/reserved answers and regression controls never become SFT
    exports. Only train/tuning may be requested, with no partition default.
    """
    if split not in ('train', 'tuning'):
        raise ValueError('Interpretation SFT exports require explicit train or tuning partition')
    eligible, _ = prepare_interpretation_bank(cases, references, annotations, ledger)
    return [dict(messages=deepcopy(row['messages'])+[dict(role='assistant', content=json.dumps(
        {k: row['analysis'][k] for k in ('coverage', 'readings')}, ensure_ascii=False, separators=(',', ':')))])
        for row in eligible if row['split'] == split]


def compile_membership_review(cases, manual, source_metadata):
    """Compile pinned first-pass decisions through the existing reference owner.

    Every semantic cell supplies an explicit label, canonical reason and rationale.
    Source admission is separate evidence supplied by the offline producer; this
    helper neither infers provenance nor claims an independent review. Consumed
    historical producers retain their archived implementations for replay.
    """
    if type(manual) is not dict or type(manual.get('cases')) is not dict or type(source_metadata) is not dict:
        raise ValueError('Manual review and source admission must be objects')
    ids = {c['id'] for c in cases}
    reviewer = manual.get('reviewer')
    if (len(ids) != len(cases) or set(manual.get('cases', {})) != ids
            or set(source_metadata) != ids or not isinstance(reviewer, str) or not reviewer.strip()):
        raise ValueError('Every unique case requires complete manual review and source admission')
    labels = {'M': 'match', 'N': 'nonmatch', 'U': 'uncertain'}
    references, annotations = [], []
    for case in cases:
        record, source = manual['cases'][case['id']], source_metadata[case['id']]
        if type(record) is not dict or type(source) is not dict:
            raise ValueError('Each manual case and source admission must be objects')
        state, loss = record.get('state'), record.get('membership_loss')
        if (state not in ('implementer_reviewed', 'unresolved', 'invalid') or type(loss) is not bool
                or type(source.get('admitted')) is not bool
                or any(not isinstance(source.get(k), str) or not source[k].strip()
                       for k in ('attribution_status', 'source_basis'))):
            raise ValueError('Explicit first-pass state, loss mask and source admission required')
        rows = membership_inputs(case, application=case['application'], format_version=4)
        pairs = [dict(slot=r['slot'], record=r['record'], input_sha256=_digest(r['pair'])) for r in rows]
        if _digest(record.get('pairs')) != _digest(pairs):
            raise ValueError('Manual review does not pin the exact supplied format-4 windows')
        vectors, notes = record.get('vectors'), record.get('notes')
        records = case['records']
        if (type(vectors) is not list or type(notes) is not list or len(vectors) != 2 or len(notes) != 2
                or any(not isinstance(v, str) or len(v) != len(records) or set(v) - set(labels) for v in vectors)
                or any(type(n) is not list or len(n) != len(records) for n in notes)):
            raise ValueError('Every occurrence needs an explicit manual cell and rationale')
        indices = {r['id']: i for i, r in enumerate(records)}
        texts = {r['id']: r['text'] for r in records}
        citations = record.get('context_citations', {})
        valid_keys = {f"{r['slot']}/{r['record']}" for r in rows}
        if type(citations) is not dict or set(citations) - valid_keys:
            raise ValueError('Review cites an unknown membership cell')
        judgments = []
        for row in rows:
            slot, rid = row['slot'], row['record']
            label, note = labels[vectors[slot][indices[rid]]], notes[slot][indices[rid]]
            if row['constraint'] is not None:
                if note != '-' or label != row['constraint']:
                    raise ValueError('Structural review must retain the code-owned actor decision')
                reason = 'wrong_outer_actor' if label == 'nonmatch' else 'unknown_outer_actor'
                rationale = ('The known focus speaker differs from the requested outer actor.' if label == 'nonmatch'
                             else 'The requested outer actor is not established.')
            else:
                if not isinstance(note, str) or '|' not in note:
                    raise ValueError('Every semantic decision needs an explicit reason and rationale')
                reason, rationale = note.split('|', 1)
                if (reason not in MEMBERSHIP_REASONS[label] or not rationale.strip()
                        or reason in ('wrong_outer_actor', 'unknown_outer_actor')):
                    raise ValueError('Invalid semantic reason or empty rationale')
            support = citations.get(f'{slot}/{rid}', [])
            if (type(support) is not list or any(type(i) is not int or i not in row['context_record_ids'] for i in support)
                    or len(set(support)) != len(support)):
                raise ValueError('Review cites context outside the supplied window or repeats support')
            judgments.append(dict(slot=slot, record=rid, label=label, reason=reason, rationale=rationale,
                evidence=[dict(start=0, end=len(texts[rid]), quote=texts[rid])],
                context_evidence=[dict(record=i, start=0, end=len(texts[i]), quote=texts[i]) for i in support]))
        reference = dict(id=case['id'], contract_version=4, reviewed=True, reviewer=reviewer, judgments=judgments)
        reviewed_membership_rows(case, reference, application=case['application'])
        references.append(reference)
        annotations.append(dict(id=case['id'], state=state, reviewers=[reviewer],
            membership_loss=loss and state == 'implementer_reviewed' and source['admitted'],
            reference_loss=False, independent_review=False, pairs=pairs,
            attribution_status=source['attribution_status'], source_basis=source['source_basis'],
            dispute=record.get('dispute'), historical_labels_reused=False))
    return references, annotations
