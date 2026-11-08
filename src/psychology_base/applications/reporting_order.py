"""Opt-in, source-bound reporting-order answers through one local pipeline.

This is a bounded grammar, not a semantic intent classifier or general memory
verifier. Unrecognized requests retain the application's injected responder.
"""
from dataclasses import dataclass
import hashlib
import json
import re
from threading import Lock

from psychology_base.applications.measurement.memory_grounding import decode_runtime_memory_context
from psychology_base.companion.turn_provenance import source_conflicts, source_relations


def _quotation(name):
    return rf'(?:“(?P<{name}_curly>[^“”"\r\n]+)”|"(?P<{name}_straight>[^“”"\r\n]+)")'


_FIRST, _SECOND = _quotation('first'), _quotation('second')
_ENGLISH = re.compile(
    rf'Which (?:statement )?was (?:said|reported) (?P<polarity>earlier|later)'
    rf'(?: in the source dialogue)?: {_FIRST} or {_SECOND}\?', re.IGNORECASE | re.ASCII)
_CHINESE = re.compile(rf'原对话里，{_FIRST}和{_SECOND}，哪一句说得(?P<polarity>更早|更晚)？')
_PHRASE_EN = re.compile(
    r'Which phrase was (?:mentioned|reported) (?P<polarity>earlier|later): '
    r'(?P<first>[^“”"?\r\n]+?) or (?P<second>[^“”"?\r\n]+)\?', re.IGNORECASE | re.ASCII)
_PHRASE_ZH = re.compile(
    r'原对话中，(?P<first>[^“”"，？\r\n]+)和(?P<second>[^“”"，？\r\n]+)，'
    r'哪段话被(?P<polarity>更早|更晚)提到？')
_DESCRIPTION_EN = re.compile(
    r'Which came (?P<polarity>earlier|later) in the conversation: '
    r'(?P<first>[^“”"?\r\n]+?) or (?P<second>[^“”"?\r\n]+)\?', re.IGNORECASE | re.ASCII)
_DESCRIPTION_ZH = re.compile(
    r'原对话中，(?P<first>[^“”"，？\r\n]+)和(?P<second>[^“”"，？\r\n]+)，'
    r'哪一句出现得(?P<polarity>较早|较晚|更早|更晚)？')
_NATURAL_EN = re.compile(
    r'Which (?P<selector>[a-z]+(?: [a-z]+){0,2}) did (?P<actor>I|you) '
    r'(?P<verb>name|mention|suggest) (?P<polarity>earlier|later)\?', re.IGNORECASE | re.ASCII)
_NATURAL_ZH = re.compile(
    r'(?P<actor>我|你)(?P<polarity>较早|较晚|更早|更晚)(?P<verb>提到|说到|建议)的是'
    r'哪(?:家|个|位|本|种|条)(?P<selector>[\u3400-\u9fffA-Za-z]{1,24})？')
_UNCERTAINTY = (
    ' If the supplied memory does not establish the order, say so.',
    ' If the order is unclear, say so.',
    '如果给出的记忆不能确定先后顺序，请说明。',
    '如果顺序不清楚，请说明。',
)


@dataclass(frozen=True)
class ReportingOrderIntent:
    language: str
    polarity: str
    selector: str | None = None
    requested_actor: str | None = None
    verb: str = 'mention'
    quotations: tuple[str, ...] = ()
    phrases: tuple[str, ...] = ()
    descriptions: tuple[str, ...] = ()
    operation: str = 'reporting_order'
    scope: str = 'delivered_occurrences'


@dataclass(frozen=True)
class ReportingTarget:
    record_id: int
    start: int
    end: int
    text: str
    speaker: str | None
    source_group: str | None
    source_position: int | None


@dataclass(frozen=True)
class ReportingOrderBinding:
    request_identity: str
    intent: ReportingOrderIntent
    alternatives: tuple[tuple[ReportingTarget, ...], ...]
    resolution: 'ReportingOrderResolution'
    selected: ReportingTarget | None = None
    unbound_record_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class ReportingOrderResolution:
    language: str
    polarity: str
    status: str
    reason: str
    record_ids: tuple[int, ...] = ()
    query_target: int | None = None


def _intent(application, query):
    # Remove only a complete, explicitly supported uncertainty instruction.
    # An independent second request must never be silently discarded.
    text = query.strip()
    for suffix in _UNCERTAINTY:
        if text.endswith(suffix):
            text = text[:-len(suffix)]
            break
    for language, grammar in (('en', _ENGLISH), ('zh', _CHINESE), ('en', _PHRASE_EN), ('zh', _PHRASE_ZH),
                              ('en', _DESCRIPTION_EN), ('zh', _DESCRIPTION_ZH),
                              ('en', _NATURAL_EN), ('zh', _NATURAL_ZH)):
        match = grammar.fullmatch(text)
        if match is None:
            continue
        polarity = {'较早': 'earlier', '更早': 'earlier', '较晚': 'later', '更晚': 'later'}.get(
            match['polarity'], match['polarity'].lower())
        if polarity not in ('earlier', 'later'):
            return None
        if grammar in (_ENGLISH, _CHINESE):
            quotes = tuple(match[f'{name}_curly'] or match[f'{name}_straight'] for name in ('first', 'second'))
            return ReportingOrderIntent(language, polarity, quotations=quotes)
        if grammar in (_PHRASE_EN, _PHRASE_ZH, _DESCRIPTION_EN, _DESCRIPTION_ZH):
            phrases = (match['first'], match['second'])
            delimiter = ' or ' if language == 'en' else '和'
            if any(p != p.strip() or delimiter in p.lower() or not p.strip() for p in phrases):
                return None
            if grammar in (_DESCRIPTION_EN, _DESCRIPTION_ZH):
                return ReportingOrderIntent(language, polarity, descriptions=phrases)
            return ReportingOrderIntent(language, polarity, phrases=phrases)
        selector = match['selector']
        # These qualifiers change the operation or the scope beyond reporting
        # among available occurrences. They are not category words in this grammar.
        if set(selector.lower().split()) & {'first', 'last', 'ever', 'current', 'corrected', 'updated', 'calendar'}:
            return None
        actor = 'user' if match['actor'].lower() in ('i', '我') else application
        verb = 'suggest' if match['verb'].lower() in ('suggest', '建议') else 'mention'
        return ReportingOrderIntent(language, polarity, selector, actor, verb)
    return None


reporting_order_intent = _intent


def _target(record, start, end):
    coordinate = record.get('source_order') or {}
    return ReportingTarget(record['id'], start, end, record['text'][start:end], record['speaker'],
        coordinate.get('group'), coordinate.get('position'))


def _phrase_targets(records, phrase):
    # Literal phrases retain case and offsets. Latin word fragments cannot
    # silently bind to a longer word; Chinese phrase matching needs no tokenizer.
    left = r'(?<!\w)' if phrase[0].isascii() and phrase[0].isalnum() else ''
    right = r'(?!\w)' if phrase[-1].isascii() and phrase[-1].isalnum() else ''
    pattern = left + re.escape(phrase) + right
    return tuple(_target(record, *match.span()) for record in records
                 for match in re.finditer(pattern, record['text']))


def _literal_alternatives(intent, records):
    if intent.quotations:
        return tuple(tuple(_target(r, 0, len(r['text'])) for r in records if r['text'] == quote)
                     for quote in intent.quotations)
    return tuple(_phrase_targets(records, phrase) for phrase in intent.phrases)


def _actor_candidates(intent, records):
    return tuple(r for r in records if r['speaker'] in (None, intent.requested_actor))


def reporting_required_occurrences(application, query, fragments):
    """Preserve all bounded-grammar competitors in the eligible candidate pool.

    Canonical v4 uses the existing renderer/strict reader in linear space;
    unlike derived relation formats, it builds no all-pairs candidate graph.
    This is not a claim of complete history or nonliteral semantic coverage.
    """
    from psychology_base.applications.context import render_context_block
    intent = _intent(application, query)
    if intent is None:
        raise ValueError('Reporting requirements need a recognized operation')
    fragments = list(fragments)
    records = decode_runtime_memory_context(render_context_block(
        fragments, assistant_speaker=application, format_version=4))['records']
    if intent.descriptions:
        # No qualified semantics can certify another record irrelevant yet.
        required = set(range(len(records)))
    elif intent.quotations or intent.phrases:
        required = {t.record_id for group in _literal_alternatives(intent, records) for t in group}
    else:
        # Unparsed or unknown-actor records can contain competing declarations.
        required = {r['id'] for r in _actor_candidates(intent, records)}
    groups = {(records[i].get('source_order') or {}).get('group') for i in required}
    for group, witnesses in source_conflicts(r.get('source_order') for r in records).items():
        if group in groups:
            required.update(witnesses)
    return tuple(f.id for i, f in enumerate(fragments) if i in required)


def _selection_omission(trace):
    if type(trace) is not dict:
        raise ValueError('Selection evidence requires its actual trace')
    requirement = trace.get('required_occurrences')
    if requirement is None:
        return bool(trace.get('skips')), None
    if (type(requirement) is not dict
            or set(requirement) != {'ids','complete','reason','candidate_count','frontier_sha256'}
            or type(requirement['complete']) is not bool
            or type(requirement['candidate_count']) is not int or requirement['candidate_count'] < 0
            or type(requirement['ids']) is not list
            or any(type(i) is not str for i in requirement['ids'])
            or len(set(requirement['ids'])) != len(requirement['ids'])
            or len(requirement['ids']) > requirement['candidate_count']
            or type(requirement['frontier_sha256']) is not str
            or re.fullmatch('[0-9a-f]{64}', requirement['frontier_sha256']) is None
            or (requirement['reason'] is not None if requirement['complete'] else
                requirement['reason'] not in ('record_limit','byte_limit'))):
        raise ValueError('Malformed candidate-completeness receipt')
    return not requirement['complete'], requirement


def _named_target(record, intent):
    """Bind a flat, explicit type/name declaration, retaining raw offsets.

    This deliberately does not perform NER or classify paraphrases. It uses
    literal phrase boundaries, so no alternate Chinese segmentation is needed.
    Unparsed utterances remain possible competitors, never certified irrelevant.
    """
    category = re.escape(intent.selector)
    if intent.language == 'en':
        prefix = (r'(?:My|The) suggested ' if intent.verb == 'suggest' else
                  r'(?:My|The) (?:(?:preferred|favorite|chosen) )?')
        pattern = prefix + category + r' is (?P<label>[^.!?\r\n]+)[.]?'
        match = re.fullmatch(pattern, record['text'], re.IGNORECASE | re.ASCII)
    else:
        prefix = r'我建议的' if intent.verb == 'suggest' else r'(?:我(?:偏爱|喜欢|选择)的|这家|这个|这本|这种)'
        match = re.fullmatch(prefix + category + r'是(?P<label>[^。！？\r\n]+)。?', record['text'])
    if match is None:
        return None
    label = match['label']
    if intent.language == 'en':
        words = label.replace('-', ' ').replace("'", ' ').replace('’', ' ').split()
        if (not 1 <= len(words) <= 4 or any(not w.isalpha() for w in words)
                or any(not w[0].isupper() for w in words)
                or set(w.lower() for w in words) & {
                    'and', 'or', 'nor', 'versus', 'vs', 'not', 'no', 'none', 'neither',
                    'unknown', 'undecided', 'undetermined', 'unspecified', 'unconfirmed',
                    'tbd', 'maybe', 'perhaps', 'is', 'was'}):
            return None
    elif (not 1 <= len(label) <= 12 or not all(c.isalpha() for c in label)
            or label in ('未知', '未定', '待定', '暂无', '没有', '无', '不知道', '不详', '尚未确定')
            or any(cue in label for cue in ('不是', '不确定', '也许', '可能', '或', '和', '与', '及', '但', '是'))):
        return None
    return _target(record, *match.span('label'))


def _unique_extreme(targets, relations, polarity):
    side = 'later' if polarity == 'earlier' else 'earlier'
    winners = [t for t in targets if all(other.record_id == t.record_id or
        other.record_id in relations[t.record_id][side] for other in targets)]
    return winners[0] if len(winners) == 1 else None


def reporting_descriptor_constraints(description, application, *, predicate_only=False):
    """Separate existing outer-actor/initial constraints from a predicate.

    This is syntax only. It never declares semantic membership or chronology.
    Predicate-only inputs reject ordering cues whose outer scope is not clear;
    legacy callers keep their historical parsing unchanged.
    """
    actor, predicate = None, description
    for prefix, actor in (("the teacher's ",'tutor'),("the tutor's ",'tutor'),("the learner's ",'user'),
                          ("the student's ",'user'),('my ','user'),('your ',application),
                          ('老师的','tutor'),('学生的','user'),('我的','user'),('你的',application)):
        if description.lower().startswith(prefix):
            predicate = description[len(prefix):]
            break
    else:
        actor = None
    pattern = r'\b(?:first|initial|opening)\b'
    initial = bool(re.search(pattern, description, re.IGNORECASE | re.ASCII)
                   or any(cue in description for cue in ('开头', '首次', '最初')))
    if predicate_only and initial:
        outer = re.match(r'^(?:the\s+)?(?:first|initial)(?:\s+|$)|^(?:开头|首次|最初)',
                         predicate, re.IGNORECASE | re.ASCII)
        remainder = predicate[outer.end():] if outer is not None else ''
        if (outer is None or re.search(pattern, remainder, re.IGNORECASE | re.ASCII)
                or any(cue in remainder for cue in ('开头', '首次', '最初'))):
            raise ValueError('Predicate normalization requires one clearly scoped outer ordinal')
    predicate = re.sub(pattern, '', predicate, flags=re.IGNORECASE | re.ASCII)
    for cue in ('开头', '首次', '最初'):
        predicate = predicate.replace(cue, '')
    return actor, initial, predicate.strip()


def _descriptor_actor(description, application):
    return reporting_descriptor_constraints(description, application)[0]


def _semantic_targets(proposal, intent, records, application, relations):
    """Validate presence/roles and initial-occurrence structure, not semantics.

    A qualified proposer's semantic precision is a separate requirement. JSON
    validity and quoted spans never certify entailment or candidate completeness.
    """
    if (type(proposal) is not dict or set(proposal) != {'complete','targets'}
            or type(proposal['complete']) is not bool or not proposal['complete']
            or type(proposal['targets']) is not list or len(proposal['targets']) != 2):
        raise ValueError('Incomplete or malformed semantic proposal')
    alternatives, selected = [], []
    for description, group in zip(intent.descriptions, proposal['targets']):
        if type(group) is not list or not 1 <= len(group) <= len(records):
            raise ValueError('Missing or unbounded semantic target set')
        targets=[]
        actor=_descriptor_actor(description,application)
        for citation in group:
            if (type(citation) is not dict or set(citation) != {'record','quote'}
                    or type(citation['record']) is not int or not 0 <= citation['record'] < len(records)
                    or type(citation['quote']) is not str or not citation['quote'].strip()):
                raise ValueError('Invalid semantic citation')
            record=records[citation['record']]
            if actor is not None and record['speaker'] != actor:
                raise ValueError('Semantic citation crosses the requested outer actor')
            matches=_phrase_targets([record],citation['quote'])
            if len(matches) != 1 or any(t.record_id == record['id'] for t in targets):
                raise ValueError('Ambiguous or repeated semantic citation')
            targets.append(matches[0])
        alternatives.append(tuple(targets))
        initial = reporting_descriptor_constraints(description, application)[1]
        target = (_unique_extreme(targets,relations,'earlier') if initial else
                  targets[0] if len(targets) == 1 else None)
        if target is None: raise ValueError('Ambiguous semantic alternatives or initial occurrence')
        if initial:
            # A visible minimum need not be the source's opening occurrence.
            # Require its original prefix, rather than inventing missing turns.
            if target.source_position is None or target.source_position>=len(records):
                raise ValueError('Opening occurrence has an incomplete source prefix')
            prefix=[r for r in records if (r.get('source_order') or {}).get('group')==target.source_group
                    and (r.get('source_order') or {}).get('position',-1)<=target.source_position]
            positions={(r.get('source_order') or {}).get('position') for r in prefix}
            if (len(positions) != target.source_position+1
                    or actor is not None and any(r['speaker'] is None for r in prefix)):
                raise ValueError('Opening occurrence has an incomplete source prefix')
        selected.append(target)
    if selected[0].record_id == selected[1].record_id:
        raise ValueError('Two descriptions collapse to the same occurrence')
    return tuple(alternatives), tuple(selected)


@dataclass(frozen=True)
class _MembershipProof:
    reason: str
    possible: tuple = ()
    witnesses: tuple = ()
    winner: int | None = None
    slot: int | None = None
    required: tuple = ()


def _membership_invariance(application, intent, records, labels, dependencies=None):
    """Reduce all independent M/N/U completions to at most N squared pairs.

    This proves invariance conditional on supplied memberships, not their truth.
    Correlations between unknown labels are conservatively overapproximated.
    """
    size=len(records)
    if (not 0 < size <= 32 or any(type(r['id']) is not int or r['id']!=i for i,r in enumerate(records)) or len(labels)!=2
            or any(len(row)!=size or any(type(v) is not str or v not in ('match','nonmatch','uncertain') for v in row)
                   for row in labels)):
        raise ValueError('Membership must cover both descriptions and the complete bounded pool')
    if dependencies is None:dependencies=((),)*size
    if (len(dependencies)!=size or any(len(set(group))!=len(group)
            or any(type(i) is not int or not 0<=i<size for i in group) for group in dependencies)):
        raise ValueError('Invalid membership context dependencies')
    targets=[_target(r,0,len(r['text'])) for r in records]
    relations=source_relations(r.get('source_order') for r in records)
    possible=[];witnesses=[];initial=[]
    for slot,description in enumerate(intent.descriptions):
        actor,first,_=reporting_descriptor_constraints(description,application)
        initial.append(first)
        for i,record in enumerate(records):
            if actor is not None and record['speaker']!=actor:
                expected='uncertain' if record['speaker'] is None else 'nonmatch'
                if labels[slot][i]!=expected:raise ValueError('Membership contradicts an authoritative outer actor')
        known=[targets[i] for i in range(size) if labels[slot][i]=='match']
        if not known:return _MembershipProof('missing_guaranteed_target')
        witness=_unique_extreme(known,relations,'earlier')
        if witness is None:return _MembershipProof('unresolved_membership_source')
        witnesses.append(witness.record_id)
        candidates=[]
        for i in range(size):
            if labels[slot][i]=='nonmatch':continue
            if first and i!=witness.record_id:
                earlier=_unique_extreme((targets[i],witness),relations,'earlier')
                if earlier is None:return _MembershipProof('unresolved_initial_source')
                if earlier.record_id!=i:continue
            candidates.append(i)
        possible.append(tuple(candidates))
    signature=None
    for left in possible[0]:
        for right in possible[1]:
            winner=_unique_extreme((targets[left],targets[right]),relations,intent.polarity)
            if winner is None:return _MembershipProof('overlapping_or_incomparable_binding')
            value=(0 if winner.record_id==left else 1,winner.record_id)
            if signature is not None and signature!=value:
                return _MembershipProof('influential_membership_uncertainty')
            signature=value
    try:
        # Retain the canonical role, citation and initial-prefix policy.
        _semantic_targets(dict(complete=True,targets=[[dict(record=i,quote=records[i]['text'])]
            for i in witnesses]),intent,records,application,relations)
    except ValueError:
        return _MembershipProof('unsupported_initial_occurrence')
    slot,winner=signature
    if labels[slot][winner]!='match':return _MembershipProof('unsupported_winning_occurrence')
    required=set(witnesses)
    for index,witness in enumerate(witnesses):
        required.update(dependencies[witness])
        if initial[index]:
            coordinate=records[witness]['source_order']
            required.update(i for i,r in enumerate(records) if r.get('source_order') is not None
                and r['source_order']['group']==coordinate['group']
                and r['source_order']['position']<=coordinate['position'])
    return _MembershipProof('binding_invariant',tuple(possible),tuple(witnesses),winner,slot,tuple(sorted(required)))


def _occurrence_identity(record):
    return json.dumps({k:record.get(k) for k in ('speaker','session','text','source_order')},
        sort_keys=True,ensure_ascii=False,separators=(',',':'))


class ReportingMembershipSession:
    """One immutable local assessment shared by admission and final binding."""

    def __init__(self,application,query,assessor,*,user_id=None):
        intent=_intent(application,query)
        if (application not in ('tutor','companion') or intent is None or not intent.descriptions
                or not callable(assessor) or type(getattr(assessor,'identity',None)) is not str
                or not assessor.identity or type(getattr(assessor,'format_version',None)) is not int
                or assessor.format_version not in (2,3,4)):
            raise ValueError('A description query requires an identified membership assessor and input contract')
        self.application=application;self.query=query;self.intent=intent
        self.assessor=assessor;self._assessor_identity=assessor.identity;self._version=assessor.format_version
        self.user_id=user_id
        self.identity=hashlib.sha256(json.dumps([application,query,assessor.identity,self._version],
            ensure_ascii=False).encode()).hexdigest()
        self._request=(application,query,self.identity)
        self.assessment_identity=None
        self._lock=Lock();self._attempted=False;self._proof=None
        self._snapshot=None;self._ids=();self._records_json=None

    def _check_assessor(self):
        if ((self.application,self.query,self.identity)!=self._request
                or self.assessor.identity!=self._assessor_identity or self.assessor.format_version!=self._version):
            raise ValueError('Membership assessor identity changed during the request')
        if getattr(self.assessor,'reuse_cache',None) is not None:
            self.assessor.check_reuse_scope(self.application,self.user_id)

    def required_occurrences(self,fragments):
        from psychology_base.applications.context import render_context_block
        from psychology_base.applications.measurement.semantic_membership import membership_inputs
        from psychology_base.llm.execution import ExecutionDeferred

        fragments=list(fragments)
        if len(fragments)>32:
            raise ExecutionDeferred('Membership requires the complete eligible pool of at most 32 records')
        if any(type(f.id) is not str for f in fragments) or len({f.id for f in fragments})!=len(fragments):
            raise ValueError('Membership requires unique stable occurrence identities')
        ordered=sorted(fragments,key=lambda f:f.id)
        ids=tuple(f.id for f in ordered)
        records=decode_runtime_memory_context(render_context_block(ordered,
            assistant_speaker=self.application,format_version=4))['records']
        snapshot=json.dumps(dict(ids=ids,records=records),sort_keys=True,ensure_ascii=False,separators=(',',':'))
        with self._lock:
            self._check_assessor()
            if self._snapshot is not None and snapshot!=self._snapshot:
                raise ValueError('The eligible membership pool changed during the request')
            if self._proof is not None:
                return tuple(self._ids[i] for i in self._proof.required) if self._proof.winner is not None else self._ids
            if self._attempted:raise ExecutionDeferred('A failed membership assessment is not retried implicitly')
            self._snapshot=snapshot;self._ids=ids
            self._records_json=json.dumps(records,ensure_ascii=False,separators=(',',':'))
            if not records:
                self._proof=_MembershipProof('missing_guaranteed_target')
                self.assessment_identity=hashlib.sha256((self.identity+'\n'+snapshot).encode()).hexdigest()
                return ()
            rows=membership_inputs(dict(records=records,descriptions=self.intent.descriptions,
                language=self.intent.language),application=self.application,format_version=self._version)
            if getattr(self.assessor,'reuse_cache',None) is not None:
                by_id={record['id']:record for record in records}
                def occurrence(index):
                    return dict(occurrence_id=ids[index],record=json.loads(_occurrence_identity(by_id[index])))
                for row in rows:
                    row['reuse_provenance']=dict(focus=occurrence(row['record']),
                        prior=[occurrence(index) for index in row['context_record_ids']])
            row_identity=json.dumps(rows,sort_keys=True,ensure_ascii=False)
            self._attempted=True
            labels=self.assessor(rows)
            self._check_assessor()
            if json.dumps(rows,sort_keys=True,ensure_ascii=False)!=row_identity:
                raise ValueError('Membership inputs were mutated during assessment')
            if type(labels) not in (tuple,list) or len(labels)!=len(rows):
                raise ValueError('Membership assessor returned incomplete decisions')
            matrix=tuple(tuple(labels[s*len(records):(s+1)*len(records)]) for s in (0,1))
            dependencies=tuple(r.get('context_record_ids',()) for r in rows[:len(records)])
            proof=_membership_invariance(self.application,self.intent,records,matrix,dependencies)
            self.assessment_identity=hashlib.sha256((self.identity+'\n'+snapshot+'\n'+
                json.dumps(matrix)+'\n'+json.dumps(dependencies)).encode()).hexdigest()
            self._proof=proof
            return tuple(ids[i] for i in proof.required) if proof.winner is not None else ids

    def _binding_targets(self,application,query,records,selection_trace):
        self._check_assessor()
        if application!=self.application or query!=self.query or self._proof is None:
            raise ValueError('Membership session does not belong to this prepared request')
        full=json.loads(self._records_json)
        by_key={_occurrence_identity(r):i for i,r in enumerate(full)}
        if len(by_key)!=len(full):raise ValueError('Ambiguous occurrence identity in membership snapshot')
        delivery={}
        for record in records:
            index=by_key.get(_occurrence_identity(record))
            if index is None or index in delivery:raise ValueError('Delivered membership evidence changed')
            delivery[index]=record
        required=self._proof.required if self._proof.winner is not None else tuple(range(len(full)))
        if selection_trace is None:
            if len(delivery)!=len(full):raise ValueError('Reduced delivery requires a complete selection receipt')
        else:
            omitted,receipt=_selection_omission(selection_trace)
            expected_frontier=hashlib.sha256(json.dumps(sorted(self._ids),separators=(',',':')).encode()).hexdigest()
            if (receipt is None or omitted or receipt['candidate_count']!=len(full)
                    or receipt['frontier_sha256']!=expected_frontier
                    or set(receipt['ids'])!={self._ids[i] for i in required}):
                raise ValueError('Membership selection receipt is stale or incomplete')
        if not set(required)<=set(delivery):raise ValueError('Required membership dependencies were omitted')
        proof=self._proof
        if proof.winner is None:return proof.reason,(),(),None
        alternatives=tuple(tuple(_target(delivery[i],0,len(delivery[i]['text'])) for i in group if i in delivery)
            for group in proof.possible)
        targets=tuple(_target(delivery[i],0,len(delivery[i]['text'])) for i in proof.witnesses)
        selected=_target(delivery[proof.winner],0,len(delivery[proof.winner]['text']))
        return proof.reason,alternatives,targets,selected


def bind_reporting_order(application: str, query: str, context: str, *, proposer=None,
                         evidence_pruned=False, selection_trace=None, membership_session=None) -> ReportingOrderBinding | None:
    """Bind query-only references to this exact delivery, then compare once.

    None delegates an unsupported operation. Unresolved recognized queries
    preserve missing/ambiguous evidence. No candidate reply or cached binding
    is used to manufacture request evidence.
    """
    if application not in ('companion', 'tutor') or type(query) is not str or type(context) is not str:
        raise ValueError('Reporting order requires application and text evidence')
    if type(evidence_pruned) is not bool:
        raise ValueError('Evidence omission must be explicitly Boolean')
    intent = _intent(application, query)
    if intent is None:
        return None
    if intent.descriptions and membership_session is not None and (
            not isinstance(membership_session,ReportingMembershipSession) or proposer is not None):
        raise ValueError('Use one prepared membership session or the legacy proposer')
    if intent.descriptions and proposer is not None and (
            not callable(proposer) or type(getattr(proposer,'identity',None)) is not str or not proposer.identity):
        raise ValueError('A semantic proposer requires a stable identity')
    requirement = None
    if selection_trace is not None:
        omitted, requirement = _selection_omission(selection_trace)
        evidence_pruned = evidence_pruned or omitted
    identity_fields = dict(schema='reporting-query-v2', application=application,
        evidence_pruned=evidence_pruned,
        proposer=getattr(proposer,'identity',None) if intent.descriptions else None,
        query=query, delivered_context=context)
    if requirement is not None:
        identity_fields.update(schema='reporting-query-v3', selection_requirements=requirement)
    if intent.descriptions and membership_session is not None:
        identity_fields.update(schema='reporting-query-v4',membership_assessment=membership_session.assessment_identity)
    identity = hashlib.sha256(json.dumps(identity_fields, sort_keys=True, ensure_ascii=False,
        separators=(',', ':')).encode('utf-8')).hexdigest()
    alternatives, unbound = (), ()

    def result(status, reason, targets=(), selected=None):
        ids = tuple(t.record_id for t in targets)
        slot = targets.index(selected) + 1 if selected is not None else None
        resolution = ReportingOrderResolution(intent.language, intent.polarity, status, reason, ids, slot)
        return ReportingOrderBinding(identity, intent, alternatives, resolution, selected, unbound)
    if evidence_pruned:
        return result('unresolved','selection_omitted_evidence')
    try:
        records = decode_runtime_memory_context(context)['records']
    except (ValueError, TypeError):
        return result('unresolved', 'invalid_evidence')
    relations = source_relations(record.get('source_order') for record in records)
    if intent.descriptions:
        if membership_session is not None:
            try:
                reason,alternatives,targets,selected=membership_session._binding_targets(
                    application,query,records,selection_trace)
            except (ValueError,TypeError):
                return result('unresolved','invalid_membership_assessment')
            return result('resolved' if selected is not None else 'unresolved',reason,targets,selected)
        if proposer is None:
            return result('unresolved','semantic_binding_unavailable')
        for description in intent.descriptions:
            actor=_descriptor_actor(description,application)
            if actor is not None and not any(r['speaker']==actor for r in records):
                return result('unresolved','missing_actor_evidence')
            if reporting_descriptor_constraints(description, application)[1]:
                if not any((r.get('source_order') or {}).get('position')==0 for r in records):
                    return result('unresolved','missing_source_prefix')
        try:
            proposal=proposer(application,query,intent.descriptions,records)
            alternatives, targets=_semantic_targets(proposal,intent,records,application,relations)
        except (ValueError,TypeError):
            return result('unresolved','invalid_semantic_binding')
    elif intent.quotations or intent.phrases:
        alternatives = _literal_alternatives(intent, records)
        if any(not group for group in alternatives):
            return result('unresolved', 'missing_target')
        if (any(len(group) != 1 for group in alternatives)
                or alternatives[0][0].record_id == alternatives[1][0].record_id):
            return result('unresolved', 'ambiguous_target')
        targets = tuple(group[0] for group in alternatives)
    else:
        targets, uncertain = [], []
        for record in _actor_candidates(intent, records):
            target = _named_target(record, intent)
            if record['speaker'] is None or target is None:
                uncertain.append(record['id'])
            else:
                targets.append(target)
        targets = tuple(targets)
        alternatives = (targets,)
        # A free-form/unknown-actor utterance could contain a competing mention
        # with no literal type overlap. Do not assume lexical coverage is complete.
        if uncertain:
            unbound = tuple(uncertain)
            return result('unresolved', 'incomplete_binding', targets)
        if len(targets) < 2:
            return result('unresolved', 'missing_target', targets)
    winner = _unique_extreme(targets,relations,intent.polarity)
    if winner is None:
        return result('unresolved', 'unresolved_order', targets)
    return result('resolved', 'source_positions', targets, winner)


def resolve_reporting_order(application: str, query: str, context: str) -> ReportingOrderResolution | None:
    """Compatibility view: preserve the original exact-quotation result shape."""
    binding = bind_reporting_order(application, query, context)
    return binding.resolution if binding is not None else None


def respond_with_reporting_order(responder, system_prompt, context, query, *, application, enabled=False, proposer=None,
                                 evidence_pruned=False, selection_trace=None,membership_session=None):
    """Use local structure only when explicitly enabled, inside application gates.

    Exact quotes retain slot references; natural queries use grounded labels.
    Tutors use occurrence anchors without echoing possible final solutions.
    No model assessor, retry, alternative recall or memory mutation runs here.
    """
    if type(enabled) is not bool:
        raise ValueError('local_reporting_order must be Boolean')
    binding = bind_reporting_order(application, query, context, proposer=proposer,
        evidence_pruned=evidence_pruned, selection_trace=selection_trace,membership_session=membership_session) if enabled else None
    if binding is None:
        return responder(system_prompt, context, query)
    resolution = binding.resolution
    if resolution.reason=='selection_omitted_evidence':
        return ('现有记忆不完整，不足以可靠比较这些提及的先后。' if resolution.language=='zh' else
                'The available memory is incomplete for a reliable comparison of these mentions.')
    if not binding.intent.quotations:
        if resolution.status == 'unresolved':
            return ('现有记忆不足以确定这些提及的先后；我无法据此判断哪一个更早或更晚。'
                    if resolution.language == 'zh' else
                    'The available memory does not establish the order of these mentions, so I cannot identify an earlier or later one.')
        # Tutors identify the grounded mention without echoing possible solutions.
        # Source position is a record anchor, not a date or global conversation ID.
        position = binding.selected.source_position
        label = (f'“{binding.selected.text}”' if application == 'companion' else
                 f'原对话中位置为 {position} 的提及' if resolution.language == 'zh' else
                 f'the mention at source position {position}')
        if resolution.language == 'zh':
            direction = '较早' if resolution.polarity == 'earlier' else '较晚'
            return f'在这里可见的记录中，{label}被{direction}提到。'
        return f'In the records available here, {label} was mentioned {resolution.polarity}.'
    if resolution.status == 'unresolved':
        return ('现有记忆不足以确定这两段引文的先后。你指的是哪一次对话中的具体表述？'
                if resolution.language == 'zh' else
                'The available memory does not establish the order of those two quotations. '
                'Which source conversation or specific occurrences do you mean?')
    if resolution.language == 'zh':
        target = '第一' if resolution.query_target == 1 else '第二'
        order = '更早' if resolution.polarity == 'earlier' else '更晚'
        return f'按这里可见的原对话记录，{target}段引文{order}出现。'
    target = 'first' if resolution.query_target == 1 else 'second'
    return f'In the records available here, the {target} quoted statement was reported {resolution.polarity}.'
