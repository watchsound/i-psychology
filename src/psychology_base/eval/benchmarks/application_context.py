"""Frozen application probes through normal factories, responders and memory.

Source-derived history and authored queries are not fresh human observations.
Software controls and live model outcomes have separate provenance.
"""
import ast
import copy
from dataclasses import asdict, replace
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import itertools
from collections import Counter
import json
from pathlib import Path
import platform
import re
import random
import time
import math
import importlib.metadata
import hashlib
import unicodedata

import numpy as np

from psychology_base.applications.context import CONTEXT_RECORD_RULE, SPEAKER_OWNERSHIP_RULE
from psychology_base.applications.emotional.companion_cli import build_companion_agent
from psychology_base.applications.emotional.companion_agent import _render_context_block as render_companion
from psychology_base.applications.learning.tutor_cli import build_tutor_agent
from psychology_base.applications.learning.tutor_agent import _render_context_block as render_tutor
from psychology_base.applications.llm_responders import make_llm_responder
from psychology_base.applications.measurement.empath_judge import (
    LLMEMPATHJudge, EMPATHSubCriterion, _find_balanced_json_object, _unique_json_object,
    validate_contract_version, empath_consistency_errors, empath_prompt, parse_empath_verdict,
)
# Preserve historical benchmark imports as aliases to one shared owner.
from psychology_base.applications.measurement.memory_grounding import (
    MEMORY_FIELDS,
    MEMORY_PROMPT,
    MEMORY_CLARIFICATION_V3,
    MEMORY_CURRENT_QUERY_CLARIFICATION_V1,
    _MEMORY_SPEAKER_RULE_V1,
    MEMORY_SPEAKER_PERSPECTIVE_CLARIFICATION_V1,
    MEMORY_SPEAKER_PERSPECTIVE_CLARIFICATION_V2,
    MEMORY_SPEAKER_PERSPECTIVE_CLARIFICATION_V3,
    MEMORY_SPEAKER_PERSPECTIVE_CLARIFICATION_V4,
    grading_version,
    revised_grading_protocol,
    memory_consistency_errors,
    parse_memory,
    parse_runtime_memory,
    runtime_memory_payload,
    runtime_memory_protocol,
    runtime_memory_resolved,
    runtime_memory_action,
    runtime_interpretation_payload,
    parse_runtime_interpretation,
    runtime_binding_claims,
)
from psychology_base.applications.pilot.layout import UserManifest, PilotApp, build_aging_policy
from psychology_base.applications.pilot.session_runner import SessionLogWriter
from psychology_base.applications.safety.minor_exclusion import AgeAttestation
from psychology_base.companion.pam_export import pam_export, pam_import, validate_content_addressing
from psychology_base.companion.turn_provenance import sequence_tag, parse_turn, local_neighbors
from psychology_base.eval.adapters import Observation, make_fragment
from psychology_base.eval.benchmarks import chinese_retrieval as shared
from psychology_base.eval.benchmarks.application_calls import (
    Budget, RecordingRouter, validate_screening_call, request_input_bound, inference_call_options,
    hosted_request_options,
    billable_output_limit,
)
from psychology_base.eval.benchmarks.evaluation_support import AuditedSystem, complete_keys, measure_recall
from psychology_base.eval.benchmarks.public_datasets.mathdial import iter_mathdial_conversations, iter_mathdial_teacher_contexts
from psychology_base.eval.benchmarks.public_datasets.msc import iter_msc_conversations
from psychology_base.i18n import tokenizer_identity
from psychology_base.llm import load_config
from psychology_base.llm.config import CacheConfig, I18nConfig, MemoryConfig
from psychology_base.llm.router import LLMRouter
from psychology_base.llm.types import ChatMessage, TaskTier

ROOT = Path(__file__).resolve().parents[4]
FIXTURES = ROOT / 'benchmarks/fixtures/application_context'
SCHEMA = 'application-context-v1'
STAMP = datetime(2026, 10, 19, tzinfo=timezone.utc)
ARMS = ('identity', 'scaled_focus')
LOCAL_SCHEMA = 'application-context-local-screen-v1'
HYBRID_SCHEMA = 'application-context-hybrid-screen-v1'
HYBRID_SCHEMA_V2 = 'application-context-hybrid-screen-v2'
HYBRID_SCHEMA_V3 = 'application-context-hybrid-screen-v3'
HYBRID_SCHEMA_V4 = 'application-context-hybrid-screen-v4'
RESOURCE_V2 = 'interleaved-batches-v2'
BATCH_RECALLS = 20
GATES = {'minimum_gain': .1, 'bootstrap_draws': 10000, 'confidence': .95,
         'abstention': .95, 'tutor_usefulness': .95, 'max_recall_ratio': 2,
         'max_allocation_ratio': 2, 'max_application_latency_ratio': 1.25,
         'max_estimated_generation_cost_ratio': 1.25, 'critical_errors': 0,
         'billing_reconciliation_required_for_default': True}
def catalog_case(case):
    case=copy.deepcopy(case)
    if case['adaptation']:
        target=case['adaptation'].pop('target')
        case['adaptation']['target_sha256']=shared.stable_key(target)
    return case


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def runtime_identity(cfg, *, cjk_tokenizer=None):
    # Config contains environment-variable names, not resolved credentials.
    values = asdict(cfg)
    values['execution'] = cfg.execution.to_mapping()
    return {'config_sha256':shared.stable_key(json.dumps(values,sort_keys=True,default=str)),
            'tokenizer':tokenizer_identity(cfg.i18n.tokenizer if cjk_tokenizer is None else cjk_tokenizer),'python':platform.python_version(),
            'openai_sdk':optional_version('openai'),'numpy':np.__version__}


def optional_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def protocol_sha(protocol):
    base = protocol.get('profile_sha256', shared.digest(FIXTURES/'protocol.json'))
    return shared.stable_key({'base_protocol_sha256':base, **grading_identity(protocol)}) if grading_version(protocol)==2 else base


def grading_identity(protocol):
    if grading_version(protocol)==1:
        return {}
    return {'grading_contract_version':2,
            'memory_contract_prompt_sha256':shared.stable_key(protocol['memory_prompt']),
            'tutor_contract_prompt_sha256':shared.stable_key(empath_prompt(2))}


def local_profile(path, protocol):
    """Local-generation overlays preserve the original source/rubric freeze."""
    profile = read(path)
    hybrid = profile.get('schema') in (HYBRID_SCHEMA, HYBRID_SCHEMA_V2, HYBRID_SCHEMA_V3, HYBRID_SCHEMA_V4)
    revision = 4 if profile.get('schema') == HYBRID_SCHEMA_V4 else 3 if profile.get('schema') == HYBRID_SCHEMA_V3 else 2
    revised = profile.get('schema') in (HYBRID_SCHEMA_V2, HYBRID_SCHEMA_V3, HYBRID_SCHEMA_V4)
    fields = {'schema','parent_protocol_sha256','samples','split','generator','judge','budget','resource_procedure'}
    if revised:
        fields |= {'memory_rubric','qualification'}
    if revision == 3:
        fields.add('memory_evidence')
    if revision == 4:
        fields.add('grading_contract')
    expected_budget = {'max_calls':615 if revision==4 else 600,'max_usd':'1' if hybrid else '0','local_only':not hybrid}
    if (set(profile) != fields
            or profile['schema'] not in (LOCAL_SCHEMA, HYBRID_SCHEMA, HYBRID_SCHEMA_V2, HYBRID_SCHEMA_V3, HYBRID_SCHEMA_V4)
            or profile['parent_protocol_sha256'] != shared.digest(FIXTURES/'protocol.json')
            or type(profile['samples']) is not int or profile['samples'] != 1 or profile['split'] != 'development'
            or profile['resource_procedure'] != RESOURCE_V2
            or profile['budget'] != expected_budget):
        raise ValueError('Local screening profile differs from its source freeze or scope')
    fields = {'provider','model','digest','family','base_url','context_length','seed','json_mode','input_rate','output_rate','max_input'}
    for role in ('generator','judge'):
        spec = profile[role]
        if hybrid and role == 'judge':
            if spec != {'provider':'openai','model':'gpt-6-luna','family':'gpt',
                        'reasoning_effort':'none','json_mode':True,
                        'input_rate':'0.125','output_rate':'0.50','max_input':12000}:
                raise ValueError('Hosted judge differs from the approved model/decoding/rate freeze')
            continue
        if (set(spec) != fields or spec['provider'] != 'ollama'
                or spec['base_url'] != 'http://127.0.0.1:11434'
                or not re.fullmatch(r'[0-9a-f]{64}', spec['digest'])
                or not spec['model'] or ':cloud' in spec['model']
                or not spec['family'] or spec['context_length'] != 16384 or spec['seed'] != 0
                or spec['json_mode'] is not (role == 'judge')
                or spec['input_rate'] != '0' or spec['output_rate'] != '0' or spec['max_input'] != 12000):
            raise ValueError('Invalid pinned local model/transport/decoding specification')
    if profile['generator']['family'] == profile['judge']['family']:
        raise ValueError('The local generator cannot judge its own model family')
    result = {**protocol, **{k:profile[k] for k in ('samples','generator','judge','budget','resource_procedure')},
              'local_only':not hybrid, 'screening':True, 'profile_sha256':shared.digest(path), 'profile':profile}
    if revised:
        for name,filename in (('memory_rubric','memory-rubric-v2.json'),('qualification',f'qualification-v{revision}.json')):
            spec = profile[name]
            if (spec != {'path':str((FIXTURES/filename).relative_to(ROOT)).replace('\\','/'),
                         'sha256':shared.digest(FIXTURES/filename)}):
                raise ValueError('Revised rubric/qualification differs from the pinned input')
        rubric = read(FIXTURES/'memory-rubric-v2.json')
        if (set(rubric) != {'schema','parent_prompt_sha256','clarification'}
                or rubric['schema'] != 'application-context-memory-rubric-v2'
                or rubric['parent_prompt_sha256'] != shared.stable_key(MEMORY_PROMPT)
                or not isinstance(rubric['clarification'],str) or not rubric['clarification'].strip()):
            raise ValueError('Memory rubric does not extend the original frozen rubric')
        result.update(memory_prompt=MEMORY_PROMPT+'\n'+rubric['clarification'],
                      calibration_revision=revision,qualification=profile['qualification'],
                      memory_rubric=profile['memory_rubric'])
        if revision == 3:
            evidence = profile['memory_evidence']
            expected = {'report':'.data/application_context/hybrid-screen-v2-calibration.json',
                        'freeze':'docs/experiments/phase-9-application-context/hybrid-screen-v2-input-freeze.json'}
            if (set(evidence) != {'report','report_sha256','freeze','freeze_sha256'}
                    or any(evidence[k] != v for k,v in expected.items())
                    or any(shared.digest(ROOT/evidence[k]) != evidence[k+'_sha256'] for k in expected)):
                raise ValueError('Reused memory evidence differs from the pinned prior run')
            result['memory_evidence'] = evidence
        if revision==4:
            result=revised_grading_protocol(result)
            pin=profile['grading_contract']
            path=FIXTURES/'grading-contract-v2.json'
            if (pin!={'path':str(path.relative_to(ROOT)).replace('\\','/'),'sha256':shared.digest(path)}
                    or read(path)!={'schema':'application-grading-contract-v2',**grading_identity(result)}):
                raise ValueError('Revised grading contract differs from its pinned prompts')
            result['grading_contract']=pin
    return result


def qualification_identity(protocol):
    if protocol.get('calibration_revision',0) < 2:
        return grading_identity(protocol)
    result = {'judge_scoring_prompt_sha256':shared.stable_key(protocol['memory_prompt']),
              'qualification_catalog_sha256':protocol['qualification']['sha256']}
    if protocol['calibration_revision'] == 3:
        result['memory_evidence_sha256'] = protocol['memory_evidence']['report_sha256']
    return {**result, **grading_identity(protocol)}


def fingerprints():
    # Include current actual owners, not only the dispatcher and a scorer.
    paths = [*sorted((ROOT/'src/psychology_base/applications').rglob('*.py')),
             *sorted((ROOT/'src/psychology_base/companion').rglob('*.py')),
             *sorted((ROOT/'src/psychology_base/llm').rglob('*.py')),
             *sorted((ROOT/'src/psychology_base/i18n').rglob('*.py')),
             Path(__file__), Path(shared.__file__), ROOT/'src/psychology_base/eval/benchmarks/evaluation_support.py',
             ROOT/'src/psychology_base/eval/benchmarks/application_calls.py',
             ROOT/'src/psychology_base/eval/adapters.py',
             ROOT/'benchmarks/scripts/phase9_carryover_run.py',
             ROOT/'src/psychology_base/eval/benchmarks/public_datasets/msc.py',
             ROOT/'src/psychology_base/eval/benchmarks/public_datasets/mathdial.py',
             ROOT/'src/psychology_base/eval/benchmarks/public_datasets/mathdial_empath_scorer.py', ROOT/'pyproject.toml']
    return {str(p.relative_to(ROOT)).replace('\\','/'): shared.digest(p) for p in paths}


def load_inputs(protocol_path=FIXTURES/'protocol.json', catalog_path=FIXTURES/'catalog.json'):
    protocol, catalog = read(protocol_path), read(catalog_path)
    if (protocol.get('schema') != SCHEMA or shared.digest(catalog_path) != protocol['catalog_sha256']
            or catalog['schema'] != SCHEMA or protocol['arms'] != list(ARMS)
            or protocol['byte_cap'] != 4096 or protocol['samples'] != 3
            or protocol['gates'] != GATES or protocol['seed'] != 0
            or protocol['top_k'] != 6 or protocol['tokenizer'] != 'jieba'
            or protocol['aging'] != 'S1' or protocol['consolidation'] is not False
            or protocol['witness'] != 'independent-identity'):
        raise ValueError('Application protocol/catalog differs from frozen configuration')
    annotations_path = ROOT/protocol['annotations']['path']
    if shared.digest(annotations_path) != protocol['annotations']['sha256']:
        raise ValueError('Authored source annotations differ from freeze')
    authored = read(annotations_path)
    if catalog['cases'] != [catalog_case(c) for c in authored['cases']] or protocol['source_pins'] != authored['source_pins']:
        raise ValueError('Catalog does not match actual source authoring')
    if (protocol['scoring_prompt_sha256']!=shared.stable_key(MEMORY_PROMPT)
            or protocol['exclusion_audit_sha256']!=shared.digest(ROOT/'benchmarks/fixtures/scaled_context/fresh-audit.json')):
        raise ValueError('Scoring prompt or holdout exclusion audit changed')
    for name, sha in protocol['source_pins'].items():
        if shared.digest(ROOT/name) != sha:
            raise ValueError('Raw source bytes differ from freeze')
    cases = authored['cases']
    counts = {(a, s, split): sum(c['app']==a and c['stratum']==s and c['split']==split for c in cases)
              for a in ('companion','tutor') for s in ('native_zh_bridge','native_en_application','adapted_zh_application')
              for split in ('development','evaluation')}
    if len({c['id'] for c in cases}) != 72 or any(n != (4 if key[2]=='development' else 8) for key,n in counts.items()):
        raise ValueError('Source-family quotas or identities differ')
    exclusions = read(ROOT/'benchmarks/fixtures/scaled_context/fresh-audit.json')['excluded_groups']
    keys = []
    for c in cases:
        if len(c['queries']) != 2 or any(not isinstance(q,str) or not q.strip() for q in c['queries']):
            raise ValueError('Missing query forms')
        if 'kdconv' in c['source']:
            domain = c['source'].split('/')[-2]
            if c['row'] in exclusions[domain+'/train.json']:
                raise ValueError('Reserved historical dialogue used')
        keys.append((c['source'],c['row']))
    complete_keys(keys,set(keys))
    return protocol, cases


class Sources:
    def __init__(self, protocol):
        self.data = {}
        for name in protocol['source_pins']:
            path=ROOT/name
            self.data[name] = (list(iter_msc_conversations(path)) if '/msc/' in name else
                               list(iter_mathdial_conversations(path)) if '/mathdial/' in name else read(path))

    def materialize(self, case):
        """Source windows preserve actual words/roles; adaptations are explicit."""
        source = self.data[case['source']][case['row']]
        target = case['target']
        problem = ''
        if '/msc/' in case['source']:
            turns = [(str(i), 'user' if t.speaker_idx==0 else 'companion', t.utterance)
                     for i,t in enumerate(source.sessions[0].turns[:12])]
        elif '/mathdial/' in case['source']:
            problem = source.problem
            turns = [('problem','user',problem)] + [(str(i),'user' if t.role=='student' else 'tutor',t.utterance)
                                                       for i,t in enumerate(source.turns[:12])]
        else:
            turns = [(str(i),'source-actor-'+str(i%2),t['message']) for i,t in enumerate(source['messages'][:12])]
        if not any(ref==str(target) for ref,_,_ in turns):
            raise ValueError('Curated target is outside its real source window')
        original = next(text for ref,_,text in turns if ref==str(target))
        if case['adaptation']:
            turns = [(ref,role,case['adaptation']['target'] if ref==str(target) else text) for ref,role,text in turns]
        target_text = next(text for ref,_,text in turns if ref==str(target))
        if '/mathdial/' in case['source']:
            gold = {str(target)}
            # Remove every attempted solution and every teacher-specific hint;
            # the problem alone cannot establish the learner's earlier method.
            absent_refs = {'problem'}
        else:
            cues = [c.casefold() for c in case['cues']]
            # Only the manually reviewed target is a gold citation. A question
            # mentioning its answer is not automatically a supporting assertion.
            gold = {str(target)}
            absent_refs = {ref for ref,_,text in turns if ref not in gold and
                           not any(c in text.casefold() for c in cues)}
            absent_refs -= {str(ref) for ref in case.get('absent_exclude_refs',[])}
        if not gold or not absent_refs or any(ref in absent_refs for ref in gold):
            raise ValueError('Invalid present/absent source support')
        return turns, gold, absent_refs, target_text, original, problem


def prepare(case, sources, cfg, arm, absent, *, system_factory=None, byte_cap=4096):
    cfg = replace(cfg, i18n=replace(cfg.i18n,tokenizer='jieba'), memory=MemoryConfig(arm))
    factory = system_factory or AuditedSystem.from_llm_config
    system = factory(cfg, byte_cap=byte_cap, seed=0, aging_policy=build_aging_policy('S1'))
    turns,gold,absent_refs,target,original,problem=sources.materialize(case)
    ids={}
    for n,(ref,role,text) in enumerate(turns):
        if absent and ref not in absent_refs:
            continue
        before={f.id for f in system.companion.store.all()}
        tags=(f'user:{case["id"]}', 'session:source', f'speaker:{role}',
              sequence_tag(shared.stable_key([case['source'],case['row']]),str(case['row']),n))
        timestamp=STAMP+timedelta(minutes=n)
        # Use the same canonical hash owner so immediately evicted evidence
        # remains in the source-present retrieval denominator.
        expected_id=make_fragment(text,timestamp,tags=tags).id
        ids[ref]=expected_id
        system.observe(Observation(timestamp,role,text,tags=tags))
        added=[f for f in system.companion.store.all() if f.id not in before]
        if len(added)>1:
            raise ValueError('Observation introduced more than one source fragment')
        if added and added[0].id!=expected_id:
            raise ValueError('Observed source ID differs from canonical content addressing')
    # A scheduled boundary is explicit; it is not a month-horizon test.
    system.mark_session_boundary()
    inventory = {f.id:f for f in system.companion.store.all()}
    expected={ids[r] for r in gold if r in ids} if not absent else set()
    return system, {'gold_ids':sorted(expected), 'source_support_refs':sorted(gold),
                    'retained_support_ids':sorted(expected & set(inventory)),
                    'inventory':pam_export(inventory.values())['fragments'],
                    'source_observations':sum(not absent or ref in absent_refs for ref,_,_ in turns),
                    'retained_count':len(inventory), 'accounted_bytes':system.companion.store.total_bytes(system.edges),
                    'reference':target, 'original_reference':original, 'problem':problem}


def render(app, fragments, *, format_version=3):
    """Historical study payloads retain their explicit format independently of serving."""
    return (render_companion if app=='companion' else render_tutor)(fragments, format_version=format_version)


def context_diagnostics(cases, sources, cfg):
    """Model-free treatment check using the actual intake, recall and render owners.

    Each query gets an independent copy of the prepared state, as in trial().
    Changed context proves exposure to a treatment, not that it is beneficial.
    """
    if not cases or any(c['split'] != 'development' for c in cases):
        raise ValueError('Context diagnostics require development cases only')
    records = []
    source_status = {}
    for case in cases:
        for absent in (False, True):
            states = {}
            for arm in ARMS:
                system, evidence = prepare(case, sources, cfg, arm, absent)
                fragments = system.companion.store.all()
                for f in fragments:
                    source_status[f.id] = parse_turn(f.tags).status
                neighbors = local_neighbors(fragments)
                states[arm] = (system, evidence, sum(bool(v) for v in neighbors.values()))
            if states[ARMS[0]][1] != states[ARMS[1]][1]:
                raise ValueError('Paired context inputs differ before recall')
            for query_index, query in enumerate(case['queries']):
                arms = {}
                for arm, (system, evidence, neighbor_count) in states.items():
                    recalled = copy.deepcopy(system).recall(query)
                    selected = recalled.fragments_used[:6]
                    arms[arm] = {'selected_ids':[f.id for f in selected],
                        'context_sha256':shared.stable_key(render(case['app'], selected)),
                        'inventory_sha256':shared.stable_key(evidence['inventory']),
                        'retained_support_ids':evidence['retained_support_ids'],
                        'delivered_support_ids':sorted({f.id for f in selected} & set(evidence['gold_ids'])),
                        'fragments_with_neighbors':neighbor_count}
                records.append({'case':case['id'], 'app':case['app'], 'absent':absent,
                    'query_index':query_index, 'arms':arms,
                    'context_changed':arms[ARMS[0]]['context_sha256'] != arms[ARMS[1]]['context_sha256']})
    per_app = {}
    for app in ('companion', 'tutor'):
        present = [r for r in records if r['app'] == app and not r['absent']]
        per_app[app] = {'source_present_pairs':len(present),
            'pairs_with_neighbors':sum(r['arms'][ARMS[1]]['fragments_with_neighbors'] > 0 for r in present),
            'changed_context_pairs':sum(r['context_changed'] for r in present)}
    passed = (all(s == 'valid' for s in source_status.values())
              and all(v['pairs_with_neighbors'] > 0 and v['changed_context_pairs'] > 0 for v in per_app.values()))
    return {'schema':'application-context-contrast-v1', 'pairs':len(records),
            'changed_context_pairs':sum(r['context_changed'] for r in records),
            'unique_source_fragments':len(source_status),
            'source_parse_status':{s:sum(v == s for v in source_status.values()) for s in sorted(set(source_status.values()))},
            'per_app':per_app, 'passed':passed, 'records':records,
            'provider_requests':0, 'generated_application_replies':0}


def validate_navigation_controls(controls):
    """Validate declared software-control content before any transport or scoring."""
    from psychology_base.companion.reconstruction import tokens

    if not isinstance(controls, list) or not controls:
        raise ValueError('Missing authored navigation controls')
    ids = set()
    for control in controls:
        if (not isinstance(control, dict) or not isinstance(control.get('id'), str)
                or not control['id'] or control['id'] in ids):
            raise ValueError('Invalid or duplicate authored control identity')
        ids.add(control['id'])
        language = control.get('language')
        source_language = control.get('source_language', language)
        query_language = control.get('query_language', language)
        distractors = control.get('distractors')
        if (language not in ('en','zh') or query_language != language
                or source_language not in ('en','zh') or not isinstance(distractors,list)
                or len(distractors) != 8 or control.get('app') not in ('companion','tutor')):
            raise ValueError('Invalid authored control shape or language')
        streams = []
        for text, declared in [(control.get('query'),query_language),
                               (control.get('source_text'),source_language),
                               *((text,source_language) for text in distractors)]:
            if (not isinstance(text,str) or not text.strip() or '\ufffd' in text
                    or not any(c.isalpha() for c in text)):
                raise ValueError('Authored navigation text is empty, corrupted or meaningless')
            if (declared == 'zh' and not any('\u3400' <= c <= '\u9fff' for c in text)
                    or declared == 'en' and not re.search('[A-Za-z]',text)):
                raise ValueError('Authored navigation text disagrees with declared language')
            words = set(tokens(text,'jieba'))
            if not words or not any(any(c.isalpha() for c in word) for word in words):
                raise ValueError('Authored navigation tokens must be meaningful and nonempty')
            streams.append(words)
        normalized = [re.sub(r'\s+',' ',unicodedata.normalize('NFKC',text).casefold()).strip()
                      for text in [control['source_text'],*distractors]]
        if len(set(normalized)) != len(normalized):
            raise ValueError('Authored competing memories must be distinct text inputs')
        if streams[0] & streams[1]:
            raise ValueError('Authored paraphrase control is not zero-overlap')


def navigation_inputs(plan_path):
    """Only the frozen, exposed development packet can enter this local stage."""
    if plan_path is None:
        raise ValueError('Navigation diagnostics require an explicit --navigation-plan')
    plan = read(plan_path)
    name = Path(plan_path).name
    if name not in ('memory-navigation-local-diagnostic-v1.json','memory-navigation-local-diagnostic-v2.json',
                    'memory-navigation-local-baseline-v2.json', 'memory-navigation-local-qwen-v3.json',
                    'memory-navigation-local-bge-v3.json'):
        raise ValueError('Unknown frozen navigation packet')
    reference = read(FIXTURES/name)
    if plan != reference:
        raise ValueError('Navigation profile differs from the frozen local-only packet')
    validate_navigation_controls(plan.get('authored_controls'))
    if name.endswith(('-v2.json', '-v3.json')):
        controls = FIXTURES/'memory-navigation-controls-v2.json'
        if (shared.digest(controls) != plan['controls_sha256']
                or read(controls)['controls'] != plan['authored_controls']):
            raise ValueError('Navigation control text differs from its reviewed freeze')
    if (plan['base_protocol_sha256'] != shared.digest(FIXTURES/'protocol.json')
            or plan['base_catalog_sha256'] != shared.digest(FIXTURES/'catalog.json')
            or plan['hosted_provider_requests'] != 0 or plan['heldout_scoring_authorized'] is not False):
        raise ValueError('Navigation source freeze or local-only boundary changed')
    protocol, cases = load_inputs()
    selected = [next(c for c in cases if c['id'] == case_id) for case_id in plan['case_ids']]
    complete_keys([c['id'] for c in selected], set(plan['case_ids']))
    if any(c['split'] != 'development' for c in selected):
        raise ValueError('Navigation diagnostic cannot score held-out cases')
    return plan, protocol, selected


def navigation_discrimination(plan, cache):
    """Score actual cached encoder outputs; no extra inference or invented vectors."""
    from psychology_base.companion.facts import cosine
    from psychology_base.llm.cache import embedding_cache_key
    from psychology_base.llm.types import validate_embedding

    identity = plan['model_identity']
    observed = {}
    def vector(text):
        key = embedding_cache_key(provider='ollama',model_id=identity['model'],text=text,identity=identity)
        result = cache.get_embedding(key)
        if result is None:
            raise ValueError('Discrimination requires an actual cached model output for every input')
        result = validate_embedding(result,dimension=identity['dimension'])
        observed[text] = tuple(result)
        return result
    records = []
    unrelated = plan['authored_control_design']['unrelated_absent_indices']
    for control in plan['authored_controls']:
        query = vector(plan['query_prefix'] + control['query'])
        inputs = [*control['distractors'],control['source_text']]
        scores = [cosine(query,vector(text)) for text in inputs]
        margin = scores[-1] - max(scores[i] for i in unrelated)
        rank = 1 + sum(score >= scores[-1] for score in scores[:-1])
        records.append({'case':control['id'], 'source_language':control.get('source_language',control['language']),
            'query_language':control.get('query_language',control['language']),
            'target_rank':rank,'target_cosine':scores[-1], 'cosines':scores,
            'unrelated_margin':margin,
            'passed':rank==1 and margin >= plan['minimum_unrelated_margin']})
    grouped = {}
    for text, values in observed.items():
        grouped.setdefault(values,[]).append(shared.stable_key(text))
    collisions = [keys for keys in grouped.values() if len(keys)>1]
    return {'scope':'authored exposed discrimination controls, not application quality',
        'records':records,'distinct_input_vector_collisions':collisions,
        'unique_input_count':len(observed),
        'passed':all(r['passed'] for r in records) and not collisions}


def navigation_diagnostic(args):
    """Real local embeddings, ordinary intake/recall/render; no generation/judge."""
    from psychology_base.companion.system import CompanionSystem
    from psychology_base.eval.benchmarks.graph_priors import bind_navigation, system_for_policy
    from psychology_base.llm.cache import LLMCache
    from psychology_base.llm.config import default_config
    from psychology_base.llm.providers.ollama import OllamaProvider

    plan, protocol, selected = navigation_inputs(args.navigation_plan)
    output_dir = args.output.with_suffix('')
    output_dir.mkdir(parents=True, exist_ok=False)
    code = fingerprints()
    extra = [ROOT/'src/psychology_base'/relative for relative in (
        'companion/navigation.py', 'companion/turn_provenance.py', 'llm/router.py',
        'llm/cache.py', 'llm/types.py', 'llm/providers/ollama.py', 'eval/benchmarks/graph_priors.py')]
    code.update({str(p.relative_to(ROOT)): shared.digest(p) for p in extra})
    plan_sha = shared.digest(args.navigation_plan)
    header = {'schema':plan['schema'], 'phase':'exposed-local-development-diagnostic',
              'plan_sha256':plan_sha, 'code_sha256':code, 'tokenizer':tokenizer_identity('jieba'),
              'model_identity':plan['model_identity'], 'limitations':plan['limitations']}
    shared.write_new(output_dir/'run-manifest.json', header)
    model = plan['model_identity']
    provider = OllamaProvider(base_url=plan['endpoint'], simple_model='unused-chat-model',
        embedding_model=model['model'], local_only=True, expected_digest=model['digest'],
        context_length=model['context_length'], request_timeout_s=plan['request_timeout_s'],
        model_metadata_timeout_s=plan['metadata_timeout_s'],
        embedding_metadata_ttl_s=plan.get('embedding_metadata_ttl_s', 0))
    opened = provider._open
    counts = Counter()
    rows = []
    with SessionLogWriter(output_dir/'local-requests.jsonl') as ledger:
        @contextmanager
        def record_open(request, *, timeout):
            endpoint = request.full_url.rsplit('/', 1)[-1]
            if endpoint not in ('tags', 'show', 'embed'):
                raise ValueError('Navigation transport attempted an undeclared endpoint')
            if endpoint == 'embed' and counts[endpoint] >= plan['max_local_embedding_requests']:
                raise ValueError('Frozen local embedding attempt cap exhausted')
            if endpoint == 'embed':
                body = json.loads(request.data)
                size = len(body['input']) if isinstance(body['input'],list) else 1
                if counts['embedding_inputs'] + size > plan.get('max_local_embedding_inputs',256):
                    raise ValueError('Frozen local embedding input cap exhausted')
                counts['embedding_inputs'] += size
            counts[endpoint] += 1
            attempt = counts[endpoint]
            ledger.write({'event':'local_request', 'endpoint':endpoint, 'attempt':counts[endpoint],
                          'body_sha256':hashlib.sha256(request.data or b'').hexdigest()})
            started = time.perf_counter()
            try:
                with opened(request, timeout=timeout) as response:
                    yield response
            finally:
                ledger.write({'event':'local_response','endpoint':endpoint,'attempt':attempt,
                              'transport_scope_ms':(time.perf_counter()-started)*1000})
        provider._open = record_open
        try:
            if provider.embedding_identity(model['model']) != model:
                raise ValueError('Installed embedding model differs from the frozen identity')
            from psychology_base.llm.execution import ExecutionPolicy
            cfg = replace(default_config(), execution=ExecutionPolicy(mode='research'))
            router = LLMRouter(cfg, providers={'ollama':provider}, cache=LLMCache(output_dir/'embedding-cache'))
            sources = Sources(protocol)
            authored = {}
            for index, control in enumerate(plan['authored_controls']):
                case = {**control, 'split':'development', 'stratum':'authored_software_control',
                        'source':plan.get('authored_source','authored-memory-navigation-v1'),
                        'row':index, 'queries':[control['query']]}
                authored[case['id']] = case
                selected.append(case)

            class DiagnosticSources:
                def materialize(self, case):
                    if case['id'] not in authored:
                        return sources.materialize(case)
                    text = case['source_text']
                    texts = [*case['distractors'], text]
                    unrelated = set(map(str, plan['authored_control_design']['unrelated_absent_indices']))
                    return [(str(i),'user',t) for i,t in enumerate(texts)], {str(len(texts)-1)}, unrelated, text, text, ''

            inputs = DiagnosticSources()
            arms = ['identity','scaled_focus','lexical'] + [f'semantic-{f}' for f in plan['cosine_floors']]
            def control_factory(cfg, **kwargs):
                return system_for_policy('no_initial_edges', cjk_tokenizer=cfg.i18n.tokenizer,
                                         fragment_context_selection=cfg.memory.context_selection, **kwargs)
            for case in selected:
                for absent in (False, True):
                    is_control = case['id'] in authored
                    prepared = {context_policy:prepare(case, inputs, cfg, context_policy, absent,
                                system_factory=control_factory if is_control else CompanionSystem.from_llm_config,
                                byte_cap=plan['authored_control_design']['byte_cap'] if is_control else plan['byte_cap'])
                                for context_policy in ('identity','scaled_focus')}
                    if prepared['identity'][1]['inventory'] != prepared['scaled_focus'][1]['inventory']:
                        raise ValueError('Navigation comparison changed the ingested inventory')
                    if is_control and len(prepared['identity'][1]['inventory']) != (6 if absent else 9):
                        raise ValueError('Authored discrimination control lost its competing memories')
                    for arm in arms:
                        base, evidence = prepared['identity' if arm=='identity' else 'scaled_focus']
                        system = copy.deepcopy(base)
                        if arm == 'lexical':
                            bind_navigation(system, 'lexical')
                        elif arm.startswith('semantic-'):
                            bind_navigation(system, 'semantic', router=router, floor=float(arm.split('-')[1]),
                                            model_identity=model, query_prefix=plan['query_prefix'])
                        for state in plan['states']:
                            before = counts['embed']
                            recalled, timing = measure_recall(system, case['queries'][0])
                            selected_ids = [f.id for f in recalled.fragments_used[:6]]
                            if not set(selected_ids) <= {f.id for f in system.companion.store.all()}:
                                raise ValueError('Navigation selected a nonlive source ID')
                            gold = set(evidence['gold_ids'])
                            row = {'case':case['id'], 'app':case['app'], 'language':case['language'],
                                   'stratum':case['stratum'], 'absent':absent, 'arm':arm, 'state':state,
                                   'selected_ids':selected_ids, 'gold_ids':sorted(gold),
                                   'complete_evidence':bool(gold) and gold <= set(selected_ids),
                                   'retained_support_ids':evidence['retained_support_ids'],
                                   'context_sha256':shared.stable_key(render(case['app'], recalled.fragments_used[:6])),
                                   'navigation':recalled.metadata.get('fragment_navigation'),
                                   'local_embedding_requests':counts['embed']-before,
                                   **shared.score_context(selected_ids, gold), **timing}
                            rows.append(row)
                            ledger.write({'event':'trial', **row})
                print(f"Completed navigation diagnostic {case['id']}", flush=True)
            if provider.embedding_identity(model['model']) != model:
                raise ValueError('Embedding model changed before the diagnostic completed')
        except Exception as exc:
            shared.write_new(output_dir/'failure.json', {'error_type':type(exc).__name__,
                'message':str(exc), 'local_requests':dict(counts), 'completed_trials':len(rows)})
            raise
    if (shared.digest(args.navigation_plan) != plan_sha
            or any(shared.digest(ROOT/name) != sha for name,sha in protocol['source_pins'].items())
            or any(shared.digest(ROOT/name) != sha for name,sha in code.items())):
        raise ValueError('Code or navigation packet changed during execution')
    expected = {(c['id'], absent, arm, state) for c in selected
                for absent in (False,True)
                for arm in arms for state in plan['states']}
    complete_keys([(r['case'],r['absent'],r['arm'],r['state']) for r in rows], expected)
    if sum(r['local_embedding_requests'] for r in rows) != counts['embed']:
        raise ValueError('Navigation inference counts differ from the actual request ledger')
    summaries = {}
    screens = {}
    for arm in arms:
        cold = [r for r in rows if r['arm']==arm and r['state']=='cold']
        summaries[arm] = {}
        for language in ('en','zh'):
            public = [r for r in cold if r['language']==language and r['stratum']!='authored_software_control' and not r['absent']]
            controls = [r for r in cold if r['language']==language and r['stratum']=='authored_software_control' and not r['absent']]
            summaries[arm][language] = {'public_complete':sum(r['complete_evidence'] for r in public),
                'public_cases':len(public), 'authored_complete':sum(r['complete_evidence'] for r in controls),
                'authored_cases':len(controls)}
        baseline = {(r['case'],r['absent']):r for r in rows if r['arm']=='scaled_focus' and r['state']=='cold'}
        public = [r for r in cold if not r['absent'] and r['stratum']!='authored_software_control']
        gain = sum(int(r['complete_evidence'])-int(baseline[r['case'],False]['complete_evidence']) for r in public)/len(public)
        losses = [r['case'] for r in public if baseline[r['case'],False]['complete_evidence'] and not r['complete_evidence']]
        absent_hints = sum(bool(r['navigation'] and r['navigation']['hinted_occurrences']) for r in cold
                           if r['absent'] and r['stratum']!='authored_software_control')
        unrelated_hints = sum(bool(r['navigation'] and r['navigation']['hinted_occurrences']) for r in cold
                              if r['absent'] and r['stratum']=='authored_software_control')
        controls_ok = all(r['complete_evidence'] for r in cold if r['stratum']=='authored_software_control' and not r['absent'])
        identity_rows = {r['case']:r for r in rows if r['arm']=='identity' and r['state']=='cold' and not r['absent']}
        identity_losses = [r['case'] for r in public if identity_rows[r['case']]['complete_evidence'] and not r['complete_evidence']]
        screens[arm] = {'observed_public_gain':gain, 'lost_baseline_cases':losses,
                        'lost_identity_cases':identity_losses,
                        'absent_cases_with_hints':absent_hints, 'authored_controls_complete':controls_ok,
                        'unrelated_controls_with_hints':unrelated_hints,
                        'passed_observed_screen':gain >= plan['minimum_observed_gain'] and not losses and not identity_losses and not unrelated_hints and controls_ok}
    discrimination = navigation_discrimination(plan,router.cache)
    for arm in arms:
        if arm.startswith('semantic-'):
            screens[arm]['passed_observed_screen'] &= discrimination['passed']
    return {**header, 'local_embedding_requests':counts['embed'], 'local_metadata_requests':counts['tags']+counts['show'],
            'local_embedding_inputs':counts['embedding_inputs'],
            'hosted_provider_requests':0, 'judge_requests':0, 'generated_application_replies':0,
            'summary':summaries, 'observed_development_screens':screens,
            'representation_discrimination':discrimination,'integration_eligible':False,
            'heldout_scored':False, 'resource_adoption_gate_measured':False, 'trials':rows,
            'request_ledger_sha256':shared.digest(output_dir/'local-requests.jsonl')}


def judge_tutor(router, response, *, problem_context, student_utterance, protocol=None):
    """One decoding contract for qualification and application judgments."""
    return LLMEMPATHJudge(router, tier=TaskTier.ADVANCED, max_tokens=300,
                          strict=True, temperature=0, contract_version=grading_version(protocol)).judge(
        response, problem_context=problem_context, student_utterance=student_utterance)


def trial(case, query_index, sample, arm, absent, sources, cfg, generation, judge, output_dir, *, protocol=None):
    system, evidence = prepare(case,sources,cfg,arm,absent)
    query = case['queries'][query_index]
    visible = []
    live = make_llm_responder(generation,tier=TaskTier.ADVANCED,max_tokens=400,temperature=0.3)
    def responder(prompt, context, utterance):
        recalled=system.last_recall
        if context != render(case['app'],recalled.fragments_used[:6]) or system.audit_recall_count != 1:
            raise ValueError('Actual responder context/count differs from the single live recall')
        known={f['id']:f for f in evidence['inventory']}
        snapshot=pam_export(recalled.fragments_used[:6])['fragments']
        if any(f['id'] not in known or f!=known[f['id']] for f in snapshot):
            raise ValueError('Responder received unknown/future/mutated source memory')
        visible.append({'prompt':prompt,'context':context,'query':utterance,'fragments':snapshot,
                        'recall_metadata':recalled.metadata})
        return live(prompt,context,utterance)
    manifest=UserManifest(case['id'], PilotApp.EMOTIONAL_COMPANION if case['app']=='companion' else PilotApp.LEARNING_TUTOR,
                          'S1', STAMP.isoformat(),'26-35','CN',case['language'])
    fields=dict(manifest=manifest,system=system,llm_responder=responder,session_id='probe', context_format_version=3)
    agent=(build_companion_agent(**fields,crisis_log_path=output_dir/'crisis.jsonl') if case['app']=='companion'
           else build_tutor_agent(**fields))
    stamp=STAMP+timedelta(days=1)
    started=time.perf_counter()
    response=agent.respond(query,timestamp=stamp,attestation=AgeAttestation(30,stamp.timestamp(),'evaluation_fixture'))
    elapsed=(time.perf_counter()-started)*1000
    if len(visible)!=1 or not response.exchange_id or not response.response_text.strip():
        raise ValueError('Application did not complete one normal recorded exchange')
    completion=generation.last_completion
    if completion.finish_reason not in ('stop','end_turn'):
        raise ValueError('Truncated/nonfinal generation cannot qualify')
    capture=visible[0]
    judgment=judge_memory(judge,case,absent,capture,response.response_text,evidence['reference'],protocol=protocol)
    memory_call=copy.deepcopy(judge.last_call)
    tutor=None
    if '/mathdial/' in case['source']:
        verdict=judge_tutor(judge,
            response.response_text,problem_context=evidence['problem'],student_utterance=query,protocol=protocol)
        tutor={'criteria':{k.value:v for k,v in verdict.sub_criteria.items()},
               'parse_failed':verdict.parse_failed,'proxy':verdict.proxy or judge.controls,
               'rationale':verdict.rationale,'finish_reason':judge.last_completion.finish_reason,
               **verdict.contract_metadata()}
    row={'case':case['id'],'app':case['app'],'stratum':case['stratum'],'language':case['language'],
         'split':case['split'],'query_index':query_index,'sample':sample,'candidate':arm,'absent':absent,
         'evidence':evidence,'capture':capture,'response':response.response_text,'exchange_id':response.exchange_id,
         'completion':{**asdict(completion),'tier':completion.tier.value},
         'generation_call':copy.deepcopy(generation.last_call),'application_ms':elapsed,
         'memory_judgment':judgment,'tutor_judgment':tutor,'proxy':generation.controls or judge.controls}
    row['memory_call']=memory_call
    row['tutor_call']=copy.deepcopy(judge.last_call) if tutor else None
    row['scores']=score_row(row)
    return row


def _validate_runtime_memory_protocol(protocol):
    expected=runtime_memory_protocol(prompt_revision=protocol.get('runtime_memory_prompt_revision',1),
                                    contract_version=protocol['runtime_memory_contract_version'])
    if (type(protocol['runtime_memory_contract_version']) is not int
            or protocol['runtime_memory_contract_version']!=expected['runtime_memory_contract_version']
            or grading_version(protocol)!=2 or protocol.get('memory_prompt')!=expected['memory_prompt']
            or (expected['runtime_memory_contract_version'] in (2,3,4) and
                (type(protocol.get('runtime_memory_decision_policy_version')) is not int or
                 protocol['runtime_memory_decision_policy_version']!=expected['runtime_memory_decision_policy_version']))
            or (expected['runtime_memory_contract_version']==4
                and protocol.get('memory_interpretation_prompt')!=expected['memory_interpretation_prompt'])):
        raise ValueError('Runtime memory contract differs from its explicit freeze')


def memory_interpretation_messages(case,capture,response,*,protocol):
    _validate_runtime_memory_protocol(protocol)
    if protocol['runtime_memory_contract_version']!=4:
        raise ValueError('Source-blind interpretation requires contract 4')
    payload=runtime_interpretation_payload(case['app'],capture['query'],response)
    return [ChatMessage('system',protocol['memory_interpretation_prompt']),
            ChatMessage('user',json.dumps(payload,ensure_ascii=False))]


def runtime_control_evidence(item,protocol,*,interpretation=None):
    version=protocol['runtime_memory_contract_version']
    return (runtime_memory_payload(item['app'],item['capture']['query'],item['capture']['context'],
            item['response'],contract_version=version,interpretation=interpretation) if version in (3,4) else None)


def memory_judge_messages(case, absent, capture, response, reference, *, protocol=None,interpretation=None):
    """One message owner for live dispatch, packet construction and audit replay."""
    version=grading_version(protocol)
    runtime=protocol is not None and 'runtime_memory_contract_version' in protocol
    if runtime:
        _validate_runtime_memory_protocol(protocol)
        payload=runtime_memory_payload(case['app'],capture['query'],capture['context'],response,
                                      contract_version=protocol['runtime_memory_contract_version'],interpretation=interpretation)
        if protocol['runtime_memory_contract_version']==4:
            runtime_binding_claims(payload)
    else:
        payload={'APPLICATION':case['app'],'QUERY':capture['query'],'VISIBLE_CONTEXT':capture['context'],
                 'REPLY':response,'SOURCE_ABSENT':absent,'WITHHELD_REFERENCE':reference}
    prompt=(protocol or {}).get('memory_prompt',MEMORY_PROMPT)
    return [ChatMessage('system',prompt),ChatMessage('user',json.dumps(payload,ensure_ascii=False))]


def memory_judge_options(protocol=None):
    """Share the declared call options across dispatch, request freeze and audit."""
    options = inference_call_options((protocol or {}).get('judge', {}))
    return options if options is not None else dict(max_tokens=300, temperature=0)


def judge_memory(router, case, absent, capture, response, reference, *, protocol=None):
    if (protocol or {}).get('runtime_memory_contract_version')==4:
        messages=memory_interpretation_messages(case,capture,response,protocol=protocol)
        router.last_memory_calls=[]
        completion=router.chat(messages,tier=TaskTier.ADVANCED,**memory_judge_options(protocol))
        router.last_memory_calls.append(copy.deepcopy(router.last_call))
        interpreted=parse_runtime_interpretation(completion.content,completion.finish_reason,router.controls,
            evidence=json.loads(messages[1].content))
        router.last_memory_interpretation=interpreted
        if interpreted['parse_failed']:
            evidence=runtime_memory_payload(case['app'],capture['query'],capture['context'],response,contract_version=4)
            return parse_runtime_memory(completion.content,completion.finish_reason,router.controls,contract_version=4,evidence=evidence)
        messages=memory_judge_messages(case,absent,capture,response,reference,protocol=protocol,
            interpretation=interpreted['interpretation'])
        completion=router.chat(messages,tier=TaskTier.ADVANCED,**memory_judge_options(protocol))
        router.last_memory_calls.append(copy.deepcopy(router.last_call))
        return parse_runtime_memory(completion.content,completion.finish_reason,router.controls,contract_version=4,
            evidence=json.loads(messages[1].content))
    messages=memory_judge_messages(case,absent,capture,response,reference,protocol=protocol)
    completion=router.chat(messages, tier=TaskTier.ADVANCED, **memory_judge_options(protocol))
    if protocol is not None and 'runtime_memory_contract_version' in protocol:
        return parse_runtime_memory(completion.content,completion.finish_reason,router.controls,
                                    contract_version=protocol['runtime_memory_contract_version'],
                                    evidence=json.loads(messages[1].content) if protocol['runtime_memory_contract_version']==3 else None)
    return parse_memory(completion.content,completion.finish_reason,router.controls,contract_version=grading_version(protocol))


def judgment_resolved(judgment, *, tutor=False):
    """Check provider-output coherence; record-level reviews are a separate gate."""
    if 'runtime_memory_contract_version' in judgment:
        return not tutor and runtime_memory_resolved(judgment)
    version=judgment.get('contract_version',1)
    validate_contract_version(version)
    if judgment['parse_failed'] or judgment.get('consistency_errors'):
        return False
    if version==2:
        checker=empath_consistency_errors if tutor else memory_consistency_errors
        errors=list(checker(judgment['criteria']))
        return not errors and judgment.get('consistency_errors')==errors
    return True


def _review_fields(kind):
    if kind == 'memory':
        return MEMORY_FIELDS
    if kind == 'tutor':
        return tuple(c.value for c in EMPATHSubCriterion)
    raise ValueError('Unknown criterion review kind')


def _review_subject_sha256(record):
    return shared.stable_key({k:v for k,v in record.items() if k not in ('criterion_reviews','scores')})


def make_criterion_review(record, kind, criteria, *, rubric_sha256, reviewer, reason, evidence):
    """Create external review metadata, without changing a verdict or score."""
    receipt={'version':1,'subject_sha256':_review_subject_sha256(record),'rubric_sha256':rubric_sha256,
             'reviewer':reviewer,'reason':reason,'evidence':copy.deepcopy(evidence),'criteria':copy.deepcopy(criteria)}
    check={**record,'criterion_reviews':{**record.get('criterion_reviews',{}),kind:receipt}}
    criterion_review_statuses(check,kind,rubric_sha256=rubric_sha256)
    return receipt


def criterion_review_statuses(record, kind, *, rubric_sha256=None):
    """Validate one receipt against the complete immutable assessment input/output."""
    fields=_review_fields(kind)
    statuses={k:'unreviewed' for k in fields}
    if 'criterion_reviews' not in record:
        return statuses
    reviews=record['criterion_reviews']
    if not isinstance(reviews,dict) or not reviews or not set(reviews)<= {'memory','tutor'}:
        raise ValueError('Invalid criterion review map')
    if kind not in reviews:
        return statuses
    review=reviews[kind]
    required={'version','subject_sha256','rubric_sha256','reviewer','reason','evidence','criteria'}
    if not isinstance(review,dict) or set(review)!=required:
        raise ValueError('Invalid criterion review receipt')
    if type(review['version']) is not int or review['version']!=1:
        raise ValueError('Unknown criterion review version')
    if any(not isinstance(review[k],str) or re.fullmatch('[0-9a-f]{64}',review[k]) is None
           for k in ('subject_sha256','rubric_sha256')):
        raise ValueError('Invalid criterion review hash')
    if (review['subject_sha256']!=_review_subject_sha256(record)
            or (rubric_sha256 is not None and review['rubric_sha256']!=rubric_sha256)):
        raise ValueError('Criterion review target or rubric changed')
    if any(not isinstance(review[k],str) or not review[k].strip() for k in ('reviewer','reason')):
        raise ValueError('Criterion review needs reviewer and reason')
    evidence=review['evidence']
    if (not isinstance(evidence,dict) or set(evidence)!= {'reference','sha256'}
            or not isinstance(evidence['reference'],str) or not evidence['reference'].strip()
            or not isinstance(evidence['sha256'],str) or re.fullmatch('[0-9a-f]{64}',evidence['sha256']) is None):
        raise ValueError('Criterion review needs a pinned evidence reference')
    reference=Path(evidence['reference'])
    path=(ROOT/reference).resolve()
    if reference.is_absolute() or not path.is_relative_to(ROOT) or not path.is_file() or shared.digest(path)!=evidence['sha256']:
        raise ValueError('Criterion review evidence differs from its local freeze')
    criteria=review['criteria']
    if (not isinstance(criteria,dict) or not criteria or not set(criteria)<=set(fields)
            or any(type(v) is not str or v not in ('needs_review','reviewed') for v in criteria.values())):
        raise ValueError('Invalid criterion review statuses')
    return {**statuses,**criteria}


def criterion_review_summary(records, *, protocol=None):
    """Count every assessment and criterion; unreviewed is not independently confirmed."""
    records=list(records)
    if not any('criterion_reviews' in record for record,kind in records):
        return None
    allowed={}
    for record,kind in records:
        _review_fields(kind)
        allowed.setdefault(id(record),set()).add(kind)
    counts={kind:{field:dict(total=0,reviewed=0,needs_review=0,unreviewed=0) for field in _review_fields(kind)}
            for kind in ('memory','tutor')}
    pending=[]; manifest=[]
    for index,(record,kind) in enumerate(records):
        if 'criterion_reviews' in record:
            mapping=record['criterion_reviews']
            if not isinstance(mapping,dict) or not set(mapping)<=allowed[id(record)]:
                raise ValueError('Unexpected criterion review kind for this assessment')
        rubric=None
        if protocol is not None:
            rubric=shared.stable_key(protocol.get('memory_prompt',MEMORY_PROMPT) if kind=='memory'
                                     else empath_prompt(grading_version(protocol)))
        statuses=criterion_review_statuses(record,kind,rubric_sha256=rubric)
        manifest.append({'kind':kind,'subject_sha256':_review_subject_sha256(record),
                         'review':record.get('criterion_reviews',{}).get(kind)})
        for field,status in statuses.items():
            counts[kind][field]['total']+=1
            counts[kind][field][status]+=1
            if status=='needs_review':
                pending.append({'index':index,'kind':kind,'field':field,
                                'subject_sha256':_review_subject_sha256(record)})
    return {'schema':'application-criterion-review-summary-v1','assessments':len(records),
            'status':'needs_review' if pending else 'no_pending_reviews','pending':pending,
            'per_criterion':counts,'manifest_sha256':shared.stable_key(manifest)}


def _apply_criterion_review_gate(metrics, records, *, protocol=None, expected_manifest_sha256=None):
    summary=criterion_review_summary(records,protocol=protocol)
    if expected_manifest_sha256 is not None and (summary is None or summary['manifest_sha256']!=expected_manifest_sha256):
        raise ValueError('Criterion review manifest differs from the required freeze')
    if summary is None:
        return metrics
    result=copy.deepcopy(metrics)
    result.update(criterion_review=summary,qualification_status=summary['status'])
    if summary['status']=='needs_review':
        for key in ('passed','quality_passed'):
            if key in result:
                result[key]=False
        if 'accuracy_passed' in result:
            result['raw_accuracy_passed']=result['accuracy_passed']; result['accuracy_passed']=None
        if 'gates' in result:
            result['gates']['criterion_review_complete']=False
        pending=summary['per_criterion']
        if any(v['needs_review'] for v in pending['memory'].values()):
            if 'memory_agreement' in result:
                result['raw_memory_agreement']=result['memory_agreement']; result['memory_agreement']=None
            if 'memory_critical_agreement' in result:
                raw=result['memory_critical_agreement']
                result['raw_memory_critical_agreement']=raw
                result['memory_critical_agreement']={k:None if pending['memory'][k]['needs_review'] else v for k,v in raw.items()}
        if any(v['needs_review'] for v in pending['tutor'].values()) and 'tutor_kappa' in result:
            result['raw_tutor_kappa']=result['tutor_kappa']; result['tutor_kappa']=None
    return result


def score_row(row):
    memory=row['memory_judgment']; tutor=row['tutor_judgment']
    reviews=criterion_review_summary([(row,'memory')]+([(row,'tutor')] if tutor else []))
    resolved=(judgment_resolved(memory) and not (tutor and not judgment_resolved(tutor,tutor=True))
              and not (reviews and reviews['status']=='needs_review'))
    m=memory['criteria']
    supported=resolved and m['grounded'] and m['attribution_correct'] and not m['critical_error']
    useful=True; leak=False
    if tutor:
        t=tutor['criteria']
        leak=bool(t.get('states_final_answer'))
        useful=bool(t.get('requests_reasoning') or t.get('provides_scaffold_hint')) and not (
            leak or t.get('corrects_with_fact'))
    used=[f['id'] for f in row['capture']['fragments']]
    gold=set(row['evidence']['gold_ids'])
    return {**shared.score_context(used,gold), 'sufficient_support':bool(gold & set(used)),
            'task_success':bool(supported and m.get('addresses_request') and useful),
            'appropriate_abstention':bool(supported and m.get('abstains_appropriately')),
            'critical_error':bool(m.get('critical_error',False) or leak),
            'tutor_useful':bool(resolved and useful),'resolved':bool(resolved),
            **({'criterion_review':reviews} if reviews is not None else {})}


def row_key(r):
    return (r['case'],r['app'],r['query_index'],r['absent'],r['candidate'],r['sample'])


def matrix(rows,cases,samples=3):
    expected={(c['id'],c['app'],q,a,arm,n) for c in cases for q,a,arm,n in
              itertools.product(range(2),(False,True),ARMS,range(samples))}
    if any(type(r['absent']) is not bool or type(r['sample']) is not int for r in rows):
        raise ValueError('Wrong trial key types')
    return complete_keys([row_key(r) for r in rows],expected)


def primary_by_family(rows,app,arm):
    chosen=[r for r in rows if r['app']==app and r['candidate']==arm and not r['absent']
            and r['stratum']!='native_zh_bridge']
    return {c:float(np.mean([r['scores']['task_success'] for r in chosen if r['case']==c]))
            for c in sorted({r['case'] for r in chosen})}


def decisions(rows, *, local_only=False):
    if any('criterion_reviews' in r for r in rows):
        rows=[{**r,'scores':score_row(r)} if 'criterion_reviews' in r else r for r in rows]
    checks={'resolved':all(r['scores']['resolved'] for r in rows),
            'real_providers':all(not r['proxy'] for r in rows),
            'no_critical_errors':not any(r['scores']['critical_error'] for r in rows if r['candidate']==ARMS[1])}
    summary={}
    for app in ('companion','tutor'):
        base,new=[primary_by_family(rows,app,arm) for arm in ARMS]
        if not base or set(base)!=set(new):
            raise ValueError('Missing paired application families')
        deltas=np.array([new[c]-base[c] for c in sorted(base)])
        draws=np.random.default_rng(0).choice(deltas,(GATES['bootstrap_draws'],len(deltas)),replace=True).mean(axis=1)
        tail=(1-GATES['confidence'])/2
        low,high=np.quantile(draws,[tail,1-tail]); gain=float(deltas.mean())
        summary[app]={'families':len(deltas),'identity':float(np.mean(list(base.values()))),
                      'scaled_focus':float(np.mean(list(new.values()))),'gain':gain,'paired_95_ci':[float(low),float(high)]}
        checks[app+'_benefit']=gain>=GATES['minimum_gain']-1e-12 and low>0
        absent=[r for r in rows if r['app']==app and r['absent']]
        rates=[np.mean([r['scores']['appropriate_abstention'] for r in absent if r['candidate']==a]) for a in ARMS]
        checks[app+'_abstention']=rates[1]>=GATES['abstention'] and rates[1]>=rates[0]
        breakdown={}
        for stratum in ('native_zh_bridge','native_en_application','adapted_zh_application'):
            group=[r for r in rows if r['app']==app and r['stratum']==stratum]
            abstention=[float(np.mean([r['scores']['appropriate_abstention'] for r in group
                                      if r['absent'] and r['candidate']==a])) for a in ARMS]
            family_values=[{c:float(np.mean([r['scores']['task_success'] for r in group
                                            if not r['absent'] and r['candidate']==a and r['case']==c]))
                            for c in sorted({r['case'] for r in group})} for a in ARMS]
            diffs=np.array([family_values[1][c]-family_values[0][c] for c in family_values[0]])
            draws=np.random.default_rng(0).choice(diffs,(GATES['bootstrap_draws'],len(diffs)),replace=True).mean(axis=1)
            breakdown[stratum]={'families':len(diffs), 'source_present_success':[
                float(np.mean(list(v.values()))) for v in family_values], 'abstention':abstention,
                'paired_95_ci':[float(v) for v in np.quantile(draws,[tail,1-tail])]}
            if stratum=='native_zh_bridge':
                checks[app+'_bridge_abstention']=abstention[1]>=GATES['abstention'] and abstention[1]>=abstention[0]
        summary[app]['strata']=breakdown
        for field,values in (('language',('en','zh')),('query_index',(0,1))):
            for value in values:
                selected=[r for r in rows if r['app']==app and not r['absent'] and r[field]==value
                          and r['stratum']!='native_zh_bridge']
                for metric in ('task_success','recall_at_6'):
                    means=[np.mean([r['scores'][metric] or 0 for r in selected if r['candidate']==a]) for a in ARMS]
                    checks[f'{app}_{field}_{value}_{metric}']=means[1]>=means[0]
        group=[r for r in rows if r['app']==app]
        latency=[shared.percentile95([r['application_ms'] for r in group if r['candidate']==a]) for a in ARMS]
        cost=[np.mean([float(r['generation_call']['accounted_usd']) for r in group if r['candidate']==a]) for a in ARMS]
        checks[app+'_application_latency']=latency[1]<=GATES['max_application_latency_ratio']*latency[0]
        if not local_only:
            checks[app+'_estimated_cost']=cost[1]<=GATES['max_estimated_generation_cost_ratio']*cost[0]
        summary[app].update(abstention=[float(v) for v in rates],latency_p95=latency,
                            cost_estimate=None if local_only else [float(v) for v in cost])
    tutor=[r for r in rows if r['tutor_judgment']]
    rates=[np.mean([r['scores']['tutor_useful'] for r in tutor if r['candidate']==a]) for a in ARMS]
    checks['tutor_usefulness']=rates[1]>=GATES['tutor_usefulness'] and rates[1]>=rates[0]
    pairs={}
    for row in rows:
        key=(row['case'],row['query_index'],row['absent'],row['sample'])
        pairs.setdefault(key,{})[row['candidate']]=row
    losses=[{'case':key[0],'query_index':key[1],'absent':key[2],'sample':key[3]}
            for key,pair in sorted(pairs.items())
            if pair[ARMS[0]]['scores']['task_success'] and not pair[ARMS[1]]['scores']['task_success']]
    # Token-based frozen-rate estimates are not an independently reconciled bill.
    result={'summary':summary,'paired_task_losses':losses,'gates':{k:bool(v) for k,v in checks.items()},
            'quality_passed':all(checks.values()),'default_activation':False,
            'billing_reconciled':False,
            **({'scope':'Development screening with pinned local generation; not default qualification',
                'hardware_cost_comparison':'unmeasured'} if local_only else {})}
    return _apply_criterion_review_gate(result,[(row,kind) for row in rows
        for kind in ('memory','tutor') if kind=='memory' or row['tutor_judgment']])


def validate_report(report,cases,protocol,*,development_sha=None,cfg=None):
    if report['schema']!=SCHEMA or report['protocol_sha256']!=protocol_sha(protocol):
        raise ValueError('Wrong application report identity')
    if any(report.get(k)!=v for k,v in qualification_identity(protocol).items()):
        raise ValueError('Application judge rubric/qualification identity changed')
    phase=report['phase']; subset=[c for c in cases if c['split']==phase]
    if phase not in ('development','evaluation'):
        raise ValueError('Not a complete development/evaluation report')
    if protocol.get('screening') and phase != 'development':
        raise ValueError('The local screening profile leaves evaluation families unopened')
    if phase=='evaluation' and (not development_sha or report['development_report_sha256']!=development_sha):
        raise ValueError('Raw development input hash required')
    if report['matrix']!=matrix(report['trials'],subset,protocol['samples']):
        raise ValueError('Inconsistent application matrix')
    if report['code_sha256']!=fingerprints():
        raise ValueError('Application report is not from the current source freeze')
    cfg=load_config() if cfg is None else cfg
    if any(report.get(k)!=v for k,v in runtime_identity(cfg, cjk_tokenizer=protocol['tokenizer']).items()):
        raise ValueError('Application runtime configuration changed')
    sources=Sources(protocol); prepared={}; recalls={}
    for r in report['trials']:
        validate_trial(r,cases,protocol,sources,cfg,prepared,recalls)
    validate_resources(report['resources'],subset,procedure=protocol.get('resource_procedure'))
    if report['decision']!=decisions(report['trials'],local_only=protocol.get('screening',False)):
        raise ValueError('Decision differs from raw outcomes')


def validate_trial(row,cases,protocol,sources,cfg,prepared=None,recalls=None):
    """Replay source/context only; never repeat generation or mutate a scored system."""
    prepared={} if prepared is None else prepared
    recalls={} if recalls is None else recalls
    case=next((c for c in cases if c['id']==row['case']),None)
    if (case is None or any(row[k]!=case[k] for k in ('app','stratum','language','split'))
            or type(row['absent']) is not bool or row['candidate'] not in ARMS
            or type(row['query_index']) is not int or row['query_index'] not in range(2)
            or type(row['sample']) is not int or row['sample'] not in range(protocol['samples'])):
        raise ValueError('Unknown/changed application trial key')
    key=(case['id'],row['candidate'],row['absent'])
    if key not in prepared: prepared[key]=prepare(case,sources,cfg,row['candidate'],row['absent'])
    system,evidence=prepared[key]
    if row['evidence']!=evidence: raise ValueError('Reported inventory/reference differs from actual source')
    rkey=key+(row['query_index'],)
    if rkey not in recalls:
        recalls[rkey]=copy.deepcopy(system).recall(case['queries'][row['query_index']])
    recalled=recalls[rkey]
    capture=row['capture']
    if (capture['query']!=case['queries'][row['query_index']]
            or capture['context']!=render(case['app'],recalled.fragments_used[:6])
            or capture['fragments']!=pam_export(recalled.fragments_used[:6])['fragments']
            or capture['recall_metadata']!=recalled.metadata):
        raise ValueError('Captured context differs from a real replay of the frozen source')
    memory=row['memory_judgment']
    if memory!=parse_memory(memory['raw'],memory['finish_reason'],memory['proxy'],contract_version=grading_version(protocol)):
        raise ValueError('Memory judgment differs from raw model output')
    tutor=row['tutor_judgment']
    if bool(tutor)!=('/mathdial/' in case['source']):
        raise ValueError('Missing/unexpected tutoring judgment')
    if tutor:
        verdict=parse_empath_verdict(tutor['rationale'],tutor['finish_reason'],strict=True,
                                    contract_version=grading_version(protocol))
        if (tutor['criteria']!={k.value:v for k,v in verdict.sub_criteria.items()}
                or tutor['parse_failed']!=verdict.parse_failed
                or any(tutor.get(k)!=verdict.contract_metadata().get(k) for k in ('contract_version','consistency_errors'))):
            raise ValueError('Tutor judgment differs from existing parser output')
    criterion_review_summary([(row,'memory')]+([(row,'tutor')] if tutor else []),protocol=protocol)
    completion=row['completion']; call=row['generation_call']
    if (not isinstance(row['response'],str) or not row['response'].strip()
            or row['response']!=completion['content'].strip() or not row['exchange_id']
            or completion!=call['completion'] or completion['provider']!=protocol['generator']['provider']
            or completion['model_id']!=protocol['generator']['model'] or completion['finish_reason']!='stop'
            or completion['tier']!=TaskTier.ADVANCED.value or call['cached'] or not call['usage_valid']
            or row['proxy']!=bool(call['proxy'] or memory['proxy'] or (tutor and tutor['proxy']))
            or not math.isfinite(row['application_ms']) or row['application_ms']<=0):
        raise ValueError('Response/provider provenance is incomplete or inconsistent')
    if row['scores']!=score_row(row): raise ValueError('Trial score differs from raw judgments')
    if protocol.get('screening'):
        validate_screening_call(call,protocol['generator'])
        validate_screening_call(row['memory_call'],protocol['judge'])
        if (row['memory_call']['completion']['content'] != memory['raw']
                or row['memory_call']['completion']['finish_reason'] != memory['finish_reason']):
            raise ValueError('Memory judgment differs from its actual completion')
        if tutor:
            validate_screening_call(row['tutor_call'],protocol['judge'])
            if (row['tutor_call']['completion']['content'].strip() != tutor['rationale']
                    or row['tutor_call']['completion']['finish_reason'] != tutor['finish_reason']):
                raise ValueError('Tutor judgment differs from its actual completion')


def validate_resources(resources,cases,*,procedure=None):
    rows=resources['measurements']
    expected={(c['id'],a,arm) for c in cases for a in (False,True) for arm in ARMS}
    complete_keys([(r['case'],r['absent'],r['arm']) for r in rows],expected)
    if resources['byte_cap']!=4096: raise ValueError('Capped workload changed configuration')
    actual_procedure = resources.get('procedure')
    if procedure is not None and actual_procedure != procedure:
        raise ValueError('Resource measurement procedure differs from local freeze')
    if actual_procedure not in (None, RESOURCE_V2):
        raise ValueError('Unknown capped resource measurement procedure')
    for r in rows:
        if (type(r['absent']) is not bool or len(r['runs'])!=5 or not 0<r['accounted_bytes']<=4096
                or type(r['retained_count']) is not int or r['retained_count']<=0
                or not math.isfinite(r['ingestion_ms']) or r['ingestion_ms']<=0
                or r['source_observations']<r['retained_count']
                or r['evicted_count']!=r['source_observations']-r['retained_count']
                or any(not math.isfinite(m[k]) or m[k]<=0 for m in r['runs'] for k in ('recall_ms','recall_peak'))):
            raise ValueError('Invalid or empty capped resource measurements')
        if actual_procedure == RESOURCE_V2:
            for run in r['runs']:
                samples=run.get('samples',[])
                if (len(samples)!=BATCH_RECALLS
                        or any(not math.isfinite(m[k]) or m[k]<=0 for m in samples for k in ('recall_ms','recall_peak'))
                        or run['recall_ms']!=sum(m['recall_ms'] for m in samples)/BATCH_RECALLS
                        or run['recall_peak']!=max(m['recall_peak'] for m in samples)):
                    raise ValueError('Batched resources differ from independent cold-call measurements')
    passed=resource_decision(rows,cases)
    if type(resources['passed']) is not bool or resources['passed']!=passed:
        raise ValueError('Capped resource verdict differs from measurements')
    return passed


def resource_decision(rows,cases):
    checks=[]
    for c in cases:
        for absent in (False,True):
            pair=[next(r for r in rows if (r['case'],r['absent'],r['arm'])==(c['id'],absent,a)) for a in ARMS]
            checks.extend(max(x[k] for x in pair[1]['runs'])<=GATES[limit]*max(x[k] for x in pair[0]['runs'])
                          for k,limit in (('recall_ms','max_recall_ratio'),('recall_peak','max_allocation_ratio')))
    return all(checks)


def prepare_resource_state(case,sources,cfg,arm,absent):
    started=time.perf_counter()
    system,evidence=prepare(case,sources,cfg,arm,absent)
    record={'case':case['id'],'absent':absent,'arm':arm,
            'accounted_bytes':evidence['accounted_bytes'], 'retained_count':evidence['retained_count'],
            'source_observations':evidence['source_observations'],
            'evicted_count':evidence['source_observations']-evidence['retained_count'],
            'ingestion_ms':(time.perf_counter()-started)*1000,'runs':[]}
    return system,record


def capped_resources(cases,sources,cfg,*,procedure=None):
    records=[]
    for c in cases:
        for absent in (False,True):
            if procedure == RESOURCE_V2:
                systems={}; pair={}
                for arm in ARMS:
                    system,record=prepare_resource_state(c,sources,cfg,arm,absent)
                    systems[arm]=system
                    measure_recall(copy.deepcopy(system),c['queries'][0])
                    pair[arm]=record
                offset=int(shared.stable_key([c['id'],absent])[:8],16)
                for round_index in range(5):
                    samples={arm:[] for arm in ARMS}
                    for repeat in range(BATCH_RECALLS):
                        order=ARMS if (offset+round_index+repeat)%2==0 else tuple(reversed(ARMS))
                        for arm in order:
                            _,m=measure_recall(copy.deepcopy(systems[arm]),c['queries'][0])
                            samples[arm].append(m)
                    for arm in ARMS:
                        pair[arm]['runs'].append({'samples':samples[arm],
                            'recall_ms':sum(m['recall_ms'] for m in samples[arm])/BATCH_RECALLS,
                            'recall_peak':max(m['recall_peak'] for m in samples[arm])})
                records.extend(pair.values())
                continue
            if procedure is not None:
                raise ValueError('Unsupported resource procedure')
            for arm in ARMS:
                system,record=prepare_resource_state(c,sources,cfg,arm,absent)
                for n in range(6):
                    _,m=measure_recall(copy.deepcopy(system),c['queries'][0])
                    if n: record['runs'].append(m)
                records.append(record)
        print(f'Completed capped resources {c["id"]}',flush=True)
    result={'byte_cap':4096,'measurements':records,'passed':resource_decision(records,cases)}
    if procedure is not None: result['procedure']=procedure
    validate_resources(result,cases,procedure=procedure)
    return result


def prepare_protocol(annotations_path):
    """Freeze authoring, raw sources, scoring rules and model/price identities."""
    from collections import Counter
    if (FIXTURES/'protocol.json').exists() or (FIXTURES/'catalog.json').exists():
        raise ValueError('Application protocol already exists; evidence is create-only')
    authored=read(annotations_path)
    catalog={'schema':SCHEMA,'cases':[catalog_case(c) for c in authored['cases']]}
    shared.write_new(FIXTURES/'catalog.json',catalog)
    protocol={'schema':SCHEMA,'status':'frozen-before-application-scoring','arms':list(ARMS),
              'byte_cap':4096,'samples':3,'seed':0,'top_k':6,'tokenizer':'jieba',
              'aging':'S1','consolidation':False,'witness':'independent-identity',
              'catalog_sha256':shared.digest(FIXTURES/'catalog.json'),
              'annotations':{'path':str(annotations_path.relative_to(ROOT)).replace('\\','/'),
                             'sha256':shared.digest(annotations_path)},
              'source_pins':authored['source_pins'],
              'scoring_prompt_sha256':shared.stable_key(MEMORY_PROMPT),
              'exclusion_audit_sha256':shared.digest(ROOT/'benchmarks/fixtures/scaled_context/fresh-audit.json'),
              'generator':{'provider':'tongyi','model':'qwen-max','input_rate':'1.6','output_rate':'6.4','max_input':12000,
                           'price_source':'https://www.alibabacloud.com/help/en/model-studio/model-pricing',
                           'price_kind':'conservative international USD upper rate; actual endpoint is China',
                           'revision_kind':'mutable provider alias, no immutable revision returned'},
              'judge':{'provider':'openai','model':'gpt-4o-mini-2024-07-18','input_rate':'0.15','output_rate':'0.60','max_input':12000,
                       'price_source':'https://developers.openai.com/api/docs/models/gpt-4o-mini',
                       'revision_kind':'dated snapshot'},
              'prices_checked_on':'2026-10-19','budget':{'max_calls':4200,'max_usd':'50'},
              'gates':dict(GATES),
              'adaptation_scope':authored['translation_scope']}
    shared.write_new(FIXTURES/'protocol.json',protocol)
    p,c=load_inputs()
    sources=Sources(p)
    math_problems=[]
    for case in c:
        turns,gold,absent,_,_,problem=sources.materialize(case)
        if problem:
            normalized=re.sub(r'\s+',' ',problem).strip().casefold()
            if normalized in math_problems: raise ValueError('Repeated MathDial source problem')
            math_problems.append(normalized)
        # Review records enumerate real removed/kept turns; no source labels
        # are inferred from candidate outputs.
    return {'schema':SCHEMA,'phase':'prepare','families':len(c),'cases_scored':0,
            'counts':dict(Counter((x['app']+'/'+x['stratum']+'/'+x['split']) for x in c)),
            'protocol_sha256':shared.digest(FIXTURES/'protocol.json'),
            'reserved_old_holdout_scored':0,'default_activation':False}


def make_routers(protocol,cfg,budget,*,controls=False,roles=('generator','judge')):
    # Frozen evaluations use their existing serial RecordingRouter/Budget
    # contract, not serving policy or a second runtime reservation.
    from psychology_base.llm.execution import ExecutionPolicy
    roles=tuple(roles)
    if not roles or len(set(roles))!=len(roles) or any(role not in ('generator','judge') for role in roles):
        raise ValueError('Evaluation router roles must be a nonempty distinct generator/judge selection')
    cfg=replace(cfg,execution=ExecutionPolicy(mode='research'))
    wrappers=[]
    for role in roles:
        spec=protocol[role]; name=spec['provider']
        if name == 'ollama' and protocol.get('screening'):
            selected=replace(cfg.get_provider('ollama'),base_url=spec['base_url'],
                             simple_model=spec['model'],advanced_model=spec['model'],
                             request_timeout_s=spec.get('chat_timeout_s',cfg.get_provider('ollama').request_timeout_s))
            current=replace(cfg,active_provider='ollama',providers={'ollama':selected},
                            local_ollama=replace(cfg.local_ollama,enabled=False),cache=CacheConfig(False))
            router=LLMRouter(current)
            router.providers['ollama']=replace(router.providers['ollama'],local_only=True,
                expected_digest=spec['digest'],context_length=spec['context_length'],seed=spec['seed'],
                json_mode=spec['json_mode'],model_metadata_timeout_s=spec.get(
                    'model_metadata_timeout_s',router.providers['ollama'].model_metadata_timeout_s))
            wrappers.append(RecordingRouter(router,budget,controls=controls,local_spec=spec,
                            **{k:spec[k] for k in ('provider','model','input_rate','output_rate','max_input')}))
            continue
        selected=replace(cfg.get_provider(name),advanced_model=spec['model'],max_retries=0)
        current=replace(cfg,active_provider=name,providers=({name:selected} if protocol.get('screening') else {**cfg.providers,name:selected}),
                        local_ollama=replace(cfg.local_ollama,enabled=False),
                        cache=CacheConfig(role=='judge' and not protocol.get('screening'),ROOT/'.cache/application-context-llm'))
        if protocol.get('screening') and name == 'tongyi':
            from psychology_base.llm.providers.openai import OpenAIProvider
            hosted_request_options(spec)
            provider=OpenAIProvider(name=name,simple_model=spec['model'],advanced_model=spec['model'],
                api_key=selected.api_key(),base_url=spec['base_url'],max_retries=0,
                enable_thinking=spec['enable_thinking'],json_mode=spec['json_mode'],
                response_schema=spec.get('response_schema'),
                total_completion_limit=spec.get('total_completion_limit',False),
                request_timeout_s=spec['chat_timeout_s'],follow_redirects=False)
            router=LLMRouter(current,providers={name:provider})
        else:
            router=LLMRouter(current)
        if protocol.get('screening') and name != 'tongyi':
            router.providers[name]=replace(router.providers[name],reasoning_effort=spec['reasoning_effort'],
                                           json_mode=spec['json_mode'])
        wrappers.append(RecordingRouter(router,budget,controls=controls,
                        remote_spec=spec if protocol.get('screening') else None,
                        **{k:spec[k] for k in ('provider','model','input_rate','output_rate','max_input')}))
    return wrappers


@contextmanager
def study_budget(protocol):
    """One ledger/lock for all stages; a crash reservation never becomes free."""
    directory=ROOT/'.data/application_context'
    directory.mkdir(parents=True,exist_ok=True)
    lock=directory/'provider-budget.lock'
    handle=lock.open('x',encoding='utf-8')
    try:
        with SessionLogWriter(directory/'provider-events.jsonl') as writer:
            yield Budget(writer,**protocol['budget'])
    finally:
        handle.close()
        lock.unlink()


def calibration_items(protocol,sources):
    """One construction owner for execution and revalidation of source controls."""
    from psychology_base.applications.measurement.socratic_compliance import ComplianceLabel
    from psychology_base.eval.benchmarks.public_datasets.mathdial_empath_scorer import gold_mathdial_to_compliance
    _,cases=load_inputs()
    used={c['row'] for c in cases if '/msc/' in c['source']}
    revision=protocol.get('calibration_revision',1)
    if grading_version(protocol)==2 and revision!=4:
        raise ValueError('Revised grading requires fresh version-4 qualification controls')
    fresh=revision >= 2
    previous_memory=[]; previous_tutor=[]
    excluded_sources=set(); excluded_text=set(); excluded_problems=set(); excluded_math_ids=set()
    if fresh:
        original,_=load_inputs()
        old_memory,old_tutor=calibration_items(original,sources)
        old_smoke=smoke_cases(original,sources)
        excluded_sources={r['source'] for r in old_memory}
        excluded_text={' '.join(r['reference'].casefold().split()) for r in old_memory}
        excluded_problems={' '.join(r['problem'].casefold().split()) for r in old_tutor}
        excluded_problems.update(' '.join(sources.data[c['source']][c['row']].problem.casefold().split())
                                 for c in old_smoke if '/mathdial/' in c['source'])
        excluded_math_ids={r['source'] for r in old_tutor}
        excluded_math_ids.update(sources.data[c['source']][c['row']].conversation_id
                                 for c in cases+old_smoke if '/mathdial/' in c['source'])
        if revision == 3:
            previous_memory,previous_tutor=calibration_items({**original,'calibration_revision':2},sources)
            excluded_problems.update(' '.join(r['problem'].casefold().split()) for r in previous_tutor)
            excluded_math_ids.update(r['source'] for r in previous_tutor)
        if revision==4:
            for prior_revision in (2,3):
                prior_memory,prior_tutor=calibration_items({**original,'calibration_revision':prior_revision},sources)
                excluded_sources.update(r['source'] for r in prior_memory)
                excluded_text.update(' '.join(r['reference'].casefold().split()) for r in prior_memory)
                excluded_problems.update(' '.join(r['problem'].casefold().split()) for r in prior_tutor)
                excluded_math_ids.update(r['source'] for r in prior_tutor)
    salt=f'application-calibration-v{revision}' if fresh else 'application-calibration'
    msc=sources.data['.data/msc/train.jsonl']; memory=[]
    candidates=[]; candidate_rows={}
    anchors={r['source']:r for r in read(FIXTURES/'qualification-memory-anchors-v4.json')['anchors']} if revision==4 else {}
    for idx,convo in ([] if revision==3 else sorted(enumerate(msc),key=lambda x:shared.stable_key([salt,x[1].conversation_id]))):
        if idx in used or convo.conversation_id in excluded_sources: continue
        if revision==4:
            anchor=anchors.get(convo.conversation_id)
            if anchor is None: continue
            turn=convo.sessions[0].turns[anchor['source_turn']]
            if (anchor['source_row']!=idx or turn.speaker_idx!=0
                    or shared.stable_key(turn.utterance)!=anchor['reference_sha256']):
                raise ValueError('Reviewed qualification anchor differs from the real user statement')
            text=turn.utterance
        else:
            text=next((t.utterance for t in convo.sessions[0].turns if t.speaker_idx==0 and len(t.utterance)>32
                       and '?' not in t.utterance and 'i ' in t.utterance.casefold()),None)
        if text and (not fresh or ' '.join(text.casefold().split()) not in excluded_text):
            candidates.append((convo.conversation_id,text))
            candidate_rows[convo.conversation_id]=idx
            if fresh:
                excluded_text.add(' '.join(text.casefold().split()))
                excluded_sources.add(convo.conversation_id)
        if len(candidates)==12: break
    if revision!=3 and len(candidates)!=12: raise ValueError('Insufficient disjoint memory calibration sources')
    for n,(cid,text) in enumerate(candidates):
        for absent in (False,True):
            quoted=(f'你之前说过：“{text}”。你接下来想聊哪一部分？' if n>=6 else f'You previously said: "{text}". What would you like to explore?')
            refusal=('我没有你之前说过的那个细节。你愿意再告诉我一点吗？' if n>=6 else
                     "I do not have that earlier detail. Could you tell me a little more?")
            reversal=n%6 in (2,4)
            response=(refusal if absent else quoted) if reversal else (quoted if n%2==0 else refusal)
            capture={'query':'你记得我之前说过自己的什么事情吗？' if n>=6 else 'What do you remember I said about myself?',
                     'context':'' if absent else ('assistant said: ' if reversal else 'user said: ')+text}
            expected=absent if reversal or n%2 else not absent
            unsupported=(reversal and not absent) or (not reversal and absent and n%2==0)
            memory.append({'source':cid,'source_absent':absent,'authored_control':True,'language':'zh' if n>=6 else 'en',
                           'control_kind':'speaker_reversal' if reversal else ('quoted_fact' if n%2==0 else 'refusal'),
                           'expected_success':expected,
                           'expected_critical':{'grounded':not unsupported,
                                                'attribution_correct':not(reversal and not absent),
                                                'critical_error':unsupported},
                           'capture':capture,'response':response,'reference':text})
            if fresh:
                row=candidate_rows[cid]
                memory[-1].update(source_row=row,source_turn=(anchors[cid]['source_turn'] if revision==4 else
                    next(i for i,t in enumerate(msc[row].sessions[0].turns) if t.speaker_idx==0 and t.utterance==text)))
    if revision==3:
        memory=previous_memory
    used_problems={sources.data[c['source']][c['row']].problem for c in cases if '/mathdial/' in c['source']}
    if fresh:
        excluded_problems.update(' '.join(p.casefold().split()) for p in used_problems)
    md=sources.data['.data/mathdial/train.jsonl']; tutor=[]; seen=set()
    for source_row,convo in sorted(enumerate(md),key=lambda pair:shared.stable_key([salt,pair[1].problem])):
        problem_key=' '.join(convo.problem.casefold().split()) if fresh else convo.problem
        if (convo.problem in used_problems or problem_key in seen or problem_key in excluded_problems
                or (fresh and convo.conversation_id in excluded_math_ids)): continue
        seen.add(problem_key)
        n=len(tutor)
        if n<12:
            expected=ComplianceLabel.SOCRATIC if n%2==0 else ComplianceLabel.DIRECT_ANSWER
            context=next((c for c in iter_mathdial_teacher_contexts(convo)
                          if gold_mathdial_to_compliance(c.turn.move)==expected),None)
            if context is None: continue
            turn=context.turn
            text=turn.utterance; gold=gold_mathdial_to_compliance(turn.move); origin='native_mathdial_move'
        else:
            from psychology_base.eval.benchmarks.public_datasets.mathdial_fade_harness import extract_final_number
            final=extract_final_number(convo.ground_truth)
            if final is None: continue
            text='请解释一下你为什么选择这一步，好吗？' if n%2==0 else f'最终答案就是{final}。'
            gold=ComplianceLabel.SOCRATIC if n%2==0 else ComplianceLabel.DIRECT_ANSWER
            origin='authored_chinese_control_on_disjoint_problem'
        tutor.append({'source':convo.conversation_id,'origin':origin,'gold':gold.value,'response':text,'problem':convo.problem})
        if fresh:
            tutor[-1].update(source_row=source_row,source_turn=context.turn_index if n<12 else None)
            excluded_math_ids.add(convo.conversation_id)
        if revision>=3:
            tutor[-1].update(student_utterance=context.student_utterance if n<12 else '',
                             native_move=turn.move.value if n<12 else None)
        if len(tutor)==24: break
    if len(tutor)!=24: raise ValueError('Insufficient disjoint tutor calibration')
    if fresh and 'qualification' in protocol:
        spec=protocol['qualification']
        if (shared.digest(ROOT/spec['path'])!=spec['sha256']
                or read(ROOT/spec['path'])!=calibration_catalog(memory,tutor,protocol=protocol)):
            raise ValueError('Qualification controls differ from their pre-model freeze')
    return memory,tutor


def calibration_catalog(memory,tutor,*,protocol=None):
    """Publish control identities/hashes while source-bearing text stays local."""
    revised=any('student_utterance' in r for r in tutor)
    return {'schema':('application-context-qualification-v4' if grading_version(protocol)==2 else
                      'application-context-qualification-v3' if revised else 'application-context-qualification-v2'),
            **grading_identity(protocol or {}),
            **({'memory_anchors_sha256':shared.digest(FIXTURES/'qualification-memory-anchors-v4.json')}
               if grading_version(protocol)==2 else {}),
            'memory_controls':[{**{k:r[k] for k in ('source','source_absent','authored_control','language',
                                                   'control_kind','expected_success','expected_critical','source_row','source_turn')},
                                'control_sha256':shared.stable_key(r)} for r in memory],
            'tutor_controls':[{**{k:r[k] for k in ('source','origin','gold','source_row','source_turn')},
                               'control_sha256':shared.stable_key(r),
                               **({'native_move':r['native_move'],
                                   'student_context_sha256':shared.stable_key(r['student_utterance'])} if revised else {})} for r in tutor]}


def memory_control_success(item,result):
    c=result['criteria']
    return (judgment_resolved(result) and bool(c.get('grounded') and c.get('attribution_correct')
            and not c.get('critical_error') and
            (c.get('abstains_appropriately') if item['source_absent'] else c.get('addresses_request'))))


def calibration_sentinels(memory,tutor):
    """Eight predetermined language/outcome strata from the frozen controls."""
    selected=[]
    for language in ('en','zh'):
        for absent in (False,True):
            selected.append(('memory',next(i for i,r in enumerate(memory)
                if r['language']==language and r['source_absent']==absent)))
        for gold in ('socratic','direct_answer'):
            selected.append(('tutor',next(i for i,r in enumerate(tutor)
                if ('zh' if r['origin']=='authored_chinese_control_on_disjoint_problem' else 'en')==language
                and r['gold']==gold)))
    return selected


def repeatability_status(memory,tutor,repeated):
    """Compare all criteria; retain disagreement instead of voting it away."""
    failed=False
    if len(memory)==len(tutor)==24:
        selected=calibration_sentinels(memory,tutor)
        expected=[(kind,index,repeat) for repeat,order in ((2,selected),(3,list(reversed(selected)))) for kind,index in order]
        keys=[(r['kind'],r['index'],r['repeat']) for r in repeated]
        if len(keys)>16 or keys!=expected[:len(keys)]:
            raise ValueError('Repeated controls differ from the frozen sentinel schedule')
        for row in repeated:
            primary=(memory if row['kind']=='memory' else tutor)[row['index']]
            before=primary['judgment'] if row['kind']=='memory' else primary
            after=row['assessment']['judgment'] if row['kind']=='memory' else row['assessment']
            failed |= (not judgment_resolved(after,tutor=row['kind']=='tutor')
                       or before['criteria']!=after['criteria'])
    elif repeated:
        raise ValueError('Repeats cannot precede completion of primary controls')
    return {'procedure':'eight-sentinels-three-repeats-v1','expected_additional_judgments':16,
            'completed_additional_judgments':len(repeated),'complete':len(repeated)==16,
            'irreversible_failure':bool(failed),'passed':len(repeated)==16 and not failed}


def calibration_metrics(memory,tutor,*,repeated=None):
    from psychology_base.applications.measurement.socratic_compliance import ComplianceLabel, cohen_kappa
    pairs=[(ComplianceLabel(r['predicted']),ComplianceLabel(r['gold'])) for r in tutor]
    kappa=cohen_kappa([a for a,b in pairs],[b for a,b in pairs]).kappa if pairs else None
    agreement=sum(r['expected_success']==r['observed_success'] for r in memory)/len(memory) if memory else None
    critical={key:sum(r['expected_critical'][key]==r['judgment']['criteria'].get(key) for r in memory)/len(memory)
              for key in ('grounded','attribution_correct','critical_error')}
    passed=(len(memory)==24 and len(tutor)==24 and agreement>=.90
            and all(v>=.90 for v in critical.values()) and kappa>=.60 and
            all(judgment_resolved(r['judgment']) for r in memory) and all(judgment_resolved(r,tutor=True) for r in tutor))
    prospective=any(r['judgment'].get('contract_version')==2 for r in memory) or any(r.get('contract_version')==2 for r in tutor)
    status=repeatability_status(memory,tutor,repeated or []) if prospective else None
    return {'memory_agreement':agreement,'memory_critical_agreement':critical,'tutor_kappa':kappa,
            'passed':passed and (status is None or status['passed']),
            **({'accuracy_passed':passed,'repeatability':status} if status is not None else {})}


def calibration_failure_is_final(memory,tutor):
    """Stop only when the fixed 24+24-control qualification is impossible."""
    if any(not judgment_resolved(r['judgment']) for r in memory) or any(not judgment_resolved(r,tutor=True) for r in tutor):
        return True
    if sum(r['expected_success']!=r['observed_success'] for r in memory)>2:
        return True
    if any(sum(r['expected_critical'][key]!=r['judgment']['criteria'].get(key) for r in memory)>2
           for key in ('grounded','attribution_correct','critical_error')):
        return True
    if len(memory)==len(tutor)==24 and any(r.get('contract_version')==2 for r in tutor):
        return not calibration_metrics(memory,tutor)['accuracy_passed']
    return False


def _calibration_review_records(memory,tutor,repeated=()):
    return ([(r,'memory') for r in memory]+[(r,'tutor') for r in tutor]
            +[(r['assessment'],r['kind']) for r in repeated])


def _calibration_needs_review(memory,tutor,repeated=()):
    summary=criterion_review_summary(_calibration_review_records(memory,tutor,repeated))
    return bool(summary and summary['status']=='needs_review')


def reviewed_calibration_metrics(memory,tutor,*,repeated=None,protocol=None,expected_review_manifest_sha256=None):
    """Apply the review gate to the existing, unchanged raw agreement statistics."""
    if not memory:
        raise ValueError('Review qualification requires memory controls')
    return _apply_criterion_review_gate(calibration_metrics(memory,tutor,repeated=repeated),
        _calibration_review_records(memory,tutor,repeated or ()),protocol=protocol,
        expected_manifest_sha256=expected_review_manifest_sha256)


def reused_memory_controls(protocol,sources,cfg=None):
    """Carry forward verified memory evidence, never the failed tutoring result.

    Only the three MathDial/context dispatch owners may differ. The memory
    prompt, parser, scoring functions, provider owners, raw controls, completed
    calls and runtime remain tied to the original tested freeze.
    """
    spec=protocol['memory_evidence']
    for name in ('report','freeze'):
        if shared.digest(ROOT/spec[name])!=spec[name+'_sha256']:
            raise ValueError('Reused memory evidence changed')
    report=read(ROOT/spec['report']); freeze=read(ROOT/spec['freeze'])
    old=local_profile(FIXTURES/'hybrid-screen-v2.json',load_inputs()[0])
    if (report['code_sha256']!=freeze['code_sha256'] or report['profile']!=freeze['profile']
            or protocol_sha(old)!=freeze['profile_sha256']
            or any(old[k]!=protocol[k] for k in ('judge','generator','budget','memory_prompt','source_pins'))
            or (cfg is not None and any(report.get(k)!=v for k,v in runtime_identity(cfg, cjk_tokenizer=protocol['tokenizer']).items()))):
        raise ValueError('Reused memory protocol/provider/runtime differs from its tested freeze')
    current=fingerprints(); prior=freeze['code_sha256']
    allowed={'src/psychology_base/eval/benchmarks/'+p for p in
             ('application_context.py','public_datasets/mathdial.py','public_datasets/mathdial_empath_scorer.py')}
    if set(current)!=set(prior) or any(current[p]!=prior[p] for p in current if p not in allowed):
        raise ValueError('Memory/provider owners changed; prior qualification cannot be reused')
    owner='src/psychology_base/eval/benchmarks/application_context.py'
    archive=ROOT/freeze['snapshot_directory']/owner
    if shared.digest(archive)!=prior[owner]:
        raise ValueError('Prior memory owner archive changed')
    def functions(path):
        return {node.name:ast.dump(node,include_attributes=False)
                for node in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(node,ast.FunctionDef)}
    before,after=functions(archive),functions(Path(__file__))
    if (any(before[name]!=after[name] for name in ('judge_memory','parse_memory','memory_control_success',
                                                  'calibration_metrics','calibration_failure_is_final'))
            or freeze['judge_scoring_prompt_sha256']!=shared.stable_key(protocol['memory_prompt'])):
        raise ValueError('Memory scoring contract changed; prior qualification cannot be reused')
    # Reparse the complete old report using the same validation owner. Its
    # tutoring gate stays failed; only independently passing memory is carried.
    validate_calibration(report,old,sources,expected_code=prior)
    memory=report['memory_controls']
    metrics=calibration_metrics(memory,[])
    if (len(memory)!=24 or metrics['memory_agreement']<.90
            or any(v<.90 for v in metrics['memory_critical_agreement'].values())
            or calibration_failure_is_final(memory,[])):
        raise ValueError('Prior memory judgments do not qualify')
    return copy.deepcopy(memory)


def assess_calibration_control(kind,item,protocol,judge):
    """The same invocation for primary controls and their repetitions."""
    if kind=='memory':
        if 'runtime_memory_contract_version' in protocol:
            result=judge_memory(judge,{'app':item['app']},None,item['capture'],item['response'],None,protocol=protocol)
            interpreted=(judge.last_memory_interpretation if protocol['runtime_memory_contract_version']==4 else None)
            evidence=runtime_control_evidence(item,protocol,
                interpretation=interpreted['interpretation'] if interpreted is not None else None)
            row={'judgment':result,'observed_success':runtime_memory_resolved(result,evidence=evidence) and result['decision']=='accept',
                 'call':copy.deepcopy(judge.last_call)}
            if interpreted is not None:
                row.update(interpretation=copy.deepcopy(interpreted),calls=copy.deepcopy(judge.last_memory_calls))
            if protocol['runtime_memory_contract_version'] in (2,3,4):
                row['disposition']=runtime_memory_action(result,evidence=evidence)
            return row
        result=judge_memory(judge,{'app':'companion'},item['source_absent'],item['capture'],item['response'],item['reference'],protocol=protocol)
        return {'judgment':result,'observed_success':memory_control_success(item,result),'call':copy.deepcopy(judge.last_call)}
    verdict=judge_tutor(judge,item['response'],problem_context=item['problem'],
                        student_utterance=item.get('student_utterance',''),protocol=protocol)
    return {'predicted':verdict.label.value,'parse_failed':verdict.parse_failed,'rationale':verdict.rationale,
            'finish_reason':judge.last_completion.finish_reason,'call':copy.deepcopy(judge.last_call),
            **verdict.contract_metadata(),
            **({'criteria':{k.value:v for k,v in verdict.sub_criteria.items()}} if verdict.contract_version==2 else {})}


def _runtime_qualification_spec():
    path=ROOT/'docs/experiments/phase-9-application-context/runtime-verifier-qualification-spec-v1.json'
    if shared.digest(path)!='1ea1165a5a9b6ad82cbeb404b3d6f5f661533c7856f120ecede815e269cce116':
        raise ValueError('Runtime qualification requirements changed after their freeze')
    return read(path)


def runtime_qualification_items(anchors_path=FIXTURES/'runtime-qualification-anchors-v1.json'):
    """Build disclosed authored controls from pinned real, unallocated MSC turns."""
    bank=read(anchors_path)
    if bank.get('schema')!='runtime-qualification-anchors-v1':
        raise ValueError('Unknown runtime qualification anchor schema')
    for name in ('source_pin','excluded_source_pin','provisioner'):
        if shared.digest(ROOT/bank[name]['path'])!=bank[name]['sha256']:
            raise ValueError('Runtime qualification source/provisioner pin changed')
    source=list(iter_msc_conversations(ROOT/bank['source_pin']['path']))
    old=list(iter_msc_conversations(ROOT/bank['excluded_source_pin']['path']))
    normalized=lambda text:' '.join(text.casefold().split())
    excluded_ids={row.conversation_id for row in old}
    excluded_text={normalized(t.utterance) for row in old for s in row.sessions for t in s.turns}
    initial=lambda row:shared.stable_key([(t.speaker_idx,t.utterance) for t in row.sessions[0].turns])
    chains={initial(row) for row in old}
    persona_pair=lambda row:shared.stable_key(sorted(sorted(p) for p in row.personas))
    personas={persona_pair(row) for row in old}
    by_id={row.conversation_id:row for row in source}
    if len(by_id)!=len(source) or len(source)!=16 or len(bank['anchors'])!=16:
        raise ValueError('Runtime qualification needs sixteen distinct real source families')
    seen=set(); anchors_seen=set(); controls=[]
    safe=dict(addresses_request=True,grounded=True,attribution_correct=True,abstains_appropriately=True,critical_error=False)
    for anchor in bank['anchors']:
        cid=anchor['source']
        if cid in seen or cid in excluded_ids or cid not in by_id:
            raise ValueError('Reused or unknown runtime qualification source family')
        row=by_id[cid]; turn=row.sessions[anchor['source_session']].turns[anchor['source_turn']]
        original_chain=hashlib.sha256(json.dumps([{'speaker_idx':t.speaker_idx,'utterance':t.utterance}
            for t in row.sessions[0].turns],sort_keys=True).encode()).hexdigest()
        if (turn.speaker_idx!=0 or shared.stable_key(turn.utterance)!=anchor['reference_sha256']
                or normalized(turn.utterance) in excluded_text|anchors_seen or initial(row) in chains
                or original_chain!=anchor['initial_chain_sha256'] or persona_pair(row) in personas):
            raise ValueError('Runtime qualification anchor differs from a disjoint direct user statement')
        seen.add(cid); anchors_seen.add(normalized(turn.utterance)); chains.add(initial(row)); personas.add(persona_pair(row))
        lang=anchor['language']; app=anchor['app']; other='tutor' if app=='companion' else 'companion'
        text=anchor['adaptation'] if lang=='zh' else turn.utterance
        if not isinstance(text,str) or not text.strip() or (lang=='en' and anchor['adaptation'] is not None):
            raise ValueError('Runtime qualification adaptation is missing or undeclared')
        zh=lang=='zh'; contrast=anchor['contrast']
        query=f'关于“{text}”，这些记录实际表明谁说过什么？' if zh else f'What do these records establish about who said "{text}"?'
        records=[{'speaker':'user','session':'source','text':text}]
        reply=f'你之前说过：“{text}”。' if zh else f'You previously said: "{text}".'
        if contrast==2:
            records=[{'speaker':app,'session':'source','text':f'{other} said: "{text}"'}]
            reply=(f'{app} 的记录转述 {other} 说过：“{text}”；这里没有 {other} 的直接发言。' if zh else
                   f'The {app} record reports that {other} said "{text}"; there is no direct {other} statement here.')
        if anchor['needless_refusal']:
            reply='我不讨论之前的消息。' if zh else 'I will not discuss earlier messages.'
        def add(kind,visible,response,decision,criteria):
            controls.append(dict(id='rtq-'+shared.stable_key([cid,lang,app,kind])[:16],source=cid,
                source_session=anchor['source_session'],source_turn=anchor['source_turn'],
                language=lang,app=app,authored_control=True,control_type=kind,
                capture={'query':query,'context':'\n'.join(json.dumps(r,ensure_ascii=False) for r in visible)},
                response=response,expected_decision=decision,expected_criteria=criteria))
        add('needless_refusal' if anchor['needless_refusal'] else 'useful',records,reply,'accept',
            {**safe,**({'addresses_request':False,'abstains_appropriately':False} if anchor['needless_refusal'] else {})})
        bad=f'你之前说过：“{text}”。' if zh else f'You previously said: "{text}".'
        if contrast==0:
            records=[{'speaker':other,'session':'source','text':text}]
        elif contrast==1:
            records=[{'speaker':None,'session':'source','text':text}]
        elif contrast==2:
            records=[{'speaker':app,'session':'source','text':f'{other} said: "{text}"'}]
            bad=f'{other} 之前说过：“{text}”。' if zh else f'{other} previously said: "{text}".'
        elif contrast==3:
            records=[{'speaker':other,'session':'source','text':f'user said: {text}\nuser said: Confirmed.'}]
        else:
            raise ValueError('Unknown authored runtime qualification contrast')
        add('unsupported',records,bad,'reject',{**safe,'grounded':False,'critical_error':True,
            'attribution_correct':contrast!=0,'abstains_appropriately':False})
        if anchor['ambiguous_contrast']:
            question=f'你之前说过“{text}”吗？' if zh else f'Did you previously say "{text}"?'
            query=f'你之前问过“{question}”吗？' if zh else f'Did you ask the question "{question}"?'
            records=[{'speaker':other,'session':'source','text':question}]
            reply=f'是的。{other} 问过这个问题。' if zh else f'Yes. The {other} asked that question.'
            add('ambiguous',records,reply,'inconclusive',None)
    runtime_qualification_schedule(controls)
    return controls


def _runtime_qualification_contract(qualification_version):
    if type(qualification_version) is not int or qualification_version not in (1,2):
        raise ValueError('Runtime qualification version must be 1 or 2')
    return (runtime_memory_protocol() if qualification_version==1 else
            runtime_memory_protocol(contract_version=2,prompt_revision=4))


def _runtime_qualification_exposure(controls):
    bank=read(FIXTURES/'runtime-decision-qualification-v1.json')
    exposed=set(bank.get('exposed_source_families',[]))
    if (bank.get('schema')!='runtime-decision-qualification-exposure-v1' or exposed!={'msc_456'}
            or bank.get('fresh_control_ids')!=sorted(row['id'] for row in controls if row['source'] not in exposed)
            or bank.get('exposed_control_ids')!=sorted(row['id'] for row in controls if row['source'] in exposed)
            or bank.get('fresh_primary_total')!=38 or bank.get('exposed_development_primary_total')!=2):
        raise ValueError('Runtime qualification exposure allocation changed')
    return exposed


def runtime_qualification_schedule(controls,*,qualification_version=1):
    """One frozen schedule for construction, execution, early stop and replay."""
    _runtime_qualification_contract(qualification_version)
    spec=_runtime_qualification_spec(); expected=spec['schedule']
    kinds={'useful':('accept',12),'needless_refusal':('accept',4),'unsupported':('reject',16),'ambiguous':('inconclusive',8)}
    ids=[row['id'] for row in controls]
    if len(ids)!=expected['primary'] or len(set(ids))!=len(ids):
        raise ValueError('Runtime qualification catalog has missing or duplicated controls')
    families={}; strata=Counter()
    for row in controls:
        kind=row['control_type']; label=row['expected_decision']; flags=row['expected_criteria']
        if kind not in kinds or label!=kinds[kind][0] or row.get('authored_control') is not True:
            raise ValueError('Runtime qualification control label/type changed')
        if row['language'] not in ('en','zh') or row['app'] not in ('companion','tutor'):
            raise ValueError('Runtime qualification has an unsupported stratum')
        if kind=='ambiguous':
            valid=flags is None
        else:
            valid=(isinstance(flags,dict) and not memory_consistency_errors(flags)
                and flags['grounded']==(kind!='unsupported') and flags['critical_error']==(kind=='unsupported')
                and flags['addresses_request']==(kind!='needless_refusal')
                and flags['abstains_appropriately']==(kind=='useful'))
        if not valid:
            raise ValueError('Runtime qualification criterion gold is inconsistent with its control')
        family=(row['language'],row['app'])
        if row['source'] in families and families[row['source']]!=family:
            raise ValueError('Adapted/reused source cannot supply another independent stratum')
        families[row['source']]=family; strata[(family,kind)]+=1
    if (Counter(row['control_type'] for row in controls)!={k:v[1] for k,v in kinds.items()}
            or len(families)!=16 or Counter(families.values())!={(lang,app):4 for lang in ('en','zh') for app in ('companion','tutor')}
            or any(strata[((lang,app),kind)]!=count//4 for lang in ('en','zh') for app in ('companion','tutor') for kind,(_,count) in kinds.items())):
        raise ValueError('Runtime qualification source/label balance changed')
    primary=sorted(ids); random.Random(0).shuffle(primary)
    exposed=_runtime_qualification_exposure(controls) if qualification_version==2 else set()
    sentinel_pool=[row for row in controls if row['source'] not in exposed]
    sentinels=sorted(min(row['id'] for row in sentinel_pool if (row['language'],row['app'],row['expected_decision'])==(lang,app,label))
        for lang in ('en','zh') for app in ('companion','tutor') for label in ('accept','reject','inconclusive'))
    sequence=[(cid,1) for cid in primary]+[(cid,2) for cid in sentinels]+[(cid,3) for cid in reversed(sentinels)]
    return [dict(index=i,control=cid,repeat=repeat) for i,(cid,repeat) in enumerate(sequence)]


def runtime_qualification_metrics(controls,rows,*,qualification_version=1):
    spec=_runtime_qualification_spec(); schedule=runtime_qualification_schedule(controls,qualification_version=qualification_version)
    if [{k:row[k] for k in ('index','control','repeat')} for row in rows]!=schedule[:len(rows)]:
        raise ValueError('Runtime qualification result schedule changed')
    by_id={row['id']:row for row in controls}; primary=rows[:40]; repeated=rows[40:]
    bad_format=0; false_accepts=0; rejections=0; unsupported_unclear=0; useful=0; refused=0; ambiguous=0
    failures=[]; critical={key:0 for key in ('grounded','attribution_correct','critical_error')}
    matched={key:0 for key in critical}; assessed=0
    exposed=_runtime_qualification_exposure(controls) if qualification_version==2 else set()
    fresh_completed=0; fresh_matches=0; exposed_completed=0; exposed_matches=0
    strata={key:{} for key in ('language','app','source')}
    for item in controls:
        for key,groups in strata.items():
            group=groups.setdefault(item[key],dict(total=0,completed=0,false_accepts=0,
                decisions={label:0 for label in ('accept','reject','inconclusive','unresolved')}))
            group['total']+=1
    first={}
    def valid(result):
        try:
            return (result==parse_runtime_memory(result['raw'],result['finish_reason'],result['proxy'],
                                                contract_version=result.get('runtime_memory_contract_version',1))
                and result.get('parse_failed') is False and result.get('consistency_errors')==[])
        except (KeyError,TypeError,ValueError):
            return False
    for row in primary:
        item=by_id[row['control']]; result=row['judgment']; kind=item['control_type']
        coherent=valid(result)
        if not coherent:
            bad_format+=1
        c=result.get('criteria',{}); decision=result.get('decision')
        if qualification_version==2:
            full_match=(coherent and decision==item['expected_decision'] and
                (item['expected_criteria'] is None or
                 runtime_memory_resolved(result) and c==item['expected_criteria']))
            if item['source'] in exposed:
                exposed_completed+=1; exposed_matches+=full_match
                if not full_match: failures.append('exposed_development_reference_mismatch')
            else:
                fresh_completed+=1; fresh_matches+=full_match
                if not full_match: failures.append('fresh_reference_mismatch')
        for key,groups in strata.items():
            group=groups[item[key]]; group['completed']+=1
            group['decisions'][decision if coherent else 'unresolved']+=1
            group['false_accepts']+=kind=='unsupported' and decision=='accept'
        first[row['control']]=result
        if kind=='unsupported':
            false_accepts+=decision=='accept'; rejections+=coherent and decision=='reject'
            unsupported_unclear+=decision=='inconclusive'
        elif kind=='useful':
            okay=coherent and decision=='accept' and c.get('addresses_request') is True
            useful+=okay
            if not okay: failures.append('useful_supported_not_accepted_and_relevant')
        elif kind=='needless_refusal':
            okay=coherent and decision=='accept' and c.get('addresses_request') is False and c.get('abstains_appropriately') is False
            refused+=okay
            if not okay: failures.append('needless_refusal_wrong_flags_or_decision')
        else:
            okay=coherent and decision=='inconclusive'
            ambiguous+=okay
            if not okay: failures.append('ambiguity_not_inconclusive')
        if item['expected_criteria'] is not None:
            assessed+=1
            for key in critical:
                correct=coherent and runtime_memory_resolved(result) and c.get(key)==item['expected_criteria'][key]
                critical[key]+=not correct; matched[key]+=correct
    repeat_failure=False
    for row in repeated:
        result=row['judgment']; original=first[row['control']]
        repeat_failure|=(not valid(result)
            or result.get('decision')!=original.get('decision')
            or (original.get('decision')!='inconclusive' and result.get('criteria')!=original.get('criteria')))
    if bad_format: failures.append('malformed_or_inconsistent')
    if false_accepts: failures.append('unsupported_claim_accepted')
    if unsupported_unclear>1: failures.append('too_many_unsupported_inconclusive')
    minimum=float(spec['gates']['minimum_each_critical_criterion_agreement_on_resolved'])
    if any((32-errors)/32<minimum for errors in critical.values()): failures.append('critical_criterion_agreement_below_floor')
    if repeat_failure: failures.append('sentinel_disagreement')
    complete=len(rows)==64
    accuracy=len(primary)==40 and not failures and rejections>=15 and useful==12 and refused==4 and ambiguous==8
    metrics=dict(primary_total=40,primary_completed=len(primary),judgments_total=64,judgments_completed=len(rows),
        remaining_judgments=64-len(rows),false_accepts_on_unsupported=int(false_accepts),
        resolved_rejections=int(rejections),unsupported_inconclusive=int(unsupported_unclear),
        useful_supported_accepted_and_relevant=int(useful),needless_refusal_correct=int(refused),
        ambiguity_inconclusive=int(ambiguous),malformed_or_inconsistent=bad_format,
        critical_errors=critical,critical_assessed=assessed,critical_denominator=32,
        critical_agreement={k:matched[k]/32 for k in critical},
        by_language=strata['language'],by_application=strata['app'],by_source_family=strata['source'],
        accuracy_passed=accuracy,repeatability_passed=len(repeated)==24 and not repeat_failure,
        complete=complete,passed=complete and accuracy and not repeat_failure,
        irreversible_failure=bool(failures),failure_reasons=sorted(set(failures)))
    if qualification_version==2:
        metrics.update(fresh_primary_total=38,fresh_primary_completed=fresh_completed,fresh_full_matches=int(fresh_matches),
            fresh_source_families=len({row['source'] for row in controls if row['source'] not in exposed}),
            exposed_development_total=2,exposed_development_completed=exposed_completed,
            exposed_development_full_matches=int(exposed_matches),fresh_sentinel_judgments=len(repeated))
    return metrics


def validate_runtime_qualification_review(controls,review):
    """Require a separately frozen, blinded review; disagreements block dispatch."""
    if (not isinstance(review,dict) or review.get('schema')!='runtime-qualification-review-v1'
            or review.get('controls_sha256')!=shared.stable_key(controls)
            or review.get('contract_sha256')!=shared.stable_key(runtime_memory_protocol())
            or review.get('independent_of_preparer') is not True or review.get('blinded_to_proposed_labels') is not True
            or review.get('reviewer_kind') not in ('human','llm') or not review.get('reviewer')
            or review['reviewer']=='preparing-codex-root-runtime-qualification-v1'):
        raise ValueError('Independent blinded runtime qualification review required')
    rows=review.get('judgments',[]); by_id={row['id']:row for row in controls}
    if len(rows)!=len(controls) or {row['control'] for row in rows}!=set(by_id):
        raise ValueError('Independent runtime qualification review is incomplete')
    for row in rows:
        item=by_id[row['control']]
        if (row.get('decision')!=item['expected_decision'] or row.get('criteria')!=item['expected_criteria']
                or not isinstance(row.get('reason'),str) or not row['reason'].strip()
                or not isinstance(row.get('evidence'),str) or not row['evidence'].strip()):
            raise ValueError('Runtime qualification review needs adjudication; preserve disagreements')
    return shared.stable_key(review)


def runtime_qualification_identity(*,qualification_version=1):
    _runtime_qualification_contract(qualification_version)
    path=FIXTURES/'runtime-qualification-anchors-v1.json'; bank=read(path)
    identity=dict(anchors_sha256=shared.digest(path),
        **{key+'_sha256':shared.digest(ROOT/bank[key]['path']) for key in ('source_pin','excluded_source_pin','provisioner')})
    if qualification_version==2:
        path=FIXTURES/'runtime-decision-qualification-v1.json'; exposure=read(path)
        for pin in exposure['inputs'].values():
            if shared.digest(ROOT/pin['path'])!=pin['sha256']:
                raise ValueError('Runtime qualification reference or exposure pin changed')
        identity['exposure_manifest_sha256']=shared.digest(path)
    return identity


def runtime_qualification(controls,protocol,judge,*,review,on_assessment=None):
    """Memory-only adapter over existing assessment/audit owners; no retries."""
    if on_assessment is not None and not callable(on_assessment):
        raise TypeError('Qualification progress callback must be callable')
    qualification_version=protocol.get('runtime_qualification_version',1)
    schedule=runtime_qualification_schedule(controls,qualification_version=qualification_version)
    review_sha=validate_runtime_qualification_review(controls,review)
    if controls!=runtime_qualification_items():
        raise ValueError('Runtime qualification controls differ from their source freeze')
    expected=_runtime_qualification_contract(qualification_version)
    if any(protocol.get(k)!=v for k,v in expected.items()):
        raise ValueError('Runtime qualification assessor contract changed')
    if type(judge) is not RecordingRouter or judge.controls is not False or 'judge' not in protocol:
        raise ValueError('Runtime qualification needs the existing real audited judge')
    spec=protocol['judge']
    if (judge.provider!=spec['provider'] or judge.model!=spec['model'] or judge.max_input!=spec['max_input']
            or (judge.local_spec if spec['provider']=='ollama' else judge.remote_spec)!=spec):
        raise ValueError('Runtime qualification judge differs from its pre-dispatch freeze')
    by_id={row['id']:row for row in controls}; rows=[]
    for entry in schedule:
        assessment=assess_calibration_control('memory',by_id[entry['control']],protocol,judge)
        rows.append(dict(entry,**assessment))
        if on_assessment is not None:
            on_assessment(copy.deepcopy(rows[-1]))
        if runtime_qualification_metrics(controls,rows,qualification_version=qualification_version)['irreversible_failure']:
            break
    metrics=runtime_qualification_metrics(controls,rows,qualification_version=qualification_version)
    report=dict(schema=f'runtime-memory-qualification-v{qualification_version}',phase='runtime-memory-qualification',rows=rows,
        controls_sha256=shared.stable_key(controls),review_sha256=review_sha,
        source_identity=runtime_qualification_identity(qualification_version=qualification_version),reference_kind=review['reviewer_kind'],
        contract_sha256=shared.stable_key(expected),protocol_sha256=shared.stable_key(protocol),code_sha256=fingerprints(),
        metrics=metrics,proxy=judge.controls,generated_application_replies=0,
        passed=metrics['passed'] and judge.controls is False)
    if qualification_version==2:
        report.update(runtime_qualification_version=2,runtime_enabled=False,
                      reference_contract_sha256=shared.stable_key(runtime_memory_protocol()))
    validate_runtime_qualification(report,controls,protocol,review=review)
    return report


def validate_runtime_qualification(report,controls,protocol,*,review,ledger_rows=None):
    """Replay parser, exact requests, raw ledger and every gate before qualification."""
    review_sha=validate_runtime_qualification_review(controls,review)
    qualification_version=protocol.get('runtime_qualification_version',1)
    expected=_runtime_qualification_contract(qualification_version)
    if (controls!=runtime_qualification_items() or any(protocol.get(k)!=v for k,v in expected.items())
            or report.get('schema')!=f'runtime-memory-qualification-v{qualification_version}' or report.get('phase')!='runtime-memory-qualification'
            or report.get('controls_sha256')!=shared.stable_key(controls) or report.get('review_sha256')!=review_sha
            or report.get('source_identity')!=runtime_qualification_identity(qualification_version=qualification_version) or report.get('reference_kind')!=review['reviewer_kind']
            or report.get('contract_sha256')!=shared.stable_key(expected) or report.get('code_sha256')!=fingerprints()
            or report.get('protocol_sha256')!=shared.stable_key(protocol)
            or report.get('proxy') is not False or report.get('generated_application_replies')!=0):
        raise ValueError('Runtime qualification provenance or source/contract freeze changed')
    if qualification_version==2 and (type(report.get('runtime_qualification_version')) is not int or
            report['runtime_qualification_version']!=2 or report.get('runtime_enabled') is not False or
            report.get('reference_contract_sha256')!=shared.stable_key(runtime_memory_protocol())):
        raise ValueError('Prospective qualification reference/contract identity changed')
    metrics=runtime_qualification_metrics(controls,report['rows'],qualification_version=qualification_version)
    if (report.get('metrics')!=metrics or report.get('passed') is not metrics['passed']
            or (not metrics['complete'] and not metrics['irreversible_failure'])
            or any(runtime_qualification_metrics(controls,report['rows'][:n],qualification_version=qualification_version)['irreversible_failure']
                   for n in range(1,len(report['rows'])))):
        raise ValueError('Runtime qualification gates or stopped schedule changed')
    validate_runtime_memory_calls(report['rows'],controls,protocol,ledger_rows=ledger_rows)
    return report['passed']


def runtime_assessment_calls(row,protocol):
    """Physical calls per assessment, including an invalid first stage's one call."""
    if protocol.get('runtime_memory_contract_version')!=4:
        return [row['call']]
    calls=row.get('calls')
    if type(calls) is not list or not 1<=len(calls)<=2 or row.get('call')!=calls[-1]:
        raise ValueError('Bound assessment has an invalid physical call trace')
    return calls


def validate_runtime_memory_calls(rows,controls,protocol,*,ledger_rows=None,protocol_for_row=None):
    """One raw completion/request audit for qualification and paired diagnostics."""
    ledger_rows=([json.loads(line) for line in (ROOT/'.data/application_context/provider-events.jsonl').read_text(encoding='utf-8').splitlines()]
                 if ledger_rows is None else ledger_rows)
    reservations={}; completions={}
    for row in ledger_rows:
        target=reservations if row['event']=='reservation' else completions if row['event']=='completion' else None
        if target is not None:
            if row['dispatch'] in target:
                raise ValueError('Runtime qualification ledger has duplicate dispatch events')
            target[row['dispatch']]=row
    by_id={row['id']:row for row in controls}; ids=set(); dispatches=set()
    for row in rows:
        current=protocol if protocol_for_row is None else protocol_for_row(row)
        item=by_id[row['control']]; result=row['judgment']
        calls=runtime_assessment_calls(row,current)
        if current['runtime_memory_contract_version']==4:
            first=memory_interpretation_messages({'app':item['app']},item['capture'],item['response'],protocol=current)
            completion=calls[0]['completion']
            interpreted=parse_runtime_interpretation(completion['content'],completion['finish_reason'],False,
                evidence=json.loads(first[1].content))
            if row.get('interpretation')!=interpreted or len(calls)!=(1 if interpreted['parse_failed'] else 2):
                raise ValueError('Bound interpretation differs from its actual first-stage response')
            evidence=runtime_control_evidence(item,current,interpretation=interpreted['interpretation'])
            messages_per_call=[first]
            if not interpreted['parse_failed']:
                messages_per_call.append(memory_judge_messages({'app':item['app']},None,item['capture'],item['response'],
                    None,protocol=current,interpretation=interpreted['interpretation']))
        else:
            messages=memory_judge_messages({'app':item['app']},None,item['capture'],item['response'],None,protocol=current)
            evidence=json.loads(messages[1].content) if current['runtime_memory_contract_version']==3 else None
            messages_per_call=[messages]
        for call,messages in zip(calls,messages_per_call):
            validate_screening_call(call,current['judge'])
            dispatch=call['dispatch']; completion_id=call['completion']['metadata'].get('completion_id')
            if (dispatch in dispatches or (current['judge']['provider']!='ollama'
                    and (completion_id in ids or not completion_id))):
                raise ValueError('Runtime qualification repeats reused a provider completion')
            ids.add(completion_id); dispatches.add(dispatch)
            request=reservations.get(dispatch,{})
            wanted=dict(provider=current['judge']['provider'],model=current['judge']['model'],tier=TaskTier.ADVANCED.value,
                        messages=[asdict(message) for message in messages],**memory_judge_options(current),proxy=False)
            if current['judge']['provider']=='ollama':
                wanted.update(local_only=True,model_digest=current['judge']['digest'])
            else:
                wanted.update(hosted_request_options(current['judge']))
            completion=call['completion']; usage=(completion['prompt_tokens'],completion['completion_tokens'])
            if (completions.get(dispatch)!=call or any(request.get(k)!=v for k,v in wanted.items())
                    or request_input_bound(messages,response_schema=current['judge'].get('response_schema'))>current['judge']['max_input']
                    or not all(type(n) is int for n in usage) or not 0<usage[0]<=current['judge']['max_input']
                    or not 0<=usage[1]<=billable_output_limit(current['judge'],wanted['max_tokens'])):
                raise ValueError('Runtime qualification differs from raw completion/request ledger')
        completion=calls[-1]['completion']
        if (result!=parse_runtime_memory(completion['content'],completion['finish_reason'],False,
                                        contract_version=current['runtime_memory_contract_version'],evidence=evidence)
                or (current['runtime_memory_contract_version'] in (2,3,4) and
                    row.get('disposition')!=runtime_memory_action(result,evidence=evidence))
                or row['observed_success']!=(runtime_memory_resolved(result,evidence=evidence) and result['decision']=='accept')):
            raise ValueError('Runtime qualification differs from raw completion/request ledger')


def execute_runtime_study(approval_path,plan_path,*,validator,run_directory,assess,ledger,judge_factory=None,local_only=False):
    """Shared approved execution, recording and budget boundary; never retry."""
    plan_path,run_directory,ledger=map(Path,(plan_path,run_directory,ledger))
    if type(local_only) is not bool:
        raise ValueError('Execution locality must be explicit')
    if judge_factory is not None and not callable(judge_factory):
        raise TypeError('Runtime study judge factory must be callable')
    if local_only:
        if approval_path is not None:
            raise ValueError('Local-only execution cannot reuse a hosted approval')
    else:
        if approval_path is None:
            raise ValueError('Explicit approval of the exact frozen study plan required')
        approval=read(Path(approval_path))
        if approval.get('approved') is not True or approval.get('plan_sha256')!=shared.digest(plan_path):
            raise ValueError('Explicit approval of the exact frozen study plan required')
    plan,cfg,controls,review,protocol=validator(plan_path)
    if local_only and (plan.get('local_only') is not True or protocol.get('local_only') is not True
            or protocol.get('screening') is not True or protocol['judge']['provider']!='ollama'
            or protocol['budget'].get('local_only') is not True or Decimal(protocol['budget']['max_usd'])!=0
            or Decimal(plan['additional_conservative_usd_cap'])!=0 or plan['endpoint']!='http://127.0.0.1:11434'
            or protocol['judge']['base_url']!=plan['endpoint']):
        raise ValueError('Local execution requires its isolated zero-API-spend loopback plan')
    if run_directory.exists():
        raise ValueError('Run directory already exists; partial runs require review, never automatic retry')
    def save(name,value):
        with (run_directory/name).open('x',encoding='utf-8') as stream:
            json.dump(value,stream,ensure_ascii=False,indent=2); stream.write('\n')
    with study_budget(protocol) as budget:
        if (shared.digest(ledger)!=plan['baseline_ledger_sha256'] or budget.pending
                or budget.attempts!=plan['baseline']['provider_attempts']
                or budget.spent!=Decimal(plan['baseline']['accounted_usd'])):
            raise ValueError('Shared ledger changed before the locked run')
        run_directory.mkdir()
        try:
            name='ollama' if local_only else protocol['judge']['provider']
            if not local_only:
                if name == 'tongyi':
                    hosted_request_options(protocol['judge'])
                    if plan['endpoint'] != protocol['judge']['base_url']:
                        raise ValueError('Runtime study requires the frozen Alibaba endpoint')
                elif name != 'openai' or plan['endpoint']!='https://api.openai.com/v1':
                    raise ValueError('Runtime study requires the frozen first-party endpoint')
            cfg=replace(cfg,providers={name:replace(cfg.get_provider(name),base_url=plan['endpoint'])})
            if judge_factory is None:
                judge,=make_routers(protocol,cfg,budget,roles=('judge',))
            else:
                judge=judge_factory(protocol,cfg,budget)
            with SessionLogWriter(run_directory/'judgments.jsonl') as progress:
                report=assess(controls,protocol,judge,review=review,on_assessment=progress.write)
            bound=protocol.get('runtime_memory_contract_version')==4
            count=(sum(len(runtime_assessment_calls(row,protocol)) for row in report['rows']) if bound else len(report['rows']))
            if ((bound and count>plan['additional_attempts']) or budget.attempts-plan['baseline']['provider_attempts']!=count
                    or budget.dispatches-plan['baseline']['dispatches']!=count
                    or budget.cache_hits!=plan['baseline']['cache_hits'] or budget.pending
                    or budget.spent-Decimal(plan['baseline']['accounted_usd'])>Decimal(plan['additional_conservative_usd_cap'])):
                raise ValueError('Study run accounting differs from its bounded schedule')
            save('report.json',report)
            save('result.json',dict(plan_sha256=shared.digest(plan_path),report_sha256=shared.digest(run_directory/'report.json'),
                judgments=len(report['rows']),passed=report['passed'],accounting=budget.summary(),runtime_enabled=False,
                **({'provider_calls':count} if bound else {})))
        except Exception as exc:
            save('abort.json',dict(plan_sha256=shared.digest(plan_path),exception_type=type(exc).__name__,
                accounting=budget.summary(),ledger_sha256=shared.digest(ledger),automatic_retry=False))
            raise


def calibration(protocol,sources,judge):
    memory_items,tutor_items=calibration_items(protocol,sources)
    if any('criterion_reviews' in r for r in memory_items+tutor_items):
        raise ValueError('Criterion reviews belong to completed assessments; adjudicate controls before new model dispatch')
    reused=protocol.get('calibration_revision')==3
    memory=reused_memory_controls(protocol,sources) if reused else []; tutor=[]
    for item in ([] if reused else memory_items):
        item.update(assess_calibration_control('memory',item,protocol,judge))
        memory.append(item)
        print(f'Calibrated memory control {len(memory)}/24',flush=True)
        if _calibration_needs_review(memory,tutor): break
        if protocol.get('screening') and calibration_failure_is_final(memory,tutor): break
    for item in (tutor_items if len(memory)==24 and not _calibration_needs_review(memory,tutor)
                 and not (protocol.get('screening') and calibration_failure_is_final(memory,tutor)) else []):
        item.update(assess_calibration_control('tutor',item,protocol,judge))
        tutor.append(item)
        print(f'Calibrated tutor control {len(tutor)}/24',flush=True)
        if _calibration_needs_review(memory,tutor): break
        if protocol.get('screening') and calibration_failure_is_final(memory,tutor): break
    revised=grading_version(protocol)==2
    repeated=[]
    if revised and len(memory)==len(tutor)==24 and not _calibration_needs_review(memory,tutor) and not calibration_failure_is_final(memory,tutor):
        selected=calibration_sentinels(memory,tutor)
        for repeat,order in ((2,selected),(3,list(reversed(selected)))):
            for kind,index in order:
                item=(memory if kind=='memory' else tutor)[index]
                repeated.append({'kind':kind,'index':index,'repeat':repeat,
                                 'assessment':assess_calibration_control(kind,item,protocol,judge)})
                print(f'Repeated calibration sentinel {len(repeated)}/16',flush=True)
                if _calibration_needs_review(memory,tutor,repeated): break
                if repeatability_status(memory,tutor,repeated)['irreversible_failure']: break
            if _calibration_needs_review(memory,tutor,repeated): break
            if repeatability_status(memory,tutor,repeated)['irreversible_failure']: break
    complete=len(memory)==len(tutor)==24 and (not revised or len(repeated)==16)
    return {'schema':SCHEMA,'phase':'calibration','memory_controls':memory,'tutor_controls':tutor,
            **qualification_identity(protocol),
            **reviewed_calibration_metrics(memory,tutor,repeated=repeated,protocol=protocol),'generated_application_replies':0,'proxy':judge.controls,
            **({'repeated_controls':repeated} if revised else {}),
            'complete':complete,'stopped_early':not complete,
            'remaining_controls':{'memory':24-len(memory),'tutor':24-len(tutor),
                                  **({'repeat_judgments':16-len(repeated)} if revised else {})}}


def validate_calibration(report,protocol,sources,*,expected_code=None,expected_review_manifest_sha256=None):
    review=criterion_review_summary(_calibration_review_records(report['memory_controls'],report['tutor_controls'],
        report.get('repeated_controls',())),protocol=protocol)
    if expected_review_manifest_sha256 is not None and (review is None or review['manifest_sha256']!=expected_review_manifest_sha256):
        raise ValueError('Criterion review manifest differs from the required freeze')
    expected_memory,expected_tutor=calibration_items(protocol,sources)
    if any(report.get(k)!=v for k,v in qualification_identity(protocol).items()):
        raise ValueError('Calibration judge rubric/qualification identity changed')
    nm,nt=len(report['memory_controls']),len(report['tutor_controls'])
    revised=grading_version(protocol)==2
    repeated=report.get('repeated_controls',[])
    status=repeatability_status(report['memory_controls'],report['tutor_controls'],repeated) if revised else None
    complete=nm==24 and nt==24 and (not revised or status['complete'])
    if (report.get('phase')!='calibration' or not 0<nm<=24 or not 0<=nt<=24 or (nt>0 and nm!=24)
            or report.get('complete') is not complete or report.get('stopped_early') is not (not complete)
            or report.get('remaining_controls') != {'memory':24-nm,'tutor':24-nt,
                    **({'repeat_judgments':16-len(repeated)} if revised else {})}
            or (not complete and not ((review and review['status']=='needs_review') or (protocol.get('screening') and (
                calibration_failure_is_final(report['memory_controls'],report['tutor_controls']) or (revised and status['irreversible_failure'])))))
            or report['code_sha256']!=(fingerprints() if expected_code is None else expected_code) or report['protocol_sha256']!=protocol_sha(protocol)
            or report.get('proxy') is not False or report.get('generated_application_replies')!=0):
        raise ValueError('Calibration has wrong provenance, counts or source freeze')
    if protocol.get('calibration_revision')==3 and report['memory_controls']!=reused_memory_controls(protocol,sources):
        raise ValueError('Carried memory judgments differ from the original evidence')
    memory_pairs=list(zip(expected_memory,report['memory_controls']))
    tutor_pairs=list(zip(expected_tutor,report['tutor_controls']))
    for r in repeated:
        items,pairs=(expected_memory,memory_pairs) if r['kind']=='memory' else (expected_tutor,tutor_pairs)
        item=items[r['index']]
        pairs.append((item,{**item,**r['assessment']}))
    if revised:
        calls=[r['call'] for _,r in memory_pairs+tutor_pairs]
        dispatches=[c['dispatch'] for c in calls]
        ids=[c['completion']['metadata'].get('completion_id') for c in calls]
        if (len(dispatches)!=len(set(dispatches))
                or (protocol.get('screening') and (None in ids or len(ids)!=len(set(ids))))):
            raise ValueError('Qualification repetitions must be distinct uncached provider calls')
    for item,row in memory_pairs:
        result=row['judgment']
        if (any(row[k]!=v for k,v in item.items()) or result.get('proxy') is not False
                or result!=parse_memory(result['raw'],result['finish_reason'],result['proxy'],contract_version=grading_version(protocol))
                or row['observed_success']!=memory_control_success(item,result)):
            raise ValueError('Memory calibration differs from raw output or source control')
        if protocol.get('screening'):
            validate_screening_call(row['call'],protocol['judge'])
            if (row['call']['completion']['content']!=result['raw']
                    or row['call']['completion']['finish_reason']!=result['finish_reason']):
                raise ValueError('Memory calibration differs from its actual completion')
    for item,row in tutor_pairs:
        verdict=parse_empath_verdict(row['rationale'],row['finish_reason'],strict=True,contract_version=grading_version(protocol))
        if (any(row[k]!=v for k,v in item.items()) or row['parse_failed']!=verdict.parse_failed
                or row['predicted']!=verdict.label.value
                or any(row.get(k)!=verdict.contract_metadata().get(k) for k in ('contract_version','consistency_errors'))
                or (verdict.contract_version==2 and row.get('criteria')!={k.value:v for k,v in verdict.sub_criteria.items()})):
            raise ValueError('Tutor calibration differs from existing parser or source control')
        if protocol.get('screening'):
            validate_screening_call(row['call'],protocol['judge'])
            if (row['call']['completion']['content'].strip()!=row['rationale']
                    or row['call']['completion']['finish_reason']!=row['finish_reason']):
                raise ValueError('Tutor calibration differs from its actual completion')
    metrics=reviewed_calibration_metrics(report['memory_controls'],report['tutor_controls'],repeated=repeated,protocol=protocol)
    review_fields={'criterion_review','qualification_status','raw_accuracy_passed','raw_memory_agreement',
                   'raw_memory_critical_agreement','raw_tutor_kappa'}
    if (any(report.get(k)!=v or k not in report for k,v in metrics.items())
            or any(k in report and k not in metrics for k in review_fields)):
        raise ValueError('Calibration verdict does not match its judgments')
    return report['passed']


def require_calibration(args,protocol,sources,cfg):
    if not args.application_calibration:
        raise ValueError('A passing real calibration report is required')
    report=read(args.application_calibration)
    if any(report.get(k)!=v for k,v in runtime_identity(cfg, cjk_tokenizer=protocol['tokenizer']).items()):
        raise ValueError('Calibration environment/configuration changed')
    if not validate_calibration(report,protocol,sources,
            expected_review_manifest_sha256=getattr(args,'application_review_manifest_sha256',None)):
        raise ValueError('Calibration failed or differs from current freeze')


def require_local_resources(args,protocol,cases,cfg,before):
    path=getattr(args,'application_resources',None)
    if not path:
        raise ValueError('Local model stages require the complete versioned resource check first')
    report=read(path)
    selected=[c for c in cases if c['split']=='development']
    if (report.get('phase')!='resources' or report.get('code_sha256')!=before
            or report.get('protocol_sha256')!=protocol_sha(protocol) or report.get('local_only') is not protocol['local_only']
            or any(report.get(k)!=v for k,v in runtime_identity(cfg, cjk_tokenizer=protocol['tokenizer']).items())
            or report.get('provider_requests')!=0 or report.get('generated_application_replies')!=0):
        raise ValueError('Resource check differs from the current local freeze')
    if not validate_resources(report['resources'],selected,procedure=RESOURCE_V2) or report.get('passed') is not True:
        raise ValueError('Capped resources failed; preserve the failure before running model stages')
    return report['resources']


def preflight_requests(protocol,role):
    # Judge rubrics are English system instructions, including on Chinese
    # evidence. Probe that actual message shape; calibration tests comprehension.
    if protocol.get('screening') and role=='judge':
        return [[ChatMessage('system','Return exactly this JSON object and no other keys: {"ready":true}'),
                 ChatMessage('user',json.dumps({'readiness_probe':probe},ensure_ascii=False))]
                for probe in ('READY','就绪')]
    return [[ChatMessage('user',prompt)] for prompt in ('Reply with the word READY only.','只回复：就绪')]


def run(args):
    """Existing Phase 9 CLI dispatch; provider calls only in explicit live stages."""
    if args.command=='application-prepare':
        return prepare_protocol(args.application_annotations)
    protocol,cases=load_inputs()
    profile_path=getattr(args,'application_profile',None)
    if profile_path:
        protocol=local_profile(profile_path,protocol)
    if args.command not in ('application-validate','application-resources') and not protocol.get('screening'):
        raise ValueError('External model execution is disabled without an explicit approved screening profile')
    if protocol.get('screening') and args.command=='application-evaluation':
        raise ValueError('Development screening does not open the held-out evaluation split')
    before=fingerprints()
    if args.command=='application-validate':
        if not args.reference_report: raise ValueError('Validation requires --reference-report')
        report=read(args.reference_report)
        devsha=shared.digest(args.development_report) if args.development_report else None
        cfg=load_config(args.llm_config) if args.llm_config else load_config()
        validate_report(report,cases,protocol,development_sha=devsha,cfg=cfg)
        return {'schema':SCHEMA,'validated':True,'reference_sha256':shared.digest(args.reference_report)}
    if args.output.suffix!='.json': raise ValueError('Application output must have a .json suffix')
    output_dir=args.output.with_suffix('')
    output_dir.mkdir(parents=True,exist_ok=False)
    cfg=load_config(args.llm_config) if args.llm_config else load_config()
    sources=Sources(protocol)
    header={'schema':SCHEMA,'command':args.command,'protocol_sha256':shared.digest(FIXTURES/'protocol.json'),
            **qualification_identity(protocol),
            'code_sha256':before,**runtime_identity(cfg, cjk_tokenizer=protocol['tokenizer'])}
    header['protocol_sha256']=protocol_sha(protocol)
    if protocol.get('screening'):
        header.update(local_only=protocol['local_only'],profile=protocol['profile'],
                      original_cloud_protocol_sha256=shared.digest(FIXTURES/'protocol.json'))
    if args.application_resume:
        header['resumed_from']={'path':str(args.application_resume.resolve()),
                                'manifest_sha256':shared.digest(args.application_resume/'run-manifest.json')}
    shared.write_new(output_dir/'run-manifest.json',header)
    if args.command=='application-resources':
        selected=[c for c in cases if c['split']=='development']
        diagnostics=context_diagnostics(selected,sources,cfg)
        resources=capped_resources(selected,sources,cfg,procedure=protocol.get('resource_procedure'))
        if fingerprints()!=before: raise ValueError('Code changed during resource measurement')
        return {**header,'phase':'resources','families':len(selected),'resources':resources,
                'context_diagnostics':diagnostics,
                'passed':resources['passed'] and diagnostics['passed'],
                'provider_requests':0,'generated_application_replies':0}
    # Reconstruct and verify a revised control catalog before any model request.
    if protocol.get('calibration_revision',0) >= 2:
        calibration_items(protocol,sources)
    if protocol.get('calibration_revision')==3:
        reused_memory_controls(protocol,sources,cfg)
    diagnostics=context_diagnostics([c for c in cases if c['split']=='development'],sources,cfg)
    if not diagnostics['passed']:
        raise ValueError('No valid application treatment contrast; stop before provider initialization')
    with SessionLogWriter(output_dir/'events.jsonl') as writer, study_budget(protocol) as budget:
        try:
            if budget.attempts>=budget.max_calls:
                raise ValueError('Provider attempt budget exhausted before provider initialization')
            generation,judge=make_routers(protocol,cfg,budget)
            # Initialization verifies SDK retry policy without issuing model calls.
            generation.verify_provider(TaskTier.ADVANCED); judge.verify_provider(TaskTier.ADVANCED)
            if protocol.get('screening') and args.command!='application-preflight':
                require_local_resources(args,protocol,cases,cfg,before)
            if args.command=='application-preflight':
                outcomes=[]
                for role,router in (('generator',generation),('judge',judge)):
                    for messages in preflight_requests(protocol,role):
                        completion=router.chat(messages,tier=TaskTier.ADVANCED,
                                               max_tokens=16,temperature=0,use_cache=False)
                        outcomes.append(copy.deepcopy(router.last_call))
                result={'schema':SCHEMA,'phase':'preflight','outcomes':outcomes,
                        'passed':all(o['usage_valid'] and not o['proxy'] and not o['cached'] and o['completion']['content'].strip()
                                     and o['completion']['finish_reason']=='stop' for o in outcomes),
                        'generated_application_replies':0}
            elif args.command=='application-smoke':
                require_preflight(args,protocol,before,cfg)
                if protocol.get('screening'): require_calibration(args,protocol,sources,cfg)
                result=smoke_run(protocol,sources,cfg,generation,judge,output_dir)
            elif args.command=='application-calibration':
                require_preflight(args,protocol,before,cfg)
                if not protocol.get('screening'): require_smoke(args,protocol,sources,cfg,before)
                result=calibration(protocol,sources,judge)
            else:
                require_preflight(args,protocol,before,cfg)
                require_smoke(args,protocol,sources,cfg,before)
                require_calibration(args,protocol,sources,cfg)
                phase='development' if args.command=='application-development' else 'evaluation'
                devsha=None
                if phase=='evaluation':
                    if not args.development_report: raise ValueError('Development evidence required before evaluation')
                    dev=read(args.development_report); devsha=shared.digest(args.development_report)
                    validate_report(dev,cases,protocol,cfg=cfg)
                    if (not dev['decision']['quality_passed'] or not dev['resources']['passed']
                            or dev['code_sha256']!=before):
                        raise ValueError('Development gates/fingerprints do not qualify evaluation')
                selected=[c for c in cases if c['split']==phase]
                # Measure serial memory resources before any provider workload.
                if args.application_resume:
                    rows,resources=resume_trials(args.application_resume,header,selected,protocol,sources,cfg)
                else:
                    resources=(require_local_resources(args,protocol,cases,cfg,before) if protocol.get('screening') else
                               capped_resources(selected,sources,cfg))
                    rows=[]
                shared.write_new(output_dir/'resources.json',resources)
                completed={row_key(r) for r in rows}
                for c in selected:
                    jobs=list(itertools.product(range(2),(False,True),range(protocol['samples'])))
                    np.random.default_rng(int(shared.stable_key(c['id'])[:8],16)).shuffle(jobs)
                    for q,absent,sample in jobs:
                        order=ARMS if (sample+q)%2==0 else tuple(reversed(ARMS))
                        for arm in order:
                            if (c['id'],c['app'],q,absent,arm,sample) in completed: continue
                            row=trial(c,q,sample,arm,absent,sources,cfg,generation,judge,output_dir,protocol=protocol)
                            rows.append(row); writer.write({'event':'trial','row':row})
                    print(f'Completed application family {c["id"]}',flush=True)
                result={'schema':SCHEMA,'phase':phase,'matrix':matrix(rows,selected,protocol['samples']),'trials':rows,
                        'resources':resources,'decision':decisions(rows,local_only=protocol.get('screening',False)),
                        'development_report_sha256':devsha}
            result.update({**header,'budget':budget.summary()})
            if result['phase'] in ('development','evaluation'):
                validate_report(result,cases,protocol,development_sha=result['development_report_sha256'],cfg=cfg)
            elif result['phase']=='calibration':
                validate_calibration(result,protocol,sources)
            if fingerprints()!=before: raise ValueError('Source code changed during live execution')
            return result
        except BaseException as exc:
            shared.write_new(output_dir/'interrupted.json',{'schema':SCHEMA,'exception_type':type(exc).__name__,
                                                            'budget':budget.summary(),'manifest':header})
            raise


def resume_trials(directory,header,cases,protocol,sources,cfg,visited=None):
    """Read complete independent trial events; never deserialize warm memory."""
    visited=set() if visited is None else visited
    directory=directory.resolve()
    if directory in visited: raise ValueError('Cyclic application resume lineage')
    visited.add(directory)
    previous=read(directory/'run-manifest.json')
    for key in ('schema','command','protocol_sha256','code_sha256','tokenizer','python','openai_sdk','config_sha256','numpy'):
        if previous[key]!=header[key]: raise ValueError('Resume source/configuration differs from tested freeze')
    interruption=directory/'interrupted.json'
    if interruption.exists() and read(interruption)['budget']['usage_uncertain']:
        raise ValueError('Unknown provider usage prevents continuation of this attempt')
    rows=[]
    if 'resumed_from' in previous:
        parent=Path(previous['resumed_from']['path'])
        if shared.digest(parent/'run-manifest.json')!=previous['resumed_from']['manifest_sha256']:
            raise ValueError('Parent resume manifest changed')
        rows,_=resume_trials(parent,previous,cases,protocol,sources,cfg,visited)
    events=directory/'events.jsonl'
    if events.exists():
        for line in events.read_text(encoding='utf-8').splitlines():
            event=json.loads(line)
            if event['event']!='trial': raise ValueError('Unexpected application trial event')
            rows.append(event['row'])
    expected={(c['id'],c['app'],q,a,arm,n) for c in cases for q,a,arm,n in
              itertools.product(range(2),(False,True),ARMS,range(protocol['samples']))}
    keys=[row_key(r) for r in rows]
    if len(keys)!=len(set(keys)) or not set(keys)<=expected:
        raise ValueError('Duplicate or foreign completed resume trial')
    prepared={}; recalls={}
    for row in rows: validate_trial(row,cases,protocol,sources,cfg,prepared,recalls)
    resources=read(directory/'resources.json'); validate_resources(resources,cases,procedure=protocol.get('resource_procedure'))
    return rows,resources


def require_preflight(args,protocol,before,cfg):
    if not args.application_preflight: raise ValueError('A matching provider preflight is required')
    report=read(args.application_preflight)
    expected=[protocol[r]['provider'] for r in ('generator','generator','judge','judge')]
    models=[protocol[r]['model'] for r in ('generator','generator','judge','judge')]
    outcomes=report.get('outcomes',[])
    if (report.get('phase')!='preflight' or not report.get('passed') or report.get('code_sha256')!=before
            or report.get('protocol_sha256')!=protocol_sha(protocol) or len(outcomes)!=4
            or [o['completion']['provider'] for o in outcomes]!=expected
            or [o['completion']['model_id'] for o in outcomes]!=models
            or any(o['completion']['tier']!=TaskTier.ADVANCED.value or not o['completion']['content'].strip() for o in outcomes)
            or any(o['proxy'] or o['cached'] or not o['usage_valid'] or o['completion']['finish_reason']!='stop' for o in outcomes)
            or any(report.get(k)!=v for k,v in runtime_identity(cfg, cjk_tokenizer=protocol['tokenizer']).items())):
        raise ValueError('Provider preflight did not pass with this frozen configuration')
    if protocol.get('screening'):
        for outcome,role in zip(outcomes,('generator','generator','judge','judge')):
            validate_screening_call(outcome,protocol[role])


def smoke_cases(protocol,sources):
    """Disjoint, exact-source plumbing controls, not a paraphrase outcome sample."""
    _,main=load_inputs()
    _,cal_tutor=calibration_items(protocol,sources)
    path='.data/kdconv/data/film/train.json'
    excluded=set(read(ROOT/'benchmarks/fixtures/scaled_context/fresh-audit.json')['excluded_groups']['film/train.json'])
    excluded.update(c['row'] for c in main if c['source']==path)
    row=next(i for i,d in enumerate(sources.data[path]) if i not in excluded and len(d['messages'])>=2)
    turns=sources.data[path][row]['messages']
    companion={'id':f'smoke-film-{row}','app':'companion','stratum':'native_zh_bridge','language':'zh',
               'split':'smoke','source':path,'row':row,'target':1,'topic':'exact-source plumbing control',
               'cues':[turns[1]['message']],'queries':[turns[0]['message']]*2,'adaptation':None}
    path='.data/mathdial/train.jsonl'
    excluded={sources.data[c['source']][c['row']].problem for c in main if '/mathdial/' in c['source']}
    excluded.update(c['problem'] for c in cal_tutor)
    for row,convo in enumerate(sources.data[path]):
        if convo.problem in excluded: continue
        target=next((i for i,t in enumerate(convo.turns[:12]) if t.role=='student' and len(t.utterance)>50),None)
        if target is not None: break
    else: raise ValueError('No disjoint source for tutoring smoke')
    tutor={'id':f'smoke-math-{row}','app':'tutor','stratum':'native_en_application','language':'en','split':'smoke',
           'source':path,'row':row,'target':target,'topic':'earlier working','cues':[],
           'queries':['Help me revisit the working I showed you earlier. Ask a guiding question without giving the final answer.']*2,
           'adaptation':None}
    return [companion,tutor]


def smoke_run(protocol,sources,cfg,generation,judge,output_dir):
    rows=[]
    for case in smoke_cases(protocol,sources):
        for absent in ((False,True) if case['app']=='companion' else (False,)):
            for arm in ARMS:
                row=trial(case,0,0,arm,absent,sources,cfg,generation,judge,output_dir,protocol=protocol)
                validate_trial(row,[case],protocol,sources,cfg)
                rows.append(row)
    return {'schema':SCHEMA,'phase':'smoke','trials':rows,'generated_application_replies':6,
            'passed':all(not r['proxy'] and r['scores']['resolved'] for r in rows),
            'scope':'Exact-source application plumbing; semantic scores are diagnostics, not main-study outcomes.'}


def require_smoke(args,protocol,sources,cfg,before):
    if not args.application_smoke: raise ValueError('Disjoint real-application smoke evidence required')
    report=read(args.application_smoke)
    cases=smoke_cases(protocol,sources)
    rows=report.get('trials',[])
    expected={(c['id'],c['app'],0,a,arm,0) for c in cases
              for a in ((False,True) if c['app']=='companion' else (False,)) for arm in ARMS}
    if (report.get('phase')!='smoke' or report.get('code_sha256')!=before
            or report.get('protocol_sha256')!=protocol_sha(protocol)
            or report.get('generated_application_replies')!=6
            or any(report.get(k)!=v for k,v in runtime_identity(cfg, cjk_tokenizer=protocol['tokenizer']).items())):
        raise ValueError('Application smoke source freeze or counts differ')
    if any(report.get(k)!=v for k,v in qualification_identity(protocol).items()):
        raise ValueError('Smoke judge rubric/qualification identity changed')
    complete_keys([row_key(r) for r in rows],expected)
    for row in rows: validate_trial(row,cases,protocol,sources,cfg)
    passed=all(not r['proxy'] and r['scores']['resolved'] for r in rows)
    if report.get('passed') is not True or not passed:
        raise ValueError('Application smoke did not complete genuine resolved model calls')
