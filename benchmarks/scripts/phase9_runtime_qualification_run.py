"""Prepare/check offline; execute the existing qualification only after exact approval.

No prompt, parser, scoring, provider, budget or retry workflow is reimplemented.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from psychology_base.eval.benchmarks import application_context as study

ROOT=study.ROOT
DOCS=ROOT/'docs/experiments/phase-9-application-context'
PLAN=DOCS/'runtime-qualification-hosted-plan-v1.json'
PACKET=DOCS/'runtime-qualification-packet-v2.json'
REVIEW=ROOT/'.data/application_context/runtime-qualification-v1/blinded-review-receipt-v1.json'
LEDGER=ROOT/'.data/application_context/provider-events.jsonl'
RUN_DIR=ROOT/'.data/application_context/runtime-qualification-hosted-v1'
CALLS=64
CAP=Decimal('0.1056')
ENDPOINT='https://api.openai.com/v1'
SPEC=dict(provider='openai',model='gpt-6-luna',family='gpt',reasoning_effort='none',json_mode=True,
          input_rate='0.125',output_rate='0.50',max_input=12000)
PRICE_SOURCE='https://developers.openai.com/api/docs/models/gpt-6-luna'


def profile(revision=1):
    if type(revision) is not int or revision not in (1,2):
        raise ValueError('Qualification study revision must be 1 or 2')
    if revision==1:
        return SimpleNamespace(plan=PLAN,run=RUN_DIR,requests=ROOT/read(PACKET)['requests']['path'],
            baseline_calls=831,baseline_usd='0.093054775',calls=CALLS,cap=CAP)
    directory=ROOT/'.data/application_context/runtime-decision-separation-qualification-v1'
    return SimpleNamespace(plan=DOCS/'runtime-decision-separation-qualification-hosted-plan-v1.json',
        run=directory/'hosted-run',requests=directory/'requests-v1.json',
        baseline_calls=895,baseline_usd='0.106581150',calls=64,cap=Decimal('0.1056'))


def read(path):
    return study.read(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_new(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2)
        stream.write('\n')


def baseline(*,revision=1):
    selected=profile(revision)
    budget=study.Budget(SimpleNamespace(path=LEDGER),max_calls=selected.baseline_calls,max_usd='1')
    if budget.attempts!=selected.baseline_calls or budget.pending or budget.spent!=Decimal(selected.baseline_usd):
        raise ValueError(f'Expected exhausted {selected.baseline_calls}-attempt baseline without pending calls')
    return budget.summary()


def checked_exposure(controls):
    """Bind freshness to the actual pre-run ledger and the whole exposed family."""
    bank=read(study.FIXTURES/'runtime-decision-qualification-v1.json')
    prefix=[line for line in LEDGER.read_bytes().splitlines(keepends=True)
            if json.loads(line)['dispatch']<=bank['baseline_attempts']]
    if hashlib.sha256(b''.join(prefix)).hexdigest()!=bank['baseline_ledger_sha256']:
        raise ValueError('Qualification exposure ledger prefix changed')
    fields=('APPLICATION','QUERY','VISIBLE_CONTEXT','REPLY'); seen={}
    for line in prefix:
        event=json.loads(line)
        if event['event']!='reservation': continue
        for message in event.get('messages',[]):
            try: payload=json.loads(message.get('content',''))
            except (ValueError,TypeError): continue
            if isinstance(payload,dict) and all(key in payload for key in fields):
                key=study.shared.stable_key({name:payload[name] for name in fields})
                seen.setdefault(key,[]).append(event['dispatch'])
    actual={}
    for item in controls:
        payload=study.runtime_memory_payload(item['app'],item['capture']['query'],
                                            item['capture']['context'],item['response'])
        dispatches=seen.get(study.shared.stable_key(payload),[])
        if dispatches: actual[item['id']]=dispatches
    if actual!=bank['known_exact_prior_payload_exposures']:
        raise ValueError('Qualification candidate exposures differ from the frozen allocation')
    if any(item['id'] in actual and item['source'] not in bank['exposed_source_families'] for item in controls):
        raise ValueError('A previously assessed case cannot count as fresh qualification')
    return dict(fresh_primary_cases=bank['fresh_primary_total'],exposed_development_cases=2,
                exact_prior_payload_exposures=actual,baseline_ledger_sha256=bank['baseline_ledger_sha256'])


def checked_inputs(*,revision=1):
    packet=read(PACKET)
    for key in ('anchors','controls','requests','blinded_review','raw_source_audit'):
        if sha(ROOT/packet[key]['path'])!=packet[key]['sha256']:
            raise ValueError('Frozen qualification source/control/request packet changed')
    controls=study.runtime_qualification_items(); review=read(REVIEW)
    study.validate_runtime_qualification_review(controls,review)
    if (packet['sources']!=study.runtime_qualification_identity()
            or read(ROOT/packet['controls']['path'])['controls']!=controls):
        raise ValueError('Frozen qualification controls differ from their real source construction')
    if revision==2: checked_exposure(controls)
    selected=profile(revision)
    outbound=read(selected.requests)['requests']
    schedule=study.runtime_qualification_schedule(controls,qualification_version=revision); by_id={row['id']:row for row in controls}
    if len(outbound)!=selected.calls or len(schedule)!=selected.calls:
        raise ValueError('Qualification schedule must contain exactly 64 maximum requests')
    protocol=(study.runtime_memory_protocol() if revision==1 else
              study.runtime_memory_protocol(contract_version=2,prompt_revision=4))
    for entry,request in zip(schedule,outbound):
        item=by_id[entry['control']]
        messages=study.memory_judge_messages({'app':item['app']},None,item['capture'],item['response'],None,protocol=protocol)
        if (any(request.get(k)!=v for k,v in entry.items())
                or request['messages']!=[study.asdict(message) for message in messages]
                or request['bounded_input_bytes']!=study.request_input_bound(messages)
                or request['bounded_input_bytes']>SPEC['max_input']
                or request['max_tokens']!=300 or request['temperature']!=0):
            raise ValueError('Frozen qualification request differs from the actual judge boundary')
    return packet,controls,review


def protocol_for(summary,*,revision=1):
    selected=profile(revision)
    protocol=(study.runtime_memory_protocol() if revision==1 else
              study.runtime_memory_protocol(contract_version=2,prompt_revision=4))
    result=dict(protocol,screening=True,judge=dict(SPEC),
        budget=dict(max_calls=summary['provider_attempts']+selected.calls,
                    max_usd=str(Decimal(summary['accounted_usd'])+selected.cap),local_only=False))
    if revision==2: result['runtime_qualification_version']=2
    return result


def requests(controls,protocol,*,revision=1):
    selected=profile(revision); by_id={item['id']:item for item in controls}; outbound=[]
    if protocol.get('runtime_qualification_version',1)!=revision:
        raise ValueError('Qualification schedule differs from its protocol')
    for entry in study.runtime_qualification_schedule(controls,qualification_version=revision):
        item=by_id[entry['control']]
        messages=study.memory_judge_messages({'app':item['app']},None,item['capture'],item['response'],None,protocol=protocol)
        bound=study.request_input_bound(messages)
        if bound>SPEC['max_input']:
            raise ValueError('Qualification request exceeds its frozen input ceiling')
        outbound.append(dict(entry,messages=[study.asdict(message) for message in messages],
                             bounded_input_bytes=bound,max_tokens=300,temperature=0))
    if len(outbound)!=selected.calls:
        raise ValueError('Qualification requests differ from the selected schedule')
    return outbound


def plan_fields(packet,cfg,controls,review,summary,*,revision=1):
    """One construction owner for the proposal and its pre-dispatch revalidation."""
    ceiling=(Decimal(SPEC['input_rate'])*SPEC['max_input']+Decimal(SPEC['output_rate'])*300)/1_000_000
    selected=profile(revision)
    if ceiling*selected.calls!=selected.cap:
        raise ValueError('Qualification cost ceiling differs from the frozen model/rate bound')
    result=dict(schema=f'runtime-qualification-hosted-plan-v{revision}',status='prepared-pending-explicit-provider-approval',
        packet_path=PACKET.relative_to(ROOT).as_posix(),packet_sha256=sha(PACKET),
        requests_path=selected.requests.relative_to(ROOT).as_posix(),requests_sha256=sha(selected.requests),
        review_path=REVIEW.relative_to(ROOT).as_posix(),review_sha256=sha(REVIEW),
        review_identity=study.validate_runtime_qualification_review(controls,review),
        controls_sha256=study.shared.stable_key(controls),source_identity=study.runtime_qualification_identity(qualification_version=revision),
        runner_sha256=sha(Path(__file__)),code_sha256=study.fingerprints(),runtime_identity=study.runtime_identity(cfg),
        baseline_ledger_sha256=sha(LEDGER),baseline=summary,endpoint=ENDPOINT,candidate=dict(SPEC),
        protocol=protocol_for(summary,revision=revision),additional_attempts=selected.calls,shared_attempt_cap=summary['provider_attempts']+selected.calls,
        additional_conservative_usd_cap=str(selected.cap),shared_conservative_usd_cap=str(Decimal(summary['accounted_usd'])+selected.cap),
        run_directory=selected.run.relative_to(ROOT).as_posix(),
        pricing=dict(checked_utc_date='2026-10-22',source=PRICE_SOURCE,
            caching_source='https://developers.openai.com/api/docs/guides/prompt-caching',
            standard_input_per_million='0.10',default_30m_cache_write_per_million='0.125',output_per_million='0.50',
            bound='Charge every input token at the higher default cache-write rate; no cache discounts assumed. Global first-party endpoint and default service tier. Frozen-rate API token-charge bound, not a tax/billing reconciliation.'),
        scope='40 pre-reviewed public-source-seeded authored English/Chinese controls plus 24 predetermined repeats. No application generations, readiness probes or private user histories.',
        payload_fields=['APPLICATION','QUERY','VISIBLE_CONTEXT','REPLY'],
        payload_excludes=['expected labels','independent reference decisions/evidence','author rationales','control types','withheld references','source-absence labels'],
        stopping='Existing qualification owner stops at the first irreversible frozen gate failure. No retries, repair loop, fallback model, cache reuse, automatic continuation or post-failure extra calls.',
        generated_application_replies=0,provider_calls_made=0)
    if revision==2:
        exposure=study.FIXTURES/'runtime-decision-qualification-v1.json'
        exposed=set(read(exposure)['exposed_source_families'])
        result.update(qualification_version=2,exposure_manifest_path=exposure.relative_to(ROOT).as_posix(),
            exposure_manifest_sha256=sha(exposure),fresh_primary_cases=38,fresh_source_families=15,
            exposed_development_primary_cases=2,fresh_sentinel_judgments=24,
            fresh_strata=dict(Counter(row['language']+'/'+row['app'] for row in controls
                                     if row['source'] not in exposed)),
            reference_contract_sha256=study.shared.stable_key(study.runtime_memory_protocol()),
            scope='40 unchanged pre-reviewed public-source/authored English/Chinese controls: 38 fresh source-disjoint cases and two exposed development cases reported separately, plus 24 fresh-case sentinel repeats.',
            reference_limit='Reuse prior blinded LLM reference decisions unchanged under the same intended semantics. No new reference adjudication or human gold is claimed. The model wire format and derived decision are prospective.',
            gates='All original gates plus exact full-reference matches for each fresh case and each exposed development case. Repeats use only fresh source families. First irreversible failure stops without retry.',
            runtime_enabled=False)
    return result


def prepare(plan_path=None,*,revision=1):
    selected=profile(revision); plan_path=selected.plan if plan_path is None else Path(plan_path)
    if plan_path.exists() or selected.run.exists():
        raise ValueError('Frozen plan or run already exists; refuse overwrite/retry')
    packet,controls,review=checked_inputs(); cfg=study.load_config(); summary=baseline(revision=revision)
    if revision==2:
        save_new(selected.requests,dict(schema='runtime-qualification-requests-v2',provider_calls=0,
            requests=requests(controls,protocol_for(summary,revision=revision),revision=revision)))
        checked_inputs(revision=revision)
    plan=dict(plan_fields(packet,cfg,controls,review,summary,revision=revision),timestamp_utc=datetime.now(timezone.utc).isoformat())
    save_new(plan_path,plan)
    return plan


def validate(plan_path=None,*,revision=1):
    selected=profile(revision); plan_path=selected.plan if plan_path is None else Path(plan_path)
    plan=read(plan_path)
    packet,controls,review=checked_inputs(revision=revision); cfg=study.load_config(); summary=baseline(revision=revision)
    expected=plan_fields(packet,cfg,controls,review,summary,revision=revision)
    if set(plan)!=set(expected)|{'timestamp_utc'} or any(plan.get(k)!=v for k,v in expected.items()):
        raise ValueError('Exact qualification plan/source/reference/model/code/budget baseline drift')
    return plan,cfg,controls,review,expected['protocol']


def run(approval_path,plan_path=None,*,revision=1):
    selected=profile(revision); plan_path=selected.plan if plan_path is None else Path(plan_path)
    validator=validate if revision==1 else lambda path:validate(path,revision=revision)
    return study.execute_runtime_study(approval_path,plan_path,validator=validator,
        run_directory=selected.run,assess=study.runtime_qualification,ledger=LEDGER)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','check','run'])
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--revision',type=int,choices=(1,2),default=1)
    parser.add_argument('--approval',type=Path)
    args=parser.parse_args()
    if args.command=='prepare':
        plan=prepare(args.plan,revision=args.revision)
        print(json.dumps(dict(status=plan['status'],maximum_requests=CALLS,maximum_additional_usd=str(CAP))))
    elif args.command=='check':
        validate(args.plan,revision=args.revision)
        print('Exact 64-request qualification scope verified offline; no provider calls.')
    elif args.approval is None:
        parser.error('run requires --approval for this exact plan')
    else:
        run(args.approval,args.plan,revision=args.revision)
