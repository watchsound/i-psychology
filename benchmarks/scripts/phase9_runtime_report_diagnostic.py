"""Freeze/check an exposed paired diagnostic; dispatch only with exact approval."""
import argparse
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

from psychology_base.eval.benchmarks import application_context as study
from psychology_base.eval.benchmarks import runtime_report_diagnostic as diagnostic
from psychology_base.eval.benchmarks.application_calls import (
    tongyi_judge_spec, tongyi_reasoning_spec, tongyi_reliability_spec, tongyi_binding_spec, hosted_request_options, billable_output_limit,
)

ROOT = study.ROOT
PLAN = ROOT / 'docs/experiments/phase-9-application-context/runtime-report-diagnostic-hosted-plan-v1.json'
DIRECTORY = ROOT / '.data/application_context/runtime-report-diagnostic-v1'
REQUESTS = DIRECTORY / 'requests-v1.json'
CONTROLS = DIRECTORY / 'controls-v1.json'
RUN = DIRECTORY / 'hosted-run'
LEDGER = ROOT / '.data/application_context/provider-events.jsonl'
CALLS = 40
CAP = Decimal('0.0660')
ENDPOINT = 'https://api.openai.com/v1'


def profile(revision=1):
    """Select one frozen study without duplicating execution or mutating globals."""
    if type(revision) is not int or revision not in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10):
        raise ValueError('Report diagnostic revision must be 1 through 10')
    if revision == 10:
        directory=ROOT/'.data/application_context/runtime-claim-binding-diagnostic-v1'
        return SimpleNamespace(revision=10,
            plan=ROOT/'docs/experiments/phase-9-application-context/runtime-claim-binding-diagnostic-plan-v1.json',
            directory=directory,requests=directory/'request-templates-v1.json',controls=directory/'controls-v1.json',
            run=directory/'hosted-run',calls=32,cap=Decimal('0.277142592'),baseline_calls=1276,
            baseline_usd='0.1814892260',prompts=(8,))
    if revision == 9:
        directory=ROOT/'.data/application_context/runtime-assessment-reliability-diagnostic-v1'
        return SimpleNamespace(revision=9,
            plan=ROOT/'docs/experiments/phase-9-application-context/runtime-assessment-reliability-diagnostic-plan-v1.json',
            directory=directory,requests=directory/'requests-v1.json',controls=directory/'controls-v1.json',
            run=directory/'hosted-run',calls=32,cap=Decimal('0.277142592'),baseline_calls=1244,
            baseline_usd='0.1549251950',prompts=(6,7))
    if revision == 8:
        directory=ROOT/'.data/application_context/runtime-evidence-assessment-diagnostic-v1'
        return SimpleNamespace(revision=8,
            plan=ROOT/'docs/experiments/phase-9-application-context/runtime-evidence-assessment-diagnostic-plan-v1.json',
            directory=directory,requests=directory/'requests-v1.json',controls=directory/'controls-v1.json',
            run=directory/'hosted-run',calls=49,cap=Decimal('0.04820522'),baseline_calls=1195,
            baseline_usd='0.1506036725',prompts=(6,))
    if revision == 7:
        directory=ROOT/'.data/application_context/runtime-tongyi-reasoning-diagnostic-v1'
        return SimpleNamespace(revision=7,
            plan=ROOT/'docs/experiments/phase-9-application-context/runtime-tongyi-reasoning-diagnostic-plan-v1.json',
            directory=directory,requests=directory/'requests-v1.json',controls=directory/'controls-v1.json',
            run=directory/'hosted-run',calls=98,cap=Decimal('0.09641044'),baseline_calls=1097,
            baseline_usd='0.1303510625',prompts=(4,))
    if revision == 6:
        directory=ROOT/'.data/application_context/runtime-tongyi-judge-diagnostic-v1'
        return SimpleNamespace(revision=6,
            plan=ROOT/'docs/experiments/phase-9-application-context/runtime-tongyi-judge-diagnostic-plan-v1.json',
            directory=directory,requests=directory/'requests-v1.json',controls=directory/'controls-v1.json',
            run=directory/'hosted-run',calls=49,cap=Decimal('0.03509576'),baseline_calls=1048,
            baseline_usd='0.127250525',prompts=(4,))
    if revision == 1:
        return SimpleNamespace(revision=1, plan=PLAN, directory=DIRECTORY, requests=REQUESTS,
            controls=CONTROLS, run=RUN, calls=CALLS, cap=CAP, baseline_calls=832,
            baseline_usd='0.093352275', prompts=(1, 2))
    if revision == 3:
        directory = ROOT / '.data/application_context/runtime-semantic-resolution-diagnostic-v1'
        return SimpleNamespace(revision=3,
            plan=ROOT / 'docs/experiments/phase-9-application-context/runtime-semantic-resolution-diagnostic-hosted-plan-v1.json',
            directory=directory, requests=directory/'requests-v1.json', controls=directory/'controls-v1.json',
            run=directory/'hosted-run', calls=98, cap=Decimal('0.1617'), baseline_calls=898,
            baseline_usd='0.107133775', prompts=(4, 5))
    if revision == 4:
        directory = ROOT / '.data/application_context/runtime-reasoning-budget-diagnostic-v1'
        return SimpleNamespace(revision=4,
            plan=ROOT / 'docs/experiments/phase-9-application-context/runtime-reasoning-budget-diagnostic-hosted-plan-v1.json',
            directory=directory, requests=directory/'requests-v1.json', controls=directory/'controls-v1.json',
            run=directory/'hosted-run', calls=98, cap=Decimal('0.247352'), baseline_calls=996,
            baseline_usd='0.127250525', prompts=(4,))
    if revision == 5:
        directory=ROOT/'.data/application_context/runtime-local-judge-diagnostic-v1'
        return SimpleNamespace(revision=5,
            plan=ROOT/'docs/experiments/phase-9-application-context/runtime-local-judge-diagnostic-plan-v1.json',
            directory=directory,requests=directory/'requests-v1.json',controls=directory/'controls-v1.json',
            run=directory/'local-run',calls=49,cap=Decimal('0'),baseline_calls=999,
            baseline_usd='0.127250525',prompts=(4,))
    directory = ROOT / '.data/application_context/runtime-report-residuals-v1'
    return SimpleNamespace(revision=2,
        plan=ROOT / 'docs/experiments/phase-9-application-context/runtime-report-residuals-hosted-plan-v1.json',
        directory=directory, requests=directory/'requests-v1.json', controls=directory/'controls-v1.json',
        run=directory/'hosted-run', calls=23, cap=Decimal('0.03795'), baseline_calls=872,
        baseline_usd='0.102426275', prompts=(2, 3))


def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def baseline(*, revision=1):
    selected = profile(revision)
    budget = study.Budget(SimpleNamespace(path=LEDGER), max_calls=selected.baseline_calls, max_usd='1')
    if budget.attempts != selected.baseline_calls or budget.pending or budget.spent != Decimal(selected.baseline_usd):
        raise ValueError(f'Expected the completed {selected.baseline_calls}-attempt baseline without pending calls')
    return budget.summary()


def protocol_for(summary, *, revision=1):
    # Historical profiles retain their specifications; new profiles opt in explicitly.
    spec = study.read(study.FIXTURES / 'hybrid-screen-v4.json')['judge']
    selected = profile(revision)
    if revision==6:
        spec=tongyi_judge_spec()
    if revision in (7,8):
        spec=tongyi_reasoning_spec(False)
    if revision==9:
        spec=tongyi_reliability_spec()
    if revision==10:
        spec=tongyi_binding_spec()
    if revision==5:
        spec=dict(study.read(study.FIXTURES/'hybrid-screen-v4.json')['generator'],json_mode=True,
                  chat_timeout_s=300,model_metadata_timeout_s=60)
    if revision == 4:
        spec = dict(spec, call_policy=dict(version=1, max_tokens=2048, temperature=None))
    output_limit = study.memory_judge_options({'judge': spec})['max_tokens']
    ceiling = (Decimal(spec['input_rate']) * spec['max_input'] + Decimal(spec['output_rate']) * billable_output_limit(spec,output_limit)) / 1_000_000
    if ceiling * selected.calls != selected.cap:
        raise ValueError('Diagnostic rates differ from the frozen cost bound')
    base = (study.runtime_memory_protocol(contract_version=2, prompt_revision=4) if revision in (3, 4, 5, 6, 7)
            else study.runtime_memory_protocol())
    if revision in (8,9):
        base=study.runtime_memory_protocol(contract_version=3,prompt_revision=6)
    if revision==10:
        base=study.runtime_memory_protocol(contract_version=4,prompt_revision=8)
    result = dict(base, screening=True, judge=spec,
        budget=dict(max_calls=summary['provider_attempts'] + selected.calls,
                    max_usd='0' if revision==5 else str(Decimal(summary['accounted_usd']) + selected.cap), local_only=revision==5))
    if revision==5:
        result['local_only']=True
    if revision != 1:
        result['runtime_report_diagnostic_revision'] = revision
    return result


def requests(controls, protocol):
    selected = profile(protocol.get('runtime_report_diagnostic_revision', 1))
    by_id = {item['id']: item for item in controls}
    rows = []
    for entry in diagnostic.schedule(controls, revision=selected.revision):
        item = by_id[entry['control']]
        current = diagnostic.condition_protocol(protocol, entry)
        if selected.revision==10:
            messages=study.memory_interpretation_messages({'app':item['app']},item['capture'],item['response'],protocol=current)
            options=dict(**study.memory_judge_options(current),**hosted_request_options(current['judge']))
            bound=study.request_input_bound(messages,response_schema=current['judge']['response_schema'])
            if bound>current['judge']['max_input']:
                raise ValueError('Interpretation request exceeds the frozen input ceiling')
            rows.append(dict(entry,stage='interpretation',messages=[study.asdict(m) for m in messages],
                bounded_input_bytes=bound,**options))
            fixed=study.runtime_control_evidence(item,current)
            rows.append(dict(entry,stage='assessment-template',system_prompt=current['memory_prompt'],
                fixed_payload=fixed,interpretation_binding='Parse the actual first completion; validate text anchors and input SHA; insert only its validated interpretation. No assessment call after invalid interpretation.',
                max_input_bytes=current['judge']['max_input'],**options))
            continue
        messages = study.memory_judge_messages({'app': item['app']}, None,
            item['capture'], item['response'], None, protocol=current)
        bound = study.request_input_bound(messages,response_schema=current['judge'].get('response_schema'))
        if bound > protocol['judge']['max_input']:
            raise ValueError('Diagnostic input exceeds the unchanged request ceiling')
        row = dict(entry, messages=[study.asdict(message) for message in messages],
                   bounded_input_bytes=bound, **study.memory_judge_options(current))
        if selected.revision in (6,7,8,9):
            row.update(hosted_request_options(current['judge']))
        if selected.revision == 4:
            row.update(reasoning_effort=current['judge']['reasoning_effort'], json_mode=current['judge']['json_mode'])
        rows.append(row)
    if len(rows) != selected.calls:
        raise ValueError('Diagnostic request count differs from its frozen study')
    return rows


def fields(controls, protocol, cfg, summary, *, revision=1):
    selected = profile(revision)
    result = dict(schema=f'runtime-report-diagnostic-hosted-plan-v{revision}', status='prepared-pending-explicit-provider-approval',
        purpose='Exposed development comparison; no qualification or runtime activation', qualification_eligible=False,
        fixture_path=(diagnostic.RELIABILITY if revision in (9,10) else diagnostic.SEMANTIC if revision in (3, 4, 5, 6, 7, 8) else diagnostic.BANK).relative_to(ROOT).as_posix(),
        fixture_sha256=study.shared.digest(diagnostic.RELIABILITY if revision in (9,10) else diagnostic.SEMANTIC if revision in (3, 4, 5, 6, 7, 8) else diagnostic.BANK),
        controls_path=selected.controls.relative_to(ROOT).as_posix(), controls_sha256=study.shared.digest(selected.controls),
        requests_path=selected.requests.relative_to(ROOT).as_posix(), requests_sha256=study.shared.digest(selected.requests),
        case_identity=study.shared.stable_key(controls), code_sha256=study.fingerprints(),
        diagnostic_owner_sha256=study.shared.digest(Path(diagnostic.__file__)), runner_sha256=study.shared.digest(Path(__file__)),
        runtime_identity=study.runtime_identity(cfg), baseline_ledger_sha256=study.shared.digest(LEDGER), baseline=summary,
        protocol=protocol, endpoint=ENDPOINT, prompt_revisions=list(selected.prompts),
        prompt_sha256={str(n): study.shared.stable_key(diagnostic.arm_protocol({}, n)) for n in selected.prompts},
        additional_attempts=selected.calls, shared_attempt_cap=protocol['budget']['max_calls'],
        additional_conservative_usd_cap=str(selected.cap), shared_conservative_usd_cap=protocol['budget']['max_usd'],
        prior_shared_approved_attempt_cap=895, unused_prior_scope_not_reused=True,
        scope='20 exposed/authored English/Chinese development cases, once per prompt revision. Includes the exact first qualification failure; excludes all other qualification controls.',
        payload_fields=['APPLICATION', 'QUERY', 'VISIBLE_CONTEXT', 'REPLY'],
        payload_excludes=['expected labels', 'reference judgments/evidence', 'author rationales', 'private histories'],
        scoring='All five settled criteria and decisions must match; ambiguity has no boolean gold. Explicit unrun denominators and usefulness/refusal counts. Never qualifies the verifier.',
        stopping=f'Development comparison assesses all {selected.calls} conditions, including semantic errors; abort on execution/audit/budget error. No retries, fallback, cache reuse, readiness call or automatic continuation.',
        pricing=dict(checked_utc_date='2026-10-22', source='https://developers.openai.com/api/docs/models/gpt-6-luna',
                     bound='All input at the higher default cache-write rate; no discount assumed. Frozen-rate API token-charge estimate, excluding tax/billing reconciliation.'),
        generated_application_replies=0, provider_calls_made=0, runtime_enabled=False)
    if revision == 2:
        result.update(report_diagnostic_revision=revision,
            residuals_path=diagnostic.RESIDUALS.relative_to(ROOT).as_posix(), residuals_sha256=study.shared.digest(diagnostic.RESIDUALS),
            scope='All 20 existing exposed/authored English/Chinese controls under v3, plus the three exact v2 residuals repeated under unchanged v2. No other qualification controls.',
            comparison_limit='The earlier full v2 run is a historical baseline. Only three residuals have contemporary v2/v3 pairs; no full paired current-model comparison or qualification is claimed.')
    if revision in (3, 4):
        manifest = study.read(diagnostic.SEMANTIC)
        result.update(report_diagnostic_revision=3, prior_shared_approved_attempt_cap=959,
            semantic_source_pins=manifest['inputs'], distinct_cases=len(controls),
            exact_observed_controls=manifest['exact_observed_control_ids'],
            source_family_exclusions=manifest['newly_exposed_source_families'],
            scope='All 49 unchanged exposed regression cases under both contract-v2 prompt revisions 4 and 5: 98 distinct case/prompt conditions. Includes three actual outcomes, reuses twenty earlier development controls and adds twenty-seven authored contrasts without duplicate payloads within an arm.',
            reference_limit='Only the original three observed cases retain earlier independently blinded LLM references. Other development labels are preparing-agent authored targets; no new independent adjudication or human gold.',
            comparison_limit='Complete contemporary paired development comparison, one assessment per case/arm with alternating arm order. No repeatability, population accuracy, fresh qualification or runtime activation claim.',
            candidate_screen='All 49 candidate outcomes must fully match: settled decision and all five criteria, or inconclusive for ambiguity with no Boolean gold. Report paired improvements and regressions against unchanged v4 separately. Baseline failures do not erase candidate matches or qualify the verifier.',
            pricing=dict(result['pricing'], checked_utc_date='2026-10-23'))
    if revision == 4:
        previous = profile(3).run/'report.json'
        result.update(report_diagnostic_revision=4, prior_shared_approved_attempt_cap=996,
            purpose='Isolate none versus low reasoning at fixed v4 prompt and equal total completion budget; exposed diagnostic only',
            previous_diagnostic_report=dict(path=previous.relative_to(ROOT).as_posix(), sha256=study.shared.digest(previous)),
            inference_conditions={effort: diagnostic.condition_protocol(protocol, dict(prompt_revision=4, inference_arm=effort))['judge']
                                  for effort in ('none', 'low')},
            scope='The same 49 exposed English/Chinese regression cases, paired under fixed contract-v2 prompt-v4 and reasoning none/low. Both arms omit temperature and use max_completion_tokens=2048, including hidden reasoning. No newly exposed cases or modified references.',
            comparison_limit='One contemporary none/low assessment per case, alternating arm order; no repeatability, population accuracy or qualification. Historical none/300/temperature=0 results are not an equivalent control.',
            candidate_screen='All 49 low-arm outcomes must fully match settled decisions and five criteria, or coherent inconclusive for ambiguity. Report none-arm outcomes, paired improvements/regressions, empty and length-limited outputs, total completion tokens and their reasoning subset. Never activate the runtime guard.',
            reasoning_budget_limit='2048 total completion tokens is a cost-constrained operating point, not evidence that low reasoning has enough budget. Truncations receive no passing credit and are counted as budget-limited; empty outputs are counted separately. No retries or automatic token-budget increase.',
            official_api_references=[
                'https://developers.openai.com/api/docs/models/gpt-6-luna',
                'https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create',
                'https://developers.openai.com/api/docs/guides/reasoning'],
            documentation_limit='Official reasoning guidance recommends initially allowing at least 25000 tokens for reasoning/output. This smaller fixed limit deliberately tests an affordable operating point and cannot establish an unrestricted model capability limit.',
            billing='completion_tokens already includes reasoning_tokens; charge it once at the output rate. Accounted USD is a conservative frozen-rate token estimate; no tax or invoice claim.')
    if revision==5:
        manifest=study.read(diagnostic.SEMANTIC)
        health=ROOT/'.data/application_context/runtime-local-judge-v1/health-result-v3.json'
        result.update(schema='runtime-local-judge-diagnostic-plan-v1',status='prepared-for-authorized-local-execution',
            local_only=True,report_diagnostic_revision=5,purpose='Evaluate the installed local Qwen judge under unchanged prompt v4',
            endpoint=protocol['judge']['base_url'],distinct_cases=len(controls),semantic_source_pins=manifest['inputs'],
            prior_shared_approved_attempt_cap=996,shared_conservative_usd_cap=summary['accounted_usd'],
            health_result=dict(path=health.relative_to(ROOT).as_posix(),sha256=study.shared.digest(health)),
            scope='The same 49 exposed English/Chinese cases, once each, using pinned local qwen2.5:7b, contract 2 / prompt 4 / decision policy 1. No hosted requests, downloads, changed references or generated application replies.',
            reference_limit='Most references are authored development targets; three retain earlier independent blinded LLM references. No new human gold or independent adjudication.',
            comparison_limit='Single local assessment per case. Results apply to this local configuration; historical hosted results are not contemporary paired controls. No repeatability or qualification claim.',
            candidate_screen='All 49 settled decisions and five flags, or coherent inconclusive for ambiguity, must match. Missing, malformed and length-limited assessments receive no passing credit.',
            pricing=dict(kind='Zero external API spend; hardware and electricity costs unmeasured'),
            stopping='At most 49 serial local assessments; abort on transport, identity, usage, audit or budget failure. No retry, download, fallback, hosted call or automatic continuation. Persist partial progress on failure.')
    if revision in (6,7,8,9,10):
        manifest=study.read(diagnostic.SEMANTIC)
        result.update(schema='runtime-tongyi-judge-diagnostic-plan-v1',report_diagnostic_revision=6,
            endpoint=protocol['judge']['base_url'],distinct_cases=len(controls),semantic_source_pins=manifest['inputs'],
            prior_shared_approved_attempt_cap=1048,
            purpose='Screen pinned hosted Qwen3.7 Flash under unchanged contract 2 / prompt 4 / policy 1',
            scope='The same 49 exposed public-source/authored English/Chinese context/query/reply cases, once each. No expected labels, reference judgments, author rationales or private histories are sent.',
            provider_transport='Existing OpenAI-compatible transport under tongyi identity; native Alibaba application defaults are unchanged.',
            key_region='Beijing, matching the existing native adapter default. Authentication/model access is unverified; first scheduled request tests access. No cross-region fallback or separate health request.',
            reference_limit='Most references are authored targets; three retain earlier independent blinded LLM references. Not human gold or fresh qualification.',
            comparison_limit='Single assessment per exposed case. Different model and output budget from historical runs; no controlled model-effect, repeatability or population accuracy claim.',
            candidate_screen='All 49 settled decisions and five flags, or coherent inconclusive for ambiguity, must match. No runtime activation even on a complete development pass.',
            inference_options=hosted_request_options(protocol['judge'])['inference_options'],
            official_api_references=[
                'https://www.alibabacloud.com/help/en/model-studio/compatibility-of-openai-with-dashscope',
                'https://help.aliyun.com/en/model-studio/deep-thinking'],
            pricing=dict(checked_utc_date='2026-10-23',source='https://www.alibabacloud.com/help/en/model-studio/model-pricing',
                bound='Conservative international input .030 multiplied by 1.25 for cache-write allowance, output .130. These exceed published Beijing short-context .028/.110. No discount assumed; output charged once using total completion tokens. Frozen-rate estimate, excluding taxes/invoice reconciliation.'),
            stopping='At most 49 serial requests. Abort on transport, identity, usage, audit or budget failure; preserve partial evidence. No retry, redirect, region switch, fallback, cache reuse, extra health request or automatic continuation.')
    if revision==7:
        previous=profile(6).run/'report.json'
        result.update(schema='runtime-tongyi-reasoning-diagnostic-plan-v1',report_diagnostic_revision=7,
            prior_shared_approved_attempt_cap=1097,
            purpose='Measure Qwen3.7 Flash thinking off/on with fixed prompt, model and total-output limit',
            previous_diagnostic_report=dict(path=previous.relative_to(ROOT).as_posix(),sha256=study.shared.digest(previous)),
            scope='The same 49 exposed public-source/authored English/Chinese context/query/reply cases, once per thinking off/on condition: 98 serial requests with alternating arm order. No new cases, labels, reference rationales or private histories are sent.',
            inference_conditions={arm:diagnostic.condition_protocol(protocol,dict(prompt_revision=4,inference_arm=arm))['judge']
                                  for arm in diagnostic.reasoning_arms(7)},
            inference_options={arm:hosted_request_options(diagnostic.condition_protocol(protocol,
                dict(prompt_revision=4,inference_arm=arm))['judge'])['inference_options'] for arm in diagnostic.reasoning_arms(7)},
            key_region='The same Beijing endpoint and configured key used successfully by the previous 49-case run. No endpoint or region switch.',
            comparison_limit='Contemporary off/on pairs hold model, prompt, input, JSON format, temperature and total-output cap fixed. One assessment per case/arm does not establish repeatability or fresh generalization. The older 2048-answer-token run is historical context only.',
            candidate_screen='All 49 on-arm outcomes must fully match settled decision and five criteria, or coherent inconclusive for ambiguity. Count all paired gains and regressions. A development pass never activates runtime enforcement.',
            output_budget='Both arms send max_completion_tokens=4096, including reasoning and answer. Reserve and audit up to 4106 output tokens for the documented ten-token tolerance; never send max_tokens as an additional limit. Count length-limited/empty outputs and overshoots separately. No automatic budget increase or format repair.',
            reasoning_evidence='The on arm requires reported positive integer reasoning tokens within total completion usage and nonempty reasoning_content evidence. Missing or inconsistent evidence aborts. Off-arm unreported reasoning remains unknown. Charge total completion tokens once; never add the reasoning subset again.',
            official_api_references=[
                'https://help.aliyun.com/en/model-studio/qwen-api-via-openai-chat-completions',
                'https://help.aliyun.com/en/model-studio/qwen-structured-output',
                'https://www.alibabacloud.com/help/en/model-studio/model-pricing'],
            stopping='At most 98 serial assessments; persist each result. Abort on transport, identity, reasoning-evidence, usage, audit or budget failure. No retry, redirect, fallback, cache reuse, extra health request, JSON repair or automatic continuation.')
    if revision==8:
        previous=profile(7).run/'report.json'
        review=ROOT/'docs/experiments/phase-9-application-context/runtime-evidence-assessment-review-v1.json'
        result.update(schema='runtime-evidence-assessment-diagnostic-plan-v1',report_diagnostic_revision=8,
            prior_shared_approved_attempt_cap=1195,
            purpose='Screen caller-bound complete-reply evidence assessments through the existing single-call judge workflow',
            previous_diagnostic_report=dict(path=previous.relative_to(ROOT).as_posix(),sha256=study.shared.digest(previous)),
            implementation_review=dict(path=review.relative_to(ROOT).as_posix(),sha256=study.shared.digest(review)),
            scope='The same 49 exposed public-source/authored English/Chinese context/query/reply cases, once each, using contract 3 / prompt 6 / decision policy 1 and thinking off. Context is the same delivered text decoded into partial-history records with positional IDs; no new facts or expected labels are added.',
            payload_contract='Four existing fields; VISIBLE_CONTEXT becomes {coverage:partial,records:[id,speaker,session,text]}. APPLICATION, QUERY and complete REPLY retain their actual values.',
            candidate_screen='All 49 settled decisions and five derived criteria must match, or coherent inconclusive for ambiguity. Invalid coverage, citations, actor evidence and truncated outputs receive no passing credit. An exposed pass cannot qualify or activate runtime enforcement.',
            comparison_limit='Single assessment per exposed case. The previous off/on study is preserved historical context, not a contemporary paired comparator. No repeatability, fresh qualification or general causal improvement claim.',
            evidence_limit='Verbatim segmentation proves textual coverage, not that the model identified every assertion. Citations prove source identity and quote presence, not semantic entailment, correct report-chain interpretation or enumeration of every plausible reading. Model interpretation and usefulness/transfer assessments still require empirical qualification.',
            key_region='Same Beijing endpoint and configured key as the completed 98-request comparison; no region switch or separate health call.',
            output_budget='Thinking off; max_completion_tokens=4096, with reservation and audit ceiling 4106 for documented billing tolerance. No increase, repair, retry or automatic continuation.',
            stopping='At most 49 serial requests, including semantic errors. Abort on transport, identity, usage, audit or budget failure; preserve partial results. No retry, redirect, fallback, health request, format repair or automatic continuation.')
    if revision==9:
        bank=study.read(diagnostic.RELIABILITY)
        review=ROOT/'docs/experiments/phase-9-application-context/runtime-assessment-reliability-review-v1.json'
        result.update(schema='runtime-assessment-reliability-diagnostic-plan-v1',report_diagnostic_revision=9,
            prior_shared_approved_attempt_cap=1244,purpose='Diagnose structural and semantic assessment failures with pinned Plus and a same-contract v6/v7 pair',
            implementation_review=dict(path=review.relative_to(ROOT).as_posix(),sha256=study.shared.digest(review)),
            semantic_source_pins=bank['inputs'],
            scope='Sixteen already-exposed public-source/authored English/Chinese context/query/reply cases, once per prompt 6/7 condition. Thirty-two requests to Qwen3.7 Plus at the existing Beijing endpoint, thinking off and strict JSON Schema. No expected labels, reference judgments, author rationales or private histories are sent.',
            comparison_limit='Contemporary prompt pairs hold model, contract 3, schema, inputs, output limit, decoding and action policy fixed. Historical Flash outcomes change model and format as well, so they do not isolate a model effect. Sixteen selected exposed controls are a diagnostic, not a full 49-case pass, fresh qualification, repeatability or production accuracy.',
            candidate_screen='All sixteen candidate decisions and five criteria, or coherent inconclusive for ambiguity, must match. Report every paired gain and regression, invalid output, useful-supported loss and unsupported acceptance. Never enables runtime enforcement.',
            provider_transport='Existing compatible provider, router, recorder, budget, scorer and executor; optional schema only. Native application defaults are unchanged.',
            key_region='Existing Beijing endpoint and configured key. Model access and schema support are documented but unverified for this account. The first scheduled request tests access; abort without retry or endpoint/format fallback on error.',
            schema_limit='Strict JSON Schema is requested, not treated as semantic proof. Full schema bytes are included in conservative input bounds and the exact request freeze. Existing parser still checks complete text, actual citations, outer actors, transfer evidence and caller identity.',
            inference_options=hosted_request_options(protocol['judge'])['inference_options'],
            pricing=dict(checked_utc_date='2026-10-23',source='https://www.alibabacloud.com/help/en/model-studio/model-pricing',
                bound='Beijing snapshot input .276/M multiplied by 1.25 for conservative cache-write allowance = .345/M; output 1.101/M. No promotional discount or cache hit assumed. Total completion usage is charged once. Conservative token estimate, not an invoice.'),
            official_api_references=['https://help.aliyun.com/en/model-studio/qwen-structured-output',
                'https://help.aliyun.com/en/model-studio/qwen3-7-plus',
                'https://help.aliyun.com/en/model-studio/qwen-api-via-openai-chat-completions'],
            output_budget='Send max_completion_tokens=4096, reserve/audit 4106 including the documented tolerance; incomplete outputs get no passing credit. Keep a hard limit despite documentation recommending uncapped structured output to avoid truncation. No format repair or automatic retry.',
            stopping='At most 32 serial requests, alternating prompt order. Preserve every response and partial result. Abort on transport, identity, schema-request evidence, usage, audit or budget error. Semantic mismatches remain recorded through completion. No retry, redirect, fallback, cache reuse, extra health request or automatic continuation.')
    if revision==10:
        bank=study.read(diagnostic.RELIABILITY)
        previous=profile(9).run/'report.json'
        review=ROOT/'docs/experiments/phase-9-application-context/runtime-claim-binding-review-v1.json'
        result.update(schema='runtime-claim-binding-diagnostic-plan-v1',report_diagnostic_revision=10,
            prior_shared_approved_attempt_cap=1276,purpose='Measure source-blind interpretation followed by fixed-claim evidence assessment',
            implementation_review=dict(path=review.relative_to(ROOT).as_posix(),sha256=study.shared.digest(review)),
            previous_diagnostic_report=dict(path=previous.relative_to(ROOT).as_posix(),sha256=study.shared.digest(previous)),
            semantic_source_pins=bank['inputs'],max_calls_per_case=2,distinct_cases=16,
            scope='The same sixteen exposed public-source/authored EN/ZH cases. First call receives application, current query and complete candidate reply only. Second receives the same fields, actual delivered context records and the validated first-call interpretation. At most 32 Qwen3.7 Plus Beijing requests, strict JSON Schema and thinking off. Expected labels, reference judgments, author rationales and private histories are excluded.',
            payload_fields=dict(interpretation=['APPLICATION','QUERY','REPLY'],assessment=['APPLICATION','QUERY','VISIBLE_CONTEXT','REPLY','INTERPRETATION']),
            request_freeze_kind='Exact interpretation requests and fixed assessment inputs, schema, prompt and transformation; assessment INTERPRETATION is bounded actual first-call output, not known before execution.',
            dynamic_input_policy='Only validated verbatim reply spans, actor/kind/engagement enums and caller-computed input identity enter stage two. The second-stage input bound including schema is checked before dispatch. Invalid interpretation skips stage two and receives no passing credit. Oversized second-stage input aborts without truncation or another call.',
            scoring='All sixteen final settled decisions and five criteria, or coherent inconclusive for ambiguity, must match. Count physical calls, both stages tokens/cost, skipped second stages, invalid interpretations/evidence and protected useful replies. No qualification or runtime activation.',
            comparison_limit='Historical single-pass prompt 6/7 results are context only. Contract, staging and prompts change together; this is not an isolated causal ablation or a full 49-case, fresh-source or repeatability qualification.',
            evidence_limit='Source blindness is an input boundary, not statistical independence or proof of correct extraction. A mistaken claim inventory can still omit or misclassify history. Exact anchors/citations do not prove entailment. Explanation consistency is deterministic, while semantic correctness remains empirical.',
            candidate_screen='All sixteen unchanged reference outcomes must match. Missing, invalid, truncated or incomplete stage results get no passing credit. No automatic expansion to other cases or provider calls.',
            inference_options=hosted_request_options(protocol['judge'])['inference_options'],
            pricing=dict(checked_utc_date='2026-10-23',source='https://www.alibabacloud.com/help/en/model-studio/model-pricing',
                bound='Same frozen Plus Beijing list rates as the completed comparison: input .345/M including conservative 1.25 cache-write allowance; output 1.101/M. Reserve 12000 input / 4106 output per physical call, at most 32 calls. No discount or invoice claim.'),
            stopping='At most two serial calls per case and 32 total. Invalid first stages are retained and skip the second. Abort on transport, request/identity/input-bound/usage/audit/budget error, retaining partial evidence. No retry, redirect, fallback, cache reuse, health call, JSON repair, regeneration or automatic continuation.')
    return result


def prepare(*, revision=1):
    selected = profile(revision)
    if selected.plan.exists() or selected.directory.exists():
        raise ValueError('Diagnostic plan/directory already exists; refuse overwrite')
    controls = diagnostic.items(revision=revision)
    cfg = study.load_config()
    summary = baseline(revision=revision)
    protocol = protocol_for(summary, revision=revision)
    outbound = requests(controls, protocol)
    selected.directory.mkdir()
    save(selected.controls, controls)
    save(selected.requests, outbound)
    save(selected.plan, dict(fields(controls, protocol, cfg, summary, revision=revision), timestamp_utc=datetime.now(timezone.utc).isoformat()))


def validate(plan_path=None, *, revision=1):
    selected = profile(revision)
    plan_path = selected.plan if plan_path is None else plan_path
    plan = study.read(plan_path)
    controls = diagnostic.items(revision=revision)
    cfg = study.load_config()
    summary = baseline(revision=revision)
    protocol = protocol_for(summary, revision=revision)
    if study.read(selected.controls) != controls or study.read(selected.requests) != requests(controls, protocol):
        raise ValueError('Diagnostic controls or outbound requests changed')
    expected = fields(controls, protocol, cfg, summary, revision=revision)
    if set(plan) != set(expected) | {'timestamp_utc'} or any(plan.get(key) != value for key, value in expected.items()):
        raise ValueError('Exact diagnostic plan/model/code/source/budget freeze drift')
    return plan, cfg, controls, None, protocol


def run(approval_path, plan_path=None, *, revision=1):
    selected = profile(revision)
    options = {'judge_factory': diagnostic.make_reasoning_judges} if revision in (4,7) else {}
    if revision==5:
        options['local_only']=True
    return study.execute_runtime_study(approval_path, selected.plan if plan_path is None else plan_path,
        validator=lambda path: validate(path, revision=revision),
        run_directory=selected.run, assess=diagnostic.run, ledger=LEDGER, **options)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'check', 'run'])
    parser.add_argument('--approval', type=Path)
    parser.add_argument('--revision', type=int, choices=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10), default=1)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(revision=args.revision)
        selected = profile(args.revision)
        print(f'Frozen {len(study.read(selected.controls))} development cases / {selected.calls} maximum requests / US${selected.cap}; no provider calls.')
    elif args.command == 'check':
        validate(revision=args.revision)
        print('Exact paired diagnostic verified offline; no provider calls.')
    elif args.approval is None and args.revision!=5:
        parser.error('run requires explicit approval of the exact frozen diagnostic plan')
    else:
        run(args.approval, revision=args.revision)
