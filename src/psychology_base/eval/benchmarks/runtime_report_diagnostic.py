"""Exposed development contrasts over the existing memory assessment workflow.

Authored references diagnose regressions; they never qualify a runtime verifier.
"""
import copy
import json
from decimal import Decimal
from pathlib import Path

from psychology_base.eval.benchmarks import application_context as study

BANK = study.FIXTURES / 'runtime-report-diagnostic-v1.json'
RESIDUALS = study.FIXTURES / 'runtime-report-residuals-v1.json'
SEMANTIC = study.FIXTURES / 'runtime-semantic-resolution-v1.json'
RELIABILITY = study.FIXTURES / 'runtime-assessment-reliability-v1.json'


def items(*, revision=1):
    if type(revision) is not int or revision not in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10):
        raise ValueError('Report diagnostic revision must be 1 through 10')
    if revision == 10:
        return items(revision=9)
    if revision == 9:
        bank=study.read(RELIABILITY)
        for pin in bank['inputs'].values():
            if study.shared.digest(study.ROOT/pin['path']) != pin['sha256']:
                raise ValueError('Reliability diagnostic source differs from its freeze')
        by_id={item['id']:item for item in semantic_items()}
        ids=bank['selected_case_ids']
        if (bank['schema']!='runtime-assessment-reliability-v1' or bank['qualification_eligible'] is not False
                or len(ids)!=16 or len(set(ids))!=16 or any(key not in by_id for key in ids)):
            raise ValueError('Reliability diagnostic requires its sixteen exposed controls')
        return [by_id[key] for key in ids]
    if revision in (3, 4, 5, 6, 7, 8):
        return semantic_items()
    bank = study.read(BANK)
    if bank['schema'] != 'runtime-report-diagnostic-v1' or bank['qualification_eligible'] is not False:
        raise ValueError('Report diagnostic must remain an exposed development study')
    inputs = {}
    for name, pin in bank['inputs'].items():
        path = study.ROOT / pin['path']
        if study.shared.digest(path) != pin['sha256']:
            raise ValueError('Report diagnostic source differs from its freeze')
        inputs[name] = study.read(path)
    failed = next(item for item in inputs['controls']['controls'] if item['id'] == bank['failed_control'])
    report = inputs['failed_report']
    if (len(report['rows']) != 1 or report['rows'][0]['control'] != failed['id']
            or report['rows'][0]['judgment']['decision'] != 'accept'
            or failed['expected_decision'] != 'reject'):
        raise ValueError('Report diagnostic must retain the actual observed false acceptance')
    result = [dict(copy.deepcopy(failed), qualification_eligible=False,
                   origin='exact_exposed_failed_control')]
    for case in inputs['development_bank']['cases'] + bank['additional_cases']:
        context = (failed['capture']['context'] if case.get('context_from_failed_control') is True else
                   '\n'.join(json.dumps(record, ensure_ascii=False) for record in case['records']))
        result.append(dict(id=case['id'], language=case['language'], app=case['app'],
            source='msc_456' if case in bank['additional_cases'] else 'exposed-authored-development',
            capture=dict(query=case['query'], context=context), response=case['response'],
            expected_decision=case['expected_decision'], expected_criteria=copy.deepcopy(case['expected_criteria']),
            rationale=case['rationale'], authored_control=True, qualification_eligible=False,
            origin='authored_development_contrast'))
    if len(result) != 20 or len({item['id'] for item in result}) != 20:
        raise ValueError('Report diagnostic requires the exact 20 distinct controls')
    return result


def semantic_items():
    """Reuse old controls and actual exposed outcomes; never relabel a holdout."""
    bank = study.read(SEMANTIC)
    if (bank['schema'] != 'runtime-semantic-resolution-regressions-v1'
            or bank['qualification_eligible'] is not False
            or bank['reuse_all_twenty_development_controls'] is not True):
        raise ValueError('Semantic diagnostics require the frozen exposed regression manifest')
    inputs = {}
    for name, pin in bank['inputs'].items():
        path = study.ROOT / pin['path']
        if study.shared.digest(path) != pin['sha256']:
            raise ValueError('Semantic diagnostic source differs from its freeze')
        inputs[name] = study.read(path)
    result = items()
    if inputs['reused_development'] != result:
        raise ValueError('Semantic diagnostic changed a reused development reference')
    report = inputs['report']
    observed = [row['control'] for row in report['rows']]
    if (observed != bank['exact_observed_control_ids'] or len(observed) != 3
            or report['passed'] is not False or report['runtime_enabled'] is not False
            or report['rows'][-1]['control'] != bank['failed_control']
            or report['rows'][-1]['judgment']['decision'] != 'accept'):
        raise ValueError('Semantic diagnostic must retain the actual stopped failure')
    original = {item['id']: item for item in inputs['controls']['controls']}
    study.validate_runtime_qualification_review(inputs['controls']['controls'], inputs['reference_review'])
    by_id = {item['id']: item for item in result}
    for identifier in observed:
        item = original[identifier]
        if identifier in by_id:
            if any(by_id[identifier][key] != item[key] for key in
                   ('capture', 'response', 'expected_decision', 'expected_criteria')):
                raise ValueError('Observed semantic reference differs from its original control')
        else:
            by_id[identifier] = dict(copy.deepcopy(item), qualification_eligible=False)
    if original[bank['failed_control']]['expected_decision'] != 'inconclusive':
        raise ValueError('The actual ambiguity failure must not be relabeled')
    for item in bank['additional_cases']:
        if item['id'] in by_id or item.get('qualification_eligible') is not False:
            raise ValueError('Semantic diagnostic duplicate or qualification claim')
        by_id[item['id']] = copy.deepcopy(item)
    result = list(by_id.values())
    payloads = {study.shared.stable_key(study.runtime_memory_payload(item['app'], item['capture']['query'],
                item['capture']['context'], item['response'])) for item in result}
    if len(result) != 49 or len(payloads) != 49:
        raise ValueError('Semantic diagnostic requires 49 distinct exposed payloads')
    return result


def residuals():
    """Select only the exact observed v2 residuals; never change their labels."""
    bank = study.read(RESIDUALS)
    if bank['schema'] != 'runtime-report-residuals-v1' or bank['qualification_eligible'] is not False:
        raise ValueError('Residual controls must remain exposed development evidence')
    source = {}
    for name, pin in bank['inputs'].items():
        path = study.ROOT / pin['path']
        if study.shared.digest(path) != pin['sha256']:
            raise ValueError('Residual source differs from its frozen actual result')
        source[name] = study.read(path)
    if source['controls'] != items():
        raise ValueError('Residual references differ from the existing controls')
    expected = [dict(control=row['control'], dispatch=row['dispatch'], expected_decision=row['expected_decision'],
        observed_decision=row['observed_decision'], expected_criteria=row['expected_criteria'])
        for row in source['analysis']['mismatches'] if row['prompt_revision'] == 2]
    if bank['residuals'] != expected or len(expected) != 3:
        raise ValueError('Residual selection differs from the three observed mismatches')
    by_id = {item['id']: item for item in source['controls']}
    for entry in expected:
        row = next(row for row in source['report']['rows'] if row['prompt_revision'] == 2 and row['control'] == entry['control'])
        if (row['call']['dispatch'] != entry['dispatch'] or row['judgment']['decision'] != entry['observed_decision']
                or by_id[entry['control']]['expected_decision'] != entry['expected_decision']
                or by_id[entry['control']]['expected_criteria'] != entry['expected_criteria']):
            raise ValueError('Residual reference or actual judgment changed')
    return tuple(row['control'] for row in expected)


def reasoning_arms(revision):
    if type(revision) is not int or revision not in (4,7):
        raise ValueError('Reasoning arms require a paired reasoning diagnostic')
    return ('none','low') if revision==4 else ('off','on')


def schedule(controls, *, revision=1):
    if controls != items(revision=revision):
        raise ValueError('Report diagnostic controls differ from the frozen inputs')
    if revision == 10:
        return [dict(index=index,control=item['id'],prompt_revision=8) for index,item in enumerate(controls)]
    if revision == 8:
        return [dict(index=index,control=item['id'],prompt_revision=6) for index,item in enumerate(controls)]
    if revision in (5, 6):
        return [dict(index=index,control=item['id'],prompt_revision=4) for index,item in enumerate(controls)]
    if revision in (4,7):
        arms=reasoning_arms(revision)
        return [dict(index=index * 2 + arm, control=item['id'], prompt_revision=4, inference_arm=effort)
                for index, item in enumerate(controls)
                for arm, effort in enumerate(arms if index % 2 == 0 else tuple(reversed(arms)))]
    if revision in (1, 3, 9):
        prompts = (1, 2) if revision == 1 else (6, 7) if revision == 9 else (4, 5)
        return [dict(index=index * 2 + arm, control=item['id'], prompt_revision=prompt)
                for index, item in enumerate(controls)
                for arm, prompt in enumerate(prompts if index % 2 == 0 else tuple(reversed(prompts)))]
    selected = set(residuals())
    result = []
    for index, item in enumerate(controls):
        prompts = ((2, 3) if index % 2 == 0 else (3, 2)) if item['id'] in selected else (3,)
        for prompt in prompts:
            result.append(dict(index=len(result), control=item['id'], prompt_revision=prompt))
    return result


def arm_protocol(protocol, revision):
    result = dict(protocol)
    result.pop('runtime_memory_prompt_revision', None)
    result.pop('runtime_memory_decision_policy_version', None)
    result.pop('memory_interpretation_prompt',None)
    result.update(study.runtime_memory_protocol(prompt_revision=revision,
        contract_version=4 if revision==8 else 3 if revision in (6,7) else 2 if revision in (4, 5) else 1))
    return result


def condition_protocol(protocol, entry):
    """Resolve one scheduled condition for dispatch, freezing and raw audit."""
    if protocol.get('runtime_report_diagnostic_revision') == 10:
        if (type(entry.get('prompt_revision')) is not int or entry['prompt_revision']!=8
                or 'inference_arm' in entry or protocol.get('runtime_memory_prompt_revision')!=8):
            raise ValueError('Bound-claim diagnostic fixes the two-stage prompt 8 protocol')
        return arm_protocol(protocol,8)
    if protocol.get('runtime_report_diagnostic_revision') == 9:
        if (type(entry.get('prompt_revision')) is not int or entry['prompt_revision'] not in (6,7)
                or 'inference_arm' in entry or protocol.get('runtime_memory_prompt_revision')!=6):
            raise ValueError('Reliability diagnostic fixes the v6/v7 assessment pair')
        return arm_protocol(protocol,entry['prompt_revision'])
    if protocol.get('runtime_report_diagnostic_revision') == 8:
        if (type(entry.get('prompt_revision')) is not int or entry['prompt_revision'] != 6
                or 'inference_arm' in entry or protocol.get('runtime_memory_prompt_revision') != 6):
            raise ValueError('Evidence assessment diagnostic fixes prompt revision 6')
        return arm_protocol(protocol, 6)
    if protocol.get('runtime_report_diagnostic_revision') in (5, 6):
        if (type(entry.get('prompt_revision')) is not int or entry['prompt_revision']!=4
                or 'inference_arm' in entry or protocol.get('runtime_memory_prompt_revision')!=4):
            raise ValueError('Local diagnostic fixes prompt revision 4')
        return arm_protocol(protocol,4)
    revision=protocol.get('runtime_report_diagnostic_revision',1)
    if revision not in (4,7):
        return arm_protocol(protocol, entry['prompt_revision'])
    if type(entry.get('prompt_revision')) is not int or entry['prompt_revision'] != 4 or entry.get('inference_arm') not in reasoning_arms(revision):
        raise ValueError('Reasoning diagnostic fixes prompt v4 and its declared arms')
    result = arm_protocol(protocol, 4)
    result['judge'] = (dict(protocol['judge'],reasoning_effort=entry['inference_arm']) if revision==4 else
                       dict(protocol['judge'],enable_thinking=entry['inference_arm']=='on'))
    if study.inference_call_options(result['judge']) is None:
        raise ValueError('Reasoning diagnostic requires its explicit bounded call policy')
    return result


def make_reasoning_judges(protocol, cfg, budget):
    """Build two stable existing routers inside the same execution/budget boundary."""
    revision=protocol.get('runtime_report_diagnostic_revision')
    if revision not in (4,7):
        raise ValueError('Reasoning routers require the frozen reasoning diagnostic')
    judges = {}
    for effort in reasoning_arms(revision):
        current = condition_protocol(protocol, dict(prompt_revision=4, inference_arm=effort))
        judges[effort], = study.make_routers(current, cfg, budget, roles=('judge',))
    return judges


def _add_call_metrics(arm,call,revision):
    completion=call['completion']
    arm['budget_limited']+=completion['finish_reason']=='length'
    arm['empty_outputs']+=not completion['content'].strip()
    for field in ('prompt_tokens','completion_tokens'):
        arm[field]+=completion[field]
    if revision==4:
        arm['reasoning_tokens']+=completion['metadata']['reasoning_tokens']
    if revision in (6,7,8,9,10):
        reasoning=completion['metadata'].get('reasoning_tokens')
        arm['reasoning_tokens']=(None if reasoning is None or arm['reasoning_tokens'] is None else arm['reasoning_tokens']+reasoning)
    if revision in (7,8,9,10):
        arm['output_limit_overshoots']+=(completion['completion_tokens']>completion['metadata']['configured_max_completion_tokens'])
    arm['accounted_usd']=str(Decimal(arm['accounted_usd'])+Decimal(call['accounted_usd']))


def metrics(controls, rows, *, revision=1):
    wanted = schedule(controls, revision=revision)
    keys = ('index', 'control', 'prompt_revision', 'inference_arm') if revision in (4,7) else ('index', 'control', 'prompt_revision')
    if [{key: row[key] for key in keys} for row in rows] != wanted[:len(rows)]:
        raise ValueError('Report diagnostic schedule changed')
    by_id = {item['id']: item for item in controls}
    arms = {}
    arm_key = 'inference_arm' if revision in (4,7) else 'prompt_revision'
    for prompt in dict.fromkeys(row[arm_key] for row in wanted):
        planned = [by_id[row['control']] for row in wanted if row[arm_key] == prompt]
        settled = sum(item['expected_criteria'] is not None for item in planned)
        arms[str(prompt)] = dict(total=len(planned), completed=0, settled_total=settled,
        settled_completed=0, settled_full_matches=0, ambiguous_total=len(planned)-settled, ambiguous_inconclusive=0,
        decision_matches=0, unsupported_accepted=0, malformed_or_inconsistent=0,
        useful_supported_preserved=0, needless_refusals_identified=0)
        if revision==10:
            arms[str(prompt)].update(provider_calls=0,interpretation_calls=0,assessment_calls=0,
                skipped_assessments=0,invalid_interpretations=0,invalid_evidence_assessments=0)
        if revision in (4,5,6,7,8,9,10):
            arms[str(prompt)].update(prompt_tokens=0, completion_tokens=0, reasoning_tokens=0 if revision==4 else None,
                budget_limited=0, empty_outputs=0, accounted_usd='0')
            if revision in (6,7,8,9,10):
                arms[str(prompt)]['reasoning_tokens']=0
            if revision in (7,8,9,10):
                arms[str(prompt)]['output_limit_overshoots']=0
    matches = {}
    for row in rows:
        item = by_id[row['control']]
        result = row['judgment']
        arm = arms[str(row[arm_key])]
        arm['completed'] += 1
        current=arm_protocol({},row['prompt_revision'])
        contract=current['runtime_memory_contract_version']
        if revision in (4,5,6,7,8,9,10):
            calls=study.runtime_assessment_calls(row,current)
            for call in calls:
                _add_call_metrics(arm,call,revision)
            if revision==10:
                arm['provider_calls']+=len(calls)
                arm['interpretation_calls']+=1
                arm['assessment_calls']+=len(calls)-1
                arm['skipped_assessments']+=len(calls)==1
                arm['invalid_interpretations']+=row['interpretation']['parse_failed']
                arm['invalid_evidence_assessments']+=len(calls)==2 and result['parse_failed']
        evidence=study.runtime_control_evidence(item,current,
            interpretation=row.get('interpretation',{}).get('interpretation'))
        coherent = (result == study.parse_runtime_memory(result['raw'], result['finish_reason'], result['proxy'],
                    contract_version=contract,evidence=evidence)
                    and result['parse_failed'] is False and result['consistency_errors'] == [])
        decision = result.get('decision')
        arm['malformed_or_inconsistent'] += not coherent
        arm['decision_matches'] += coherent and decision == item['expected_decision']
        arm['unsupported_accepted'] += item['expected_decision'] == 'reject' and decision == 'accept'
        if item['expected_criteria'] is None:
            matched = coherent and decision == 'inconclusive'
            arm['ambiguous_inconclusive'] += matched
        else:
            arm['settled_completed'] += 1
            matched = (coherent and study.runtime_memory_resolved(result,evidence=evidence)
                       and decision == item['expected_decision'] and result['criteria'] == item['expected_criteria'])
            arm['settled_full_matches'] += matched
            useful = item['expected_decision'] == 'accept' and item['expected_criteria']['addresses_request']
            arm['useful_supported_preserved'] += bool(useful and matched)
            needless = item['expected_decision'] == 'accept' and not item['expected_criteria']['abstains_appropriately']
            arm['needless_refusals_identified'] += bool(needless and matched)
        matches[(item['id'], row[arm_key])] = bool(matched)
    complete = len(rows) == len(wanted)
    result = dict(qualification_eligible=False, complete=complete, total=len(wanted), completed=len(rows), arms=arms,
        all_reference_matches=complete and all(arm['settled_full_matches'] == arm['settled_total']
            and arm['ambiguous_inconclusive'] == arm['ambiguous_total'] for arm in arms.values()))
    if revision in (3, 4, 7, 9):
        paired = dict(total=len(controls), completed=0, improved=0, regressed=0, both_match=0, neither_matches=0)
        for item in controls:
            baseline, candidate = ((item['id'], prompt) for prompt in (reasoning_arms(revision) if revision in (4,7)
                else (6,7) if revision==9 else (4, 5)))
            if baseline in matches and candidate in matches:
                paired['completed'] += 1
                first, second = matches[baseline], matches[candidate]
                paired['improved'] += not first and second
                paired['regressed'] += first and not second
                paired['both_match'] += first and second
                paired['neither_matches'] += not first and not second
        candidate = arms[reasoning_arms(revision)[1] if revision in (4,7) else '7' if revision==9 else '5']
        result.update(remaining=len(wanted)-len(rows), paired=paired,
            no_observed_regressions=complete and paired['regressed'] == 0,
            candidate_reference_screen_passed=complete
                and candidate['settled_full_matches'] == candidate['settled_total']
                and candidate['ambiguous_inconclusive'] == candidate['ambiguous_total'])
    if revision in (5,6,8,10):
        result.update(remaining=len(wanted)-len(rows),candidate_reference_screen_passed=result['all_reference_matches'])
    return result


def run(controls, protocol, judge, *, review=None, on_assessment=None):
    if on_assessment is not None and not callable(on_assessment):
        raise TypeError('Diagnostic progress callback must be callable')
    revision = protocol.get('runtime_report_diagnostic_revision', 1)
    if review is not None or controls != items(revision=revision) or protocol.get('screening') is not True:
        raise ValueError('Diagnostic uses exposed authored references, never qualification review')
    if revision==5 and (protocol.get('local_only') is not True or protocol['judge']['provider']!='ollama'
            or protocol['budget'].get('local_only') is not True or Decimal(protocol['budget']['max_usd'])!=0
            or protocol.get('runtime_memory_prompt_revision')!=4):
        raise ValueError('Local diagnostic requires its isolated local-only prompt-v4 protocol')
    if revision in (6,7,8,9,10):
        if (protocol.get('local_only',False) is not False or protocol['judge']['provider']!='tongyi'
                or protocol.get('runtime_memory_prompt_revision')!=(8 if revision==10 else 6 if revision in (8,9) else 4)):
            raise ValueError('Alibaba diagnostic requires its frozen hosted prompt protocol')
        study.hosted_request_options(protocol['judge'])
    if revision in (8,9) and any(protocol.get(key)!=value for key,value in
            study.runtime_memory_protocol(contract_version=3,prompt_revision=6).items()):
        raise ValueError('Evidence assessment diagnostic requires its exact contract')
    if revision==10 and any(protocol.get(key)!=value for key,value in
            study.runtime_memory_protocol(contract_version=4,prompt_revision=8).items()):
        raise ValueError('Bound-claim diagnostic requires its exact two-stage contract')
    if revision in (4,7) and (type(judge) is not dict or set(judge) != set(reasoning_arms(revision))):
        raise ValueError('Reasoning diagnostic requires both isolated hosted judges')
    judges = judge if revision in (4,7) else {'legacy': judge}
    for effort, router in judges.items():
        spec = (condition_protocol(protocol, dict(prompt_revision=4, inference_arm=effort))['judge']
                if revision in (4,7) else protocol['judge'])
        if (type(router) is not study.RecordingRouter or router.controls is not False
                or router.provider != spec['provider'] or router.model != spec['model']
                or (router.local_spec if revision==5 else router.remote_spec) != spec
                or router.max_input != spec['max_input']):
            raise ValueError('Diagnostic requires the existing audited judge')
    if revision in (4,7):
        first,second=(judge[arm] for arm in reasoning_arms(revision))
        if first is second or first.budget is not second.budget:
            raise ValueError('Reasoning arms need distinct routers sharing one budget')
    rows = []
    by_id = {item['id']: item for item in controls}
    for entry in schedule(controls, revision=revision):
        assessment = study.assess_calibration_control('memory', by_id[entry['control']],
            condition_protocol(protocol, entry), judges[entry['inference_arm']] if revision in (4,7) else judge)
        rows.append(dict(entry, **assessment))
        if on_assessment is not None:
            on_assessment(copy.deepcopy(rows[-1]))
    report = dict(schema='runtime-report-diagnostic-result-v1', phase='development', rows=rows,
        controls_sha256=study.shared.stable_key(controls), code_sha256=study.fingerprints(),
        diagnostic_owner_sha256=study.shared.digest(Path(__file__)), protocol_sha256=study.shared.stable_key(protocol),
        metrics=metrics(controls, rows, revision=revision), qualification_eligible=False, runtime_enabled=False,
        passed=False, generated_application_replies=0)
    if revision != 1:
        report['report_diagnostic_revision'] = revision
    validate(report, controls, protocol)
    return report


def validate(report, controls, protocol, *, ledger_rows=None):
    revision = protocol.get('runtime_report_diagnostic_revision', 1)
    if (controls != items(revision=revision) or report.get('schema') != 'runtime-report-diagnostic-result-v1'
            or report.get('controls_sha256') != study.shared.stable_key(controls)
            or report.get('code_sha256') != study.fingerprints()
            or report.get('diagnostic_owner_sha256') != study.shared.digest(Path(__file__))
            or report.get('protocol_sha256') != study.shared.stable_key(protocol)
            or report.get('phase') != 'development' or report.get('qualification_eligible') is not False
            or report.get('passed') is not False or report.get('runtime_enabled') is not False
            or report.get('generated_application_replies') != 0
            or type(report.get('report_diagnostic_revision', 1)) is not int
            or report.get('report_diagnostic_revision', 1) != revision
            or report.get('metrics') != metrics(controls, report['rows'], revision=revision)
            or not report['metrics']['complete']):
        raise ValueError('Report diagnostic identity or complete accounting changed')
    study.validate_runtime_memory_calls(report['rows'], controls, protocol, ledger_rows=ledger_rows,
        protocol_for_row=lambda row: condition_protocol(protocol, row))
    return report['metrics']
