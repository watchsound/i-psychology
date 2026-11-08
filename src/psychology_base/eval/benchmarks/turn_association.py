"""Three-arm, frozen-source study of portable local co-occurrence priors.

Uses the existing trial ingestion, carryover states, ID scorer, controls and CLI.
No answers or relevance labels enter the provenance encoder or graph prior.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import time
import tracemalloc

from psychology_base.companion import turn_provenance as turn
from psychology_base.companion.pam_export import pam_export
from psychology_base.eval.adapters import Observation, make_fragment
from psychology_base.eval.benchmarks import chinese_retrieval as shared
from psychology_base.eval.benchmarks import recall_carryover as carry
from psychology_base.eval.benchmarks import graph_priors as graph
from psychology_base.i18n import tokenizer_identity
from psychology_base.eval.benchmarks.evaluation_support import measure_recall


ARMS = ("ordinary_v4", "metadata_only", "local_association")
SCHEMA = "turn-association-v1"
FIXED = {"top_k": 6, "tokenizer": "jieba", "byte_cap": None, "seed": 0, "witness_active": True,
         "states": list(carry.STATES), "paired_absent": True, "association_floor": 1.0,
         "development_per_domain": 15, "evaluation_per_domain": 30,
         "stress_sizes": [24, 128, 512], "stress_repeats": 5, "stress_warmup": 1,
         "background_pool_dialogues": 32, "stress_design": "one-ingestion-independent-recalls-v2"}
GATES = {"cold_recall_gain": 0.05, "max_p95_latency_ratio": 2.0, "max_peak_python_memory_ratio": 2.0}


def allowed_groups(sources, protocol):
    """Hash partition entire source dialogues before targets/background selection."""
    return {key: {di for di in range(len(source))
                  if (int(shared.stable_key([pin["sha256"], di]), 16) % 2 == 0)
                  == (pin["split"] == "development")}
            for key, source in sources.items() for pin in (protocol["sources"][key],)}


def validate_protocol(protocol):
    from benchmarks.scripts import phase9_carryover_run as cli

    if (protocol["schema"] != SCHEMA or protocol["fixed"] != FIXED or protocol["gates"] != GATES
            or protocol["status"] != "frozen-before-development-scoring"):
        raise ValueError("Unsupported association protocol")
    expected = {f"{domain}/train-{split}" for domain in ("film", "music")
                for split in ("development", "evaluation")}
    if set(protocol["sources"]) != expected:
        raise ValueError("Association source groups differ")
    for key, pin in protocol["sources"].items():
        split = key.split("-")[-1]
        if (pin["split"] != split or pin["path"] != f'{key.split("/")[0]}/train.json'
                or pin["limit"] != (15 if split == "development" else 30)):
            raise ValueError("Association sample size or partition differs")
    if (any(protocol["selection"][key] != value for key, value in {
            "target_window_turns": 8, "distractor_windows": 2, "distractor_window_turns": 8}.items())
            or protocol["stress_source"]["path"] != "film/dev.json"
            or protocol["revision"] != "653db76432de09a004ba708a68f8bbd5500e6bec"):
        raise ValueError("Unsupported source-window or resource configuration")
    if protocol["controls_sha256"] != shared.digest(cli.FIXTURES / "controls.json"):
        raise ValueError("Association controls differ from reviewed contracts")
    if any(shared.digest(cli.ROOT / path) != expected for path, expected in protocol["prior_inputs"].items()):
        raise ValueError("Earlier observed-case inputs differ from freeze")


def load_sources(data_root, protocol):
    return {key: shared.load_source(data_root / pin["path"], {"source": pin})
            for key, pin in protocol["sources"].items()}


def catalog_for(sources, protocol):
    groups = allowed_groups(sources, protocol)
    backgrounds = {key: set(sorted(pool, key=lambda di: shared.stable_key(["background", di]))[
                       :FIXED["background_pool_dialogues"]]) for key, pool in groups.items()}
    targets = {key: pool - backgrounds[key] for key, pool in groups.items()}
    return carry.build_catalog(sources, protocol, allowed_dialogues=targets, background_dialogues=backgrounds)


def known_regression(data_root, revision):
    from benchmarks.scripts import phase9_carryover_run as cli

    old = cli.FIXTURES
    protocol, sources, cases, _ = cli.load_inputs(
        old / "protocol.json", old / "catalog.json", old / "controls.json", data_root, "evaluation")
    adapted = protocol | {"upstream": protocol["selection"]["upstream"], "revision": revision,
                           "sources": {k: v | {"path": f"{k}.json"} for k, v in protocol["sources"].items()}}
    rows = score_cases(sources, cases, adapted)
    return rows, comparison(rows)


def annotation(protocol, key):
    pin = protocol["sources"][key]
    identity = f'{protocol["upstream"]}@{protocol["revision"]}:{pin["path"]}:{pin["sha256"]}'
    return lambda ref: (turn.sequence_tag(identity, str(ref[0]), ref[1]),)


def setup(source, case, arm, protocol, absent=False, *, arms=ARMS, system_factory=None):
    if arm not in arms:
        raise ValueError("Unknown association study arm")
    return shared.prepare_trial(
        source, case, "precise", absent=absent, source_sessions=True,
        system_factory=(lambda: system_factory(arm)) if system_factory is not None else
            lambda: graph.system_for_policy("local_association" if arm == "local_association" else "production"),
        annotation=None if arm == "ordinary_v4" else annotation(protocol, case["source"]))


def score_cases(sources, cases, protocol, *, arms=ARMS, system_factory=None, inventories=None, focus_current=False):
    rows = []
    for case in cases:
        for absent in (False, True):
            annotated_ids = None
            for arm in arms:
                base, ids, _ = setup(sources[case["source"]], case, arm, protocol, absent,
                                     arms=arms, system_factory=system_factory)
                if inventories is not None:
                    inventories.append({"case": case["id"], "candidate": arm, "absent": absent,
                                        "ids": {fid: list(ref) for ref, fid in ids.items()}})
                if arm != "ordinary_v4":
                    if annotated_ids is not None and annotated_ids != ids:
                        raise AssertionError("Metadata control and intervention have different input IDs")
                    annotated_ids = ids
                for state in carry.STATES:
                    row = carry.evaluate(sources[case["source"]], case, base, ids, "production", state, absent,
                                         focus_current=focus_current)
                    row["candidate"] = arm
                    rows.append(row)
        print(f'Completed {case["id"]}', flush=True)
    carry.trial_matrix(rows, cases, arms)
    return rows


def quality_gates(rows):
    """Same frozen gates for development, held-out scoring and report validation."""
    summary = shared.summarize_trials(rows, ARMS, carry.STATES)
    checks = {}
    for baseline in ARMS[:2]:
        candidate, control = summary["local_association"], summary[baseline]
        checks[f"cold_gain_over_{baseline}"] = (
            candidate["cold"]["recall_at_6"] - control["cold"]["recall_at_6"] >= GATES["cold_recall_gain"] - 1e-12)
        for state in carry.STATES:
            for metric in ("recall_at_6", "mrr_at_6"):
                checks[f"{state}_{metric}_over_{baseline}"] = candidate[state][metric] >= control[state][metric] - 1e-12
            checks[f"{state}_absent_over_{baseline}"] = (
                candidate[state]["absent_nonempty_rate"] <= control[state]["absent_nonempty_rate"])
            checks[f"{state}_latency_over_{baseline}"] = (
                candidate[state]["p95_latency_ms"] <= control[state]["p95_latency_ms"] * GATES["max_p95_latency_ratio"])
            checks[f"{state}_allocation_over_{baseline}"] = (
                candidate[state]["peak_python_bytes"] <= control[state]["peak_python_bytes"] * GATES["max_peak_python_memory_ratio"])
        for domain in sorted({r["source"].split("/")[0] for r in rows}):
            domain_rows = [r for r in rows if r["source"].split("/")[0] == domain]
            partial = shared.summarize_trials(domain_rows, ARMS, carry.STATES)
            for state in carry.STATES:
                for metric in ("recall_at_6", "mrr_at_6"):
                    checks[f"{domain}_{state}_{metric}_over_{baseline}"] = (
                        partial["local_association"][state][metric] >= partial[baseline][state][metric] - 1e-12)
    return summary, checks


def comparison(rows, *, arms=ARMS):
    indexed = {(r["case"], r["candidate"], r["state"], r["absent"]): r for r in rows}
    result = {}
    for baseline in arms[:2]:
        wins, losses, metadata_changes = [], [], []
        for key, candidate in indexed.items():
            case, arm, state, absent = key
            if arm != arms[-1]:
                continue
            other = indexed[case, baseline, state, absent]
            if not absent and candidate["recall_at_6"] > other["recall_at_6"]:
                wins.append([case, state, absent])
            if not absent and candidate["recall_at_6"] < other["recall_at_6"]:
                losses.append([case, state, absent])
            if baseline == "ordinary_v4":
                annotated = indexed[case, "metadata_only", state, absent]
                if annotated["selected_sources"] != other["selected_sources"]:
                    metadata_changes.append([case, state, absent])
        result[baseline] = {"wins": wins, "losses": losses, "metadata_only_source_changes": metadata_changes}
    return result


def stress_config(config=None):
    config = ({"sizes": FIXED["stress_sizes"], "warmup": FIXED["stress_warmup"],
               "repeats": FIXED["stress_repeats"]} if config is None else config)
    if (set(config) != {"sizes", "warmup", "repeats"} or not isinstance(config["sizes"], list)
            or not config["sizes"] or any(type(size) is not int or size < 1 for size in config["sizes"])
            or config["sizes"] != sorted(set(config["sizes"]))
            or type(config["warmup"]) is not int or config["warmup"] < 0
            or type(config["repeats"]) is not int or config["repeats"] < 1):
        raise ValueError("Invalid stress configuration")
    return config


def stress_options(arms, config, max_latency_ratio, max_memory_ratio):
    """Validate shared measurement/decision settings before any engine work."""
    config = stress_config(config)
    if (not isinstance(arms, (tuple, list)) or len(arms) < 3
            or any(not isinstance(arm, str) or not arm for arm in arms) or len(set(arms)) != len(arms)):
        raise ValueError("Stress arms require distinct baselines and a primary")
    if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
           for value in (max_latency_ratio, max_memory_ratio)):
        raise ValueError("Invalid stress resource limits")
    return tuple(arms), config


def stress(source, pin, *, arms=ARMS, system_factory=None, config=None, focus_current=False,
           max_latency_ratio=2.0, max_memory_ratio=2.0):
    """Real source text, same budgets; separate ingestion/import/recall resources."""
    arms, config = stress_options(arms, config, max_latency_ratio, max_memory_ratio)
    if system_factory is None and any(arm not in ARMS for arm in arms):
        raise ValueError("Custom stress arms require an explicit system factory")
    refs = [[di, ti] for di, dialogue in enumerate(source) for ti in range(len(dialogue["messages"]))]
    if len(refs) < max(config["sizes"]):
        raise ValueError("Insufficient real source turns for stress sizes")
    if tracemalloc.is_tracing():
        raise ValueError("Stop existing allocation tracing before stress")
    result = []
    for size in config["sizes"]:
        for arm in arms:
            system = (system_factory(arm) if system_factory is not None else
                      graph.system_for_policy("local_association" if arm == "local_association" else "production"))
            print(f"Ingesting stress {size}/{arm}", flush=True)
            tracemalloc.start()
            try:
                start = time.perf_counter()
                for index, ref in enumerate(refs[:size]):
                    tags = ("user:stress", f"session:{ref[0]}",)
                    if arm != "ordinary_v4":
                        tags += (turn.sequence_tag(pin["sha256"], str(ref[0]), ref[1]),)
                    system.observe(Observation(shared.BASE + shared.timedelta(minutes=index), "unknown-source-actor",
                                               shared.message(source, ref)["message"], tags=tags))
                ingest_ms = (time.perf_counter() - start) * 1000
                _, ingest_peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            ingestion = dict(ingest_ms=ingest_ms, ingest_peak=ingest_peak,
                             accounted_bytes=system.companion.store.total_bytes(system.edges))
            fragments = system.companion.store.all()
            if len(fragments) != size:
                raise ValueError("Stress source ingestion did not retain its configured size")
            cue = shared.message(source, refs[size // 2])["message"]
            print(f"Ingested stress {size}/{arm}: {ingest_ms:.0f} ms", flush=True)
            records = []
            for repeat in range(config["repeats"] + config["warmup"]):
                # Clone actual ingested state, as prepare_recall already does.
                # Each timed recall starts from the same independent cold state.
                trial = copy.deepcopy(system)
                _, measurement = measure_recall(trial, cue, **({"focus_cue": cue} if focus_current else {}))
                recall_ms, recall_peak = measurement["recall_ms"], measurement["recall_peak"]
                before = (pam_export(fragments), copy.deepcopy(trial.edges), copy.deepcopy(trial.witness_edges))
                start = time.perf_counter()
                for f in fragments:
                    trial._ingest_fragment(f)
                import_ms = (time.perf_counter() - start) * 1000
                if before != (pam_export(trial.companion.store.all()), trial.edges, trial.witness_edges):
                    raise AssertionError("Repeated import changes stored/learned state")
                if repeat >= config["warmup"]:
                    records.append(dict(recall_ms=recall_ms, recall_peak=recall_peak, repeated_import_ms=import_ms))
            result.append({"size": size, "arm": arm, "ingestion": ingestion, "runs": records})
            print(f"Completed stress {size}/{arm}", flush=True)
    checks = stress_gates(result, arms=arms, config=config,
                          max_latency_ratio=max_latency_ratio, max_memory_ratio=max_memory_ratio)
    return {"measurements": result, "gates": checks, "passed": all(checks.values())}


def stress_gates(result, *, arms=ARMS, config=None, max_latency_ratio=2.0, max_memory_ratio=2.0):
    arms, config = stress_options(arms, config, max_latency_ratio, max_memory_ratio)
    expected = {(size, arm) for size in config["sizes"] for arm in arms}
    if (len(result) != len(expected) or {(r["size"], r["arm"]) for r in result} != expected
            or any(len(r["runs"]) != config["repeats"] for r in result)):
        raise ValueError("Incomplete stress measurements")
    for row in result:
        if set(row["ingestion"]) != {"ingest_ms", "ingest_peak", "accounted_bytes"}:
            raise ValueError("Invalid stress ingestion fields")
        if any(set(run) != {"recall_ms", "recall_peak", "repeated_import_ms"} for run in row["runs"]):
            raise ValueError("Invalid stress recall fields")
        for run in [row["ingestion"], *row["runs"]]:
            if (set(run) not in ({"ingest_ms", "ingest_peak", "accounted_bytes"},
                                {"recall_ms", "recall_peak", "repeated_import_ms"})
                    or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in run.values())):
                raise ValueError("Invalid stress resource values")
    checks = {}
    by_key = {(r["size"], r["arm"]): r["runs"] for r in result}
    for size in config["sizes"]:
        for baseline in arms[:2]:
            for metric in ("recall_ms", "recall_peak"):
                # Five repeats: nearest-rank p95 is the maximum, also used for peak allocation.
                checks[f"{size}_{metric}_over_{baseline}"] = (
                    max(r[metric] for r in by_key[size, arms[-1]])
                    <= (max_latency_ratio if metric == "recall_ms" else max_memory_ratio)
                    * max(r[metric] for r in by_key[size, baseline]))
    return checks


def validate_report(report, cases, protocol):
    if report["schema"] != SCHEMA or report["matrix"] != carry.trial_matrix(report["trials"], cases, ARMS):
        raise ValueError("Association report matrix differs")
    from benchmarks.scripts import phase9_carryover_run as cli

    known_cases = [c for c in json.loads((cli.FIXTURES / "catalog.json").read_text(encoding="utf-8"))["cases"]
                   if c["split"] == "evaluation"]
    carry.trial_matrix(report["known_trials"], known_cases, ARMS)
    for row in (*report["trials"], *report["known_trials"]):
        ids, gold = row["selected_ids"], row["gold_ids"]
        if (len(ids) > 6 or len(set(ids)) != len(ids) or bool(gold) == row["absent"]
                or any(row[k] != v for k, v in shared.score_context(ids, set(gold)).items())
                or any(type(row[k]) not in (float, int) or not math.isfinite(row[k]) or row[k] < 0
                       for k in ("latency_ms", "peak_python_bytes"))):
            raise ValueError("Association report score differs from IDs")
    summary, checks = quality_gates(report["trials"])
    if summary != report["summary"] or checks != report["gates"]:
        raise ValueError("Association report decision differs from trials")
    controls = json.loads((cli.FIXTURES / "controls.json").read_text(encoding="utf-8"))
    expected_controls = {c["id"] for c in controls["cases"]}
    if set(report["software_controls"]) != set(ARMS):
        raise ValueError("Association software-control arms differ")
    for results in report["software_controls"].values():
        if (len(results) != len(expected_controls) or {r["case"] for r in results} != expected_controls
                or any(type(r["passed"]) is not bool for r in results)):
            raise ValueError("Incomplete association software controls")
        by_case = {r["case"]: r for r in results}
        for case in controls["cases"]:
            row = by_case[case["id"]]
            observed = [make_fragment(text, shared.BASE + shared.timedelta(minutes=i))
                        for i, text in enumerate(case["facts"])]
            available = {f.id: f for f in observed}
            selected = row["selected_ids"]
            required = sorted(observed[i].id for i in case["required"])
            passed = (set(required) <= set(selected) <= set(available) and len(selected) <= 6
                      and len(set(selected)) == len(selected)
                      and not any(case.get("forbidden_text", "\0") in available[fid].content
                                  for fid in selected if fid in available))
            if row["required_ids"] != required or row["passed"] != passed:
                raise ValueError("Software-control result differs from actual fixture source IDs")
    controls_pass = all(c["passed"] for rows in report["software_controls"].values() for c in rows)
    known = comparison(report["known_trials"])
    known_pass = not any(known[arm]["losses"] for arm in ARMS[:2])
    stress = report["stress"]
    stress_checks = stress_gates(stress["measurements"])
    if (report["comparisons"] != comparison(report["trials"]) or report["known_comparisons"] != known
            or report["known_cases_passed"] != known_pass or report["controls_passed"] != controls_pass
            or stress["gates"] != stress_checks or stress["passed"] != all(stress_checks.values())):
        raise ValueError("Association auxiliary evidence differs")
    eligible = all(checks.values()) and known_pass and controls_pass and all(stress_checks.values())
    if (type(report["advance"]) is not bool or report["advance"] != eligible
            or report["integration_eligible"] != (report["phase"] == "evaluation" and eligible)
            or report["phase"] not in ("development", "evaluation")
            or report["fresh_evaluation_scored"] != (len(cases) if report["phase"] == "evaluation" else 0)
            or report["production_association_enabled"] is not False):
        raise ValueError("Association report advance decision differs")


def run(args):
    from benchmarks.scripts import phase9_carryover_run as cli

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    validate_protocol(protocol)
    before = fingerprints()
    inputs = cli.input_fingerprints(args)
    if (inputs["catalog_sha256"] != protocol["catalog_sha256"]
            or inputs["controls_sha256"] != protocol["controls_sha256"]):
        raise ValueError("Association inputs differ from freeze")
    sources = load_sources(args.data_root, protocol)
    old_protocol = json.loads((cli.FIXTURES / "protocol.json").read_text(encoding="utf-8"))
    data_pins = {pin["path"]: pin["sha256"] for pin in protocol["sources"].values()}
    data_pins.update({f"{key}.json": pin["sha256"] for key, pin in old_protocol["sources"].items()
                      if pin["split"] == "evaluation"})
    data_pins[protocol["stress_source"]["path"]] = protocol["stress_source"]["sha256"]
    if any(shared.digest(args.data_root / path) != sha for path, sha in data_pins.items()):
        raise ValueError("Source files differ from freeze")
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    if catalog_for(sources, protocol) != catalog:
        raise ValueError("Association catalog differs from pinned source selection")
    phase = "development" if args.command == "association-development" else "evaluation"
    identity = tokenizer_identity("jieba")
    if phase == "evaluation":
        if args.development_report is None:
            raise ValueError("Association evaluation needs a frozen development report")
        development = json.loads(args.development_report.read_text(encoding="utf-8"))
        validate_report(development, [c for c in catalog["cases"] if c["split"] == "development"], protocol)
        if (development["phase"] != "development" or not development["advance"]
                or development["code_sha256"] != before or development["tokenizer"] != identity
                or any(development[k] != inputs[k] for k in ("protocol_sha256", "catalog_sha256", "controls_sha256"))):
            raise ValueError("Development did not qualify with identical inputs/code; evaluation remains unscored")
    cases = [c for c in catalog["cases"] if c["split"] == phase]
    rows = score_cases(sources, cases, protocol)
    controls = json.loads(args.controls.read_text(encoding="utf-8"))
    control_reports = {arm: carry.control_results(controls, "production", system_factory=lambda arm=arm:
                       graph.system_for_policy("local_association" if arm == "local_association" else "production"))
                       for arm in ARMS}
    # The historical source catalog is already exposed. Score it only as a regression guard.
    known_rows, known = known_regression(args.data_root, protocol["revision"])
    known_pass = not any(known[arm]["losses"] for arm in ARMS[:2])
    summary, checks = quality_gates(rows)
    report = {"schema": SCHEMA, "phase": phase, **inputs, "source_sha256": data_pins,
              "code_sha256": before, "tokenizer": identity,
              "matrix": carry.trial_matrix(rows, cases, ARMS), "trials": rows,
              "summary": summary, "gates": checks, "comparisons": comparison(rows),
              "known_trials": known_rows, "known_comparisons": known, "known_cases_passed": known_pass,
              "software_controls": control_reports,
              "controls_passed": all(c["passed"] for r in control_reports.values() for c in r),
              "fresh_evaluation_scored": len(cases) if phase == "evaluation" else 0,
              "production_association_enabled": False,
              "limitations": ["Mechanical source co-occurrence, not annotated reply or truth.",
                              "Public factual recall and fixed software controls do not establish model or clinical quality."]}
    pin = protocol["stress_source"]
    resource_source = shared.load_source(args.data_root / pin["path"], {"source": pin})
    report["stress"] = stress(resource_source, pin)
    report["advance"] = all(checks.values()) and known_pass and report["controls_passed"] and report["stress"]["passed"]
    report["integration_eligible"] = phase == "evaluation" and report["advance"]
    validate_report(report, cases, protocol)
    validate_protocol(protocol)
    if (before != fingerprints() or inputs != cli.input_fingerprints(args)
            or any(shared.digest(args.data_root / path) != sha for path, sha in data_pins.items())):
        raise ValueError("Code or inputs changed during association study")
    return report


def fingerprints():
    from benchmarks.scripts import phase9_carryover_run as cli

    root = cli.ROOT
    extra = (Path(__file__), Path(turn.__file__), Path(graph.__file__),
             *(root / f"src/psychology_base/{name}.py" for name in (
                 "companion/fragment", "companion/pam_export", "companion/pam_cli",
                 "applications/emotional/companion_agent", "applications/learning/tutor_agent",
                 "eval/benchmarks/attachment_longitudinal", "eval/benchmarks/evaluation_support")))
    return cli.fingerprints(extra_paths=extra)
