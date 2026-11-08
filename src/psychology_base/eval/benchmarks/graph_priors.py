"""Graph-prior interventions over real ingestion and reconstruction.

The retrieval baseline is ordinary cue-gated v3 (v4 only adds identity-safe
ingestion), not the earlier carryover
experiment's unconditional-residual candidate named 'baseline'.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import re
from types import MethodType

from psychology_base.companion import reconstruction as memory
from psychology_base.companion import navigation
from psychology_base.companion.system import CompanionSystem
from psychology_base.companion.turn_provenance import locally_associated
from psychology_base.companion.perspective.routing import ScopeSpec
from psychology_base.eval.benchmarks import chinese_retrieval as shared
from psychology_base.eval.benchmarks import recall_carryover as carryover
from psychology_base.i18n import tokenizer_identity


POLICIES = ("production", "routing_filtered", "content_only", "no_initial_edges")


def routing_tags(fragment):
    """Project owner/session/speaker routing out of graph similarity only.

    Perspective/entity, pedagogical and unknown semantic tags remain visible.
    The Fragment and its content address are never changed.
    """
    return {tag for tag in fragment.tags
            if not tag.startswith(("user:", "session:", "speaker:"))
            and not re.fullmatch(r"session_\d+", tag)}


def no_tags(fragment):
    return set()


def no_edges(a, b, cjk_tokenizer="unicode"):
    """Diagnostic removal of initial edges; normal Hebbian learning remains."""
    return 0.0


def local_association_prior(a, b, cjk_tokenizer="unicode"):
    """Reviewed single research candidate; not enabled by production ingestion."""
    prior = memory.jaccard_prior(a, b, cjk_tokenizer)
    return max(prior, 1.0) if locally_associated(a, b) else prior


def system_for_policy(policy, **overrides):
    if policy not in (*POLICIES, "local_association"):
        raise ValueError(f"Unknown graph-prior policy: {policy}")
    options = {"top_k": 6, "byte_cap": None, "cjk_tokenizer": "jieba"}
    options.update(overrides)
    system = CompanionSystem(**options)
    if policy == "production":
        return system
    prior = (local_association_prior if policy == "local_association" else
             no_edges if policy == "no_initial_edges" else shared._bind(
                 memory.jaccard_prior, _graph_prior_tags=routing_tags if policy == "routing_filtered" else no_tags))
    # Bind before observe/rerender/restore. The one real ingestion method builds
    # both independent stores/graphs using this input policy.
    ingest = shared._bind(CompanionSystem._ingest_fragment, jaccard_prior=prior)
    system._ingest_fragment = MethodType(ingest, system)
    return system


def bind_navigation(system, policy, *, router=None, floor=None, model_identity=None, query_prefix=""):
    """Research binding of the existing recall; no second commit or global patch."""
    if policy not in ("lexical", "semantic") or system.fact_store is not None:
        raise ValueError("Navigation research requires a known policy and the fragment path")
    if system.recall.__func__.__code__ is not CompanionSystem.recall.__code__:
        raise ValueError("Bind navigation to the existing CompanionSystem recall owner")
    if policy == "semantic":
        from psychology_base.llm.providers.ollama import OllamaProvider
        provider = None if router is None else router.providers.get("ollama")
        if not isinstance(provider, OllamaProvider) or not provider.local_only or not model_identity:
            raise ValueError("Semantic research requires a pinned local Ollama router")
        if type(floor) not in (int, float) or not -1 <= floor <= 1:
            raise ValueError("Semantic research requires a finite cosine floor")
    expected_model = copy.deepcopy(model_identity)
    original = system.recall.__func__.__globals__["reconstruct"]
    if getattr(original, "navigation_arguments", None) is not None:
        raise ValueError("Navigation research is already bound to this system")

    def prepare(fragments, cue, *, focus_cue=None, cjk_tokenizer="unicode"):
        snapshot = memory.recall_tokens(fragments, cue, cjk_tokenizer,
                                        focus_cue=focus_cue, retain_order=True)
        if policy == "lexical":
            hints = navigation.lexical_hints(fragments, cue, dict(snapshot.ordered_fragments),
                snapshot.ordered_cue if snapshot.ordered_focus is None else snapshot.ordered_focus,
                focus_cue=focus_cue, cjk_tokenizer=cjk_tokenizer)
        else:
            actual = expected_model
            query = cue if focus_cue is None else focus_cue
            vectors = {}
            query_vector = []
            if query and fragments:
                estimate = (len(fragments) + 1) * actual["dimension"] * 8
                estimate += sum(len(f.id.encode()) + 32 for f in fragments)
                if estimate > navigation.MAX_BYTES:
                    hints = navigation.NavigationHints(snapshot.identity,
                        navigation.semantic_policy(floor, {**actual, "query_prefix":query_prefix}),
                        (), "resource-bytes", estimate)
                    return {"_navigation_hints":hints, "_token_snapshot":snapshot}
                encoded = router.embed_batch([query_prefix + query, *(f.content for f in fragments)],
                    backend="local", expected_identity=expected_model)
                query_vector = encoded[0]
                vectors = {fragment.id:vector for fragment,vector in zip(fragments,encoded[1:])}
            hints = navigation.semantic_hints(fragments, cue, vectors, query_vector,
                floor=floor, model_identity={**actual, "query_prefix": query_prefix},
                focus_cue=focus_cue, cjk_tokenizer=cjk_tokenizer)
        return {"_navigation_hints": hints, "_token_snapshot": snapshot}

    def reconstruct(fragments, cue, context, state, tick_count, *args, **kwargs):
        prepared = prepare(fragments, cue, focus_cue=kwargs.get("focus_cue"),
                           cjk_tokenizer=kwargs.get("cjk_tokenizer", "unicode"))
        return original(fragments, cue, context, state, tick_count, *args, **kwargs, **prepared)

    reconstruct.navigation_arguments = prepare
    system.recall = MethodType(shared._bind(system.recall.__func__, reconstruct=reconstruct), system)
    return system


def trace_recall(source, case, base_system, ref_ids, state="cold", *, focus_current=False):
    """Explain real ranks without a second scorer or mutation of the input system.

    Hop contributions are differences of actual spreading calls. The full-pool
    reconstruction uses private state/edges and disables learning for inspection;
    its first six IDs must equal the ordinary, learning-enabled recall.
    """
    system, cue = carryover.prepare_recall(source, case, base_system, "production", state, focus_current=focus_current)
    focus = {"focus_cue": shared.message(source, case["query"])["message"]} if focus_current else {}
    fragments = ScopeSpec("study", "memory").apply(system.companion.store.all())
    edges_before = copy.deepcopy(system.edges)
    state_before = copy.deepcopy(system.activation)
    state_before.decay()
    reconstruct = system.recall.__func__.__globals__["reconstruct"]
    prepare = getattr(reconstruct, "navigation_arguments", None)
    navigation_args = ({} if prepare is None else
                       prepare(fragments, cue, cjk_tokenizer=system.cjk_tokenizer, **focus))
    profiles = [memory.spread_activation(fragments, cue, state_before, hops=hops,
                                         edges=edges_before, cjk_tokenizer=system.cjk_tokenizer,
                                         **navigation_args)
                for hops in range(3)]
    reconstructed = reconstruct(
        fragments, cue, None, copy.deepcopy(system.activation), system.heartbeat.sleep_ticks,
        system_seed=system.seed, top_k=len(fragments), edges=copy.deepcopy(edges_before),
        hebbian_delta=0.0, cjk_tokenizer=system.cjk_tokenizer, **focus,
        **({"fragment_context_selection": system.fragment_context_selection}
           if system.fragment_context_selection != "identity" else {}))
    if reconstructed.activation_profile != {f.id: profiles[-1].get(f.id, 0.0) for f in fragments}:
        raise AssertionError("Trace and ordinary context-free activation differ")
    actual = system.recall(cue, scope=("study", "memory"), **focus)
    ranked_ids = [f.id for f in reconstructed.fragments_used]
    if [f.id for f in actual.fragments_used] != ranked_ids[:system.top_k]:
        raise AssertionError("Trace changes the returned production context")
    reverse = {fid: list(ref) for ref, fid in ref_ids.items()}
    cue_tokens = set(memory.tokens(cue, system.cjk_tokenizer))
    focus_tokens = set(memory.tokens(focus["focus_cue"], system.cjk_tokenizer)) if focus_current else set()
    prior = system._ingest_fragment.__func__.__globals__["jaccard_prior"]
    lexical_prior = shared._bind(memory.jaccard_prior, _graph_prior_tags=no_tags)
    rows = []
    for fragment in fragments:
        neighbors = [f for f in fragments if f.id != fragment.id]
        content_prior = sum(lexical_prior(fragment, f, system.cjk_tokenizer) for f in neighbors)
        # These are unlearned prior components; current edge weights are separate.
        total_prior = sum(prior(fragment, f, system.cjk_tokenizer) for f in neighbors)
        rows.append({"id": fragment.id, "source": reverse[fragment.id],
                     "tags": list(fragment.tags),
                     "cue_overlap": sorted(cue_tokens & set(memory.tokens(fragment.content, system.cjk_tokenizer))),
                     "rank": ranked_ids.index(fragment.id) + 1 if fragment.id in ranked_ids else None,
                     "seed_and_residual": profiles[0].get(fragment.id, 0.0),
                     "hop_1": profiles[1].get(fragment.id, 0.0) - profiles[0].get(fragment.id, 0.0),
                     "hop_2": profiles[2].get(fragment.id, 0.0) - profiles[1].get(fragment.id, 0.0),
                     "final_score": reconstructed.activation_profile[fragment.id],
                     "selection_score": reconstructed.selection_profile[fragment.id],
                     "degree": sum(edges_before.get(fragment.id, f.id) > 0 for f in neighbors),
                     "current_edge_weight": sum(edges_before.get(fragment.id, f.id) for f in neighbors),
                     "initial_content_prior": content_prior if prior is not no_edges else 0.0,
                     "initial_tag_prior": total_prior - content_prior if prior is not no_edges else 0.0,
                     **({"focus_overlap": sorted(focus_tokens & set(memory.tokens(fragment.content, system.cjk_tokenizer)))}
                        if focus_current else {})})
    return {"case": case["id"], "state": state,
            "gold_ids": [ref_ids[tuple(ref)] for ref in case["gold"]],
            "selected_ids": [f.id for f in actual.fragments_used], "fragments": rows}


def data_audit(root, catalog, protocol, *, data_root=None):
    """Count unscored targets after excluding all earlier source-dialogue roles."""
    fixture = root / "benchmarks/fixtures/chinese_retrieval"
    data_root = root / ".data/kdconv/data" if data_root is None else data_root
    earlier_protocol = json.loads((fixture / "protocol.json").read_text(encoding="utf-8"))
    if shared.digest(fixture / "catalog.json") != earlier_protocol["catalog_sha256"]:
        raise ValueError("Historical exposure catalog differs from freeze")
    earlier = json.loads((fixture / "catalog.json").read_text(encoding="utf-8"))
    exposed = {}
    for old_catalog, source in ((earlier, "film/dev"), (catalog, None)):
        for case in old_catalog["cases"]:
            groups = exposed.setdefault(source or case["source"], set())
            for field in ("query", "answer"):
                groups.add(case[field][0])
            for field in ("history", "absent_history", "cue_refs", "followup_refs", "switch_refs", "gold"):
                groups.update(ref[0] for ref in case.get(field, []))
    pins = {**protocol["sources"], "film/dev": earlier_protocol["source"]}
    result = {}
    for name, pin in pins.items():
        source = shared.load_source(data_root / f"{name}.json", {"source": pin})
        fresh = [item["query"][0] for item in shared.question_candidates(source)
                 if item["query"][0] not in exposed[name] and carryover.target_window(source, item) is not None]
        result[name] = {"source_sha256": pin["sha256"], "dialogues": len(source),
                        "exposed_dialogues": sorted(exposed[name]), "fresh_eligible_targets": fresh}
    return {"sources": result, "fresh_target_count": sum(len(v["fresh_eligible_targets"]) for v in result.values()),
            "planned_development_cases": 30, "planned_evaluation_cases": 60,
            "fresh_cases_scored": 0,
            "limitation": "Counts precede fresh split/background assignment; do not establish sufficient evaluation data."}


def diagnose(args):
    """Observed-case gate: reject regressions before spending fresh evaluation."""
    from benchmarks.scripts import phase9_carryover_run as cli

    if args.reference_report is None:
        raise ValueError("Graph diagnosis requires a frozen --reference-report")
    root = cli.ROOT
    historical = root / "benchmarks/fixtures/chinese_retrieval"

    def inputs():
        return {**cli.input_fingerprints(args),
                "earlier_catalog_sha256": shared.digest(historical / "catalog.json"),
                "earlier_protocol_sha256": shared.digest(historical / "protocol.json")}

    code = cli.fingerprints(extra_paths=(Path(__file__),))
    frozen = inputs()
    protocol, sources, cases, controls = cli.load_inputs(
        args.protocol, args.catalog, args.controls, args.data_root, "evaluation")
    identity = tokenizer_identity("jieba")
    reference = json.loads(args.reference_report.read_text(encoding="utf-8"))
    if (reference["phase"] != "evaluation" or reference["selected"] != "novel_cue"
            or any(reference[key] != frozen[key] for key in ("protocol_sha256", "catalog_sha256", "controls_sha256"))):
        raise ValueError("Reference is not the frozen evaluation configuration")
    cli.validate_evidence(reference, cases, controls, protocol)
    expected = {(r["case"], r["state"], r["absent"]): r for r in reference["trials"] if r["candidate"] == "novel_cue"}
    rows, traces = [], []
    omitted = {"candidate", "latency_ms", "peak_python_bytes"}
    for case in cases:
        source = sources[case["source"]]
        for policy in POLICIES:
            for absent in (False, True):
                base, ids, _ = shared.prepare_trial(source, case, "precise", absent=absent, source_sessions=True,
                                                   system_factory=lambda: system_for_policy(policy))
                for state in carryover.STATES:
                    row = carryover.evaluate(source, case, base, ids, "production", state, absent)
                    if policy == "production":
                        prior = expected[case["id"], state, absent]
                        if ({k: v for k, v in row.items() if k not in omitted}
                                != {k: v for k, v in prior.items() if k not in omitted}):
                            raise ValueError("Ordinary v3 baseline differs from the reviewed source-ID results")
                    row["candidate"] = policy
                    rows.append(row)
                if not absent:
                    traces.append({"candidate": policy, **trace_recall(source, case, base, ids)})
        print(f"Diagnosed {case['id']}", flush=True)
    matrix = carryover.trial_matrix(rows, cases, POLICIES)
    summary = shared.summarize_trials(rows, POLICIES, carryover.STATES)
    control_reports = {p: carryover.control_results(controls, "production", system_factory=lambda: system_for_policy(p))
                       for p in POLICIES}
    baseline = {(r["case"], r["state"]): r for r in rows if r["candidate"] == "production" and not r["absent"]}
    comparisons = {}
    for policy in POLICIES[1:]:
        wins, losses = [], []
        for row in rows:
            if row["candidate"] != policy or row["absent"]:
                continue
            old = baseline[row["case"], row["state"]]
            if row["recall_at_6"] > old["recall_at_6"]:
                wins.append([row["case"], row["state"]])
            elif row["recall_at_6"] < old["recall_at_6"]:
                losses.append([row["case"], row["state"]])
        comparisons[policy] = {"wins": wins, "losses": losses, "no_lost_observed_hits": not losses,
                               "controls": all(c["passed"] for c in control_reports[policy]),
                               "cold_gain": summary[policy]["cold"]["recall_at_6"] - summary["production"]["cold"]["recall_at_6"],
                               "diagnostic_only": policy == "no_initial_edges"}
    eligible = [p for p, r in comparisons.items()
                if not r["diagnostic_only"] and r["no_lost_observed_hits"] and r["controls"] and r["cold_gain"] >= 0.05 - 1e-12]
    audit = data_audit(root, json.loads(args.catalog.read_text(encoding="utf-8")), protocol, data_root=args.data_root)
    if cli.fingerprints(extra_paths=(Path(__file__),)) != code or inputs() != frozen:
        raise ValueError("Graph diagnosis code or inputs changed during execution")
    return {"schema": "cold-recall-graph-diagnosis-v1", "kind": "observed-case-diagnostic-not-selection",
            "policy": memory.RECALL_SELECTION_POLICY, "code_sha256": code, **frozen,
            "tokenizer": identity, "matrix": matrix, "summary": summary, "software_controls": control_reports,
            "comparisons": comparisons, "candidates_passing_observed_gate": eligible,
            "integration_eligible": False, "production_baseline_matches": len(expected),
            "data_audit": audit, "trials": rows, "cold_traces": traces,
            "limitations": ["Observed-case interventions, not independent development/evaluation.",
                            "No fresh cases scored; no production graph-policy promotion.",
                            "Mechanical source labels do not establish generated-answer or clinical quality."]}
