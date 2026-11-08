"""Coherent source-window evaluation of state carryover and lexical specificity.

Executes actual memory, including Witness, with per-instance input-policy binding.
Labels and source sessions are explicit; no model response quality is inferred.
"""
from __future__ import annotations

import copy
from collections import Counter
import math
from pathlib import Path
import time
import tracemalloc
from types import MethodType

from psychology_base.companion import reconstruction as memory
from psychology_base.companion import witness
from psychology_base.companion.system import CompanionSystem
from psychology_base.companion.perspective.routing import ScopeSpec
from psychology_base.eval.adapters import Observation
from psychology_base.eval.benchmarks import chinese_retrieval as shared
from psychology_base.eval.benchmarks.evaluation_support import complete_keys


POLICIES = ("baseline", "reset", "idf", "cue_gate", "cue_idf", "conservative", "novel_cue")
STATES = ("cold", "followup", "switch")


def trial_matrix(rows, cases, policies):
    """Reject incomplete/duplicate source-case matrices before aggregation."""
    expected = {(c["id"], policy, state, absent) for c in cases for policy in policies
                for state in STATES for absent in (False, True)}
    actual = [(r["case"], r["candidate"], r["state"], r["absent"]) for r in rows]
    if (not expected or len({c["id"] for c in cases}) != len(cases)
            or any(type(r["absent"]) is not bool for r in rows)):
        raise ValueError("Missing, duplicated or unexpected scored trials")
    return complete_keys(actual, expected)


def selection(summary, control_reports, rules):
    """One development selection rule for execution and report validation."""
    checks = {policy: guards(summary[policy], summary["baseline"],
                              all(c["passed"] for c in control_reports[policy]), rules)
              for policy in summary if policy not in ("baseline", "reset")}
    eligible = [p for p in checks if all(checks[p].values())]
    selected = max(eligible, key=lambda p: (summary[p]["switch"]["recall_at_6"],
                                           summary[p]["switch"]["mrr_at_6"],
                                           -POLICIES.index(p))) if eligible else None
    return checks, selected


def count_weights(fragment_tokens, cue_tokens):
    return {fid: float(len(words & cue_tokens)) for fid, words in fragment_tokens}


def idf_weights(fragment_tokens, cue_tokens):
    frequencies = Counter(word for _, words in fragment_tokens for word in words if word in cue_tokens)
    size = len(fragment_tokens)
    weights = {word: 1.0 + math.log((size + 1) / (count + 1)) for word, count in frequencies.items()}
    return {fid: sum(weights[word] for word in sorted(words & cue_tokens))
            for fid, words in fragment_tokens}


def full_carryover(seed, residual, *evidence):
    return 1.0


def reset_carryover(seed, residual, *evidence):
    return 0.0


def overlap_carryover(seed, residual, *evidence):
    seed_total, residual_total = sum(seed.values()), sum(residual.values())
    if seed_total <= 0 or residual_total <= 0:
        return 1.0
    return min(1.0, sum(min(value / seed_total, residual.get(fid, 0) / residual_total)
                        for fid, value in seed.items()))


def conservative_carryover(seed, residual, *evidence):
    return 0.5 + 0.5 * overlap_carryover(seed, residual)


# The measured winner now has one runtime owner. Research baselines continue to
# bind their explicit historical policies without copying the production gate.
novel_cue_carryover = memory._residual_gate


def bind_policy(system: CompanionSystem, policy: str) -> CompanionSystem:
    if policy not in POLICIES:
        raise ValueError(f"Unknown carryover candidate: {policy}")
    weights = idf_weights if policy in ("idf", "cue_idf", "conservative") else count_weights
    gate = {"reset": reset_carryover, "cue_gate": overlap_carryover,
            "cue_idf": overlap_carryover, "conservative": conservative_carryover,
            "novel_cue": novel_cue_carryover}.get(policy, full_carryover)
    spread = shared._bind(memory.spread_activation, _cue_activation=weights, _residual_gate=gate)
    reconstruct = shared._bind(memory.reconstruct, spread_activation=spread)
    return bind_reconstruction(system, reconstruct)


def bind_reconstruction(system: CompanionSystem, reconstruct, *, witness_reconstruct=None) -> CompanionSystem:
    """One research binding path for Companion and its independent Witness."""
    if system.fragment_context_selection != "identity":
        raise ValueError("Research binding cannot override runtime context selection")
    second = shared._bind(witness.witness_second_perspective,
                          reconstruct=reconstruct if witness_reconstruct is None else witness_reconstruct)
    couple = shared._bind(witness.witness_couple, witness_second_perspective=second)
    recall = shared._bind(CompanionSystem.recall, reconstruct=reconstruct, witness_couple=couple)
    system.recall = MethodType(recall, system)
    return system


def load_sources(data_root: Path, protocol: dict, split=None):
    return {key: shared.load_source(data_root / f"{key}.json", {"source": pin})
            for key, pin in protocol["sources"].items() if split is None or pin["split"] == split}


def target_window(source, item):
    """The existing nine-turn eligibility rule and paired eight-turn windows."""
    di, ti = item["query"]
    turns = source[di]["messages"]
    if ti < 2 or len(turns) < 9:
        return None
    start = min(max(ti - 3, 0), len(turns) - 9)
    extended = [[di, j] for j in range(start, start + 9)]
    values = {a["attrvalue"] for a in turns[ti + 1].get("attrs", [])
              if a["name"] == item["fact"]["name"] and a["attrname"] == item["fact"]["attrname"]}
    containing = [ref for ref in extended if any(value in shared.message(source, ref)["message"] for value in values)]
    if containing != [item["answer"]]:
        return None
    positive = extended[:8] if item["answer"] in extended[:8] else extended[1:]
    absent = [ref for ref in extended if ref != item["answer"]]
    entities = {a["name"] for ref in extended for a in shared.message(source, ref).get("attrs", [])}
    answer_text = turns[ti + 1]["message"]
    span = answer_text.index(item["fact"]["attrvalue"])
    return positive, absent, entities, values, [span, span + len(item["fact"]["attrvalue"])]


def build_catalog(sources: dict, protocol: dict, *, allowed_dialogues=None, background_dialogues=None) -> dict:
    cases = []
    for key, source in sources.items():
        pin = protocol["sources"][key]
        selected = []
        for item in sorted(shared.question_candidates(source), key=shared.stable_key):
            di, ti = item["query"]
            if allowed_dialogues is not None and di not in allowed_dialogues[key]:
                continue
            target = target_window(source, item)
            if target is None:
                continue
            positive, absent, target_entities, values, span = target
            windows = []
            for other, dialogue in enumerate(source):
                background_pool = background_dialogues if background_dialogues is not None else allowed_dialogues
                if background_pool is not None and other not in background_pool[key]:
                    continue
                if other == di:
                    continue
                for offset in range(len(dialogue["messages"]) - 7):
                    refs = [[other, j] for j in range(offset, offset + 8)]
                    messages = [shared.message(source, ref) for ref in refs]
                    attrs = [a for message in messages for a in message.get("attrs", [])]
                    if (any(value in message["message"] for message in messages for value in values)
                            or target_entities & {a["name"] for a in attrs}):
                        continue
                    hard = 0 if any(a["attrname"] == item["fact"]["attrname"] for a in attrs) else 1
                    windows.append((hard, shared.stable_key(refs), refs))
            backgrounds = []
            used = {di}
            for _, _, refs in sorted(windows):
                if refs[0][0] not in used:
                    backgrounds.append(refs)
                    used.add(refs[0][0])
                if len(backgrounds) == 2:
                    break
            if len(backgrounds) != 2:
                continue
            selected.append({**item, "id": f"{key}/{di}/{ti}", "source": key, "split": pin["split"],
                             "history": positive + backgrounds[0] + backgrounds[1],
                             "absent_history": absent + backgrounds[0] + backgrounds[1],
                             "gold": [item["answer"]],
                             "span": span,
                             "followup_refs": item["cue_refs"][:-1],
                             "switch_refs": backgrounds[0][:2]})
            if len(selected) == pin["limit"]:
                break
        if len(selected) != pin["limit"]:
            raise ValueError(f"Insufficient frozen coherent cases in {key}: {len(selected)}")
        cases.extend(selected)
    return {"schema": protocol["schema"], "cases": cases}


def control_results(catalog: dict, policy: str, *, system_factory=None) -> list[dict]:
    results = []
    for case in catalog["cases"]:
        system = (CompanionSystem(top_k=6, byte_cap=None, cjk_tokenizer="jieba")
                  if system_factory is None else system_factory())
        if policy != "production":
            bind_policy(system, policy)
        for index, text in enumerate(case["facts"]):
            system.observe(Observation(shared.BASE + shared.timedelta(minutes=index), "user", text))
        ids = [fragment.id for fragment in system.companion.store.all()]
        system.recall(case["warm"])
        result = system.recall(case["cue"])
        actual = {f.id for f in result.fragments_used}
        required = {ids[i] for i in case["required"]}
        passed = required <= actual and len(actual) <= 6 and actual <= set(ids)
        if "forbidden_text" in case:
            passed = passed and not any(case["forbidden_text"] in f.content for f in result.fragments_used)
        results.append({"case": case["id"], "passed": passed,
                        "required_ids": sorted(required), "selected_ids": [f.id for f in result.fragments_used]})
    return results


def prepare_recall(source, case, base_system, policy, state, *, focus_current=False):
    """One isolated, genuinely primed system and cue for scoring or tracing."""
    if state not in STATES:
        raise ValueError(f"Unknown recall state: {state}")
    if focus_current and (not case["cue_refs"] or case["query"] != case["cue_refs"][-1]):
        raise ValueError("Active question must be the last supplied cue turn")
    system = copy.deepcopy(base_system)
    if policy != "production":
        bind_policy(system, policy)
    if state != "cold":
        refs = case["followup_refs"] if state == "followup" else case["switch_refs"]
        for ref in refs:
            text = shared.message(source, ref)["message"]
            system.recall(text, scope=("study", "memory"), **({"focus_cue": text} if focus_current else {}))
    cue = " ".join(shared.message(source, ref)["message"] for ref in case["cue_refs"])
    return system, cue


def evaluate(source, case, base_system, ref_ids, policy, state, absent, *, focus_current=False):
    before = tuple(base_system.companion.store.all())
    system, cue = prepare_recall(source, case, base_system, policy, state, focus_current=focus_current)
    focus = {"focus_cue": shared.message(source, case["query"])["message"]} if focus_current else {}
    witness_view, witness_state = None, None
    if focus_current:
        # Inspect the actual bound Witness implementation on private pre-call
        # state. Its activation and Witness-only IDs must match the real pass.
        second = system.recall.__func__.__globals__["witness_couple"].__globals__["witness_second_perspective"]
        witness_state = copy.deepcopy(system.witness_state)
        witness_view = second(fragments=ScopeSpec("study", "memory").apply(system.witness.store.all()),
            cue=cue, companion_context=None, witness_state=witness_state,
            tick_count=system.heartbeat.sleep_ticks, system_seed=system.seed, top_k=system.top_k,
            witness_edges=copy.deepcopy(system.witness_edges), cjk_tokenizer=system.cjk_tokenizer, **focus)
    if tracemalloc.is_tracing():
        raise ValueError("Stop existing allocation tracing before evaluation")
    tracemalloc.start()
    try:
        started = time.perf_counter()
        result = system.recall(cue, scope=("study", "memory"), **focus)
        elapsed = (time.perf_counter() - started) * 1000
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    ids = [fragment.id for fragment in result.fragments_used]
    if witness_view is not None:
        witness_ids = [f.id for f in witness_view.fragments_used]
        if (witness_state.activation != system.witness_state.activation
                or set(system.witness_state.pain_log[-1].witness_only_fragments) != set(witness_ids) - set(ids)):
            raise AssertionError("Witness inspection differs from actual recall")
    if len(ids) > 6 or len(set(ids)) != len(ids) or not set(ids) <= set(ref_ids.values()):
        raise AssertionError("Context budget or source ownership violation")
    if tuple(system.companion.store.all()) != before:
        raise AssertionError("Source observation mutation")
    gold = {ref_ids[tuple(ref)] for ref in case["gold"] if tuple(ref) in ref_ids}
    reverse = {fid: list(ref) for ref, fid in ref_ids.items()}
    return {"case": case["id"], "source": case["source"], "candidate": policy,
            "state": state, "absent": absent, **shared.score_context(ids, gold),
            "selected_ids": ids, "selected_sources": [reverse[fid] for fid in ids],
            "gold_ids": sorted(gold), "latency_ms": elapsed, "peak_python_bytes": peak,
            **({"focus_ref": case["query"], "witness_ids": witness_ids,
                "witness_activation": witness_state.activation.residual} if focus_current else {})}


def guards(candidate, baseline, controls_pass, rules):
    return {"switch_recall": candidate["switch"]["recall_at_6"] - baseline["switch"]["recall_at_6"] >= rules["minimum_switch_recall_gain"] - 1e-12,
            "cold_recall": candidate["cold"]["recall_at_6"] >= baseline["cold"]["recall_at_6"] - 1e-12,
            "followup_recall": candidate["followup"]["recall_at_6"] >= baseline["followup"]["recall_at_6"] - 1e-12,
            "switch_mrr": candidate["switch"]["mrr_at_6"] >= baseline["switch"]["mrr_at_6"] - 1e-12,
            "absent": all(candidate[s]["absent_nonempty_rate"] <= baseline[s]["absent_nonempty_rate"] for s in STATES),
            "latency": all(candidate[s]["p95_latency_ms"] <= baseline[s]["p95_latency_ms"] * rules["max_p95_latency_ratio"] for s in STATES),
            "memory": all(candidate[s]["peak_python_bytes"] <= baseline[s]["peak_python_bytes"] * rules["max_peak_python_memory_ratio"] for s in STATES),
            "controls": controls_pass}
