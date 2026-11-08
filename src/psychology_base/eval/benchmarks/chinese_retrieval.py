"""Source-backed lexical recall study, using the installed memory implementation.

This is a film-domain pilot with mechanically defined supporting-span labels.
It neither supplies human relevance judgments nor selects a production default.
"""

from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
from statistics import mean
import time
import tracemalloc
from types import FunctionType, MethodType

from psychology_base.companion import reconstruction as memory
from psychology_base.companion.system import CompanionSystem
from psychology_base.eval.adapters import Observation
from psychology_base.i18n import _load_jieba, detect_lang, tokenizer_identity


HAN_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
ASCII_ID = re.compile(r"[A-Za-z0-9]+(?:[-_.][A-Za-z0-9]+)*")
LETTERS = re.compile(r"[^\W\d_]+", re.UNICODE)
CANDIDATES = ("unicode", "precise", "search", "bigram", "bounded", "width_ids")
BASE = datetime(2026, 10, 18, tzinfo=timezone.utc)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, value: dict) -> None:
    """Write new evaluation evidence without replacing an existing artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.write("\n")


def stable_key(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def load_source(path: Path, protocol: dict) -> list[dict]:
    raw = path.read_bytes()
    pin = protocol["source"]
    blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    if (len(raw) != pin["bytes"] or hashlib.sha256(raw).hexdigest() != pin["sha256"]
            or blob != pin["blob_sha1"]):
        raise ValueError("Source differs from the frozen KdConv byte/hash pins")
    return json.loads(raw)


def message(source: list[dict], ref: list[int]) -> dict:
    return source[ref[0]]["messages"][ref[1]]


def question_candidates(source: list[dict], previous_turns=2) -> list[dict]:
    """One upstream question/structured-answer candidate per dialogue, by hash."""
    candidates = []
    for di, dialogue in enumerate(source):
        eligible = []
        turns = dialogue["messages"]
        for ti, turn in enumerate(turns[:-1]):
            question = turn["message"]
            cue_refs = [[di, j] for j in range(max(0, ti - previous_turns), ti + 1)]
            cue = " ".join(message(source, ref)["message"] for ref in cue_refs)
            if not any(mark in question for mark in ("?", "？")):
                continue
            answer = turns[ti + 1]
            for attr in answer.get("attrs", []):
                entity, relation, value = (attr[k] for k in ("name", "attrname", "attrvalue"))
                if (relation.lower() == "information" or len(value) < 2 or value in cue
                        or entity.split("（")[0] not in cue or relation not in question
                        or value not in answer["message"]):
                    continue
                eligible.append({"dialogue": di, "query": [di, ti], "cue_refs": cue_refs,
                                 "answer": [di, ti + 1], "fact": attr})
        if eligible:
            candidates.append(min(eligible, key=stable_key))
    return candidates


def build_catalog(source: list[dict], protocol: dict) -> dict:
    """Freeze source pointers independently of candidate tokens or scores."""
    selection = protocol["selection"]
    groups = {"development": [], "evaluation": []}
    for candidate in question_candidates(source, selection["query_context_previous_turns"]):
        split = "development" if int(stable_key(candidate["dialogue"]), 16) % 3 == 0 else "evaluation"
        groups[split].append(candidate)

    cases = []
    split_dialogues = {}
    for split, eligible in groups.items():
        # Distractor dialogues also stay within the same local split.
        allowed = {di for di in range(len(source))
                   if (int(stable_key(di), 16) % 3 == 0) == (split == "development")}
        split_dialogues[split] = sorted(allowed)
        for candidate in sorted(eligible, key=stable_key)[:selection[f"{split}_limit"]]:
            attr = candidate["fact"]
            di, answer_ti = candidate["answer"]
            # Preserve the local antecedent around a pronoun-bearing answer.
            support_context = [[di, j] for j in range(max(0, answer_ti - 2), answer_ti + 1)]
            gold = [candidate["answer"]]
            forbidden_values = {
                a["attrvalue"] for a in message(source, candidate["answer"]).get("attrs", [])
                if a["name"] == attr["name"] and a["attrname"] == attr["attrname"]
            }
            distractors = []
            for ddi in sorted(allowed):
                for tti, turn in enumerate(source[ddi]["messages"]):
                    ref = [ddi, tti]
                    if ref in support_context or ref in candidate["cue_refs"]:
                        continue
                    if any(value in turn["message"] for value in forbidden_values):
                        continue
                    attrs = turn.get("attrs", [])
                    if not attrs:
                        continue
                    if any(a["name"] == attr["name"] and a["attrname"] == attr["attrname"] for a in attrs):
                        continue
                    # Hard pool: same subject/other relation, then same relation/other subject.
                    hard = 0 if any(a["name"] == attr["name"] for a in attrs) else (
                        1 if any(a["attrname"] == attr["attrname"] for a in attrs) else 2)
                    distractors.append((hard, stable_key(ref), ref))
            count = selection["history_size"] - len(support_context)
            pool = [ref for _, _, ref in sorted(distractors)[:count]]
            if len(distractors) <= count:
                raise ValueError("Insufficient independently sourced distractors")
            present = sorted(pool + support_context, key=stable_key)
            start = message(source, candidate["answer"])["message"].index(attr["attrvalue"])
            case = {**candidate, "split": split, "id": f"film-{di}-{candidate['query'][1]}",
                    "history": present, "gold": gold,
                    "span": [start, start + len(attr["attrvalue"])],
                    "absent_history": sorted([ref for ref in present if ref not in gold]
                                             + [sorted(distractors)[count][2]], key=stable_key),
                    "label_basis": "upstream structured relation + exact source span"}
            # The surrounding turns must not disclose the withheld answer either.
            if any(any(value in message(source, ref)["message"] for value in forbidden_values)
                   for ref in case["absent_history"]):
                continue
            cases.append(case)
    if not all(any(c["split"] == split for c in cases) for split in groups):
        raise ValueError("Study requires nonempty development and evaluation splits")
    return {"schema": protocol["schema"], "source_sha256": protocol["source"]["sha256"],
            "split_dialogues": split_dialogues, "cases": cases}


def fold_width(text: str) -> str:
    """One-character ASCII-width folding; original character offsets stay valid."""
    return "".join(chr(ord(ch) - 0xFEE0) if 0xFF01 <= ord(ch) <= 0xFF5E
                   else " " if ch == "\u3000" else ch for ch in text)


def features(text: str, policy: str, cap: int = 64) -> list[str]:
    """Real lexical features; no answer-specific dictionaries or source labels."""
    if policy not in CANDIDATES:
        raise ValueError(f"Unknown study candidate: {policy}")
    if policy in ("unicode", "precise"):
        return memory.tokens(text, "unicode" if policy == "unicode" else "jieba")
    original = text
    if policy == "width_ids":
        text = fold_width(text)
    if detect_lang(text) != "zh":
        return memory.tokens(original, "jieba")
    backend, _, error = _load_jieba()
    if backend is None:
        raise ValueError(f"Study requires real jieba: {error}")
    search = [word.lower() for word, _, _ in backend.tokenize(text, mode="search", HMM=True)
              if LETTERS.fullmatch(word)]
    bigrams = [run[i:i + 2] for run in HAN_RUN.findall(text) for i in range(len(run) - 1)]
    latin = [word.lower() for word in ASCII_ID.findall(text)]
    if policy == "search":
        return search
    if policy == "bigram":
        return list(dict.fromkeys(bigrams + latin))
    # A bounded two-channel union, not calibrated rank fusion. Namespaces keep
    # word features and character features from accidentally sharing a dimension.
    words = search + (latin if policy == "width_ids" else [])
    return (["w:" + word for word in list(dict.fromkeys(words))[:cap]]
            + ["h:" + word for word in list(dict.fromkeys(bigrams))[:cap]])


def _bind(function, **bindings):
    """Rebind inputs to installed production bytecode, without copying its logic.

    Per-instance experiment only: no module/global monkeypatch, no canned result.
    Coalescence and Witness keep their ordinary module bindings. Cloned globals
    are shallow; stores, algorithms and helpers are still the real implementation.
    """
    bound = FunctionType(function.__code__, {**function.__globals__, **bindings},
                         function.__name__, function.__defaults__, function.__closure__)
    bound.__kwdefaults__ = copy.copy(function.__kwdefaults__)
    return bound


def study_system(policy: str, *, cap: int = 64) -> CompanionSystem:
    if policy not in CANDIDATES:
        raise ValueError(f"Unknown study candidate: {policy}")
    system = CompanionSystem(top_k=6, seed=0, byte_cap=None,
                             cjk_tokenizer="unicode" if policy == "unicode" else "jieba")
    if policy not in ("unicode", "precise"):
        # Only Companion seed extraction changes. Graphs and Witness deliberately
        # stay precise: this tests seed representations, not a migrated graph policy.
        feature_tokens = lambda text, _mode="unicode": features(text, policy, cap)
        snapshot = _bind(memory.recall_tokens, tokens=feature_tokens)
        spread = _bind(memory.spread_activation, tokens=feature_tokens, recall_tokens=snapshot)
        reconstruct = _bind(memory.reconstruct, spread_activation=spread, recall_tokens=snapshot)
        recall = _bind(CompanionSystem.recall, reconstruct=reconstruct)
        system.recall = MethodType(recall, system)
    return system


def score_context(selected: list[str], relevant: set[str]) -> dict:
    hits = [i + 1 for i, fid in enumerate(selected) if fid in relevant]
    return {"recall_at_6": len(set(selected) & relevant) / len(relevant) if relevant else None,
            "mrr_at_6": 1 / hits[0] if hits else 0.0,
            "support_precision": len(hits) / len(selected) if selected else 0.0,
            "nonempty": bool(selected)}


def prepare_trial(source: list[dict], case: dict, policy: str, *, absent=False, cap=64,
                  source_sessions=False, system_factory=None, annotation=None):
    """One actual ingestion/scope setup shared by evaluation and diagnostics."""
    system = study_system(policy, cap=cap) if system_factory is None else system_factory()
    history = case["absent_history"] if absent else case["history"]
    ref_ids = {}
    for index, ref in enumerate(history):
        if source_sessions and index and ref[0] != history[index - 1][0]:
            system.mark_session_boundary()
        previous = {f.id for f in system.companion.store.all()}
        system.observe(Observation(
            BASE + timedelta(minutes=index),
            "unknown-source-actor" if annotation is not None else f"speaker-{ref[1] % 2}",
            message(source, ref)["message"],
            tags=("user:study", f"session:{ref[0] if source_sessions else index // 8}")
                 + (() if annotation is None else tuple(annotation(ref))),
            subject_id="study", entity_id="memory"))
        added = {f.id for f in system.companion.store.all()} - previous
        if len(added) != 1:
            raise AssertionError("Observation did not retain its actual source occurrence")
        ref_ids[tuple(ref)] = added.pop()
        if not source_sessions and (index + 1) % 8 == 0:
            system.mark_session_boundary()
    # Exact foreign evidence is an isolation challenge, including absent trials.
    system.observe(Observation(BASE + timedelta(days=1), "speaker-foreign",
                               message(source, case["answer"])["message"],
                               tags=("user:foreign",), subject_id="foreign", entity_id="memory"))
    return system, ref_ids, history


def evaluate_case(source: list[dict], case: dict, policy: str, *, absent=False,
                  warm=False, cap=64) -> dict:
    if tracemalloc.is_tracing():
        raise ValueError("Study owns recall allocation tracing; stop an existing tracing session first")
    system, ref_ids, history = prepare_trial(source, case, policy, absent=absent, cap=cap)
    eligible_ids = set(ref_ids.values())
    before = tuple(system.companion.store.all())
    initial_edges = len(system.edges)
    initial_graph = stable_key(sorted((list(pair), value) for pair, value in system.edges.weights.items()))
    if warm:
        for ref in history[:2]:
            system.recall(message(source, ref)["message"], scope=("study", "memory"))
    cue = " ".join(message(source, ref)["message"] for ref in case["cue_refs"])
    tracemalloc.start()
    try:
        start = time.perf_counter()
        recall = system.recall(cue, scope=("study", "memory"))
        elapsed = (time.perf_counter() - start) * 1000
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    selected = [f.id for f in recall.fragments_used]
    if len(selected) > 6 or len(set(selected)) != len(selected) or not set(selected) <= eligible_ids:
        raise AssertionError("Context budget or ownership invariant failed")
    if tuple(system.companion.store.all()) != before:
        raise AssertionError("Recall changed raw observations")
    relevant = {ref_ids[tuple(ref)] for ref in case["gold"] if tuple(ref) in ref_ids}
    reverse = {fid: list(ref) for ref, fid in ref_ids.items()}
    return {"case": case["id"], "split": case["split"], "candidate": policy,
            "absent": absent, "state": "warm" if warm else "cold",
            **score_context(selected, relevant), "selected_ids": selected,
            "selected_sources": [reverse[fid] for fid in selected],
            "gold_ids": sorted(relevant), "latency_ms": elapsed, "peak_python_bytes": peak,
            "initial_graph_sha256": initial_graph, "initial_graph_edges": initial_edges,
            "graph_edges": len(system.edges), "stored_occurrences": len(before),
            "unique_context_content": len({f.content for f in recall.fragments_used})}


def diagnose_case(source: list[dict], case: dict) -> dict:
    """Exploratory state ablation on copies, not another selection candidate.

    Reset only Companion residuals after the two warm-up recalls; learned edges
    and Witness state remain. Execute actual recall on each copied system.
    """
    system, ref_ids, history = prepare_trial(source, case, "precise")
    cue = " ".join(message(source, ref)["message"] for ref in case["cue_refs"])
    gold_id = ref_ids[tuple(case["answer"])]
    cue_source_ids = {ref_ids[tuple(ref)] for ref in case["cue_refs"] if tuple(ref) in ref_ids}
    cold = copy.deepcopy(system).recall(cue, scope=("study", "memory"))
    for ref in history[:2]:
        system.recall(message(source, ref)["message"], scope=("study", "memory"))
    reset = copy.deepcopy(system)
    reset.activation = memory.ActivationState()
    warm = system.recall(cue, scope=("study", "memory"))
    reset_recall = reset.recall(cue, scope=("study", "memory"))
    variants = {"cold": cold, "warm": warm, "warm_reset_residual": reset_recall}
    for recall in variants.values():
        if len(recall.fragments_used) > 6 or not {f.id for f in recall.fragments_used} <= set(ref_ids.values()):
            raise AssertionError("Diagnostic scope/context invariant failed")
    return {"case": case["id"], "split": case["split"],
            "direct_gold_token_overlap": sorted(set(memory.tokens(cue, "jieba")) & set(
                memory.tokens(message(source, case["answer"])["message"], "jieba"))),
            "variants": {name: {"gold_hit": any(f.id == gold_id for f in recall.fragments_used),
                                "query_antecedent_slots": sum(f.id in cue_source_ids for f in recall.fragments_used),
                                "selected_ids": [f.id for f in recall.fragments_used]}
                         for name, recall in variants.items()}}


def percentile95(values):
    ordered = sorted(values)
    return ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)]


def summarize_trials(rows: list[dict], policies=CANDIDATES, states=("cold", "warm")) -> dict:
    output = {}
    for policy in policies:
        output[policy] = {}
        for state in states:
            pool = [r for r in rows if r["candidate"] == policy and r["state"] == state]
            positive = [r for r in pool if not r["absent"]]
            negative = [r for r in pool if r["absent"]]
            if not positive or not negative:
                raise ValueError("Incomplete candidate/split/state trial matrix")
            output[policy][state] = {
                "positive_cases": len(positive), "absent_cases": len(negative),
                "recall_at_6": mean(r["recall_at_6"] for r in positive),
                "mrr_at_6": mean(r["mrr_at_6"] for r in positive),
                "support_precision": mean(r["support_precision"] for r in positive),
                "absent_nonempty_rate": mean(r["nonempty"] for r in negative),
                "p95_latency_ms": percentile95([r["latency_ms"] for r in pool]),
                "peak_python_bytes": max(r["peak_python_bytes"] for r in pool),
            }
            for field, label in (("graph_edges", "mean_graph_edges"),
                                 ("unique_context_content", "mean_unique_context_content")):
                if all(field in r for r in pool):
                    output[policy][state][label] = mean(r[field] for r in pool)
    return output


def summarize(rows: list[dict]) -> dict:
    return {split: summarize_trials([r for r in rows if r["split"] == split])
            for split in ("development", "evaluation")}


def gate(candidate: dict, baseline: dict, rules: dict) -> dict:
    cold, base = candidate["cold"], baseline["cold"]
    return {
        "recall_gain": cold["recall_at_6"] - base["recall_at_6"] >= rules["minimum_recall_gain"] - 1e-12,
        "mrr": cold["mrr_at_6"] >= base["mrr_at_6"] + rules["minimum_mrr_gain"] - 1e-12,
        "support_precision": cold["support_precision"] >= base["support_precision"] + rules["minimum_support_precision_gain"] - 1e-12,
        "absent": cold["absent_nonempty_rate"] <= base["absent_nonempty_rate"] + rules["maximum_absent_nonempty_rate_gain"] + 1e-12,
        "warm_recall": candidate["warm"]["recall_at_6"] >= baseline["warm"]["recall_at_6"] - 1e-12,
        "latency": all(candidate[s]["p95_latency_ms"] <= baseline[s]["p95_latency_ms"] * rules["maximum_p95_latency_ratio"] for s in ("cold", "warm")),
        "memory": all(candidate[s]["peak_python_bytes"] <= baseline[s]["peak_python_bytes"] * rules["maximum_peak_python_memory_ratio"] for s in ("cold", "warm")),
    }


def decide(summary: dict, protocol: dict) -> dict:
    rules = protocol["selection_gate"]
    development = summary["development"]
    guards = {p: gate(development[p], development["precise"], rules)
              for p in CANDIDATES if p not in ("unicode", "precise")}
    eligible = [p for p, checks in guards.items() if all(checks.values())]
    chosen = max(eligible, key=lambda p: (development[p]["cold"]["recall_at_6"],
                                         development[p]["cold"]["mrr_at_6"],
                                         -CANDIDATES.index(p))) if eligible else None
    heldout = gate(summary["evaluation"][chosen], summary["evaluation"]["precise"], rules) if chosen else None
    return {"development_guards": guards, "development_selected": chosen,
            "evaluation_guards": heldout,
            "advance_to_independent_replication": bool(heldout and all(heldout.values())),
            "production_adoption": False,
            "reason": rules["production_adoption"]}
