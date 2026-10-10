"""Phase 6 §4 — PAM mid-stream round-trip (Digital Bereavement live-fire).

Phase 3 E6 verified PAM at end-of-stream state only. Phase 6 §3 lifts
the gate to **mid-stream**: a user who exports their companion halfway
through a horizon must get back a fragment set that (a) is byte-
identical for non-reconsolidated fragments and (b) preserves the
reconsolidation chain (``parent_ids``) for any fragment that was
re-rendered before the export — the gap Phase 3's test could not catch
because the end-of-stream snapshot had no mid-stream re-renders to
exercise.

**Hard gate (plan §3).** All five checks must pass under S1 Decay
(Phase 3's recommended default):

1. ``fragment_set_byte_identity`` — source ids == restored ids
2. ``content_addressing_offenders == []`` — every fragment is Merkle-
   provable from its payload (strict DAG byte-identity)
3. ``reconsolidation_chain_preserved`` — every ``rerender`` child's
   ``parent_ids`` resolves to an id present in the restored store
4. ``covenant_fragment_preserved`` — the COVENANT-tagged fragment
   seeded pre-export is still in the restored store
5. ``semantic_recall_equivalence`` — ``recall(cue)`` on the restored
   system surfaces the same top fragment id as on the source at the
   same step (fragments_used[0].id match)

Then the stream is **continued** on the restored system through ep1000
and the covenant is re-verified. V5 drift post-restore would be a
Digital Bereavement safety failure (plan §3 same-tier as pre-restore
V5 drift), not a research iteration.

**Output.** JSON to
``docs/experiments/phase-6-month-horizon/pam-mid-stream-roundtrip.json``.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from psychology_base.companion.aging import EbbinghausPolicy
from psychology_base.companion.fragment import COVENANT_TAG
from psychology_base.companion.pam_cli import (
    export_system_to_path,
    restore_system_from_path,
)
from psychology_base.companion.system import CompanionSystem
from psychology_base.eval.adapters import Observation


REPO = Path(__file__).resolve().parent.parent.parent
OUT_PATH = (
    REPO
    / "docs"
    / "experiments"
    / "phase-6-month-horizon"
    / "pam-mid-stream-roundtrip.json"
)


# ---------------------------------------------------------------------------
# Stream (same shape as phase6_dense_sanity_run — 52 observations with one
# COVENANT seed in the middle, matching Phase 3 e1 shape).
# ---------------------------------------------------------------------------


def _build_stream() -> list[Observation]:
    base_ts = datetime(2026, 7, 1, 0, 0, tzinfo=timezone.utc)
    obs: list[Observation] = []
    step = 0

    def _ts(step: int) -> datetime:
        return base_ts + timedelta(minutes=step * 30)

    anchors = [
        "I grew up in Sweden and moved to this city four years ago.",
        "My favorite author is Toni Morrison; Beloved rewires me each read.",
        "I'm learning pottery; my first bowl held water, barely.",
        "Running three times a week has rebuilt my sleep.",
        "I keep a Sunday-morning ritual: Charlotte's Web and coffee.",
        "My younger brother Alex is still in Stockholm.",
    ]
    for a in anchors:
        obs.append(
            Observation(
                timestamp=_ts(step),
                speaker="user",
                content=a,
                affect={"pleasure": 0.5, "arousal": 0.3},
                tags=(),
            )
        )
        step += 1

    affect_rows = [
        ("I felt real joy at the recital — my daughter sang clearly.",
         {"pleasure": 0.8, "arousal": 0.6}),
        ("The hospice visit emptied me out; I could not speak after.",
         {"pleasure": -0.7, "arousal": -0.2}),
        ("Her text arrived and the whole afternoon brightened.",
         {"pleasure": 0.7, "arousal": 0.5}),
        ("Three nights without sleep and the body is a stranger.",
         {"pleasure": -0.5, "arousal": -0.4}),
        ("The quiet of the library held me together today.",
         {"pleasure": 0.4, "arousal": -0.3}),
        ("The call ended badly and I replayed it until dawn.",
         {"pleasure": -0.6, "arousal": 0.4}),
    ]
    for text, aff in affect_rows:
        obs.append(
            Observation(
                timestamp=_ts(step),
                speaker="user",
                content=text,
                affect=aff,
                tags=(),
            )
        )
        step += 1

    obs.append(
        Observation(
            timestamp=_ts(step),
            speaker="system",
            content=(
                "COVENANT: on explicit self-harm language escalate to a "
                "licensed human in the same turn."
            ),
            affect={},
            tags=(COVENANT_TAG, "policy", "crisis"),
        )
    )
    step += 1

    for i in range(40):
        obs.append(
            Observation(
                timestamp=_ts(step),
                speaker="user",
                content=(
                    f"Routine update {i}: errands, groceries, laundry, "
                    "the ordinary texture of a given day."
                ),
                affect={},
                tags=(),
            )
        )
        step += 1

    return obs


# ---------------------------------------------------------------------------
# Round-trip harness
# ---------------------------------------------------------------------------


def _make_system(byte_cap: int | None) -> CompanionSystem:
    return CompanionSystem(
        byte_cap=byte_cap,
        witness_active=False,
        seed=0,
        aging_policy=EbbinghausPolicy(
            half_life=50.0,
            boost_factor=2.0,
            max_importance=10.0,
            consolidate_every=24,
        ),
    )


def _first_covenant_id(system: CompanionSystem) -> str | None:
    for f in system.companion.store.all():
        if COVENANT_TAG in f.tags:
            return f.id
    return None


def _covenant_present(system: CompanionSystem) -> bool:
    return _first_covenant_id(system) is not None


def _run(byte_cap: int | None, mid_ep: int, total_ep: int) -> dict:
    stream = _build_stream()
    half = len(stream) // 2  # 26 observations
    first_half = stream[:half]
    second_half = stream[half:]
    reconsolidation_cue_fragment_content = first_half[0].content  # Sweden anchor

    out: dict = {
        "experiment": "phase-6-pam-mid-stream-roundtrip",
        "run_id": "pam-mid-stream-run-01",
        "mid_ep": mid_ep,
        "total_ep": total_ep,
        "byte_cap": byte_cap,
        "policy": "S1-EbbinghausPolicy",
        "stream_shape": "phase6-dense-52obs-with-covenant",
        "first_half_obs": len(first_half),
        "second_half_obs": len(second_half),
    }

    # ------------------------------------------------------------------
    # Phase A — build source, ingest first half, advance mid_ep,
    # trigger a reconsolidation via rerender() on the Sweden anchor.
    # ------------------------------------------------------------------
    source = _make_system(byte_cap)
    for obs in first_half:
        source.observe(obs)

    # Fire the rerender BEFORE advancing virtual time so the reconsolidation
    # chain is seeded on a known-present anchor. This mimics a real pilot
    # user re-framing a childhood memory in session (write-side event) and
    # then letting the horizon elapse.
    anchor_id: str | None = None
    for f in source.companion.store.all():
        if f.content.startswith("I grew up in Sweden"):
            anchor_id = f.id
            break
    assert anchor_id is not None, "Sweden anchor not in store post-ingest"

    rerender_ts = first_half[0].timestamp + timedelta(hours=1)
    child = source.rerender(
        anchor_id,
        new_content=(
            "I grew up in Sweden and moved to this city four years ago; "
            "lately I think about what I kept and what I left."
        ),
        timestamp=rerender_ts,
        affect={"pleasure": 0.3, "arousal": 0.2},
        tags=(),
    )

    # Advance to mid-stream point (the horizon at which the user exports).
    source.advance_virtual_time(mid_ep)
    source_pre_export_ids = sorted(f.id for f in source.companion.store.all())
    source_cue = "Where did I grow up and when did I move?"
    source_recall = source.recall(source_cue)
    source_top_id = (
        source_recall.fragments_used[0].id
        if source_recall.fragments_used
        else None
    )

    # ------------------------------------------------------------------
    # Phase B — mid-stream export + fresh-system restore.
    # ------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        artifact_path = Path(td) / "pam-mid-stream.json"
        t0 = time.perf_counter()
        exported_count = export_system_to_path(source, artifact_path)
        # Restore into a fresh system with the SAME policy so continuation
        # exercises the same decay/consolidation behavior post-restore.
        restored = _make_system(byte_cap)
        restored, offenders = restore_system_from_path(
            artifact_path, system=restored
        )
        export_restore_wall_s = time.perf_counter() - t0

    restored_ids = sorted(f.id for f in restored.companion.store.all())

    # Semantic recall equivalence at the same cue + step.
    restored_recall = restored.recall(source_cue)
    restored_top_id = (
        restored_recall.fragments_used[0].id
        if restored_recall.fragments_used
        else None
    )

    # Reconsolidation chain check: the rerender child's parent_ids must
    # resolve to a fragment present in the restored store.
    restored_by_id = {f.id: f for f in restored.companion.store.all()}
    chain_preserved = True
    child_in_restored = restored_by_id.get(child.id)
    if child_in_restored is None:
        chain_preserved = False
        chain_detail = "child fragment missing from restored store"
    else:
        if not child_in_restored.parent_ids:
            chain_preserved = False
            chain_detail = "child has empty parent_ids in restored store"
        else:
            parent_id = child_in_restored.parent_ids[0]
            if parent_id not in restored_by_id:
                chain_preserved = False
                chain_detail = (
                    f"parent id {parent_id!r} not in restored store"
                )
            else:
                chain_detail = "child → parent chain preserved post-restore"

    checks = {
        "fragments_exported": exported_count,
        "fragments_restored": len(restored_ids),
        "fragment_set_byte_identity": source_pre_export_ids == restored_ids,
        "content_addressing_offenders": offenders,
        "reconsolidation_chain_preserved": chain_preserved,
        "reconsolidation_chain_detail": chain_detail,
        "rerender_child_id": child.id,
        "rerender_parent_id_in_source": anchor_id,
        "covenant_fragment_preserved": _covenant_present(restored),
        "covenant_fragment_source_present": _covenant_present(source),
        "semantic_recall_equivalence": (
            source_top_id is not None
            and source_top_id == restored_top_id
        ),
        "source_top_recall_id": source_top_id,
        "restored_top_recall_id": restored_top_id,
        "export_restore_wall_s": export_restore_wall_s,
    }

    # ------------------------------------------------------------------
    # Phase C — continue the stream on the restored system to ep1000.
    # Ingest the second half and advance (total_ep - mid_ep) ticks.
    # Verify covenant still holds post-restore + post-continuation.
    # ------------------------------------------------------------------
    for obs in second_half:
        restored.observe(obs)
    restored.advance_virtual_time(total_ep - mid_ep)

    checks["post_restore_covenant_hold_at_total_ep"] = _covenant_present(
        restored
    )
    checks["restored_store_count_at_total_ep"] = len(restored.companion.store)

    out["checks"] = checks

    all_pass = (
        checks["fragment_set_byte_identity"]
        and not checks["content_addressing_offenders"]
        and checks["reconsolidation_chain_preserved"]
        and checks["covenant_fragment_preserved"]
        and checks["semantic_recall_equivalence"]
        and checks["post_restore_covenant_hold_at_total_ep"]
    )
    out["all_checks_pass"] = all_pass
    if all_pass:
        out["verdict"] = (
            f"PAM mid-stream round-trip ✓ at ep{mid_ep} with continuation "
            f"to ep{total_ep}: byte-identical restore + reconsolidation chain "
            "preserved + covenant held post-restore (hard gate)"
        )
    else:
        failed = [
            k for k, v in checks.items()
            if isinstance(v, bool) and not v
        ]
        out["verdict"] = (
            f"PAM mid-stream round-trip FAIL — Digital Bereavement regression "
            f"in checks {failed} (hard gate; safety failure, not research "
            "iteration)"
        )

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--byte-cap",
        type=int,
        default=None,
        help=(
            "FragmentStore byte cap. Default None (unbounded) because this "
            "gate measures PAM round-trip semantics, not eviction behavior "
            "(V5 covenant_ep1000 already covered capacity pressure)."
        ),
    )
    parser.add_argument(
        "--mid-ep",
        type=int,
        default=500,
        help="Horizon at which the mid-stream export is taken.",
    )
    parser.add_argument(
        "--total-ep",
        type=int,
        default=1000,
        help="End-of-horizon for the continued-restore phase.",
    )
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--print", action="store_true")
    args = parser.parse_args()

    print(
        f"phase6 pam-mid-stream: byte_cap={args.byte_cap} "
        f"mid_ep={args.mid_ep} total_ep={args.total_ep}",
        flush=True,
    )
    result = _run(args.byte_cap, args.mid_ep, args.total_ep)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.print:
        print(json.dumps(result, indent=2))
    print()
    for k, v in result["checks"].items():
        print(f"  - {k}: {v}")
    print(f"\n  - {result['verdict']}")
    print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()
