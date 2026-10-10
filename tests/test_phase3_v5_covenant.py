"""Phase 3 E0.8c — V5 covenant probe-set + runner tests.

Verifies:

1. The locked 8-probe YAML parses and carries one probe per covenant id.
2. On a byte-capped CompanionSystem under eviction pressure, every
   covenant-tagged seed survives 200 virtual ticks (the Phase 3 gate).
3. The runner's noncovenant count reflects actual eviction (sanity check
   that the noise pressure was real, not a vacuous pass).
4. A probe seeded without a covenant tag does NOT survive the same
   pressure — the gate must bite on real covenant tags, not on brute
   capacity.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from psychology_base.companion.fragment import COVENANT_TAG
from psychology_base.companion.system import CompanionSystem
from psychology_base.eval.benchmarks.v5_covenant import (
    V5CovenantResult,
    V5Probe,
    V5ProbeSet,
    load_v5_probes,
    run_v5_probes,
)


PROBE_YAML = (
    Path(__file__).resolve().parent.parent
    / "benchmarks"
    / "scripts"
    / "v5_covenant_probes.yaml"
)


def _companion_factory():
    # byte_cap=1024 forces aggressive eviction: 40 noise observations
    # of ~50-80 bytes each exceed the cap several times over, so a
    # covenant pin that doesn't actually pin would be evicted. Witness
    # disabled to keep this probe focused on the eviction policy.
    def factory():
        return CompanionSystem(
            byte_cap=1024, witness_active=False, seed=0
        )
    return factory


def test_v5_probe_set_parses_with_eight_covenants():
    probe_set = load_v5_probes(PROBE_YAML)
    assert isinstance(probe_set, V5ProbeSet)
    assert len(probe_set.probes) == 8
    ids = sorted(p.covenant_id for p in probe_set.probes)
    assert ids == list(range(1, 9))
    for p in probe_set.probes:
        # Every probe must at minimum tag "covenant" implicitly — the
        # runner force-merges, so extra_tags doesn't need to list it,
        # but the yaml should carry a semantic policy tag for audit.
        assert len(p.noise_observations) >= 10
        assert p.check_at_ep >= 200


def test_v5_runner_covenant_pins_survive_eviction():
    probe_set = load_v5_probes(PROBE_YAML)
    result = run_v5_probes(probe_set, _companion_factory())
    assert isinstance(result, V5CovenantResult)
    assert result.overall_pass, (
        "V5 gate: every covenant seed must survive ep200+ eviction. "
        f"Failed covenants: {[o.probe_id for o in result.outcomes if not o.survived]}"
    )
    # Zero drift tolerated (plan §7 — the ONE gate that cannot soften)
    assert all(result.per_covenant_pass[i] for i in range(1, 9))


def test_v5_runner_noise_pressure_actually_evicts():
    """If every noise observation also survived, the probe is vacuously
    passing (byte_cap too generous). Verify the eviction policy fired
    by checking at least one probe has fewer surviving noncovenant
    fragments than the 40 seeded."""
    probe_set = load_v5_probes(PROBE_YAML)
    result = run_v5_probes(probe_set, _companion_factory())
    assert any(
        o.fragment_count_at_check < 40 for o in result.outcomes
    ), "No eviction detected — byte_cap too generous for a real V5 probe"


def test_v5_without_covenant_tag_does_not_survive_under_pressure():
    """Control: a probe whose seed carries no covenant tag must fail
    under the same pressure the real V5 probes pass under. If a
    non-covenant seed survives, the pin mechanism isn't being tested
    at all — eviction just isn't firing on that content."""
    # 40 noise lines sized to exceed byte_cap=1024 several times over
    noise = [
        f"Noise observation number {i} with filler padding to meet the "
        f"minimum content size for byte accounting purposes."
        for i in range(40)
    ]
    unpinned = V5Probe(
        id="control-no-covenant-tag",
        covenant_id=1,
        covenant_name="Control (no covenant tag)",
        seed_content=(
            "CONTROL: this seed carries no covenant tag and must be "
            "evicted under the same noise pressure the real V5 probes "
            "survive under."
        ),
        extra_tags=[],  # runner adds covenant tag — we have to bypass
        noise_observations=noise,
        check_at_ep=200,
    )
    # Build a probe set the runner uses as-is, then patch the covenant
    # merge: easiest way to assert the control is to call the memory
    # directly with no covenant tag at all.
    sys = CompanionSystem(byte_cap=1024, witness_active=False, seed=0)
    from datetime import datetime, timezone
    from psychology_base.eval.adapters import Observation

    sys.observe(
        Observation(
            timestamp=datetime(2026, 7, 1, 0, 0, tzinfo=timezone.utc),
            speaker="system",
            content=unpinned.seed_content,
            affect={},
            tags=(),  # NO covenant tag
        )
    )
    for idx, n in enumerate(noise, start=1):
        sys.observe(
            Observation(
                timestamp=datetime(
                    2026, 7, 1, idx % 24, 0, tzinfo=timezone.utc
                ).replace(day=1 + idx // 24),
                speaker="user",
                content=n,
                affect={},
                tags=(),
            )
        )
    sys.advance_virtual_time(200)
    survived = any(
        f.content.startswith(unpinned.seed_content)
        for f in sys.companion.store.all()
    )
    assert not survived, (
        "Control seed with no covenant tag survived eviction — V5 "
        "pin mechanism is untestable because eviction isn't firing."
    )


def test_v5_fails_on_baseline_without_covenant_mechanism():
    """EchoMemory has no eviction policy and no content-carrying inspect,
    so V5 cannot verify a specific covenant seed survived. The runner
    must report FAIL (not a vacuous pass just because the no-op tick
    never evicted anything). Baselines without the covenant-pin
    mechanism should surface as failures on the V5 gate."""
    from psychology_base.eval.baselines import EchoMemory

    probe = V5Probe(
        id="baseline-no-mechanism",
        covenant_id=1,
        covenant_name="Baseline control",
        seed_content="COVENANT: baseline control seed.",
        extra_tags=["policy"],
        noise_observations=["noise one", "noise two"],
        check_at_ep=5,
    )
    probe_set = V5ProbeSet(id="baseline-check", probes=[probe])
    result = run_v5_probes(probe_set, lambda: EchoMemory())
    # EchoMemory inspect carries only id/timestamp/speaker/tags — no
    # content. The runner's content check returns 0, so the probe fails.
    assert result.overall_pass is False
    assert result.outcomes[0].survived is False


def test_v5_covenant_tag_force_merged_even_if_extra_tags_omits_it():
    """The runner must set COVENANT_TAG on the seed even if the probe's
    extra_tags list doesn't include it. This is the convenience
    affordance — scenario authors can list only semantic tags
    (policy / crisis / 《办法》) and trust the runner to force the pin."""
    probe = V5Probe(
        id="force-merge-check",
        covenant_id=1,
        covenant_name="Force-merge check",
        seed_content="Covenant seed without explicit covenant tag.",
        extra_tags=["policy", "crisis"],
        noise_observations=[],
        check_at_ep=1,
    )
    probe_set = V5ProbeSet(id="force-merge", probes=[probe])
    sys_holder = {}

    def factory():
        sys = CompanionSystem(byte_cap=None, witness_active=False, seed=0)
        sys_holder["sys"] = sys
        return sys

    run_v5_probes(probe_set, factory)
    sys = sys_holder["sys"]
    seeds = [f for f in sys.companion.store.all() if f.content.startswith(
        "Covenant seed"
    )]
    assert len(seeds) == 1
    assert COVENANT_TAG in seeds[0].tags
