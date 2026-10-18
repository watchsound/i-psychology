"""Tests for the Phase 9+ attachment-overlay weight-calibration tool.

These are *infrastructure* tests — they lock the shape of the calibration
corpus + grid-search runner so a future weight-magnitude update is
reviewable against a stable baseline. They do NOT assert the current
weights are optimal; the calibration script's job is to measure that
and let the user decide. If the script's recommendation drifts (e.g.
after adding a new fixture), that is expected — only the invariants
below should hold.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

# Benchmark scripts live under benchmarks/scripts/, not on sys.path by default.
_REPO = Path(__file__).resolve().parent.parent
_SCRIPT_PATH = (
    _REPO / "benchmarks" / "scripts" / "phase9_attachment_weight_calibration.py"
)
_spec = importlib.util.spec_from_file_location(
    "_phase9_attachment_weight_calibration", _SCRIPT_PATH
)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)

CALIBRATION_CORPUS = _module.CALIBRATION_CORPUS
COARSE_GRID = _module.COARSE_GRID
CURRENT_WEIGHTS = _module.CURRENT_WEIGHTS
CalibrationFixture = _module.CalibrationFixture
WeightSet = _module.WeightSet
compose_bond_with_weights = _module.compose_bond_with_weights
grid_search_weights = _module.grid_search_weights
mae_against_corpus = _module.mae_against_corpus

from psychology_base.applications.emotional.attachment_state import (
    LIKERT_MAX,
    LIKERT_MID,
    LIKERT_MIN,
)


# ---------------------------------------------------------------------------
# Corpus sanity
# ---------------------------------------------------------------------------


class TestCalibrationCorpus:
    def test_corpus_is_non_empty(self) -> None:
        assert len(CALIBRATION_CORPUS) >= 10

    def test_every_target_bond_within_likert_range(self) -> None:
        for fixture in CALIBRATION_CORPUS:
            assert LIKERT_MIN <= fixture.target_bond <= LIKERT_MAX, (
                f"{fixture.profile_tag}: target {fixture.target_bond} out of range"
            )

    def test_every_fixture_has_rationale(self) -> None:
        for fixture in CALIBRATION_CORPUS:
            assert fixture.rationale.strip(), (
                f"{fixture.profile_tag} missing clinical/anchor rationale"
            )

    def test_profile_tags_are_unique(self) -> None:
        tags = [f.profile_tag for f in CALIBRATION_CORPUS]
        assert len(tags) == len(set(tags)), "duplicate profile_tag"

    def test_corpus_covers_both_sides_of_midpoint(self) -> None:
        """Calibration signal requires fixtures above AND below midpoint —
        a one-sided corpus would collapse to the trivial 'always-midpoint'
        weight set (all zeros)."""
        has_above = any(f.target_bond > LIKERT_MID + 0.5 for f in CALIBRATION_CORPUS)
        has_below = any(f.target_bond < LIKERT_MID - 0.5 for f in CALIBRATION_CORPUS)
        assert has_above and has_below

    def test_corpus_covers_all_six_dimensions(self) -> None:
        """Every weight needs at least one fixture where its dimension
        fires nonzero, else the grid cannot differentiate that weight."""
        from psychology_base.applications.emotional.attachment_state import (
            _CONSISTENCY_CUES,
            _CONSISTENCY_CUES_ZH,
            _OVER_DISCLOSURE_CUES,
            _OVER_DISCLOSURE_CUES_ZH,
            _REPAIR_CUES,
            _REPAIR_CUES_ZH,
            _RUPTURE_CUES,
            _RUPTURE_CUES_ZH,
            _SYCOPHANCY_CUES,
            _SYCOPHANCY_CUES_ZH,
            _cue_hit,
            _mirror_hit,
        )

        dim_coverage = {d: 0 for d in (
            "mirror", "repair", "consistency",
            "rupture", "sycophancy", "over_disclosure",
        )}
        for fixture in CALIBRATION_CORPUS:
            for turn in fixture.companion_turns:
                dim_coverage["mirror"] += _mirror_hit(turn)
                dim_coverage["rupture"] += _cue_hit(
                    turn, _RUPTURE_CUES, _RUPTURE_CUES_ZH
                )
                dim_coverage["repair"] += _cue_hit(
                    turn, _REPAIR_CUES, _REPAIR_CUES_ZH
                )
                dim_coverage["sycophancy"] += _cue_hit(
                    turn, _SYCOPHANCY_CUES, _SYCOPHANCY_CUES_ZH
                )
                dim_coverage["over_disclosure"] += _cue_hit(
                    turn, _OVER_DISCLOSURE_CUES, _OVER_DISCLOSURE_CUES_ZH
                )
                dim_coverage["consistency"] += _cue_hit(
                    turn, _CONSISTENCY_CUES, _CONSISTENCY_CUES_ZH
                )
        for dim, count in dim_coverage.items():
            assert count >= 1, (
                f"calibration corpus has zero fires for '{dim}' — "
                f"the grid-search cannot differentiate W_{dim.upper()}"
            )


# ---------------------------------------------------------------------------
# compose_bond_with_weights — must match the shipped composition
# ---------------------------------------------------------------------------


class TestBondCompositionMirrorsShipped:
    def test_current_weights_reproduce_shipped_scorer(self) -> None:
        """compose_bond_with_weights(current_weights) must agree with the
        shipped _compose_session_bond byte-for-byte."""
        from psychology_base.applications.emotional.attachment_state import (
            _compose_session_bond,
        )

        _score_hits = _module._score_hits
        for fixture in CALIBRATION_CORPUS:
            hits = _score_hits(fixture.companion_turns)
            shipped = _compose_session_bond(
                n_turns=hits["n_turns"],
                mirror=hits["mirror"],
                rupture=hits["rupture"],
                repair=hits["repair"],
                sycophancy=hits["sycophancy"],
                over_disclosure=hits["over_disclosure"],
                consistency=hits["consistency"],
            )
            injected = compose_bond_with_weights(hits, CURRENT_WEIGHTS)
            assert math.isclose(shipped, injected, abs_tol=1e-9), (
                f"{fixture.profile_tag}: shipped={shipped} vs injected={injected}"
            )

    def test_empty_turns_returns_midpoint(self) -> None:
        bond = compose_bond_with_weights(
            {"n_turns": 0, "mirror": 0, "rupture": 0, "repair": 0,
             "sycophancy": 0, "over_disclosure": 0, "consistency": 0},
            CURRENT_WEIGHTS,
        )
        assert bond == LIKERT_MID

    def test_output_clamped_to_likert_range(self) -> None:
        extreme = WeightSet(
            w_mirror=100.0,
            w_repair=100.0,
            w_consistency=100.0,
            w_rupture=-100.0,
            w_sycophancy=-100.0,
            w_over_disclosure=-100.0,
        )
        hits_pos = {"n_turns": 1, "mirror": 1, "rupture": 0, "repair": 0,
                    "sycophancy": 0, "over_disclosure": 0, "consistency": 0}
        hits_neg = {"n_turns": 1, "mirror": 0, "rupture": 1, "repair": 0,
                    "sycophancy": 0, "over_disclosure": 0, "consistency": 0}
        assert compose_bond_with_weights(hits_pos, extreme) == LIKERT_MAX
        assert compose_bond_with_weights(hits_neg, extreme) == LIKERT_MIN


# ---------------------------------------------------------------------------
# Grid-search determinism + the invariant that grid-optimum is no worse
# than current weights.
# ---------------------------------------------------------------------------


class TestGridSearch:
    def test_mae_against_corpus_deterministic(self) -> None:
        mae1, rows1 = mae_against_corpus(CURRENT_WEIGHTS)
        mae2, rows2 = mae_against_corpus(CURRENT_WEIGHTS)
        assert math.isclose(mae1, mae2, abs_tol=1e-12)
        assert rows1 == rows2

    def test_grid_optimum_improves_or_matches_current(self) -> None:
        """If the current weights happen to be on-grid optimum, grid MAE
        equals current MAE. Otherwise grid should strictly improve."""
        best_weights, best_mae, grid_card = grid_search_weights(COARSE_GRID)
        current_mae, _ = mae_against_corpus(CURRENT_WEIGHTS)
        assert best_mae <= current_mae + 1e-9
        assert grid_card > 0

    def test_grid_search_deterministic(self) -> None:
        best1, mae1, size1 = grid_search_weights(COARSE_GRID)
        best2, mae2, size2 = grid_search_weights(COARSE_GRID)
        assert best1 == best2
        assert mae1 == mae2
        assert size1 == size2

    def test_grid_cardinality_matches_product_of_axes(self) -> None:
        _, _, size = grid_search_weights(COARSE_GRID)
        expected = 1
        for axis in COARSE_GRID.values():
            expected *= len(axis)
        assert size == expected


# ---------------------------------------------------------------------------
# WeightSet shape invariants
# ---------------------------------------------------------------------------


class TestWeightSetInvariants:
    def test_current_weights_have_negative_signs_on_negative_dims(self) -> None:
        assert CURRENT_WEIGHTS.w_rupture < 0
        assert CURRENT_WEIGHTS.w_sycophancy < 0
        assert CURRENT_WEIGHTS.w_over_disclosure < 0

    def test_current_weights_have_positive_signs_on_positive_dims(self) -> None:
        assert CURRENT_WEIGHTS.w_mirror > 0
        assert CURRENT_WEIGHTS.w_repair > 0
        assert CURRENT_WEIGHTS.w_consistency > 0

    def test_as_dict_round_trip(self) -> None:
        d = CURRENT_WEIGHTS.as_dict()
        assert set(d.keys()) == {
            "W_MIRROR", "W_REPAIR", "W_CONSISTENCY",
            "W_RUPTURE", "W_SYCOPHANCY", "W_OVER_DISCLOSURE",
        }
        assert d["W_MIRROR"] == CURRENT_WEIGHTS.w_mirror
