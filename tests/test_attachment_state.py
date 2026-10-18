"""Tests for parameterized AttachmentState — behavior-driven longitudinal bond.

Each test pins a specific invariant of the bond composition or state carry-
over so the lexical scorer cannot drift silently. The dialogues are hand-
authored to isolate one dimension at a time (pure mirror, pure rupture,
mixed-with-repair, etc.); real-session mixtures are covered by the
cross-session trajectory tests.
"""

from __future__ import annotations

import math

import pytest

from psychology_base.applications.emotional.attachment_state import (
    AttachmentState,
    LIKERT_MAX,
    LIKERT_MID,
    LIKERT_MIN,
    SessionAttachmentReport,
    TrajectoryConvergence,
    compute_trajectory_convergence,
    score_session_attachment,
)


# ---------------------------------------------------------------------------
# AttachmentState dataclass invariants
# ---------------------------------------------------------------------------


class TestAttachmentStateConstruction:
    def test_default_starts_at_likert_midpoint(self) -> None:
        state = AttachmentState(user_id="alice")
        assert state.trust_ema == LIKERT_MID
        assert state.sessions_completed == 0
        assert state.bond_history == ()
        assert state.cumulative_companion_turns == 0

    def test_requires_non_empty_user_id(self) -> None:
        with pytest.raises(ValueError, match="user_id"):
            AttachmentState(user_id="")

    def test_rejects_out_of_range_trust_ema(self) -> None:
        with pytest.raises(ValueError, match="trust_ema"):
            AttachmentState(user_id="alice", trust_ema=0.5)
        with pytest.raises(ValueError, match="trust_ema"):
            AttachmentState(user_id="alice", trust_ema=7.5)

    def test_four_week_gate_requires_16_sessions_and_bond_above_4(self) -> None:
        under_sessions = AttachmentState(
            user_id="alice", sessions_completed=15, trust_ema=5.0
        )
        assert not under_sessions.meets_four_week_gate

        under_bond = AttachmentState(
            user_id="alice", sessions_completed=16, trust_ema=3.9
        )
        assert not under_bond.meets_four_week_gate

        passes = AttachmentState(
            user_id="alice", sessions_completed=16, trust_ema=4.0
        )
        assert passes.meets_four_week_gate


# ---------------------------------------------------------------------------
# Per-component scoring (one dimension at a time)
# ---------------------------------------------------------------------------


class TestMirrorDimension:
    def test_mirror_cues_raise_bond_above_midpoint(self) -> None:
        companion_turns = [
            "That sounds really hard. I hear you.",
            "I feel for what you're going through — you're not alone in this.",
            "It makes sense that you'd feel overwhelmed. I'm here.",
        ]
        prior = AttachmentState(user_id="alice")
        report = score_session_attachment(
            session_id="s1",
            seeker_turns=["I'm struggling.", "Everything feels heavy.", "I just can't."],
            companion_turns=companion_turns,
            prior_state=prior,
        )
        assert report.mirror_hits == 3
        assert report.session_bond_estimate > LIKERT_MID
        assert report.new_state.trust_ema > LIKERT_MID

    def test_bare_acknowledgement_no_empathy_counts_zero(self) -> None:
        companion_turns = ["Noted.", "OK.", "I see."]
        prior = AttachmentState(user_id="alice")
        report = score_session_attachment(
            session_id="s1",
            seeker_turns=["x", "y", "z"],
            companion_turns=companion_turns,
            prior_state=prior,
        )
        assert report.mirror_hits == 0


class TestRuptureDimension:
    def test_rupture_cues_lower_bond_below_midpoint(self) -> None:
        companion_turns = [
            "Actually, no — you're wrong about that.",
            "You should just move on from this and stop talking about it.",
            "You're being dramatic. It's not a big deal.",
        ]
        prior = AttachmentState(user_id="alice")
        report = score_session_attachment(
            session_id="s1",
            seeker_turns=["I'm hurting.", "I don't know.", "This matters to me."],
            companion_turns=companion_turns,
            prior_state=prior,
        )
        assert report.rupture_hits == 3
        assert report.session_bond_estimate < LIKERT_MID

    def test_rupture_pure_dialogue_drives_trust_down(self) -> None:
        companion_turns = [
            "That's not right.",
            "You're overthinking this.",
            "Get over it.",
        ]
        prior = AttachmentState(user_id="alice", trust_ema=5.0)
        report = score_session_attachment(
            session_id="s1",
            seeker_turns=["a", "b", "c"],
            companion_turns=companion_turns,
            prior_state=prior,
        )
        assert report.new_state.trust_ema < 5.0


class TestRepairDimension:
    def test_rupture_then_repair_recovers_bond_vs_rupture_alone(self) -> None:
        rupture_only = [
            "You're wrong.",
            "Let's change the topic.",
            "Enough about that.",
        ]
        rupture_then_repair = [
            "You're wrong.",
            "I'm sorry — I misunderstood. Let me try again.",
            "Thank you for correcting me. I hear what you're saying now.",
        ]
        prior = AttachmentState(user_id="alice")
        rupt_report = score_session_attachment(
            session_id="s1",
            seeker_turns=["a", "b", "c"],
            companion_turns=rupture_only,
            prior_state=prior,
        )
        repair_report = score_session_attachment(
            session_id="s1",
            seeker_turns=["a", "b", "c"],
            companion_turns=rupture_then_repair,
            prior_state=prior,
        )
        assert repair_report.repair_hits >= 2
        assert repair_report.session_bond_estimate > rupt_report.session_bond_estimate


class TestSycophancyDimension:
    def test_sycophancy_cues_lower_bond_despite_positive_tone(self) -> None:
        companion_turns = [
            "You're absolutely right! That's such a great point.",
            "Amazing insight — you're so smart!",
            "Wow, you nailed it. 100% correct.",
        ]
        prior = AttachmentState(user_id="alice", trust_ema=5.0)
        report = score_session_attachment(
            session_id="s1",
            seeker_turns=["x", "y", "z"],
            companion_turns=companion_turns,
            prior_state=prior,
        )
        assert report.sycophancy_hits == 3
        assert report.session_bond_estimate < LIKERT_MID
        assert report.new_state.trust_ema < 5.0


class TestOverDisclosureDimension:
    def test_distant_memory_invocation_lowers_bond(self) -> None:
        companion_turns = [
            "Three sessions ago you mentioned feeling anxious.",
            "A while back you said you had trouble sleeping.",
            "Going back to our first conversation, you told me about your mother.",
        ]
        prior = AttachmentState(user_id="alice")
        report = score_session_attachment(
            session_id="s1",
            seeker_turns=["hi", "how are you", "ok"],
            companion_turns=companion_turns,
            prior_state=prior,
        )
        assert report.over_disclosure_hits == 3
        assert report.session_bond_estimate < LIKERT_MID


class TestConsistencyDimension:
    def test_generic_prior_reference_raises_bond(self) -> None:
        companion_turns = [
            "Last time you mentioned something similar.",
            "You said earlier that you've been working on this.",
            "We talked about this previously.",
        ]
        prior = AttachmentState(user_id="alice")
        report = score_session_attachment(
            session_id="s1",
            seeker_turns=["x", "y", "z"],
            companion_turns=companion_turns,
            prior_state=prior,
        )
        assert report.consistency_hits == 3
        assert report.session_bond_estimate > LIKERT_MID


# ---------------------------------------------------------------------------
# Cross-session state carry-over
# ---------------------------------------------------------------------------


class TestSessionCarryOver:
    def test_empty_companion_turns_leaves_state_untouched(self) -> None:
        prior = AttachmentState(user_id="alice", sessions_completed=5, trust_ema=5.0)
        report = score_session_attachment(
            session_id="s6",
            seeker_turns=["hi"],
            companion_turns=[],
            prior_state=prior,
        )
        assert report.new_state == prior
        assert report.companion_turns_scored == 0
        assert report.session_bond_estimate == LIKERT_MID

    def test_sessions_completed_increments_on_non_empty_session(self) -> None:
        prior = AttachmentState(user_id="alice", sessions_completed=5)
        report = score_session_attachment(
            session_id="s6",
            seeker_turns=["x"],
            companion_turns=["I hear you."],
            prior_state=prior,
        )
        assert report.new_state.sessions_completed == 6
        assert report.session_index == 6

    def test_cumulative_counts_accumulate_across_sessions(self) -> None:
        state = AttachmentState(user_id="alice")
        turns_s1 = ["I hear you. That's hard."]
        turns_s2 = ["You're wrong.", "I'm sorry, let me try again."]
        r1 = score_session_attachment(
            session_id="s1",
            seeker_turns=["a"],
            companion_turns=turns_s1,
            prior_state=state,
        )
        r2 = score_session_attachment(
            session_id="s2",
            seeker_turns=["a", "b"],
            companion_turns=turns_s2,
            prior_state=r1.new_state,
        )
        assert r2.new_state.cumulative_mirror_turns == 1
        assert r2.new_state.cumulative_rupture_turns == 1
        assert r2.new_state.cumulative_repair_turns == 1
        assert r2.new_state.cumulative_companion_turns == 3
        assert len(r2.new_state.bond_history) == 2

    def test_trust_ema_mixes_prior_and_current(self) -> None:
        # EMA alpha = 0.6 → new_trust = 0.6 * session_bond + 0.4 * prior
        # Prior at midpoint 4.0, pure-mirror session (session_bond ≈ 5.5)
        # → new ≈ 0.6*5.5 + 0.4*4.0 = 3.3 + 1.6 = 4.9
        prior = AttachmentState(user_id="alice")
        companion_turns = ["I hear you.", "That makes sense.", "I'm here with you."]
        report = score_session_attachment(
            session_id="s1",
            seeker_turns=["a", "b", "c"],
            companion_turns=companion_turns,
            prior_state=prior,
        )
        # session_bond = 4.0 + 1.5 * 1.0 (mirror saturated) = 5.5
        assert math.isclose(report.session_bond_estimate, 5.5, abs_tol=0.01)
        # new trust = 0.6*5.5 + 0.4*4.0 = 4.9
        assert math.isclose(report.new_state.trust_ema, 4.9, abs_tol=0.01)


# ---------------------------------------------------------------------------
# Monotonicity under single-dimension pressure (longitudinal trajectory)
# ---------------------------------------------------------------------------


class TestLongitudinalMonotonicity:
    def test_pure_mirror_across_sessions_trust_rises_monotonically(self) -> None:
        state = AttachmentState(user_id="alice")
        bond_series = []
        for i in range(5):
            r = score_session_attachment(
                session_id=f"s{i+1}",
                seeker_turns=["a", "b", "c"],
                companion_turns=[
                    "I hear you. That makes sense.",
                    "I'm here with you.",
                    "That sounds really hard — you're not alone.",
                ],
                prior_state=state,
            )
            bond_series.append(r.new_state.trust_ema)
            state = r.new_state
        # Monotonic non-decreasing (EMA converges to session_bond)
        for i in range(1, len(bond_series)):
            assert bond_series[i] >= bond_series[i - 1] - 1e-9
        # After 5 sessions of pure-mirror, trust should be well above midpoint
        assert state.trust_ema > 5.0

    def test_pure_sycophancy_across_sessions_trust_falls_monotonically(self) -> None:
        state = AttachmentState(user_id="alice", trust_ema=5.0)
        bond_series = []
        for i in range(5):
            r = score_session_attachment(
                session_id=f"s{i+1}",
                seeker_turns=["a", "b", "c"],
                companion_turns=[
                    "You're absolutely right! Amazing insight.",
                    "You're so smart! That's a great point.",
                    "Wow, you nailed it — 100% correct.",
                ],
                prior_state=state,
            )
            bond_series.append(r.new_state.trust_ema)
            state = r.new_state
        for i in range(1, len(bond_series)):
            assert bond_series[i] <= bond_series[i - 1] + 1e-9
        # Pure sycophancy drives trust toward midpoint and below (started at 5.0)
        assert state.trust_ema < 5.0

    def test_sixteen_mirror_sessions_cross_four_week_gate(self) -> None:
        """SOTA goal row sanity: 16 sessions of empathic Companion → gate PASS."""
        state = AttachmentState(user_id="alice")
        for i in range(16):
            r = score_session_attachment(
                session_id=f"s{i+1}",
                seeker_turns=["a"],
                companion_turns=["I hear you. That sounds really hard. I'm here."],
                prior_state=state,
            )
            state = r.new_state
        assert state.sessions_completed == 16
        assert state.meets_four_week_gate


# ---------------------------------------------------------------------------
# TrajectoryConvergence — behavior-trace vs LLM-trace alignment
# ---------------------------------------------------------------------------


class TestTrajectoryConvergence:
    def test_identical_series_perfect_correlation_zero_divergence(self) -> None:
        series = (4.0, 4.5, 5.0, 5.5, 6.0)
        conv = compute_trajectory_convergence(series, series)
        assert math.isclose(conv.pearson_r, 1.0, abs_tol=1e-9)
        assert conv.mean_abs_divergence == 0.0
        assert conv.max_divergence == 0.0
        assert conv.n_sessions == 5

    def test_parallel_offset_perfect_correlation_constant_divergence(self) -> None:
        behavior = (4.0, 4.5, 5.0, 5.5, 6.0)
        llm = (4.5, 5.0, 5.5, 6.0, 6.5)
        conv = compute_trajectory_convergence(behavior, llm)
        assert math.isclose(conv.pearson_r, 1.0, abs_tol=1e-9)
        assert math.isclose(conv.mean_abs_divergence, 0.5, abs_tol=1e-9)

    def test_anti_correlated_series_negative_pearson(self) -> None:
        behavior = (4.0, 5.0, 6.0, 7.0)
        llm = (7.0, 6.0, 5.0, 4.0)
        conv = compute_trajectory_convergence(behavior, llm)
        assert conv.pearson_r < 0.0

    def test_max_divergence_session_index_points_at_largest_gap(self) -> None:
        behavior = (4.0, 4.0, 4.0, 4.0)
        llm = (4.1, 4.2, 6.0, 4.1)  # session index 2 is the outlier (0-indexed)
        conv = compute_trajectory_convergence(behavior, llm)
        assert conv.max_divergence_session_index == 2
        assert math.isclose(conv.max_divergence, 2.0, abs_tol=1e-9)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            compute_trajectory_convergence((1.0, 2.0), (1.0, 2.0, 3.0))

    def test_empty_series_returns_zero_values(self) -> None:
        conv = compute_trajectory_convergence((), ())
        assert conv.n_sessions == 0
        assert conv.mean_abs_divergence == 0.0
        assert conv.pearson_r == 0.0

    def test_constant_series_pearson_returns_zero_not_nan(self) -> None:
        # Pearson is undefined on a constant series; function returns 0.0
        # as the honest numeric value rather than NaN.
        conv = compute_trajectory_convergence((5.0, 5.0, 5.0), (4.0, 5.0, 6.0))
        assert conv.pearson_r == 0.0

    def test_single_point_series_pearson_zero(self) -> None:
        conv = compute_trajectory_convergence((4.5,), (5.0,))
        assert conv.n_sessions == 1
        assert conv.pearson_r == 0.0
        assert math.isclose(conv.mean_abs_divergence, 0.5, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Clamping + edge cases on bond composition
# ---------------------------------------------------------------------------


class TestBondCompositionClamp:
    def test_pathological_mix_still_clamped_to_likert_range(self) -> None:
        # Impossible-but-defined: every component maxed on both sides.
        # Compose should still return a value in [1.0, 7.0].
        companion_turns = [
            "I hear you. You're absolutely right! That's not right. "
            "I'm sorry. Three sessions ago. You mentioned."
        ]
        prior = AttachmentState(user_id="alice")
        report = score_session_attachment(
            session_id="s1",
            seeker_turns=["x"],
            companion_turns=companion_turns,
            prior_state=prior,
        )
        assert LIKERT_MIN <= report.session_bond_estimate <= LIKERT_MAX
        assert LIKERT_MIN <= report.new_state.trust_ema <= LIKERT_MAX


# ---------------------------------------------------------------------------
# Phase 9+ i18n migration (2026-10-11) — Chinese cue banks dispatch via
# ``detect_lang`` while English continues through the legacy banks
# byte-identical. The EN-input tests above are the regression lock on
# that byte-identity; the tests below cover the ZH dispatch and the
# non-regression of mixed/non-zh text.
# ---------------------------------------------------------------------------


class TestChineseCueDispatch:
    def test_chinese_mirror_cues_fire(self) -> None:
        companion_turns = [
            "我理解你的感受,这确实很难。",
            "我在这里,慢慢来,你不是一个人。",
            "听起来你压得人喘不过气,这说得通。",
        ]
        prior = AttachmentState(user_id="user_zh")
        report = score_session_attachment(
            session_id="s1",
            seeker_turns=["我最近很累。", "每天都很沉重。", "我撑不住了。"],
            companion_turns=companion_turns,
            prior_state=prior,
        )
        assert report.mirror_hits == 3
        assert report.session_bond_estimate > LIKERT_MID

    def test_chinese_rupture_cues_fire(self) -> None:
        companion_turns = [
            "你搞错了,其实不是这样。",
            "你应该换个话题,别再说了。",
            "你想太多了,没什么大不了。",
        ]
        prior = AttachmentState(user_id="user_zh")
        report = score_session_attachment(
            session_id="s1",
            seeker_turns=["我很伤心。", "我不知道。", "这对我很重要。"],
            companion_turns=companion_turns,
            prior_state=prior,
        )
        assert report.rupture_hits == 3
        assert report.session_bond_estimate < LIKERT_MID

    def test_chinese_repair_cues_fire(self) -> None:
        companion_turns = [
            "对不起,我误会了,让我再试一次。",
            "你是对的,我应该先听你说。",
            "谢谢你指出,我没抓住重点。",
        ]
        prior = AttachmentState(user_id="user_zh")
        report = score_session_attachment(
            session_id="s1",
            seeker_turns=["a", "b", "c"],
            companion_turns=companion_turns,
            prior_state=prior,
        )
        assert report.repair_hits == 3

    def test_chinese_sycophancy_cues_fire(self) -> None:
        companion_turns = [
            "你完全正确,说得太好了!",
            "太棒了,你真聪明,好有见解。",
            "你抓住了重点,100%正确。",
        ]
        prior = AttachmentState(user_id="user_zh", trust_ema=5.0)
        report = score_session_attachment(
            session_id="s1",
            seeker_turns=["x", "y", "z"],
            companion_turns=companion_turns,
            prior_state=prior,
        )
        assert report.sycophancy_hits == 3
        assert report.session_bond_estimate < LIKERT_MID
        assert report.new_state.trust_ema < 5.0

    def test_chinese_over_disclosure_cues_fire(self) -> None:
        companion_turns = [
            "三次会话前你提到过失眠。",
            "回到我们第一次聊天,你说过妈妈的事。",
            "我一直在想你说的话,从我们早期的对话开始。",
        ]
        prior = AttachmentState(user_id="user_zh")
        report = score_session_attachment(
            session_id="s1",
            seeker_turns=["你好", "最近怎么样", "嗯"],
            companion_turns=companion_turns,
            prior_state=prior,
        )
        assert report.over_disclosure_hits == 3

    def test_chinese_consistency_cues_fire(self) -> None:
        companion_turns = [
            "上次你提到过类似的事。",
            "你之前说你一直在处理这件事。",
            "我们讨论过,像你说的那样。",
        ]
        prior = AttachmentState(user_id="user_zh")
        report = score_session_attachment(
            session_id="s1",
            seeker_turns=["x", "y", "z"],
            companion_turns=companion_turns,
            prior_state=prior,
        )
        assert report.consistency_hits == 3
        assert report.session_bond_estimate > LIKERT_MID

    def test_chinese_neutral_dialogue_scores_zero_on_all_dims(self) -> None:
        """Negative control: Chinese text without any CN cue should hit 0
        on every behavior dimension — not accidentally match a CN entry
        via substring, and not spuriously fire EN cues via lowercase."""
        companion_turns = [
            "好的,我在听。",
            "继续吧。",
            "嗯,然后呢?",
        ]
        prior = AttachmentState(user_id="user_zh")
        report = score_session_attachment(
            session_id="s1",
            seeker_turns=["x", "y", "z"],
            companion_turns=companion_turns,
            prior_state=prior,
        )
        assert report.mirror_hits == 0
        assert report.rupture_hits == 0
        assert report.repair_hits == 0
        assert report.sycophancy_hits == 0
        assert report.over_disclosure_hits == 0
        assert report.consistency_hits == 0

    def test_longitudinal_chinese_pure_mirror_crosses_four_week_gate(self) -> None:
        """SOTA goal row — the 4-week bond gate is reachable on an
        all-Chinese transcript now that CN cue banks route."""
        state = AttachmentState(user_id="user_zh")
        for i in range(16):
            r = score_session_attachment(
                session_id=f"s{i+1}",
                seeker_turns=["我很难过。"],
                companion_turns=[
                    "我理解你,这确实很难。我在这里陪着你,慢慢来。"
                ],
                prior_state=state,
            )
            state = r.new_state
        assert state.sessions_completed == 16
        assert state.meets_four_week_gate

    def test_english_byte_identical_hit_counts_under_i18n_dispatch(self) -> None:
        """Regression lock: an EN-only transcript must produce identical
        hit counts after the i18n dispatch — ``detect_lang`` on English
        returns ``"en"``, which routes to the EN banks, which are the
        pre-migration banks unchanged."""
        companion_turns = [
            "I hear you. That sounds really hard.",
            "You're absolutely right — that's a great point.",
            "Last time you mentioned something similar.",
            "I'm sorry, I misunderstood.",
            "Actually, no — you're wrong about that.",
            "Three sessions ago you said you were anxious.",
        ]
        prior = AttachmentState(user_id="alice")
        report = score_session_attachment(
            session_id="s1",
            seeker_turns=["a"] * 6,
            companion_turns=companion_turns,
            prior_state=prior,
        )
        # Each cue family is engineered to fire exactly once in this fixture.
        assert report.mirror_hits == 1
        assert report.sycophancy_hits == 1
        assert report.consistency_hits == 1
        assert report.repair_hits == 1
        assert report.rupture_hits == 1
        assert report.over_disclosure_hits == 1
