"""
Unit tests for stage_attribution module (no torch dependency).
Tests validate pure computation functions that only require pandas/numpy.
"""
import pytest
import sys
import os
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from stage_attribution import (
    compute_stage_deltas,
    compute_attribution_scores,
    compute_dpo_comparison,
    validate_stages,
    EXPECTED_STAGES,
    METRICS,
    STAGE_TRANSITIONS,
)


def _make_4stage_df():
    """Create a 4-stage audit DataFrame for testing."""
    return pd.DataFrame({
        "Stage": [
            "Stage0_Base",
            "Stage1_SFT",
            "Stage2a_DPO_NoCanary",
            "Stage2b_DPO_WithCanary",
        ],
        "MIA_Gap": [-3.79, -3.85, -3.88, -3.92],
        "Avg_LogProb": [-6.55, -6.33, -6.20, -6.05],
        "Avg_Rank": [3244.4, 1621.1, 1400.0, 1100.0],
        "Canary_PPL": [857.9, 690.1, 600.0, 450.0],
        "PPL_Ratio": [51.6, 55.4, 58.0, 62.0],
    })


def _make_3stage_df():
    """Create a legacy 3-stage DataFrame (missing Stage2b)."""
    return pd.DataFrame({
        "Stage": ["Stage0_Base", "Stage1_SFT", "Stage2a_DPO_NoCanary"],
        "MIA_Gap": [-3.79, -3.85, -3.88],
        "Avg_LogProb": [-6.55, -6.33, -6.20],
        "Avg_Rank": [3244.4, 1621.1, 1400.0],
        "Canary_PPL": [857.9, 690.1, 600.0],
        "PPL_Ratio": [51.6, 55.4, 58.0],
    })


class TestValidateStages:
    """Tests for validate_stages strict/tolerant behavior."""

    def test_strict_mode_all_present(self):
        df = _make_4stage_df()
        present = validate_stages(df, tolerant=False)
        assert present == EXPECTED_STAGES

    def test_strict_mode_missing_stage_exits(self):
        df = _make_3stage_df()
        with pytest.raises(SystemExit):
            validate_stages(df, tolerant=False)

    def test_tolerant_mode_missing_stage_warns(self):
        df = _make_3stage_df()
        with pytest.warns(UserWarning, match="Missing stage"):
            present = validate_stages(df, tolerant=True)
        assert "Stage2b_DPO_WithCanary" not in present
        assert "Stage0_Base" in present


class TestComputeStageDeltas:
    """Tests for compute_stage_deltas (4-stage)."""

    def test_all_transitions_present(self):
        df = _make_4stage_df()
        deltas = compute_stage_deltas(df)
        for key in STAGE_TRANSITIONS:
            assert key in deltas, f"Missing transition: {key}"

    def test_delta_arithmetic_correctness(self):
        df = _make_4stage_df()
        deltas = compute_stage_deltas(df)
        # Base_to_SFT MIA_Gap: -3.85 - (-3.79) = -0.06
        assert abs(deltas["Base_to_SFT"]["MIA_Gap"] - (-0.06)) < 1e-9
        # SFT_to_DPO_NoCanary Avg_Rank: 1400.0 - 1621.1 = -221.1
        assert abs(deltas["SFT_to_DPO_NoCanary"]["Avg_Rank"] - (-221.1)) < 1e-9
        # DPO comparison: 1100.0 - 1400.0 = -300.0
        assert abs(deltas["DPO_NoCanary_vs_WithCanary"]["Avg_Rank"] - (-300.0)) < 1e-9

    def test_tolerant_skips_missing_transitions(self):
        df = _make_3stage_df()
        deltas = compute_stage_deltas(df, tolerant=True)
        assert "Base_to_SFT" in deltas
        assert "SFT_to_DPO_NoCanary" in deltas
        # These should be skipped (Stage2b missing)
        assert "SFT_to_DPO_WithCanary" not in deltas
        assert "DPO_NoCanary_vs_WithCanary" not in deltas

    def test_percentage_changes_present(self):
        df = _make_4stage_df()
        deltas = compute_stage_deltas(df)
        assert "MIA_Gap_pct" in deltas["Base_to_SFT"]


class TestComputeDPOComparison:
    """Tests for compute_dpo_comparison."""

    def test_comparison_values(self):
        df = _make_4stage_df()
        cmp = compute_dpo_comparison(df)
        # Avg_Rank: 1100.0 - 1400.0 = -300.0
        assert abs(cmp["Avg_Rank"] - (-300.0)) < 1e-9
        # Canary_PPL: 450.0 - 600.0 = -150.0
        assert abs(cmp["Canary_PPL"] - (-150.0)) < 1e-9

    def test_comparison_has_all_metrics(self):
        df = _make_4stage_df()
        cmp = compute_dpo_comparison(df)
        for m in METRICS:
            assert m in cmp


class TestComputeAttributionScores:
    """Tests for compute_attribution_scores (4-stage)."""

    def test_attribution_keys_present(self):
        df = _make_4stage_df()
        deltas = compute_stage_deltas(df)
        attr = compute_attribution_scores(deltas)
        assert "SFT_contribution_nc" in attr
        assert "DPO_NoCanary_contribution" in attr
        assert "SFT_contribution_wc" in attr
        assert "DPO_WithCanary_contribution" in attr

    def test_no_canary_path_sums_to_100(self):
        df = _make_4stage_df()
        deltas = compute_stage_deltas(df)
        attr = compute_attribution_scores(deltas)
        for metric in METRICS:
            total = (
                attr["SFT_contribution_nc"][metric]
                + attr["DPO_NoCanary_contribution"][metric]
            )
            assert abs(total - 100.0) < 1e-6, (
                f"{metric}: SFT_nc+DPO_NC={total}, expected 100"
            )

    def test_with_canary_path_sums_to_100(self):
        df = _make_4stage_df()
        deltas = compute_stage_deltas(df)
        attr = compute_attribution_scores(deltas)
        for metric in METRICS:
            total = (
                attr["SFT_contribution_wc"][metric]
                + attr["DPO_WithCanary_contribution"][metric]
            )
            assert abs(total - 100.0) < 1e-6, (
                f"{metric}: SFT_wc+DPO_WC={total}, expected 100"
            )


    def test_strict_mode_deltas_exits_on_missing(self):
        """compute_stage_deltas strict mode should exit when stages are missing."""
        df = _make_3stage_df()
        with pytest.raises(SystemExit):
            compute_stage_deltas(df, tolerant=False)
