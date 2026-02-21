"""Unit tests for dp_sft_analysis module.

Tests cover: parse_args, load_audit_results, METRIC_DIRECTION_MAP,
normalize_effect_direction, compute_epsilon_summary,
classify_effect_size, compute_effect_sizes, compute_canary_level_bootstrap.
"""
import math
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dp_sft_analysis import (
    ALL_METRICS,
    METRIC_DIRECTION_MAP,
    PRIMARY_METRICS,
    SECONDARY_METRICS,
    annotate_negative_result,
    annotate_pilot_mode,
    classify_effect_size,
    compute_canary_level_bootstrap,
    compute_effect_sizes,
    compute_epsilon_summary,
    evaluate_detectability,
    evaluate_suppression,
    generate_report,
    load_audit_results,
    normalize_effect_direction,
    parse_args,
    plot_epsilon_trend,
)


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------
class TestParseArgs:
    def test_defaults(self):
        args = parse_args([])
        assert args.input_csv == "reports/dp_sft_audit_results.csv"
        assert args.output_dir == "reports"
        assert args.pilot is False

    def test_custom_values(self):
        args = parse_args([
            "--input-csv", "custom.csv",
            "--output-dir", "/tmp/out",
            "--pilot",
        ])
        assert args.input_csv == "custom.csv"
        assert args.output_dir == "/tmp/out"
        assert args.pilot is True

    def test_pilot_flag_only(self):
        args = parse_args(["--pilot"])
        assert args.pilot is True
        # other defaults unchanged
        assert args.input_csv == "reports/dp_sft_audit_results.csv"


# ---------------------------------------------------------------------------
# load_audit_results
# ---------------------------------------------------------------------------
class TestLoadAuditResults:
    def _write_csv(self, path, content):
        with open(path, "w") as f:
            f.write(content)

    def test_basic_load(self, tmp_path):
        csv_path = str(tmp_path / "results.csv")
        self._write_csv(csv_path, (
            "epsilon,seed,Avg_LogProb,Avg_Rank,Canary_PPL,"
            "Extraction_Rate,ROC_AUC,PR_AUC\n"
            "inf,42,-6.41,50.5,794.1,0.0,0.5,0.31\n"
            "8,42,-6.45,52.0,810.0,0.02,0.52,0.33\n"
        ))
        df = load_audit_results(csv_path)
        assert len(df) == 2
        assert df["epsilon"].dtype == object  # string type
        assert df.loc[0, "epsilon"] == "inf"
        assert df.loc[1, "epsilon"] == "8"

    def test_na_handling_pilot(self, tmp_path):
        csv_path = str(tmp_path / "pilot.csv")
        self._write_csv(csv_path, (
            "epsilon,seed,Avg_LogProb,Avg_Rank,Canary_PPL,"
            "Extraction_Rate,ROC_AUC,PR_AUC\n"
            "inf,42,-6.41,50.5,794.1,NA,NA,NA\n"
        ))
        df = load_audit_results(csv_path)
        assert pd.isna(df.loc[0, "Extraction_Rate"])
        assert pd.isna(df.loc[0, "ROC_AUC"])
        assert pd.isna(df.loc[0, "PR_AUC"])
        # Primary metrics should be valid
        assert not pd.isna(df.loc[0, "Avg_LogProb"])

    def test_epsilon_stays_string(self, tmp_path):
        csv_path = str(tmp_path / "eps.csv")
        self._write_csv(csv_path, (
            "epsilon,seed,Avg_LogProb,Avg_Rank,Canary_PPL,"
            "Extraction_Rate,ROC_AUC,PR_AUC\n"
            "inf,42,-6.41,50.5,794.1,0.0,0.5,0.31\n"
            "1,42,-6.55,55.0,850.0,0.01,0.48,0.29\n"
        ))
        df = load_audit_results(csv_path)
        # Both "inf" and "1" should be strings
        assert all(isinstance(v, str) for v in df["epsilon"])

    def test_file_not_found(self):
        with pytest.raises(Exception):
            load_audit_results("/nonexistent/path.csv")


# ---------------------------------------------------------------------------
# METRIC_DIRECTION_MAP
# ---------------------------------------------------------------------------
class TestMetricDirectionMap:
    def test_all_metrics_covered(self):
        for m in ALL_METRICS:
            assert m in METRIC_DIRECTION_MAP, f"{m} missing from METRIC_DIRECTION_MAP"

    def test_positive_metrics(self):
        assert METRIC_DIRECTION_MAP["Avg_LogProb"] == "positive"
        assert METRIC_DIRECTION_MAP["Extraction_Rate"] == "positive"
        assert METRIC_DIRECTION_MAP["ROC_AUC"] == "positive"
        assert METRIC_DIRECTION_MAP["PR_AUC"] == "positive"

    def test_negative_metrics(self):
        assert METRIC_DIRECTION_MAP["Avg_Rank"] == "negative"
        assert METRIC_DIRECTION_MAP["Canary_PPL"] == "negative"


# ---------------------------------------------------------------------------
# normalize_effect_direction
# ---------------------------------------------------------------------------
class TestNormalizeEffectDirection:
    def test_positive_metric_unchanged(self):
        assert normalize_effect_direction(0.5, "Avg_LogProb") == 0.5
        assert normalize_effect_direction(-0.3, "Avg_LogProb") == -0.3

    def test_negative_metric_flipped(self):
        assert normalize_effect_direction(0.5, "Avg_Rank") == -0.5
        assert normalize_effect_direction(-0.8, "Canary_PPL") == 0.8

    def test_zero_unchanged(self):
        assert normalize_effect_direction(0.0, "Avg_LogProb") == 0.0
        assert normalize_effect_direction(0.0, "Avg_Rank") == 0.0

    def test_nan_passthrough(self):
        result = normalize_effect_direction(float("nan"), "Avg_LogProb")
        assert math.isnan(result)
        result2 = normalize_effect_direction(float("nan"), "Avg_Rank")
        assert math.isnan(result2)

    def test_unknown_metric_unchanged(self):
        # Unknown metric not in map → no flip
        assert normalize_effect_direction(1.0, "UnknownMetric") == 1.0


# ---------------------------------------------------------------------------
# compute_epsilon_summary
# ---------------------------------------------------------------------------
class TestComputeEpsilonSummary:
    def _make_df(self):
        """3 seeds for eps=inf and eps=4."""
        return pd.DataFrame({
            "epsilon": ["inf", "inf", "inf", "4", "4", "4"],
            "seed": [42, 123, 456, 42, 123, 456],
            "Avg_LogProb": [-6.41, -6.39, -6.43, -6.52, -6.50, -6.54],
            "Avg_Rank": [50.5, 51.2, 49.8, 60.0, 61.5, 58.5],
            "Canary_PPL": [794.1, 801.3, 788.7, 850.0, 860.0, 840.0],
            "Extraction_Rate": [0.0, 0.0, 0.0, 0.01, 0.02, 0.0],
            "ROC_AUC": [0.5, 0.52, 0.48, 0.45, 0.47, 0.43],
            "PR_AUC": [0.31, 0.32, 0.30, 0.28, 0.29, 0.27],
        })

    def test_basic_structure(self):
        df = self._make_df()
        summary = compute_epsilon_summary(df)
        assert "inf" in summary
        assert "4" in summary
        for metric in ALL_METRICS:
            assert metric in summary["inf"]
            assert "mean" in summary["inf"][metric]
            assert "std" in summary["inf"][metric]

    def test_mean_correctness(self):
        df = self._make_df()
        summary = compute_epsilon_summary(df)
        # inf Avg_LogProb: mean of [-6.41, -6.39, -6.43]
        expected_mean = np.mean([-6.41, -6.39, -6.43])
        assert abs(summary["inf"]["Avg_LogProb"]["mean"] - expected_mean) < 1e-10

    def test_std_correctness(self):
        df = self._make_df()
        summary = compute_epsilon_summary(df)
        # inf Avg_LogProb: sample std (ddof=1) of [-6.41, -6.39, -6.43]
        expected_std = float(np.std([-6.41, -6.39, -6.43], ddof=1))
        assert abs(summary["inf"]["Avg_LogProb"]["std"] - expected_std) < 1e-10

    def test_single_seed_std_is_nan(self):
        """Single seed group should have std=NaN, not 0."""
        df = pd.DataFrame({
            "epsilon": ["1"],
            "seed": [42],
            "Avg_LogProb": [-6.55],
            "Avg_Rank": [55.0],
            "Canary_PPL": [850.0],
        })
        summary = compute_epsilon_summary(df)
        assert math.isnan(summary["1"]["Avg_LogProb"]["std"])

    def test_na_values_excluded(self):
        """NA values in secondary metrics (pilot mode) should be excluded."""
        df = pd.DataFrame({
            "epsilon": ["inf", "inf"],
            "seed": [42, 123],
            "Avg_LogProb": [-6.41, -6.39],
            "Extraction_Rate": [float("nan"), float("nan")],
        })
        summary = compute_epsilon_summary(df)
        # Primary metric should have valid mean
        assert not math.isnan(summary["inf"]["Avg_LogProb"]["mean"])
        # Secondary metric with all NaN → mean and std are NaN
        assert math.isnan(summary["inf"]["Extraction_Rate"]["mean"])
        assert math.isnan(summary["inf"]["Extraction_Rate"]["std"])

    def test_mixed_na_values(self):
        """Some NA, some valid values in a group."""
        df = pd.DataFrame({
            "epsilon": ["inf", "inf", "inf"],
            "seed": [42, 123, 456],
            "Avg_LogProb": [-6.41, -6.39, -6.43],
            "Extraction_Rate": [0.0, float("nan"), 0.02],
        })
        summary = compute_epsilon_summary(df)
        # Only 2 valid values for Extraction_Rate
        expected_mean = np.mean([0.0, 0.02])
        assert abs(summary["inf"]["Extraction_Rate"]["mean"] - expected_mean) < 1e-10

    def test_multiple_epsilon_groups(self):
        df = self._make_df()
        summary = compute_epsilon_summary(df)
        # Each epsilon group should have independent stats
        assert summary["inf"]["Avg_LogProb"]["mean"] != summary["4"]["Avg_LogProb"]["mean"]

    def test_empty_metric_column_not_present(self):
        """Metrics not in DataFrame columns are skipped."""
        df = pd.DataFrame({
            "epsilon": ["inf"],
            "seed": [42],
            "Avg_LogProb": [-6.41],
        })
        summary = compute_epsilon_summary(df)
        assert "Avg_LogProb" in summary["inf"]
        # Metrics not in df.columns should not appear
        assert "ROC_AUC" not in summary["inf"]


# ---------------------------------------------------------------------------
# classify_effect_size
# ---------------------------------------------------------------------------
class TestClassifyEffectSize:
    def test_negligible(self):
        assert classify_effect_size(0.0) == "negligible"
        assert classify_effect_size(0.19) == "negligible"
        assert classify_effect_size(-0.19) == "negligible"

    def test_small(self):
        assert classify_effect_size(0.2) == "small"
        assert classify_effect_size(0.49) == "small"
        assert classify_effect_size(-0.2) == "small"
        assert classify_effect_size(-0.49) == "small"

    def test_medium(self):
        assert classify_effect_size(0.5) == "medium"
        assert classify_effect_size(0.79) == "medium"
        assert classify_effect_size(-0.5) == "medium"
        assert classify_effect_size(-0.79) == "medium"

    def test_large(self):
        assert classify_effect_size(0.8) == "large"
        assert classify_effect_size(1.5) == "large"
        assert classify_effect_size(-0.8) == "large"
        assert classify_effect_size(-2.0) == "large"

    def test_nan(self):
        assert classify_effect_size(float("nan")) == "not_estimable"

    def test_boundary_0_2(self):
        # Exactly 0.2 should be "small", not "negligible"
        assert classify_effect_size(0.2) == "small"
        assert classify_effect_size(-0.2) == "small"

    def test_boundary_0_5(self):
        assert classify_effect_size(0.5) == "medium"

    def test_boundary_0_8(self):
        assert classify_effect_size(0.8) == "large"


# ---------------------------------------------------------------------------
# compute_effect_sizes
# ---------------------------------------------------------------------------
class TestComputeEffectSizes:
    def _make_df(self):
        """3 seeds for eps=inf and eps=8."""
        return pd.DataFrame({
            "epsilon": ["inf", "inf", "inf", "8", "8", "8"],
            "seed": [42, 123, 456, 42, 123, 456],
            "Avg_LogProb": [-6.41, -6.39, -6.43, -6.51, -6.49, -6.53],
            "Avg_Rank": [50.5, 51.2, 49.8, 60.0, 61.5, 58.5],
        })

    def test_basic_structure(self):
        df = self._make_df()
        result = compute_effect_sizes(df)
        assert "8_vs_inf" in result
        assert "Avg_LogProb" in result["8_vs_inf"]
        assert "cohens_d" in result["8_vs_inf"]["Avg_LogProb"]
        assert "category" in result["8_vs_inf"]["Avg_LogProb"]

    def test_control_not_in_result(self):
        df = self._make_df()
        result = compute_effect_sizes(df)
        # Control group (inf) should not appear as a comparison key
        assert "inf_vs_inf" not in result

    def test_cohens_d_value(self):
        df = self._make_df()
        result = compute_effect_sizes(df)
        # Manual calculation for Avg_LogProb:
        ctrl = np.array([-6.41, -6.39, -6.43])
        treat = np.array([-6.51, -6.49, -6.53])
        expected_d = (np.mean(treat) - np.mean(ctrl)) / np.sqrt(
            (np.var(ctrl) + np.var(treat)) / 2
        )
        assert abs(result["8_vs_inf"]["Avg_LogProb"]["cohens_d"] - expected_d) < 1e-10

    def test_category_matches_d(self):
        df = self._make_df()
        result = compute_effect_sizes(df)
        for key in result:
            for metric in result[key]:
                d = result[key][metric]["cohens_d"]
                cat = result[key][metric]["category"]
                assert cat == classify_effect_size(d)

    def test_single_seed_returns_nan(self):
        """When either group has < 2 samples, Cohen's d should be NaN."""
        df = pd.DataFrame({
            "epsilon": ["inf", "8"],
            "seed": [42, 42],
            "Avg_LogProb": [-6.41, -6.51],
        })
        result = compute_effect_sizes(df)
        assert math.isnan(result["8_vs_inf"]["Avg_LogProb"]["cohens_d"])
        assert result["8_vs_inf"]["Avg_LogProb"]["category"] == "not_estimable"

    def test_identical_values_returns_zero(self):
        """When both groups have zero variance and same mean, d=0."""
        df = pd.DataFrame({
            "epsilon": ["inf", "inf", "8", "8"],
            "seed": [42, 123, 42, 123],
            "Avg_LogProb": [5.0, 5.0, 5.0, 5.0],
        })
        result = compute_effect_sizes(df)
        assert result["8_vs_inf"]["Avg_LogProb"]["cohens_d"] == 0.0

    def test_multiple_epsilons(self):
        df = pd.DataFrame({
            "epsilon": ["inf", "inf", "8", "8", "4", "4"],
            "seed": [42, 123, 42, 123, 42, 123],
            "Avg_LogProb": [-6.4, -6.4, -6.5, -6.5, -6.6, -6.6],
        })
        result = compute_effect_sizes(df)
        assert "8_vs_inf" in result
        assert "4_vs_inf" in result

    def test_na_values_excluded(self):
        """NaN metric values should be excluded from Cohen's d calculation."""
        df = pd.DataFrame({
            "epsilon": ["inf", "inf", "inf", "8", "8", "8"],
            "seed": [42, 123, 456, 42, 123, 456],
            "Avg_LogProb": [-6.41, -6.39, -6.43, -6.51, float("nan"), -6.53],
        })
        result = compute_effect_sizes(df)
        # Treatment group has 2 valid values, control has 3 → should compute
        assert not math.isnan(result["8_vs_inf"]["Avg_LogProb"]["cohens_d"])


# ---------------------------------------------------------------------------
# compute_canary_level_bootstrap
# ---------------------------------------------------------------------------
class TestComputeCanaryLevelBootstrap:
    def test_basic_structure(self):
        values = np.random.default_rng(0).normal(size=50)
        result = compute_canary_level_bootstrap(values)
        assert "mean" in result
        assert "ci_lower" in result
        assert "ci_upper" in result

    def test_ci_order(self):
        values = np.random.default_rng(0).normal(size=50)
        result = compute_canary_level_bootstrap(values)
        assert result["ci_lower"] <= result["ci_upper"]

    def test_mean_value(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = compute_canary_level_bootstrap(values)
        assert abs(result["mean"] - 3.0) < 1e-10

    def test_empty_array_returns_nan(self):
        result = compute_canary_level_bootstrap(np.array([]))
        assert math.isnan(result["mean"])
        assert math.isnan(result["ci_lower"])
        assert math.isnan(result["ci_upper"])

    def test_identical_values(self):
        """When all values are the same, CI should collapse to that value."""
        values = np.full(50, 7.0)
        result = compute_canary_level_bootstrap(values)
        assert result["mean"] == 7.0
        assert result["ci_lower"] == 7.0
        assert result["ci_upper"] == 7.0

    def test_reproducibility(self):
        """Same seed should produce same results."""
        values = np.random.default_rng(99).normal(size=50)
        r1 = compute_canary_level_bootstrap(values, seed=42)
        r2 = compute_canary_level_bootstrap(values, seed=42)
        assert r1 == r2

    def test_different_seeds_differ(self):
        """Different seeds should generally produce different CIs."""
        values = np.random.default_rng(99).normal(size=50)
        r1 = compute_canary_level_bootstrap(values, seed=42)
        r2 = compute_canary_level_bootstrap(values, seed=99)
        # Very unlikely to be exactly equal with different seeds
        assert r1["ci_lower"] != r2["ci_lower"] or r1["ci_upper"] != r2["ci_upper"]

    def test_single_element(self):
        """Single element array: all bootstrap means are that value."""
        result = compute_canary_level_bootstrap(np.array([3.14]))
        assert result["mean"] == 3.14
        assert result["ci_lower"] == 3.14
        assert result["ci_upper"] == 3.14

    def test_n_bootstrap_parameter(self):
        """Custom n_bootstrap should work."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = compute_canary_level_bootstrap(values, n_bootstrap=500)
        assert result["ci_lower"] <= result["ci_upper"]


# ---------------------------------------------------------------------------
# evaluate_detectability
# ---------------------------------------------------------------------------


class TestEvaluateDetectability:
    """Tests for evaluate_detectability (Requirement 9.1)."""

    def test_detectable_when_positive_metric_above_threshold(self):
        """Avg_LogProb (positive direction) with Cohen_d > 0.5 → detectable."""
        effect_sizes = {
            "inf": {
                "Avg_LogProb": {"cohens_d": 0.8, "category": "large"},
                "Avg_Rank": {"cohens_d": 0.1, "category": "negligible"},
                "Canary_PPL": {"cohens_d": 0.1, "category": "negligible"},
            }
        }
        result = evaluate_detectability(effect_sizes)
        assert result["inf"]["detectable"] is True
        assert "Avg_LogProb" in result["inf"]["reason"]

    def test_detectable_when_negative_metric_flipped_above_threshold(self):
        """Avg_Rank (negative direction): raw Cohen_d=-0.7 → normalized=0.7 > 0.5."""
        effect_sizes = {
            "8": {
                "Avg_LogProb": {"cohens_d": 0.1, "category": "negligible"},
                "Avg_Rank": {"cohens_d": -0.7, "category": "medium"},
                "Canary_PPL": {"cohens_d": 0.0, "category": "negligible"},
            }
        }
        result = evaluate_detectability(effect_sizes)
        assert result["8"]["detectable"] is True
        assert "Avg_Rank" in result["8"]["reason"]

    def test_not_detectable_when_all_below_threshold(self):
        """All normalized Cohen_d <= 0.5 → not detectable."""
        effect_sizes = {
            "4": {
                "Avg_LogProb": {"cohens_d": 0.3, "category": "small"},
                "Avg_Rank": {"cohens_d": -0.2, "category": "small"},
                "Canary_PPL": {"cohens_d": -0.1, "category": "negligible"},
            }
        }
        result = evaluate_detectability(effect_sizes)
        assert result["4"]["detectable"] is False

    def test_not_detectable_at_boundary_0_5(self):
        """Exactly 0.5 is NOT > 0.5, so not detectable."""
        effect_sizes = {
            "1": {
                "Avg_LogProb": {"cohens_d": 0.5, "category": "medium"},
                "Avg_Rank": {"cohens_d": 0.0, "category": "negligible"},
                "Canary_PPL": {"cohens_d": 0.0, "category": "negligible"},
            }
        }
        result = evaluate_detectability(effect_sizes)
        assert result["1"]["detectable"] is False

    def test_nan_cohens_d_ignored(self):
        """NaN Cohen's d should not trigger detectability."""
        effect_sizes = {
            "1": {
                "Avg_LogProb": {"cohens_d": float("nan"), "category": "not_estimable"},
                "Avg_Rank": {"cohens_d": float("nan"), "category": "not_estimable"},
                "Canary_PPL": {"cohens_d": float("nan"), "category": "not_estimable"},
            }
        }
        result = evaluate_detectability(effect_sizes)
        assert result["1"]["detectable"] is False

    def test_multiple_epsilons(self):
        """Multiple epsilon values processed independently."""
        effect_sizes = {
            "inf": {
                "Avg_LogProb": {"cohens_d": 1.5, "category": "large"},
                "Avg_Rank": {"cohens_d": 0.0, "category": "negligible"},
                "Canary_PPL": {"cohens_d": 0.0, "category": "negligible"},
            },
            "4": {
                "Avg_LogProb": {"cohens_d": 0.2, "category": "small"},
                "Avg_Rank": {"cohens_d": 0.0, "category": "negligible"},
                "Canary_PPL": {"cohens_d": 0.0, "category": "negligible"},
            },
        }
        result = evaluate_detectability(effect_sizes)
        assert result["inf"]["detectable"] is True
        assert result["4"]["detectable"] is False

    def test_only_primary_metrics_considered(self):
        """Secondary metrics should not affect detectability."""
        effect_sizes = {
            "8": {
                "Avg_LogProb": {"cohens_d": 0.1, "category": "negligible"},
                "Avg_Rank": {"cohens_d": 0.0, "category": "negligible"},
                "Canary_PPL": {"cohens_d": 0.0, "category": "negligible"},
                "Extraction_Rate": {"cohens_d": 2.0, "category": "large"},
            }
        }
        result = evaluate_detectability(effect_sizes)
        assert result["8"]["detectable"] is False

    def test_empty_input(self):
        """Empty dict returns empty dict."""
        assert evaluate_detectability({}) == {}

    def test_reports_best_metric(self):
        """When multiple metrics are detectable, reports the one with largest normalized d."""
        effect_sizes = {
            "inf": {
                "Avg_LogProb": {"cohens_d": 0.6, "category": "medium"},
                "Avg_Rank": {"cohens_d": -1.2, "category": "large"},
                "Canary_PPL": {"cohens_d": -0.8, "category": "large"},
            }
        }
        result = evaluate_detectability(effect_sizes)
        assert result["inf"]["detectable"] is True
        # Avg_Rank normalized = 1.2, Canary_PPL normalized = 0.8, Avg_LogProb = 0.6
        assert "Avg_Rank" in result["inf"]["reason"]


# ---------------------------------------------------------------------------
# evaluate_suppression
# ---------------------------------------------------------------------------


class TestEvaluateSuppression:
    """Tests for evaluate_suppression (Requirement 9.2)."""

    def test_suppressed_when_positive_metric_below_threshold(self):
        """Avg_LogProb (positive direction) with Cohen_d < -0.8 → suppressed."""
        effect_sizes = {
            "8_vs_inf": {
                "Avg_LogProb": {"cohens_d": -1.0, "category": "large"},
                "Avg_Rank": {"cohens_d": 0.0, "category": "negligible"},
                "Canary_PPL": {"cohens_d": 0.0, "category": "negligible"},
            }
        }
        result = evaluate_suppression(effect_sizes)
        assert result["8_vs_inf"]["suppressed"] is True
        assert "Avg_LogProb" in result["8_vs_inf"]["reason"]

    def test_suppressed_when_negative_metric_flipped_below_threshold(self):
        """Avg_Rank (negative direction): raw Cohen_d=1.0 → normalized=-1.0 < -0.8."""
        effect_sizes = {
            "4_vs_inf": {
                "Avg_LogProb": {"cohens_d": 0.0, "category": "negligible"},
                "Avg_Rank": {"cohens_d": 1.0, "category": "large"},
                "Canary_PPL": {"cohens_d": 0.0, "category": "negligible"},
            }
        }
        result = evaluate_suppression(effect_sizes)
        assert result["4_vs_inf"]["suppressed"] is True
        assert "Avg_Rank" in result["4_vs_inf"]["reason"]

    def test_not_suppressed_when_above_threshold(self):
        """All normalized Cohen_d >= -0.8 → not suppressed."""
        effect_sizes = {
            "8_vs_inf": {
                "Avg_LogProb": {"cohens_d": -0.3, "category": "small"},
                "Avg_Rank": {"cohens_d": 0.2, "category": "small"},
                "Canary_PPL": {"cohens_d": 0.1, "category": "negligible"},
            }
        }
        result = evaluate_suppression(effect_sizes)
        assert result["8_vs_inf"]["suppressed"] is False

    def test_not_suppressed_at_boundary_minus_0_8(self):
        """Exactly -0.8 is NOT < -0.8, so not suppressed."""
        effect_sizes = {
            "1_vs_inf": {
                "Avg_LogProb": {"cohens_d": -0.8, "category": "large"},
                "Avg_Rank": {"cohens_d": 0.0, "category": "negligible"},
                "Canary_PPL": {"cohens_d": 0.0, "category": "negligible"},
            }
        }
        result = evaluate_suppression(effect_sizes)
        assert result["1_vs_inf"]["suppressed"] is False

    def test_nan_cohens_d_ignored(self):
        """NaN Cohen's d should not trigger suppression."""
        effect_sizes = {
            "1_vs_inf": {
                "Avg_LogProb": {"cohens_d": float("nan"), "category": "not_estimable"},
                "Avg_Rank": {"cohens_d": float("nan"), "category": "not_estimable"},
                "Canary_PPL": {"cohens_d": float("nan"), "category": "not_estimable"},
            }
        }
        result = evaluate_suppression(effect_sizes)
        assert result["1_vs_inf"]["suppressed"] is False

    def test_multiple_comparisons(self):
        """Multiple comparison keys processed independently."""
        effect_sizes = {
            "8_vs_inf": {
                "Avg_LogProb": {"cohens_d": -0.3, "category": "small"},
                "Avg_Rank": {"cohens_d": 0.0, "category": "negligible"},
                "Canary_PPL": {"cohens_d": 0.0, "category": "negligible"},
            },
            "1_vs_inf": {
                "Avg_LogProb": {"cohens_d": -1.2, "category": "large"},
                "Avg_Rank": {"cohens_d": 0.0, "category": "negligible"},
                "Canary_PPL": {"cohens_d": 0.0, "category": "negligible"},
            },
        }
        result = evaluate_suppression(effect_sizes)
        assert result["8_vs_inf"]["suppressed"] is False
        assert result["1_vs_inf"]["suppressed"] is True

    def test_only_primary_metrics_considered(self):
        """Secondary metrics should not affect suppression."""
        effect_sizes = {
            "8_vs_inf": {
                "Avg_LogProb": {"cohens_d": 0.0, "category": "negligible"},
                "Avg_Rank": {"cohens_d": 0.0, "category": "negligible"},
                "Canary_PPL": {"cohens_d": 0.0, "category": "negligible"},
                "Extraction_Rate": {"cohens_d": -2.0, "category": "large"},
            }
        }
        result = evaluate_suppression(effect_sizes)
        assert result["8_vs_inf"]["suppressed"] is False

    def test_empty_input(self):
        """Empty dict returns empty dict."""
        assert evaluate_suppression({}) == {}


# ---------------------------------------------------------------------------
# annotate_negative_result
# ---------------------------------------------------------------------------


class TestAnnotateNegativeResult:
    """Tests for annotate_negative_result (Requirement 9.3)."""

    def test_all_undetectable_returns_annotation(self):
        """All epsilon values not detectable → negative result annotation."""
        detectability = {
            "inf": {"detectable": False, "reason": "..."},
            "8": {"detectable": False, "reason": "..."},
            "4": {"detectable": False, "reason": "..."},
            "1": {"detectable": False, "reason": "..."},
        }
        result = annotate_negative_result(detectability)
        assert "negative result" in result
        assert "DP noise" in result

    def test_some_detectable_returns_empty(self):
        """At least one detectable → no annotation."""
        detectability = {
            "inf": {"detectable": True, "reason": "..."},
            "8": {"detectable": False, "reason": "..."},
        }
        assert annotate_negative_result(detectability) == ""

    def test_all_detectable_returns_empty(self):
        """All detectable → no annotation."""
        detectability = {
            "inf": {"detectable": True, "reason": "..."},
            "8": {"detectable": True, "reason": "..."},
        }
        assert annotate_negative_result(detectability) == ""

    def test_empty_dict_returns_empty(self):
        """Empty detectability dict → empty string."""
        assert annotate_negative_result({}) == ""

    def test_single_epsilon_undetectable(self):
        """Single epsilon, not detectable → annotation."""
        detectability = {"1": {"detectable": False, "reason": "..."}}
        result = annotate_negative_result(detectability)
        assert "negative result" in result


# ---------------------------------------------------------------------------
# annotate_pilot_mode
# ---------------------------------------------------------------------------


class TestAnnotatePilotMode:
    """Tests for annotate_pilot_mode (Requirement 10.3)."""

    def test_pilot_true_returns_annotation(self):
        """pilot=True → pilot annotation string."""
        result = annotate_pilot_mode(True)
        assert "Pilot mode" in result
        assert "1 seed" in result
        assert "feasibility validation" in result

    def test_pilot_false_returns_empty(self):
        """pilot=False → empty string."""
        assert annotate_pilot_mode(False) == ""

# ---------------------------------------------------------------------------
# generate_report
# ---------------------------------------------------------------------------
class TestGenerateReport:

    def _make_inputs(self):
        """Create minimal valid inputs for generate_report."""
        summary = {
            "inf": {"Avg_LogProb": {"mean": -6.41, "std": 0.02}},
            "4": {"Avg_LogProb": {"mean": -6.52, "std": 0.04}},
        }
        effect_sizes = {
            "4_vs_inf": {
                "Avg_LogProb": {"cohens_d": -0.8, "category": "large"},
            },
        }
        bootstrap_cis = {
            "inf": {"Avg_LogProb": {"mean": -6.41, "ci_lower": -6.45, "ci_upper": -6.37}},
        }
        detectability = {
            "inf": {"detectable": True, "reason": "Avg_LogProb normalized Cohen_d=1.5 vs base"},
            "4": {"detectable": False, "reason": "No metric normalized Cohen_d > 0.5 vs base"},
        }
        suppression = {
            "4_vs_inf": {"suppressed": True, "reason": "Avg_LogProb normalized Cohen_d=-1.0, < -0.8"},
        }
        return summary, effect_sizes, bootstrap_cis, detectability, suppression

    def test_returns_dict_with_required_keys(self, tmp_path, monkeypatch):
        """Report dict contains all required top-level keys."""
        monkeypatch.chdir(tmp_path)
        s, e, b, d, sup = self._make_inputs()
        report = generate_report(s, e, b, d, sup, pilot=False)
        for key in ("pilot", "epsilon_summary", "effect_sizes",
                     "canary_bootstrap_ci", "detectability", "suppression"):
            assert key in report

    def test_pilot_flag_stored(self, tmp_path, monkeypatch):
        """pilot field reflects the input flag."""
        monkeypatch.chdir(tmp_path)
        s, e, b, d, sup = self._make_inputs()
        assert generate_report(s, e, b, d, sup, pilot=True)["pilot"] is True
        assert generate_report(s, e, b, d, sup, pilot=False)["pilot"] is False

    def test_pilot_annotation_added(self, tmp_path, monkeypatch):
        """pilot=True adds pilot_annotation key."""
        monkeypatch.chdir(tmp_path)
        s, e, b, d, sup = self._make_inputs()
        report = generate_report(s, e, b, d, sup, pilot=True)
        assert "pilot_annotation" in report
        assert "Pilot mode" in report["pilot_annotation"]

    def test_no_pilot_annotation_when_false(self, tmp_path, monkeypatch):
        """pilot=False does not add pilot_annotation key."""
        monkeypatch.chdir(tmp_path)
        s, e, b, d, sup = self._make_inputs()
        report = generate_report(s, e, b, d, sup, pilot=False)
        assert "pilot_annotation" not in report

    def test_negative_result_annotation(self, tmp_path, monkeypatch):
        """All undetectable → negative_result_annotation added."""
        monkeypatch.chdir(tmp_path)
        s, e, b, _, sup = self._make_inputs()
        detect_all_false = {
            "inf": {"detectable": False, "reason": "none"},
            "4": {"detectable": False, "reason": "none"},
        }
        report = generate_report(s, e, b, detect_all_false, sup, pilot=False)
        assert "negative_result_annotation" in report

    def test_json_file_created(self, tmp_path, monkeypatch):
        """JSON file is written to reports/dp_sft_analysis.json."""
        monkeypatch.chdir(tmp_path)
        s, e, b, d, sup = self._make_inputs()
        generate_report(s, e, b, d, sup)
        json_path = tmp_path / "reports" / "dp_sft_analysis.json"
        assert json_path.exists()

    def test_json_is_valid(self, tmp_path, monkeypatch):
        """Written JSON file is parseable."""
        import json
        monkeypatch.chdir(tmp_path)
        s, e, b, d, sup = self._make_inputs()
        generate_report(s, e, b, d, sup)
        json_path = tmp_path / "reports" / "dp_sft_analysis.json"
        with open(json_path) as f:
            data = json.load(f)
        assert data["pilot"] is False

    def test_nan_values_serialized(self, tmp_path, monkeypatch):
        """NaN values in summary are serialized as null in JSON."""
        import json
        monkeypatch.chdir(tmp_path)
        summary = {"inf": {"Avg_LogProb": {"mean": float("nan"), "std": float("nan")}}}
        e, b, d, sup = {}, {}, {}, {}
        generate_report(summary, e, b, d, sup)
        json_path = tmp_path / "reports" / "dp_sft_analysis.json"
        with open(json_path) as f:
            data = json.load(f)
        assert data["epsilon_summary"]["inf"]["Avg_LogProb"]["mean"] is None
        assert data["epsilon_summary"]["inf"]["Avg_LogProb"]["std"] is None


# ---------------------------------------------------------------------------
# plot_epsilon_trend
# ---------------------------------------------------------------------------
class TestPlotEpsilonTrend:

    def _make_summary(self):
        """Create a minimal summary dict for plotting."""
        return {
            "inf": {
                "Avg_LogProb": {"mean": -6.41, "std": 0.02},
                "Avg_Rank": {"mean": 50.5, "std": 1.0},
                "Canary_PPL": {"mean": 794.1, "std": 10.0},
            },
            "8": {
                "Avg_LogProb": {"mean": -6.45, "std": 0.03},
                "Avg_Rank": {"mean": 52.0, "std": 1.5},
                "Canary_PPL": {"mean": 810.0, "std": 12.0},
            },
            "4": {
                "Avg_LogProb": {"mean": -6.52, "std": 0.04},
                "Avg_Rank": {"mean": 55.0, "std": 2.0},
                "Canary_PPL": {"mean": 850.0, "std": 15.0},
            },
            "1": {
                "Avg_LogProb": {"mean": -6.55, "std": 0.05},
                "Avg_Rank": {"mean": 58.0, "std": 2.5},
                "Canary_PPL": {"mean": 900.0, "std": 20.0},
            },
        }

    def test_plot_file_created(self, tmp_path):
        """Plot PNG file is created in the output directory."""
        summary = self._make_summary()
        plot_epsilon_trend(summary, output_dir=str(tmp_path))
        assert (tmp_path / "dp_sft_epsilon_trend.png").exists()

    def test_plot_file_nonzero_size(self, tmp_path):
        """Plot file has non-zero size."""
        summary = self._make_summary()
        plot_epsilon_trend(summary, output_dir=str(tmp_path))
        assert (tmp_path / "dp_sft_epsilon_trend.png").stat().st_size > 0

    def test_plot_with_nan_std(self, tmp_path):
        """Plot handles NaN std (single seed) without error."""
        summary = {
            "inf": {"Avg_LogProb": {"mean": -6.41, "std": float("nan")}},
            "4": {"Avg_LogProb": {"mean": -6.52, "std": float("nan")}},
        }
        plot_epsilon_trend(summary, output_dir=str(tmp_path))
        assert (tmp_path / "dp_sft_epsilon_trend.png").exists()

    def test_plot_empty_summary_no_error(self, tmp_path):
        """Empty summary dict does not raise an error."""
        plot_epsilon_trend({}, output_dir=str(tmp_path))
        # No file created when there's nothing to plot
        assert not (tmp_path / "dp_sft_epsilon_trend.png").exists()

    def test_plot_creates_output_dir(self, tmp_path):
        """Output directory is created if it doesn't exist."""
        out = str(tmp_path / "subdir" / "plots")
        summary = {"inf": {"Avg_LogProb": {"mean": -6.0, "std": 0.1}}}
        plot_epsilon_trend(summary, output_dir=out)
        assert os.path.exists(os.path.join(out, "dp_sft_epsilon_trend.png"))
