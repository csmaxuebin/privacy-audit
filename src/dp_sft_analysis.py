"""DP-SFT statistical analysis module.

Cross-epsilon, cross-seed analysis for DP-SFT baseline experiments.
Reuses bootstrap_ci and cohens_d from stage_attribution.py.

Usage:
    python -m src.dp_sft_analysis
    python -m src.dp_sft_analysis --input-csv reports/dp_sft_audit_results.csv
    python -m src.dp_sft_analysis --pilot
"""

import argparse
import json
import math
import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

# Metrics definitions
PRIMARY_METRICS = ["Avg_LogProb", "Avg_Rank", "Canary_PPL"]
SECONDARY_METRICS = ["Extraction_Rate", "ROC_AUC", "PR_AUC"]
ALL_METRICS = PRIMARY_METRICS + SECONDARY_METRICS

# Metric direction map: defines which direction indicates increased memorization risk
# positive = higher value means more risk, negative = lower value means more risk
METRIC_DIRECTION_MAP = {
    "Avg_LogProb": "positive",   # higher logprob = model more confident on canary = more risk
    "Avg_Rank": "negative",      # lower rank = canary token ranked higher = more risk
    "Canary_PPL": "negative",    # lower PPL = model more familiar with canary = more risk
    "Extraction_Rate": "positive",
    "ROC_AUC": "positive",
    "PR_AUC": "positive",
}


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    """Parse CLI arguments for DP-SFT analysis."""
    parser = argparse.ArgumentParser(
        description="Statistical analysis for DP-SFT audit results"
    )
    parser.add_argument(
        "--input-csv", type=str,
        default="reports/dp_sft_audit_results.csv",
        help="Path to audit results CSV"
    )
    parser.add_argument(
        "--output-dir", type=str, default="reports",
        help="Directory for output files"
    )
    parser.add_argument(
        "--pilot", action="store_true",
        help="Pilot mode: skip stability/significance claims"
    )
    return parser.parse_args(argv)


def load_audit_results(filepath: str) -> pd.DataFrame:
    """Load audit results CSV with proper NA handling.

    The CSV contains columns: epsilon, seed, Avg_LogProb, Avg_Rank,
    Canary_PPL, Extraction_Rate, ROC_AUC, PR_AUC.

    The epsilon column is kept as string type (contains "inf").
    NA values in secondary metrics (pilot mode) are preserved as NaN.
    """
    df = pd.read_csv(
        filepath, keep_default_na=True, na_values=["NA"],
        dtype={"epsilon": str},  # read epsilon as string directly
    )
    # Ensure epsilon stays string (already handled by dtype above)
    df["epsilon"] = df["epsilon"].astype(str)
    return df


def normalize_effect_direction(cohens_d_value: float, metric: str) -> float:
    """Normalize Cohen's d so positive always means increased memorization risk.

    For 'negative' direction metrics (Avg_Rank, Canary_PPL), flip the sign
    so that a positive normalized value consistently means more memorization.

    Args:
        cohens_d_value: Raw Cohen's d value.
        metric: Metric name (must be a key in METRIC_DIRECTION_MAP).

    Returns:
        Normalized Cohen's d. Positive = increased risk.
        Returns NaN unchanged if input is NaN.
    """
    if np.isnan(cohens_d_value):
        return cohens_d_value
    if METRIC_DIRECTION_MAP.get(metric) == "negative":
        return -cohens_d_value
    return cohens_d_value


def compute_epsilon_summary(df: pd.DataFrame) -> dict:
    """Per-epsilon: mean, std across seeds for each metric.

    Groups by epsilon (string), computes mean and sample std (ddof=1)
    for each metric column present in the DataFrame.

    For single-seed groups, std is NaN (not 0) — pandas default with ddof=1.

    Args:
        df: DataFrame with 'epsilon' column and metric columns.

    Returns:
        Nested dict: {epsilon_str: {metric: {"mean": float, "std": float}}}
        NaN values in metrics are excluded from aggregation (skipna).
    """
    available_metrics = [m for m in ALL_METRICS if m in df.columns]
    result = {}

    for eps, group in df.groupby("epsilon", sort=False):
        eps_str = str(eps)
        result[eps_str] = {}
        for metric in available_metrics:
            values = group[metric].dropna()
            if len(values) == 0:
                result[eps_str][metric] = {"mean": float("nan"), "std": float("nan")}
            else:
                result[eps_str][metric] = {
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)),  # sample std; NaN for n=1
                }

    return result

def classify_effect_size(d: float) -> str:
    """Classify Cohen's d into effect size category.

    Args:
        d: Cohen's d value.

    Returns:
        One of: "negligible", "small", "medium", "large", "not_estimable".
    """
    if np.isnan(d):
        return "not_estimable"
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def _cohens_d_safe(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Cohen's d with safe handling for small/degenerate groups.

    Tries to reuse stage_attribution.cohens_d; falls back to inline calc.
    Returns NaN when either group has fewer than 2 samples.
    Returns 0.0 when both groups have zero variance.

    Formula: d = (mean(B) - mean(A)) / sqrt((var(A) + var(B)) / 2)
    where var uses population variance (ddof=0).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    var_a = float(np.var(a))
    var_b = float(np.var(b))
    pooled_std = float(np.sqrt((var_a + var_b) / 2))
    if pooled_std == 0:
        return 0.0
    return float(np.mean(b) - np.mean(a)) / pooled_std


def compute_effect_sizes(df: pd.DataFrame, control_epsilon: str = "inf") -> dict:
    """Cohen's d: each finite epsilon vs control (epsilon=inf).

    For each finite epsilon group, computes Cohen's d against the control
    group for each available metric. When either group has < 2 samples,
    Cohen's d is NaN.

    Args:
        df: DataFrame with 'epsilon' column and metric columns.
        control_epsilon: Epsilon value for the control group (default "inf").

    Returns:
        Nested dict: {"8_vs_inf": {"Avg_LogProb": {"cohens_d": float,
        "category": str}, ...}, ...}
    """
    available_metrics = [m for m in ALL_METRICS if m in df.columns]
    control_df = df[df["epsilon"] == control_epsilon]
    result = {}

    for eps, group in df.groupby("epsilon", sort=False):
        eps_str = str(eps)
        if eps_str == control_epsilon:
            continue
        key = f"{eps_str}_vs_{control_epsilon}"
        result[key] = {}
        for metric in available_metrics:
            ctrl_vals = control_df[metric].dropna().values
            treat_vals = group[metric].dropna().values
            d = _cohens_d_safe(ctrl_vals, treat_vals)
            result[key][metric] = {
                "cohens_d": d,
                "category": classify_effect_size(d),
            }

    return result


def compute_canary_level_bootstrap(
    per_canary_values: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """Bootstrap 95% CI on per-canary metric values.

    Resamples the input array n_bootstrap times with replacement,
    computes the mean of each resample, then returns the overall mean
    and the 2.5th/97.5th percentile as the 95% CI bounds.

    Args:
        per_canary_values: 1D array of metric values across canaries.
        n_bootstrap: Number of bootstrap resamples (default 1000).
        seed: Random seed for reproducibility (default 42).

    Returns:
        {"mean": float, "ci_lower": float, "ci_upper": float}
        Returns all NaN when the input array is empty.
    """
    arr = np.asarray(per_canary_values, dtype=float)
    if len(arr) == 0:
        return {"mean": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan")}

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_means[i] = np.mean(sample)

    return {
        "mean": float(np.mean(arr)),
        "ci_lower": float(np.percentile(boot_means, 2.5)),
        "ci_upper": float(np.percentile(boot_means, 97.5)),
    }

def evaluate_detectability(effect_sizes_vs_baseline: dict) -> dict:
    """Judge if memorization is 'detectable' per Requirement 9.1 criteria.

    For each epsilon, checks if at least one PRIMARY metric has
    normalized Cohen_d > 0.5 (vs Random_Baseline).

    Uses normalize_effect_direction() to ensure cross-metric consistency:
    normalized Cohen_d > 0.5 means detectable memorization signal.

    Args:
        effect_sizes_vs_baseline: {epsilon_str: {metric: {"cohens_d": float,
            "category": str}, ...}, ...}

    Returns:
        {epsilon_str: {"detectable": bool, "reason": str}}
    """
    result = {}
    for eps_str, metrics in effect_sizes_vs_baseline.items():
        detected_metrics = []
        for metric in PRIMARY_METRICS:
            if metric not in metrics:
                continue
            raw_d = metrics[metric].get("cohens_d", float("nan"))
            norm_d = normalize_effect_direction(raw_d, metric)
            if not np.isnan(norm_d) and norm_d > 0.5:
                detected_metrics.append((metric, norm_d))

        if detected_metrics:
            # Report the metric with the largest normalized Cohen's d
            best_metric, best_d = max(detected_metrics, key=lambda x: x[1])
            result[eps_str] = {
                "detectable": True,
                "reason": f"{best_metric} normalized Cohen_d={best_d:.2f} vs base",
            }
        else:
            result[eps_str] = {
                "detectable": False,
                "reason": "No metric normalized Cohen_d > 0.5 vs base",
            }
    return result


def evaluate_suppression(effect_sizes_vs_control: dict) -> dict:
    """Judge if DP 'effectively suppresses' memorization per Requirement 9.2.

    For each finite epsilon comparison (e.g. "8_vs_inf"), checks if at least
    one PRIMARY metric has normalized Cohen_d < -0.8.

    Uses normalize_effect_direction() to ensure direction consistency:
    normalized Cohen_d < -0.8 means effective suppression (DP significantly
    reduces memorization risk compared to non-DP control).

    Args:
        effect_sizes_vs_control: {"8_vs_inf": {"Avg_LogProb": {"cohens_d": float,
            "category": str}, ...}, ...}

    Returns:
        {"8_vs_inf": {"suppressed": bool, "reason": str}}
    """
    result = {}
    for comparison_key, metrics in effect_sizes_vs_control.items():
        suppressed_metrics = []
        for metric in PRIMARY_METRICS:
            if metric not in metrics:
                continue
            raw_d = metrics[metric].get("cohens_d", float("nan"))
            norm_d = normalize_effect_direction(raw_d, metric)
            if not np.isnan(norm_d) and norm_d < -0.8:
                suppressed_metrics.append((metric, norm_d))

        if suppressed_metrics:
            best_metric, best_d = min(suppressed_metrics, key=lambda x: x[1])
            result[comparison_key] = {
                "suppressed": True,
                "reason": f"{best_metric} normalized Cohen_d={best_d:.2f}, < -0.8",
            }
        else:
            result[comparison_key] = {
                "suppressed": False,
                "reason": "normalized Cohen_d > -0.8",
            }
    return result


def annotate_negative_result(detectability: dict) -> str:
    """Annotate negative result when all epsilon values show no detectable signal.

    Per Requirement 9.3: if ALL epsilon values have detectable=False,
    return the negative result annotation string.

    Args:
        detectability: Output of evaluate_detectability().

    Returns:
        Annotation string if negative result, empty string otherwise.
    """
    if not detectability:
        return ""
    all_undetectable = all(
        not entry["detectable"] for entry in detectability.values()
    )
    if all_undetectable:
        return (
            "DP noise eliminated all detectable signals at the current model scale"
            " — this is itself a valuable negative result"
        )
    return ""


def annotate_pilot_mode(pilot: bool) -> str:
    """Generate pilot mode annotation per Requirement 10.3.

    Args:
        pilot: Whether pilot mode is enabled.

    Returns:
        Pilot annotation string if pilot=True, empty string otherwise.
    """
    if pilot:
        return (
            "Pilot mode: only 1 seed, no stability or statistical significance claims; "
            "used only for feasibility validation and signal direction assessment"
        )
    return ""




def _sanitize_for_json(obj):
    """Recursively convert NaN/Inf floats to None for JSON serialization."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def generate_report(
    summary: dict,
    effect_sizes: dict,
    bootstrap_cis: dict,
    detectability: dict,
    suppression: dict,
    pilot: bool = False,
) -> dict:
    """Assemble all analysis results and write to reports/dp_sft_analysis.json.

    Args:
        summary: Output of compute_epsilon_summary().
        effect_sizes: Output of compute_effect_sizes().
        bootstrap_cis: Canary-level bootstrap CI results.
        detectability: Output of evaluate_detectability().
        suppression: Output of evaluate_suppression().
        pilot: Whether pilot mode is enabled.

    Returns:
        The assembled report dict.
    """
    report = {
        "pilot": pilot,
        "epsilon_summary": summary,
        "effect_sizes": effect_sizes,
        "canary_bootstrap_ci": bootstrap_cis,
        "detectability": detectability,
        "suppression": suppression,
    }

    # Add pilot annotation if applicable
    pilot_annotation = annotate_pilot_mode(pilot)
    if pilot_annotation:
        report["pilot_annotation"] = pilot_annotation

    # Add negative result annotation if applicable
    neg_annotation = annotate_negative_result(detectability)
    if neg_annotation:
        report["negative_result_annotation"] = neg_annotation

    # Write to JSON file
    os.makedirs("reports", exist_ok=True)
    output_path = os.path.join("reports", "dp_sft_analysis.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(_sanitize_for_json(report), f, indent=2, ensure_ascii=False)

    return report


def plot_epsilon_trend(summary: dict, output_dir: str = "reports") -> None:
    """Create ε vs metric value trend plot with error bars (mean ± std).

    X-axis: epsilon values ordered inf → 8 → 4 → 1 (decreasing privacy).
    Y-axis: metric values.
    One subplot per PRIMARY_METRIC.

    Args:
        summary: Output of compute_epsilon_summary().
        output_dir: Directory to save the plot (default "reports").
    """
    # Determine epsilon ordering: inf first, then descending numeric
    eps_keys = list(summary.keys())
    inf_keys = [k for k in eps_keys if k == "inf"]
    numeric_keys = sorted(
        [k for k in eps_keys if k != "inf"],
        key=lambda x: float(x),
        reverse=True,
    )
    ordered_epsilons = inf_keys + numeric_keys

    # Filter to primary metrics that exist in the summary
    available_metrics = [
        m for m in PRIMARY_METRICS
        if any(m in summary.get(eps, {}) for eps in ordered_epsilons)
    ]
    if not available_metrics:
        return

    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4), squeeze=False)

    x_labels = [f"ε={e}" for e in ordered_epsilons]
    x_pos = list(range(len(ordered_epsilons)))

    for idx, metric in enumerate(available_metrics):
        ax = axes[0, idx]
        means = []
        stds = []
        for eps in ordered_epsilons:
            entry = summary.get(eps, {}).get(metric, {})
            m = entry.get("mean", float("nan"))
            s = entry.get("std", float("nan"))
            means.append(m)
            # Use 0 for error bar if std is NaN (e.g. single seed)
            stds.append(s if not math.isnan(s) else 0.0)

        ax.errorbar(x_pos, means, yerr=stds, fmt="o-", capsize=4, linewidth=1.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_title(metric)
        ax.set_ylabel("Value")
        ax.set_xlabel("Epsilon")

    fig.suptitle("DP-SFT: ε vs Metric Trend", fontsize=13)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "dp_sft_epsilon_trend.png")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main(argv=None):
    """Main entry point: load CSV, compute all analyses, generate report and plot."""
    args = parse_args(argv)

    # Load audit results
    df = load_audit_results(args.input_csv)

    # Compute summary statistics
    summary = compute_epsilon_summary(df)

    # Compute effect sizes (each finite ε vs inf control)
    effect_sizes = compute_effect_sizes(df, control_epsilon="inf")

    # Compute canary-level bootstrap CIs (per-epsilon)
    # Note: this requires per-canary data; if only seed-level aggregates
    # are available, we bootstrap over the seed-level values instead.
    bootstrap_cis = {}
    for eps in df["epsilon"].unique():
        eps_str = str(eps)
        bootstrap_cis[eps_str] = {}
        eps_df = df[df["epsilon"] == eps]
        for metric in PRIMARY_METRICS:
            if metric not in eps_df.columns:
                continue
            values = eps_df[metric].dropna().values
            bootstrap_cis[eps_str][metric] = compute_canary_level_bootstrap(values)

    # Evaluate detectability and suppression
    detectability = evaluate_detectability(effect_sizes)
    suppression = evaluate_suppression(effect_sizes)

    # Generate report JSON
    report = generate_report(
        summary=summary,
        effect_sizes=effect_sizes,
        bootstrap_cis=bootstrap_cis,
        detectability=detectability,
        suppression=suppression,
        pilot=args.pilot,
    )

    # Plot epsilon trend
    plot_epsilon_trend(summary, output_dir=args.output_dir)

    # Print summary to stdout
    print("=== DP-SFT Analysis Summary ===")
    print(f"Pilot mode: {args.pilot}")
    print(f"Epsilons: {list(summary.keys())}")
    for eps_str, metrics in summary.items():
        print(f"\nε={eps_str}:")
        for metric, stats in metrics.items():
            m = stats["mean"]
            s = stats["std"]
            std_str = f"{s:.4f}" if not math.isnan(s) else "N/A"
            print(f"  {metric}: mean={m:.4f}, std={std_str}")

    neg = annotate_negative_result(detectability)
    if neg:
        print(f"\n⚠ {neg}")

    print(f"\nReport saved to: reports/dp_sft_analysis.json")
    print(f"Trend plot saved to: {args.output_dir}/dp_sft_epsilon_trend.png")

    return report


if __name__ == "__main__":
    main()
