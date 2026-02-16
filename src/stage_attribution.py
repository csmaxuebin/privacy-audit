"""
Stage Attribution Analysis

Analyze privacy risk changes across Base → SFT → DPO stages (4-stage),
and quantify each stage's contribution to privacy risk.

Supports:
- Stage0_Base, Stage1_SFT, Stage2a_DPO_NoCanary, Stage2b_DPO_WithCanary
- Strict mode (default): fail if any expected stage is missing
- Tolerant mode (--tolerant): warn and skip missing stages
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import sys
import warnings

# Expected stages and metrics
EXPECTED_STAGES = [
    "Stage0_Base",
    "Stage1_SFT",
    "Stage2a_DPO_NoCanary",
    "Stage2b_DPO_WithCanary",
]

METRICS = ["MIA_Gap", "Avg_LogProb", "Avg_Rank", "Canary_PPL", "PPL_Ratio"]

# Stage transitions to compute
STAGE_TRANSITIONS = {
    "Base_to_SFT": ("Stage0_Base", "Stage1_SFT"),
    "SFT_to_DPO_NoCanary": ("Stage1_SFT", "Stage2a_DPO_NoCanary"),
    "SFT_to_DPO_WithCanary": ("Stage1_SFT", "Stage2b_DPO_WithCanary"),
    "DPO_NoCanary_vs_WithCanary": ("Stage2a_DPO_NoCanary", "Stage2b_DPO_WithCanary"),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Stage attribution analysis for privacy audit results"
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default="reports/privacy_audit_summary.csv",
        help="Path to audit summary CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Output directory for reports and plots",
    )
    parser.add_argument(
        "--tolerant",
        action="store_true",
        help="Tolerant mode: warn and skip missing stages instead of failing",
    )
    return parser.parse_args(argv)


def load_audit_summary(filepath: str = "reports/privacy_audit_summary.csv") -> pd.DataFrame:
    """Load audit summary data"""
    return pd.read_csv(filepath)


def validate_stages(df: pd.DataFrame, tolerant: bool = False) -> list:
    """
    Validate that expected stages are present in the DataFrame.

    Returns:
        list of present stage names

    Raises:
        SystemExit in strict mode if any stage is missing.
    """
    present = df["Stage"].tolist()
    missing = [s for s in EXPECTED_STAGES if s not in present]

    if missing:
        if tolerant:
            for s in missing:
                warnings.warn(f"Missing stage '{s}' in input data, skipping.")
        else:
            print(f"Error: Missing expected stages: {missing}", file=sys.stderr)
            sys.exit(1)

    return [s for s in EXPECTED_STAGES if s in present]


def _stage_val(df: pd.DataFrame, stage: str, metric: str):
    """Get a single metric value for a stage. Returns NaN if stage missing."""
    rows = df[df["Stage"] == stage][metric]
    if rows.empty:
        return float("nan")
    return rows.values[0]


def compute_stage_deltas(df: pd.DataFrame, tolerant: bool = False) -> dict:
    """
    Compute metric changes between stages (4-stage aware).

    Returns dict with keys matching STAGE_TRANSITIONS plus legacy keys
    for backward compatibility.
    """
    deltas = {}

    for trans_name, (src, tgt) in STAGE_TRANSITIONS.items():
        src_present = src in df["Stage"].values
        tgt_present = tgt in df["Stage"].values

        if not (src_present and tgt_present):
            if tolerant:
                continue
            # strict mode: fail if any expected transition is incomplete
            missing = []
            if not src_present:
                missing.append(src)
            if not tgt_present:
                missing.append(tgt)
            print(
                f"Error: Cannot compute '{trans_name}': missing stage(s) {missing}",
                file=sys.stderr,
            )
            sys.exit(1)

        deltas[trans_name] = {}
        for metric in METRICS:
            src_val = _stage_val(df, src, metric)
            tgt_val = _stage_val(df, tgt, metric)
            deltas[trans_name][metric] = tgt_val - src_val

            # percentage change
            if src_val != 0 and not np.isnan(src_val):
                deltas[trans_name][f"{metric}_pct"] = (
                    (tgt_val - src_val) / abs(src_val) * 100
                )

    return deltas


def compute_dpo_comparison(df: pd.DataFrame) -> dict:
    """
    Compute the difference between DPO-with-canary and DPO-no-canary.

    Returns dict with per-metric absolute and percentage differences.
    """
    comparison = {}
    src = "Stage2a_DPO_NoCanary"
    tgt = "Stage2b_DPO_WithCanary"

    for metric in METRICS:
        no_canary_val = _stage_val(df, src, metric)
        with_canary_val = _stage_val(df, tgt, metric)
        diff = with_canary_val - no_canary_val
        comparison[metric] = diff
        if no_canary_val != 0 and not np.isnan(no_canary_val):
            comparison[f"{metric}_pct"] = diff / abs(no_canary_val) * 100

    return comparison


def compute_attribution_scores(deltas: dict) -> dict:
    """
    Compute each training stage's contribution to privacy risk (4-stage).

    Attribution is computed relative to Base for each DPO variant path:
    - SFT contribution = Base_to_SFT delta
    - DPO_NoCanary contribution = SFT_to_DPO_NoCanary delta
    - DPO_WithCanary contribution = SFT_to_DPO_WithCanary delta
    """
    attribution = {
        "SFT_contribution_nc": {},
        "DPO_NoCanary_contribution": {},
        "SFT_contribution_wc": {},
        "DPO_WithCanary_contribution": {},
    }

    for metric in METRICS:
        sft_change = deltas.get("Base_to_SFT", {}).get(metric, 0)

        # No-canary path: SFT + DPO_NoCanary should sum to 100%
        dpo_nc_change = deltas.get("SFT_to_DPO_NoCanary", {}).get(metric, 0)
        total_nc = sft_change + dpo_nc_change
        if total_nc != 0:
            attribution["SFT_contribution_nc"][metric] = sft_change / total_nc * 100
            attribution["DPO_NoCanary_contribution"][metric] = (
                dpo_nc_change / total_nc * 100
            )
        else:
            attribution["SFT_contribution_nc"][metric] = 0
            attribution["DPO_NoCanary_contribution"][metric] = 0

        # With-canary path: SFT + DPO_WithCanary should sum to 100%
        dpo_wc_change = deltas.get("SFT_to_DPO_WithCanary", {}).get(metric, 0)
        total_wc = sft_change + dpo_wc_change
        if total_wc != 0:
            attribution["SFT_contribution_wc"][metric] = sft_change / total_wc * 100
            attribution["DPO_WithCanary_contribution"][metric] = (
                dpo_wc_change / total_wc * 100
            )
        else:
            attribution["SFT_contribution_wc"][metric] = 0
            attribution["DPO_WithCanary_contribution"][metric] = 0

    return attribution


def interpret_results(df: pd.DataFrame, deltas: dict) -> dict:
    """Interpret audit results and generate conclusions (4-stage)."""
    interpretations = {}

    # MIA Gap analysis
    mia_base = _stage_val(df, "Stage0_Base", "MIA_Gap")
    mia_sft = _stage_val(df, "Stage1_SFT", "MIA_Gap")
    mia_nc = _stage_val(df, "Stage2a_DPO_NoCanary", "MIA_Gap")
    mia_wc = _stage_val(df, "Stage2b_DPO_WithCanary", "MIA_Gap")

    interpretations["MIA"] = {
        "base": mia_base,
        "sft": mia_sft,
        "dpo_no_canary": mia_nc,
        "dpo_with_canary": mia_wc,
        "canary_effect": mia_wc - mia_nc if not (np.isnan(mia_wc) or np.isnan(mia_nc)) else None,
        "conclusion": (
            "Canary in DPO preference data increases MIA risk"
            if (not np.isnan(mia_wc) and not np.isnan(mia_nc) and mia_wc < mia_nc)
            else "Canary in DPO preference data does not significantly increase MIA risk"
        ),
    }

    # Rank analysis
    rank_base = _stage_val(df, "Stage0_Base", "Avg_Rank")
    rank_sft = _stage_val(df, "Stage1_SFT", "Avg_Rank")
    rank_nc = _stage_val(df, "Stage2a_DPO_NoCanary", "Avg_Rank")
    rank_wc = _stage_val(df, "Stage2b_DPO_WithCanary", "Avg_Rank")

    interpretations["Extraction"] = {
        "trend": (
            f"Rank: {rank_base:.0f} (Base) -> {rank_sft:.0f} (SFT) "
            f"-> {rank_nc:.0f} (DPO-NC) / {rank_wc:.0f} (DPO-WC)"
        ),
        "canary_effect": rank_wc - rank_nc if not (np.isnan(rank_wc) or np.isnan(rank_nc)) else None,
        "conclusion": (
            "Lower rank with canary indicates increased extraction risk from canary in preference data"
            if (not np.isnan(rank_wc) and not np.isnan(rank_nc) and rank_wc < rank_nc)
            else "Canary in preference data does not significantly increase extraction risk"
        ),
    }

    # PPL analysis
    ppl_base = _stage_val(df, "Stage0_Base", "Canary_PPL")
    ppl_sft = _stage_val(df, "Stage1_SFT", "Canary_PPL")
    ppl_nc = _stage_val(df, "Stage2a_DPO_NoCanary", "Canary_PPL")
    ppl_wc = _stage_val(df, "Stage2b_DPO_WithCanary", "Canary_PPL")

    interpretations["Perplexity"] = {
        "trend": (
            f"Canary PPL: {ppl_base:.0f} (Base) -> {ppl_sft:.0f} (SFT) "
            f"-> {ppl_nc:.0f} (DPO-NC) / {ppl_wc:.0f} (DPO-WC)"
        ),
        "canary_effect": ppl_wc - ppl_nc if not (np.isnan(ppl_wc) or np.isnan(ppl_nc)) else None,
        "conclusion": (
            "Lower PPL with canary indicates increased memorization from canary in preference data"
            if (not np.isnan(ppl_wc) and not np.isnan(ppl_nc) and ppl_wc < ppl_nc)
            else "Canary in preference data does not significantly increase memorization"
        ),
    }

    return interpretations


def generate_attribution_report(
    df: pd.DataFrame,
    deltas: dict,
    attribution: dict,
    dpo_comparison: dict,
    interpretations: dict,
    output_dir: str = "reports",
) -> dict:
    """Generate complete attribution report (4-stage)."""
    present_stages = [s for s in EXPECTED_STAGES if s in df["Stage"].values]

    report = {
        "summary": {
            "total_stages": len(present_stages),
            "stages": present_stages,
            "metrics_analyzed": METRICS,
        },
        "stage_metrics": df.to_dict(orient="records"),
        "stage_deltas": deltas,
        "attribution_scores": attribution,
        "dpo_comparison": {"canary_effect": dpo_comparison},
        "interpretations": interpretations,
        "key_findings": [],
    }

    # Key findings
    findings = []

    # 1. DPO canary effect
    if dpo_comparison:
        for metric in METRICS:
            val = dpo_comparison.get(metric)
            if val is not None and not np.isnan(val):
                findings.append(
                    f"Canary effect on {metric}: {val:+.4f} "
                    f"(DPO-with-canary minus DPO-no-canary)"
                )

    # 2. SFT contribution
    if "Base_to_SFT" in deltas:
        rank_delta = deltas["Base_to_SFT"].get("Avg_Rank", 0)
        if rank_delta < 0:
            findings.append(
                f"SFT decreased canary rank by {abs(rank_delta):.0f}, increasing extraction risk"
            )

    report["key_findings"] = findings

    # Save report
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    with open(output_path / "attribution_summary.json", "w", encoding="utf-8") as f:
        # Convert NaN to None for JSON serialization
        json.dump(report, f, indent=2, ensure_ascii=False, default=_json_default)

    return report


def _json_default(obj):
    """Handle NaN and numpy types for JSON serialization."""
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def plot_stage_comparison(df: pd.DataFrame, output_dir: str = "reports"):
    """Generate 4-stage comparison visualization."""
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    present_stages = [s for s in EXPECTED_STAGES if s in df["Stage"].values]
    plot_df = df[df["Stage"].isin(present_stages)].copy()
    plot_df = plot_df.set_index("Stage").loc[present_stages].reset_index()

    n = len(present_stages)
    labels = [s.replace("Stage0_", "").replace("Stage1_", "").replace("Stage2a_", "").replace("Stage2b_", "") for s in present_stages]
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"][:n]
    x = range(n)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    plot_configs = [
        ("MIA_Gap", "MIA Gap", "More negative = Better", axes[0, 0]),
        ("Avg_LogProb", "Avg LogProb", "More negative = Better", axes[0, 1]),
        ("Avg_Rank", "Canary Token Rank", "Higher = Safer", axes[0, 2]),
        ("Canary_PPL", "Canary Perplexity", "Higher = Safer", axes[1, 0]),
        ("PPL_Ratio", "PPL Ratio", "Higher = Safer", axes[1, 1]),
    ]

    for metric, title, subtitle, ax in plot_configs:
        vals = plot_df[metric].tolist()
        bars = ax.bar(x, vals, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
        ax.set_ylabel(metric)
        ax.set_title(f"{title}\n({subtitle})")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        for bar, val in zip(bars, vals):
            offset = abs(max(vals) - min(vals)) * 0.02 if vals else 0
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    # DPO variant comparison subplot
    ax_cmp = axes[1, 2]
    nc_stage = "Stage2a_DPO_NoCanary"
    wc_stage = "Stage2b_DPO_WithCanary"
    if nc_stage in df["Stage"].values and wc_stage in df["Stage"].values:
        cmp_metrics = METRICS
        nc_vals = [_stage_val(df, nc_stage, m) for m in cmp_metrics]
        wc_vals = [_stage_val(df, wc_stage, m) for m in cmp_metrics]
        mx = np.arange(len(cmp_metrics))
        w = 0.35
        ax_cmp.bar(mx - w / 2, nc_vals, w, label="No Canary", color="#2ecc71")
        ax_cmp.bar(mx + w / 2, wc_vals, w, label="With Canary", color="#f39c12")
        ax_cmp.set_xticks(mx)
        ax_cmp.set_xticklabels(cmp_metrics, rotation=20, ha="right", fontsize=7)
        ax_cmp.set_title("DPO Variant Comparison")
        ax_cmp.legend(fontsize=8)
    else:
        ax_cmp.text(0.5, 0.5, "DPO variants\nnot available", ha="center", va="center", transform=ax_cmp.transAxes)
        ax_cmp.set_title("DPO Variant Comparison")

    plt.tight_layout()
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    plt.savefig(output_path / "stage_attribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart saved to {output_path / 'stage_attribution.png'}")


def plot_attribution_breakdown(attribution: dict, output_dir: str = "reports"):
    """Generate attribution breakdown chart (3 training stages)."""
    sft_nc = [attribution["SFT_contribution_nc"].get(m, 0) for m in METRICS]
    nc_contrib = [attribution["DPO_NoCanary_contribution"].get(m, 0) for m in METRICS]
    sft_wc = [attribution["SFT_contribution_wc"].get(m, 0) for m in METRICS]
    wc_contrib = [attribution["DPO_WithCanary_contribution"].get(m, 0) for m in METRICS]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # No-canary path
    x = np.arange(len(METRICS))
    width = 0.35
    axes[0].bar(x - width / 2, sft_nc, width, label="SFT", color="#e74c3c")
    axes[0].bar(x + width / 2, nc_contrib, width, label="DPO-NoCanary", color="#2ecc71")
    axes[0].set_ylabel("Contribution (%)")
    axes[0].set_title("No-Canary Path Attribution")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(METRICS, rotation=15, ha="right")
    axes[0].legend()
    axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # With-canary path
    axes[1].bar(x - width / 2, sft_wc, width, label="SFT", color="#e74c3c")
    axes[1].bar(x + width / 2, wc_contrib, width, label="DPO-WithCanary", color="#f39c12")
    axes[1].set_ylabel("Contribution (%)")
    axes[1].set_title("With-Canary Path Attribution")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(METRICS, rotation=15, ha="right")
    axes[1].legend()
    axes[1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    plt.savefig(output_path / "attribution_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart saved to {output_path / 'attribution_breakdown.png'}")


def main(argv=None):
    """Main function"""
    args = parse_args(argv)

    print("=" * 60)
    print("Stage Attribution Analysis (4-Stage)")
    print("=" * 60)

    # 1. Load data
    print(f"\n1. Loading audit data from {args.input_csv}...")
    df = load_audit_summary(args.input_csv)
    print(df.to_string(index=False))

    # 2. Validate stages
    print("\n2. Validating stages...")
    present = validate_stages(df, tolerant=args.tolerant)
    print(f"   Present stages: {present}")

    # 3. Compute stage deltas
    print("\n3. Computing stage transitions...")
    deltas = compute_stage_deltas(df, tolerant=args.tolerant)
    for trans_name, trans_deltas in deltas.items():
        print(f"\n  {trans_name}:")
        for k, v in trans_deltas.items():
            if not k.endswith("_pct"):
                print(f"    {k}: {v:+.4f}")

    # 4. Compute DPO comparison
    print("\n4. Computing DPO variant comparison...")
    dpo_comparison = compute_dpo_comparison(df)
    for k, v in dpo_comparison.items():
        if not k.endswith("_pct"):
            print(f"    {k}: {v:+.4f}")

    # 5. Compute attribution scores
    print("\n5. Computing attribution scores...")
    attribution = compute_attribution_scores(deltas)
    for contrib_name, contrib_vals in attribution.items():
        print(f"\n  {contrib_name}:")
        for k, v in contrib_vals.items():
            print(f"    {k}: {v:.1f}%")

    # 6. Interpret results
    print("\n6. Interpreting results...")
    interpretations = interpret_results(df, deltas)
    for category, interp in interpretations.items():
        print(f"\n  [{category}]")
        print(f"    Conclusion: {interp['conclusion']}")

    # 7. Generate report
    print(f"\n7. Generating attribution report to {args.output_dir}...")
    report = generate_attribution_report(
        df, deltas, attribution, dpo_comparison, interpretations, args.output_dir
    )
    print(f"\n  Key findings ({len(report['key_findings'])}):")
    for i, finding in enumerate(report["key_findings"], 1):
        print(f"    {i}. {finding}")

    # 8. Generate visualizations
    print("\n8. Generating visualization charts...")
    plot_stage_comparison(df, args.output_dir)
    plot_attribution_breakdown(attribution, args.output_dir)

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)

    return report


if __name__ == "__main__":
    main()
