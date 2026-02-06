"""
Stage Attribution Analysis

Analyze privacy risk changes across Base → SFT → DPO stages,
and quantify each stage's contribution to privacy risk.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json


def load_audit_summary(filepath: str = "reports/privacy_audit_summary.csv") -> pd.DataFrame:
    """Load audit summary data"""
    return pd.read_csv(filepath)


def compute_stage_deltas(df: pd.DataFrame) -> dict:
    """
    Compute metric changes between stages
    
    Returns:
        dict: Dictionary containing changes for each stage transition
    """
    stages = df["Stage"].tolist()
    
    deltas = {
        "Base_to_SFT": {},
        "SFT_to_DPO": {},
        "Base_to_DPO": {}  # Overall change
    }
    
    metrics = ["MIA_Gap", "Avg_LogProb", "Avg_Rank", "Canary_PPL", "PPL_Ratio"]
    
    for metric in metrics:
        base_val = df[df["Stage"] == "Stage0_Base"][metric].values[0]
        sft_val = df[df["Stage"] == "Stage1_SFT"][metric].values[0]
        dpo_val = df[df["Stage"] == "Stage2_DPO"][metric].values[0]
        
        # Compute absolute changes
        deltas["Base_to_SFT"][metric] = sft_val - base_val
        deltas["SFT_to_DPO"][metric] = dpo_val - sft_val
        deltas["Base_to_DPO"][metric] = dpo_val - base_val
        
        # Compute percentage changes
        if base_val != 0:
            deltas["Base_to_SFT"][f"{metric}_pct"] = (sft_val - base_val) / abs(base_val) * 100
            deltas["Base_to_DPO"][f"{metric}_pct"] = (dpo_val - base_val) / abs(base_val) * 100
        if sft_val != 0:
            deltas["SFT_to_DPO"][f"{metric}_pct"] = (dpo_val - sft_val) / abs(sft_val) * 100
    
    return deltas


def compute_attribution_scores(deltas: dict) -> dict:
    """
    Compute each stage's contribution to privacy risk
    
    Privacy risk indicators:
    - MIA_Gap: More negative is better (smaller gap between canary and normal)
    - Avg_LogProb: More negative is better (lower canary probability)
    - Avg_Rank: Higher is better (lower canary ranking)
    - Canary_PPL: Higher is better (higher model perplexity on canary)
    - PPL_Ratio: Higher is better (higher canary perplexity relative to normal)
    """
    attribution = {
        "SFT_contribution": {},
        "DPO_contribution": {}
    }
    
    # For each metric, compute SFT and DPO contribution ratios
    metrics = ["MIA_Gap", "Avg_LogProb", "Avg_Rank", "Canary_PPL", "PPL_Ratio"]
    
    for metric in metrics:
        total_change = deltas["Base_to_DPO"][metric]
        sft_change = deltas["Base_to_SFT"][metric]
        dpo_change = deltas["SFT_to_DPO"][metric]
        
        if total_change != 0:
            attribution["SFT_contribution"][metric] = sft_change / total_change * 100
            attribution["DPO_contribution"][metric] = dpo_change / total_change * 100
        else:
            attribution["SFT_contribution"][metric] = 0
            attribution["DPO_contribution"][metric] = 0
    
    return attribution


def interpret_results(df: pd.DataFrame, deltas: dict) -> dict:
    """
    Interpret audit results and generate conclusions
    """
    interpretations = {}
    
    # MIA Gap analysis
    mia_base = df[df["Stage"] == "Stage0_Base"]["MIA_Gap"].values[0]
    mia_sft = df[df["Stage"] == "Stage1_SFT"]["MIA_Gap"].values[0]
    mia_dpo = df[df["Stage"] == "Stage2_DPO"]["MIA_Gap"].values[0]
    
    interpretations["MIA"] = {
        "trend": "SFT slightly increased MIA risk, DPO provided some mitigation",
        "base_to_sft_change": f"{deltas['Base_to_SFT']['MIA_Gap']:.4f}",
        "sft_to_dpo_change": f"{deltas['SFT_to_DPO']['MIA_Gap']:.4f}",
        "conclusion": "DPO slightly mitigates MIA risk" if mia_dpo > mia_sft else "DPO did not effectively mitigate MIA risk"
    }
    
    # Rank analysis (lower rank means model can extract canary more easily)
    rank_base = df[df["Stage"] == "Stage0_Base"]["Avg_Rank"].values[0]
    rank_sft = df[df["Stage"] == "Stage1_SFT"]["Avg_Rank"].values[0]
    rank_dpo = df[df["Stage"] == "Stage2_DPO"]["Avg_Rank"].values[0]
    
    interpretations["Extraction"] = {
        "trend": f"Rank decreased from {rank_base:.0f} to {rank_sft:.0f} (SFT) to {rank_dpo:.0f} (DPO)",
        "risk_increase": f"{(1 - rank_dpo/rank_base) * 100:.1f}%",
        "conclusion": "Post-training model can extract canary more easily, extraction risk significantly increased"
    }
    
    # PPL analysis
    ppl_base = df[df["Stage"] == "Stage0_Base"]["Canary_PPL"].values[0]
    ppl_sft = df[df["Stage"] == "Stage1_SFT"]["Canary_PPL"].values[0]
    ppl_dpo = df[df["Stage"] == "Stage2_DPO"]["Canary_PPL"].values[0]
    
    interpretations["Perplexity"] = {
        "trend": f"Canary PPL decreased from {ppl_base:.0f} to {ppl_sft:.0f} (SFT) to {ppl_dpo:.0f} (DPO)",
        "memorization_increase": f"{(1 - ppl_dpo/ppl_base) * 100:.1f}%",
        "conclusion": "Model perplexity on canary decreased, indicating increased memorization"
    }
    
    return interpretations


def generate_attribution_report(
    df: pd.DataFrame,
    deltas: dict,
    attribution: dict,
    interpretations: dict,
    output_dir: str = "reports"
) -> dict:
    """Generate complete attribution report"""
    
    report = {
        "summary": {
            "total_stages": 3,
            "stages": ["Stage0_Base", "Stage1_SFT", "Stage2_DPO"],
            "metrics_analyzed": ["MIA_Gap", "Avg_LogProb", "Avg_Rank", "Canary_PPL", "PPL_Ratio"]
        },
        "stage_metrics": df.to_dict(orient="records"),
        "stage_deltas": deltas,
        "attribution_scores": attribution,
        "interpretations": interpretations,
        "key_findings": []
    }
    
    # Key findings
    findings = []
    
    # 1. Extraction risk change
    rank_change = deltas["Base_to_DPO"]["Avg_Rank"]
    if rank_change < 0:
        findings.append(f"Canary extraction risk increased: average rank dropped from {df[df['Stage']=='Stage0_Base']['Avg_Rank'].values[0]:.0f} to {df[df['Stage']=='Stage2_DPO']['Avg_Rank'].values[0]:.0f}")
    
    # 2. SFT vs DPO contribution
    sft_rank_contrib = attribution["SFT_contribution"]["Avg_Rank"]
    dpo_rank_contrib = attribution["DPO_contribution"]["Avg_Rank"]
    findings.append(f"SFT stage contributed {sft_rank_contrib:.1f}% of rank change, DPO stage contributed {dpo_rank_contrib:.1f}%")
    
    # 3. Memorization level
    ppl_change_pct = deltas["Base_to_DPO"]["Canary_PPL_pct"]
    findings.append(f"Canary perplexity decreased by {abs(ppl_change_pct):.1f}%, indicating increased model memorization")
    
    # 4. MIA risk
    mia_change = deltas["Base_to_DPO"]["MIA_Gap"]
    if abs(mia_change) < 0.1:
        findings.append("MIA Gap change is small, membership inference attack risk remains relatively stable")
    
    report["key_findings"] = findings
    
    # Save report
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    with open(output_path / "attribution_summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    return report


def plot_stage_comparison(df: pd.DataFrame, output_dir: str = "reports"):
    """Generate stage comparison visualization"""
    
    # Set font for better compatibility
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    stages = df["Stage"].tolist()
    x = range(len(stages))
    
    # 1. MIA Gap
    ax1 = axes[0, 0]
    bars1 = ax1.bar(x, df["MIA_Gap"], color=["#3498db", "#e74c3c", "#2ecc71"])
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Base", "SFT", "DPO"])
    ax1.set_ylabel("MIA Gap")
    ax1.set_title("MIA Gap by Stage\n(More negative = Better)")
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    for bar, val in zip(bars1, df["MIA_Gap"]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Average Rank
    ax2 = axes[0, 1]
    bars2 = ax2.bar(x, df["Avg_Rank"], color=["#3498db", "#e74c3c", "#2ecc71"])
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Base", "SFT", "DPO"])
    ax2.set_ylabel("Average Rank")
    ax2.set_title("Canary Token Rank by Stage\n(Higher = Safer)")
    for bar, val in zip(bars2, df["Avg_Rank"]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                f'{val:.0f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Canary Perplexity
    ax3 = axes[1, 0]
    bars3 = ax3.bar(x, df["Canary_PPL"], color=["#3498db", "#e74c3c", "#2ecc71"])
    ax3.set_xticks(x)
    ax3.set_xticklabels(["Base", "SFT", "DPO"])
    ax3.set_ylabel("Perplexity")
    ax3.set_title("Canary Perplexity by Stage\n(Higher = Safer)")
    for bar, val in zip(bars3, df["Canary_PPL"]):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                f'{val:.0f}', ha='center', va='bottom', fontsize=9)
    
    # 4. PPL Ratio
    ax4 = axes[1, 1]
    bars4 = ax4.bar(x, df["PPL_Ratio"], color=["#3498db", "#e74c3c", "#2ecc71"])
    ax4.set_xticks(x)
    ax4.set_xticklabels(["Base", "SFT", "DPO"])
    ax4.set_ylabel("PPL Ratio (Canary/Normal)")
    ax4.set_title("Perplexity Ratio by Stage\n(Higher = Safer)")
    for bar, val in zip(bars4, df["PPL_Ratio"]):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    plt.savefig(output_path / "stage_attribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Chart saved to {output_path / 'stage_attribution.png'}")


def plot_attribution_breakdown(attribution: dict, output_dir: str = "reports"):
    """Generate attribution breakdown chart"""
    
    metrics = ["MIA_Gap", "Avg_LogProb", "Avg_Rank", "Canary_PPL", "PPL_Ratio"]
    sft_contrib = [attribution["SFT_contribution"][m] for m in metrics]
    dpo_contrib = [attribution["DPO_contribution"][m] for m in metrics]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, sft_contrib, width, label='SFT Contribution', color='#e74c3c')
    bars2 = ax.bar(x + width/2, dpo_contrib, width, label='DPO Contribution', color='#2ecc71')
    
    ax.set_ylabel('Contribution (%)')
    ax.set_title('Stage Attribution: SFT vs DPO Contribution to Metric Changes')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=15, ha='right')
    ax.legend()
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.3)
    ax.axhline(y=100, color='gray', linestyle=':', alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    output_path = Path(output_dir)
    plt.savefig(output_path / "attribution_breakdown.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Chart saved to {output_path / 'attribution_breakdown.png'}")


def main():
    """Main function"""
    print("=" * 60)
    print("Stage Attribution Analysis")
    print("=" * 60)
    
    # 1. Load data
    print("\n1. Loading audit data...")
    df = load_audit_summary()
    print(df.to_string(index=False))
    
    # 2. Compute stage changes
    print("\n2. Computing stage transitions...")
    deltas = compute_stage_deltas(df)
    
    print("\n  Base -> SFT changes:")
    for k, v in deltas["Base_to_SFT"].items():
        if not k.endswith("_pct"):
            print(f"    {k}: {v:+.4f}")
    
    print("\n  SFT -> DPO changes:")
    for k, v in deltas["SFT_to_DPO"].items():
        if not k.endswith("_pct"):
            print(f"    {k}: {v:+.4f}")
    
    # 3. Compute attribution scores
    print("\n3. Computing attribution scores...")
    attribution = compute_attribution_scores(deltas)
    
    print("\n  SFT contribution:")
    for k, v in attribution["SFT_contribution"].items():
        print(f"    {k}: {v:.1f}%")
    
    print("\n  DPO contribution:")
    for k, v in attribution["DPO_contribution"].items():
        print(f"    {k}: {v:.1f}%")
    
    # 4. Interpret results
    print("\n4. Interpreting results...")
    interpretations = interpret_results(df, deltas)
    
    for category, interp in interpretations.items():
        print(f"\n  [{category}]")
        print(f"    Trend: {interp['trend']}")
        print(f"    Conclusion: {interp['conclusion']}")
    
    # 5. Generate report
    print("\n5. Generating attribution report...")
    report = generate_attribution_report(df, deltas, attribution, interpretations)
    
    print("\n  Key findings:")
    for i, finding in enumerate(report["key_findings"], 1):
        print(f"    {i}. {finding}")
    
    # 6. Generate visualizations
    print("\n6. Generating visualization charts...")
    plot_stage_comparison(df)
    plot_attribution_breakdown(attribution)
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)
    
    return report


if __name__ == "__main__":
    main()
