"""
Stage Attribution Analysis

分析隐私风险在 Base → SFT → DPO 各阶段的变化，
量化每个阶段对隐私风险的贡献。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json


def load_audit_summary(filepath: str = "doc/privacy_audit_summary.csv") -> pd.DataFrame:
    """加载审计汇总数据"""
    return pd.read_csv(filepath)


def compute_stage_deltas(df: pd.DataFrame) -> dict:
    """
    计算各阶段之间的指标变化
    
    Returns:
        dict: 包含各阶段变化的字典
    """
    stages = df["Stage"].tolist()
    
    deltas = {
        "Base_to_SFT": {},
        "SFT_to_DPO": {},
        "Base_to_DPO": {}  # 总体变化
    }
    
    metrics = ["MIA_Gap", "Avg_LogProb", "Avg_Rank", "Canary_PPL", "PPL_Ratio"]
    
    for metric in metrics:
        base_val = df[df["Stage"] == "Stage0_Base"][metric].values[0]
        sft_val = df[df["Stage"] == "Stage1_SFT"][metric].values[0]
        dpo_val = df[df["Stage"] == "Stage2_DPO"][metric].values[0]
        
        # 计算绝对变化
        deltas["Base_to_SFT"][metric] = sft_val - base_val
        deltas["SFT_to_DPO"][metric] = dpo_val - sft_val
        deltas["Base_to_DPO"][metric] = dpo_val - base_val
        
        # 计算变化百分比
        if base_val != 0:
            deltas["Base_to_SFT"][f"{metric}_pct"] = (sft_val - base_val) / abs(base_val) * 100
            deltas["Base_to_DPO"][f"{metric}_pct"] = (dpo_val - base_val) / abs(base_val) * 100
        if sft_val != 0:
            deltas["SFT_to_DPO"][f"{metric}_pct"] = (dpo_val - sft_val) / abs(sft_val) * 100
    
    return deltas


def compute_attribution_scores(deltas: dict) -> dict:
    """
    计算各阶段对隐私风险的贡献度
    
    基于以下指标判断隐私风险：
    - MIA_Gap: 越负越好（canary 和 normal 差距小）
    - Avg_LogProb: 越负越好（canary 概率低）
    - Avg_Rank: 越高越好（canary 排名低）
    - Canary_PPL: 越高越好（模型对 canary 困惑度高）
    - PPL_Ratio: 越高越好（canary 相对 normal 困惑度高）
    """
    attribution = {
        "SFT_contribution": {},
        "DPO_contribution": {}
    }
    
    # 对于每个指标，计算 SFT 和 DPO 各自的贡献比例
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
    解读审计结果，生成结论
    """
    interpretations = {}
    
    # MIA Gap 分析
    mia_base = df[df["Stage"] == "Stage0_Base"]["MIA_Gap"].values[0]
    mia_sft = df[df["Stage"] == "Stage1_SFT"]["MIA_Gap"].values[0]
    mia_dpo = df[df["Stage"] == "Stage2_DPO"]["MIA_Gap"].values[0]
    
    interpretations["MIA"] = {
        "trend": "SFT 略微增加了 MIA 风险，DPO 有所缓解",
        "base_to_sft_change": f"{deltas['Base_to_SFT']['MIA_Gap']:.4f}",
        "sft_to_dpo_change": f"{deltas['SFT_to_DPO']['MIA_Gap']:.4f}",
        "conclusion": "DPO 对 MIA 风险有轻微的缓解作用" if mia_dpo > mia_sft else "DPO 未能有效缓解 MIA 风险"
    }
    
    # Rank 分析（越低表示模型越容易提取 canary）
    rank_base = df[df["Stage"] == "Stage0_Base"]["Avg_Rank"].values[0]
    rank_sft = df[df["Stage"] == "Stage1_SFT"]["Avg_Rank"].values[0]
    rank_dpo = df[df["Stage"] == "Stage2_DPO"]["Avg_Rank"].values[0]
    
    interpretations["Extraction"] = {
        "trend": f"Rank 从 {rank_base:.0f} 降至 {rank_sft:.0f} (SFT) 再降至 {rank_dpo:.0f} (DPO)",
        "risk_increase": f"{(1 - rank_dpo/rank_base) * 100:.1f}%",
        "conclusion": "训练后模型更容易提取 canary，提取风险显著增加"
    }
    
    # PPL 分析
    ppl_base = df[df["Stage"] == "Stage0_Base"]["Canary_PPL"].values[0]
    ppl_sft = df[df["Stage"] == "Stage1_SFT"]["Canary_PPL"].values[0]
    ppl_dpo = df[df["Stage"] == "Stage2_DPO"]["Canary_PPL"].values[0]
    
    interpretations["Perplexity"] = {
        "trend": f"Canary PPL 从 {ppl_base:.0f} 降至 {ppl_sft:.0f} (SFT) 再降至 {ppl_dpo:.0f} (DPO)",
        "memorization_increase": f"{(1 - ppl_dpo/ppl_base) * 100:.1f}%",
        "conclusion": "模型对 canary 的困惑度降低，表明记忆程度增加"
    }
    
    return interpretations


def generate_attribution_report(
    df: pd.DataFrame,
    deltas: dict,
    attribution: dict,
    interpretations: dict,
    output_dir: str = "reports"
) -> dict:
    """生成完整的归因报告"""
    
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
    
    # 关键发现
    findings = []
    
    # 1. 提取风险变化
    rank_change = deltas["Base_to_DPO"]["Avg_Rank"]
    if rank_change < 0:
        findings.append(f"Canary 提取风险增加：平均排名从 {df[df['Stage']=='Stage0_Base']['Avg_Rank'].values[0]:.0f} 降至 {df[df['Stage']=='Stage2_DPO']['Avg_Rank'].values[0]:.0f}")
    
    # 2. SFT vs DPO 贡献
    sft_rank_contrib = attribution["SFT_contribution"]["Avg_Rank"]
    dpo_rank_contrib = attribution["DPO_contribution"]["Avg_Rank"]
    findings.append(f"SFT 阶段贡献了 {sft_rank_contrib:.1f}% 的排名变化，DPO 阶段贡献了 {dpo_rank_contrib:.1f}%")
    
    # 3. 记忆程度
    ppl_change_pct = deltas["Base_to_DPO"]["Canary_PPL_pct"]
    findings.append(f"Canary 困惑度降低 {abs(ppl_change_pct):.1f}%，表明模型记忆程度增加")
    
    # 4. MIA 风险
    mia_change = deltas["Base_to_DPO"]["MIA_Gap"]
    if abs(mia_change) < 0.1:
        findings.append("MIA Gap 变化较小，成员推断攻击风险相对稳定")
    
    report["key_findings"] = findings
    
    # 保存报告
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    with open(output_path / "attribution_summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    return report


def plot_stage_comparison(df: pd.DataFrame, output_dir: str = "reports"):
    """生成阶段对比可视化"""
    
    # 设置字体以支持中文（如果可用）
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
    
    print(f"图表已保存至 {output_path / 'stage_attribution.png'}")


def plot_attribution_breakdown(attribution: dict, output_dir: str = "reports"):
    """生成归因分解图"""
    
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
    
    # 添加数值标签
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
    
    print(f"图表已保存至 {output_path / 'attribution_breakdown.png'}")


def main():
    """主函数"""
    print("=" * 60)
    print("Stage Attribution Analysis")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n1. 加载审计数据...")
    df = load_audit_summary()
    print(df.to_string(index=False))
    
    # 2. 计算阶段变化
    print("\n2. 计算阶段间变化...")
    deltas = compute_stage_deltas(df)
    
    print("\n  Base → SFT 变化:")
    for k, v in deltas["Base_to_SFT"].items():
        if not k.endswith("_pct"):
            print(f"    {k}: {v:+.4f}")
    
    print("\n  SFT → DPO 变化:")
    for k, v in deltas["SFT_to_DPO"].items():
        if not k.endswith("_pct"):
            print(f"    {k}: {v:+.4f}")
    
    # 3. 计算归因分数
    print("\n3. 计算归因分数...")
    attribution = compute_attribution_scores(deltas)
    
    print("\n  SFT 贡献度:")
    for k, v in attribution["SFT_contribution"].items():
        print(f"    {k}: {v:.1f}%")
    
    print("\n  DPO 贡献度:")
    for k, v in attribution["DPO_contribution"].items():
        print(f"    {k}: {v:.1f}%")
    
    # 4. 解读结果
    print("\n4. 结果解读...")
    interpretations = interpret_results(df, deltas)
    
    for category, interp in interpretations.items():
        print(f"\n  [{category}]")
        print(f"    趋势: {interp['trend']}")
        print(f"    结论: {interp['conclusion']}")
    
    # 5. 生成报告
    print("\n5. 生成归因报告...")
    report = generate_attribution_report(df, deltas, attribution, interpretations)
    
    print("\n  关键发现:")
    for i, finding in enumerate(report["key_findings"], 1):
        print(f"    {i}. {finding}")
    
    # 6. 生成可视化
    print("\n6. 生成可视化图表...")
    plot_stage_comparison(df)
    plot_attribution_breakdown(attribution)
    
    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)
    
    return report


if __name__ == "__main__":
    main()
