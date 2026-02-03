# Privacy Audit Pipeline

Auditing Memorization and Privacy Leakage Across Post-Training Pipelines in Large Language Models

## Project Structure

```
privacy-audit/
├── configs/                    # 配置文件
├── data/                       # 数据文件
│   ├── canary_output.txt       # Canary 列表
│   └── wiki_trimmed_with_canary.jsonl
├── doc/                        # 文档
│   └── PROJECT_PLAN.md         # 项目规划
├── models/                     # 模型文件
│   ├── Qwen2.5-0.5B-Instruct/  # Base 模型
│   ├── stage1_sft/             # SFT 后模型
│   └── stage2_dpo/             # DPO 后模型 (待实现)
├── notebooks/                  # Colab Notebooks
│   ├── 01_sft_training.ipynb   # SFT 训练
│   ├── 03_audit_stage0_stage1.ipynb  # 审计
│   └── 04_stress_test.ipynb    # Stress Test
├── reports/                    # 审计报告
├── src/                        # 源代码
│   ├── audit/                  # 审计模块
│   ├── canary.py               # Canary 生成器
│   ├── prepare_data.py         # 数据准备
│   └── train_sft.py            # SFT 训练脚本
└── requirements.txt
```

## Research Question

How does memorization-based privacy risk evolve across post-training stages (SFT / preference optimization), and which commonly used privacy signals become unreliable in these stages?

## Quick Start

```bash
# 安装依赖
pip install -r requirements.txt

# 生成 Canary
python src/canary.py

# 准备数据
python src/prepare_data.py
```

## Training (Colab)

使用 `notebooks/01_sft_training.ipynb` 在 Colab 上进行 SFT 训练。

## License

MIT
