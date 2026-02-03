# Privacy Audit Pipeline 项目规划

## 项目概述

**目标**：构建一个 Stage-attributable Privacy Audit Pipeline，评估 Qwen2.5-0.5B 在不同训练阶段的隐私风险演变。

**核心研究问题**：
- 隐私风险在 Base → SFT → Preference Optimization 各阶段如何变化？
- 哪些常用隐私指标在 post-training 后失效？

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Privacy Audit Pipeline                    │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────┐    ┌──────────┐    ┌──────────┐               │
│  │  Stage 0 │───▶│  Stage 1 │───▶│  Stage 2 │               │
│  │   Base   │    │   SFT    │    │   DPO    │               │
│  └────┬─────┘    └────┬─────┘    └────┬─────┘               │
│       │               │               │                      │
│       ▼               ▼               ▼                      │
│  ┌─────────────────────────────────────────┐                │
│  │           Auditing Suite                 │                │
│  │  ┌─────┐  ┌─────────┐  ┌──────────┐    │                │
│  │  │ MIA │  │ Canary  │  │ Internal │    │                │
│  │  │     │  │Extract  │  │ Signals  │    │                │
│  │  └─────┘  └─────────┘  └──────────┘    │                │
│  └─────────────────────────────────────────┘                │
│                      │                                       │
│                      ▼                                       │
│  ┌─────────────────────────────────────────┐                │
│  │         Stage Attribution Report         │                │
│  └─────────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

---

## 项目文件结构

```
privacy-audit/
├── data/
│   ├── canary_output.txt
│   ├── wiki_trimmed_with_canary.jsonl
│   └── preference_data.jsonl       # DPO 偏好数据
├── models/
│   ├── Qwen2.5-0.5B-Instruct/
│   ├── stage1_sft/
│   └── stage2_dpo/
├── src/
│   ├── canary.py
│   ├── prepare_data.py
│   ├── train_sft.py
│   ├── train_dpo.py
│   └── audit/
│       ├── mia.py
│       ├── extraction.py
│       ├── internal_signals.py
│       └── stress_test.py
├── notebooks/
│   ├── 01_sft_training.ipynb       # Colab
│   ├── 02_dpo_training.ipynb       # Colab
│   ├── 03_audit_stage0.ipynb
│   ├── 04_audit_stage1.ipynb
│   ├── 05_audit_stage2.ipynb
│   └── 06_stage_attribution.ipynb
├── reports/
│   ├── stage0_report.json
│   ├── stage1_report.json
│   ├── stage2_report.json
│   └── attribution_summary.json
└── configs/
    ├── sft_config.yaml
    ├── dpo_config.yaml
    └── audit_config.yaml
```

---

## 8 周执行计划

### Phase 1: 基础设施 (Week 1-2)
| Week | 任务 | 交付物 | 环境 |
|------|------|--------|------|
| 1 | 项目结构搭建 | 目录 + configs | 本地 |
| 1 | DPO 偏好数据准备 | preference_data.jsonl | 本地 |
| 2 | DPO 训练脚本 | train_dpo.py | 本地 |
| 2 | DPO 训练执行 | stage2_dpo/ | **Colab** |

### Phase 2: Auditing Suite (Week 3-4)
| Week | 任务 | 交付物 | 环境 |
|------|------|--------|------|
| 3 | MIA 模块 | audit/mia.py | 本地 |
| 3 | Canary Extraction | audit/extraction.py | 本地 |
| 4 | Internal Signals | audit/internal_signals.py | 本地 |
| 4 | Stress Test 框架 | audit/stress_test.py | 本地 |

### Phase 3: 审计执行 (Week 5-6)
| Week | 任务 | 交付物 | 环境 |
|------|------|--------|------|
| 5 | Stage 0/1 审计 | stage0/1_report.json | Colab |
| 6 | Stage 2 审计 | stage2_report.json | Colab |
| 6 | Stress Test | 27 实验点 | Colab |

### Phase 4: 分析与报告 (Week 7-8)
| Week | 任务 | 交付物 | 环境 |
|------|------|--------|------|
| 7 | Stage Attribution | attribution_summary.json | 本地 |
| 7 | 可视化 | plots/ | 本地 |
| 8 | Technical Write-up | 5-8 页报告 | 本地 |

---

## 实验矩阵 (27 实验点)

### 维度
- **Stages (3)**: S0 (Base), S1 (SFT), S2 (DPO)
- **Metrics (3)**: MIA, Canary Extraction, Internal Signals
- **Variants (3)**: Prompt Template, Decoding Strategy, Canary Difficulty

### 矩阵
```
         │  M1 (MIA)  │  M2 (Extract)  │  M3 (Internal)
─────────┼────────────┼────────────────┼────────────────
S0-V1/2/3│    ✓       │       ✓        │       ✓
S1-V1/2/3│    ✓       │       ✓        │       ✓
S2-V1/2/3│    ✓       │       ✓        │       ✓
```

---

## 技术栈

| 组件 | 工具 |
|------|------|
| 模型 | Qwen2.5-0.5B-Instruct |
| 训练 | transformers, trl, peft |
| 审计 | 自实现 + privacy_meter |
| 可视化 | matplotlib, seaborn |
| 训练环境 | Google Colab (GPU) |
| 开发环境 | 本地 macOS |

---

## 当前进度

- [x] Canary Generator (canary.py)
- [x] 数据准备 (prepare_wikipedia_with_canary.py)
- [x] SFT 训练 (training.py)
- [ ] DPO 偏好数据准备
- [ ] DPO 训练脚本
- [ ] Auditing Suite
- [ ] Stage Attribution 分析
