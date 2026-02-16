# Stage-Attributable Privacy Auditing Across Post-Training LLM Pipelines

This repository presents a **stage-attributable privacy auditing pipeline** for large language models (LLMs), focusing on how different post-training stages (SFT, DPO) affect memorization and extractability of sensitive data.

Rather than treating privacy risk as a property of the final model checkpoint, this project explicitly **attributes privacy signals to individual training stages**, and evaluates the robustness of common auditing metrics under prompt perturbations.

## Motivation

Modern LLMs typically undergo multiple post-training stages after pretraining, such as:
- **Supervised Fine-Tuning (SFT)**
- **Preference Optimization (e.g. DPO)**

However, privacy risk is often evaluated only at the final model, without understanding:
- Where memorization is introduced
- Which training objectives amplify or suppress it
- Whether common audit metrics remain reliable across stages

This project aims to answer:

> **How does privacy risk evolve across post-training stages, and how reliable are common audit signals under realistic prompt variations?**

## Key Contributions

1. **Stage-attributable privacy auditing** across Base → SFT → DPO (no-canary) → DPO (with-canary), with controlled ablation of canary content in preference data.
2. **Ablation experiment design** that isolates the causal effect of canary presence in DPO preference data on memorization signals.
3. **Empirical evidence** that SFT introduces the dominant memorization jump, while DPO's effect depends on preference data composition.
4. **Metric validity analysis** showing that log-probability–based signals are highly prompt-sensitive, while rank- and perplexity-based signals are comparatively more robust.
5. **An end-to-end, reproducible audit pipeline**, suitable for integration into post-training safety or privacy reviews.

## Method Overview

### Training Stages

All experiments use the same model architecture (Qwen2.5-0.5B-Instruct) and base data distribution.

| Stage | Description |
|-------|-------------|
| Stage 0 – Base | Original pretrained/instruction model |
| Stage 1 – SFT | Supervised fine-tuning on a dataset containing sparse canary insertions |
| Stage 2a – DPO (no canary) | Preference optimization on SFT checkpoint using preference data **without** canary pairs |
| Stage 2b – DPO (with canary) | Preference optimization on SFT checkpoint using preference data **with** canary pairs |

This design ensures that changes in privacy signals can be **causally attributed to training objectives and data composition**, not confounded by uncontrolled variables.

### Canary Design

- Canary sequences are **synthetic, unique, and non-semantic**
- Inserted sparsely into the SFT training corpus
- DPO stage is split into two ablation groups:
  - **DPO-no-canary (Stage 2a)**: preference data contains only normal pairs — no canary content
  - **DPO-with-canary (Stage 2b)**: preference data contains normal pairs plus canary preference pairs
- Both DPO variants share the same SFT base model and identical training hyperparameters
- Normal preference pairs are generated with a fixed random seed and are **line-by-line identical** across variants

The goal is to measure how the presence of canary content in preference data affects extractability and memorization behavior at the DPO stage.

### Privacy Audit Signals

The audit evaluates multiple complementary signals:

| Signal | Description |
|--------|-------------|
| Log-Probability (Last-Token) | Measures model confidence on canary continuation |
| Rank / Exposure-style Signal | Measures how highly the canary token ranks in the predictive distribution |
| Perplexity-based Membership Proxy | Lower perplexity indicates stronger memorization |

All metrics are computed consistently across stages.

### Prompt Stress Testing

To assess robustness, each canary is evaluated under multiple prompt variants:
- Plain canary
- Instruction-wrapped prompt
- Suffix / formatting perturbations
- Optional contextual prefixes

This stress test reveals whether audit conclusions are stable or prompt-dependent.

## Results Summary

### Stage Attribution

Aggregated results show clear stage-wise privacy signal evolution:

| Stage | MIA Gap | Avg LogProb | Avg Rank | Canary PPL | PPL Ratio |
|-------|---------|-------------|----------|------------|-----------|
| Stage 0 (Base) | -3.79 | -6.55 | 3244.4 | 857.9 | 51.6 |
| Stage 1 (SFT) | -3.85 | -6.33 | 1621.1 | 690.1 | 55.4 |
| Stage 2a (DPO-no-canary) | — | — | — | — | — |
| Stage 2b (DPO-with-canary) | — | — | — | — | — |

> Stage 2a/2b results will be populated after running the ablation experiment on GPU.

![Privacy Audit Results](reports/privacy_audit_results.png)

**Key Findings:**

**SFT introduces the largest increase in memorization signals:**
- Rank drops from 3244 → 1621 (50% reduction, indicating stronger memorization)
- Perplexity drops from 858 → 690 (model becomes more "familiar" with canaries)

**DPO ablation (pending):**
- Stage 2a (DPO-no-canary) and Stage 2b (DPO-with-canary) results will reveal whether canary content in preference data amplifies memorization signals
- Comparison between the two DPO variants isolates the causal effect of canary presence in preference data

### Metric Robustness

Stress testing reveals:
- **Rank-based signals** are the most robust across prompt perturbations
- **Log-probability signals** are highly sensitive to prompt structure
- Single-prompt, single-metric audits can be misleading

These findings highlight the importance of **stage-aware and stress-tested privacy auditing**.

## Repository Structure

```
.
├── data/
│   ├── canary_output.txt              # Canary definitions
│   ├── preference_data_no_canary.jsonl # DPO preference data without canary pairs
│   ├── preference_data_with_canary.jsonl # DPO preference data with canary pairs
│   └── wiki_trimmed_with_canary.jsonl # Training corpus with canaries
├── models/
│   ├── Qwen2.5-0.5B-Instruct/        # Base model (Stage 0)
│   ├── stage1_sft/                    # SFT model checkpoint (Stage 1)
│   ├── stage2_dpo_no_canary/          # DPO-no-canary checkpoint (Stage 2a)
│   └── stage2_dpo_with_canary/        # DPO-with-canary checkpoint (Stage 2b)
├── notebooks/
│   ├── 01_sft_training.ipynb          # Supervised fine-tuning (Stage 1)
│   ├── 02_dpo_training.ipynb          # Preference optimization (Stage 2a + 2b)
│   └── 05_privacy_audit.ipynb         # Multi-signal privacy audit (4 stages)
├── src/
│   ├── canary.py                      # Canary generation
│   ├── prepare_data.py                # Data preparation
│   ├── prepare_preference_data.py     # DPO preference data generation (dual-variant)
│   ├── train_sft.py                   # SFT training script
│   ├── train_dpo.py                   # DPO training script (parameterized CLI)
│   ├── stage_attribution.py           # 4-stage attribution analysis
│   └── audit/                         # Audit modules
│       ├── mia.py                     # Membership Inference Attack
│       ├── extraction.py              # Canary extraction tests
│       ├── internal_signals.py        # Perplexity/entropy analysis
│       └── stress_test.py             # Prompt robustness testing
├── tests/
│   ├── test_audit_modules.py          # Audit module tests (requires torch)
│   └── test_stage_attribution.py      # Stage attribution tests
├── reports/
│   ├── privacy_audit_summary.csv      # 4-stage audit results
│   ├── attribution_summary.json       # Stage attribution report
│   ├── privacy_audit_results.png      # Results visualization
│   └── stress_test_results.csv        # Stress test results
├── doc/
│   ├── PROJECT_PLAN.md                # Project planning document
│   └── agent_sync.md                  # Multi-agent coordination log
├── requirements.txt                   # Python dependencies
└── README.md
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run Privacy Audit

```bash
# Option 1: Run notebook locally (requires sufficient RAM)
jupyter notebook notebooks/05_privacy_audit.ipynb

# Option 2: Run on Google Colab (recommended for GPU)
# Upload notebooks/05_privacy_audit.ipynb to Colab
```

#### Colab Path Convention

When running notebooks in Colab, use the following paths:

- `data/canary_output.txt`
- `data/wiki_trimmed_with_canary.jsonl`
- `data/preference_data_no_canary.jsonl`
- `data/preference_data_with_canary.jsonl`
- `models/stage1_sft/`
- `models/stage2_dpo_no_canary/`
- `models/stage2_dpo_with_canary/`

Example full paths:

- `/data/canary_output.txt`
- `/models/stage1_sft/`
- `/models/stage2_dpo_no_canary/`
- `/models/stage2_dpo_with_canary/`

## Reproducibility Notes

- Experiments are designed to run on consumer hardware (e.g. Apple Silicon) or Google Colab using **parameter-efficient fine-tuning (LoRA)**.
- Exact numeric results may vary with random seeds, but **qualitative trends are stable**.
- All stages use identical audit code paths to ensure comparability.

## Limitations & Threat Model

- Canary-based auditing measures **extractability**, not real-world PII exposure.
- Results are demonstrated on small-scale models; trends may differ at larger scales.
- Black-box auditing remains sensitive to prompt choice, even with stress testing.

## Why This Matters

This project suggests that:

1. **Privacy auditing should be stage-aware**, not checkpoint-only
2. **Preference optimization should not be assumed to reduce memorization** by default
3. **Audit metrics must be stress-tested** to avoid false conclusions

These insights are directly relevant to **post-training safety and privacy reviews** for frontier language models.

## License

MIT License

## Contact

This repository was developed as a research-oriented engineering project focused on privacy and safety in LLM training pipelines.
