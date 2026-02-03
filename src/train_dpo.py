#!/usr/bin/env python3
"""
train_dpo.py

使用 DPO (Direct Preference Optimization) 对 SFT 模型进行偏好优化。
这是 Stage 2 训练，用于观察偏好优化对隐私风险的影响。

输入：
  - SFT 模型 (models/stage1_sft/)
  - 偏好数据 (data/preference_data.jsonl)

输出：
  - DPO 模型 (models/stage2_dpo/)
"""

import os
import sys

# Fix OpenMP library conflict on macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

print("=" * 60)
print("Privacy Audit - DPO Training (Stage 2)")
print("=" * 60)

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
from trl import DPOTrainer, DPOConfig

# ----------------------------------
# 配置
# ----------------------------------
BASE_MODEL_NAME = "models/Qwen2.5-0.5B-Instruct"
SFT_MODEL_DIR = "models/stage1_sft"
PREFERENCE_DATA = "data/preference_data.jsonl"
OUTPUT_DIR = "models/stage2_dpo"

# ----------------------------------
# 1) 加载 Tokenizer
# ----------------------------------
print("\n[INFO] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"[OK] Tokenizer loaded. Vocab size: {len(tokenizer)}")

# ----------------------------------
# 2) 加载 SFT 模型 (Stage 1 输出)
# ----------------------------------
print("\n[INFO] Loading SFT model (Stage 1)...")
# 先加载 base model
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16
)
# 加载 LoRA adapter
model = PeftModel.from_pretrained(base_model, SFT_MODEL_DIR)
print("[OK] SFT model loaded!")

# ----------------------------------
# 3) 加载偏好数据
# ----------------------------------
print("\n[INFO] Loading preference dataset...")
dataset = load_dataset("json", data_files=PREFERENCE_DATA, split="train")
print(f"[OK] Dataset loaded. Number of examples: {len(dataset)}")

# 数据格式验证
print(f"[INFO] Sample data: {dataset[0]}")

# ----------------------------------
# 4) DPO 配置
# ----------------------------------
print("\n[INFO] Configuring DPO Trainer...")

# 为 DPO 添加新的 LoRA adapter
# 注意：DPO 需要 reference model，这里使用 SFT model 作为 reference
dpo_config = DPOConfig(
    learning_rate=5e-5,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    output_dir=OUTPUT_DIR,
    logging_steps=10,
    save_steps=100,
    beta=0.1,  # DPO temperature
    max_length=512,
    max_prompt_length=256,
    remove_unused_columns=False,
)

# ----------------------------------
# 5) 创建 DPO Trainer
# ----------------------------------
trainer = DPOTrainer(
    model=model,
    ref_model=None,  # 使用 model 的初始状态作为 reference
    args=dpo_config,
    train_dataset=dataset,
    processing_class=tokenizer,
)
print("[OK] DPO Trainer initialized!")

# ----------------------------------
# 6) 开始训练
# ----------------------------------
print("\n" + "=" * 60)
print("[INFO] Starting DPO training (Stage 2)...")
print("=" * 60)
trainer.train()

# ----------------------------------
# 7) 保存模型
# ----------------------------------
print("\n[INFO] Saving DPO model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"[DONE] DPO model saved to {OUTPUT_DIR}")
print("=" * 60)
