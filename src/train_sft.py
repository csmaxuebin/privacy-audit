import os
import sys

# Fix OpenMP library conflict on macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Ensure output is not buffered
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

print("=" * 60, flush=True)
print("Starting Privacy Audit Training Script", flush=True)
print("=" * 60, flush=True)

import torch
# 设置 PyTorch 使用多线程 (你有 12 核心)
torch.set_num_threads(10)
print(f"[OK] PyTorch threads set to: {torch.get_num_threads()}", flush=True)

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

print("[OK] All imports successful!", flush=True)

# ----------------------------------
# 模型与数据设置
# ----------------------------------
model_name = "models/Qwen2.5-0.5B-Instruct"
train_data_file = "data/wiki_trimmed_with_canary.jsonl"  # 你已经准备好的训练语料

output_dir = "models/stage1_sft"

# ----------------------------------
# 1) 加载 Tokenizer 与 Model
# ----------------------------------
print("\n[INFO] Loading tokenizer and base model...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f"[OK] Tokenizer loaded. Vocab size: {len(tokenizer)}", flush=True)

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
print(f"[OK] Model loaded successfully!", flush=True)

# ----------------------------------
# 2) 加载你的训练集
# ----------------------------------
print("\n[INFO] Loading training dataset...", flush=True)
train_dataset = load_dataset("json", data_files=train_data_file, split="train")
print(f"[OK] Dataset loaded. Number of examples: {len(train_dataset)}", flush=True)

# ----------------------------------
# 3) PEFT/LoRA 配置
# ----------------------------------
print("\n[INFO] Configuring LoRA/PEFT...", flush=True)
lora_config = LoraConfig(
    r=32, lora_alpha=16, lora_dropout=0.05,
    bias="none", task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
print("[OK] LoRA configuration applied!", flush=True)
model.print_trainable_parameters()

# ----------------------------------
# 4) SFT 训练设置 (CPU 优化)
# ----------------------------------
print("\n[INFO] Setting up SFT Trainer with CPU optimization...", flush=True)
training_args = SFTConfig(
    learning_rate=2e-4,
    num_train_epochs=1,
    per_device_train_batch_size=4,        # 减小 batch size 以适应内存
    gradient_accumulation_steps=4,        # 有效 batch size = 16
    output_dir=output_dir,
    logging_steps=50,
    # 优化参数
    dataloader_num_workers=4,
    dataloader_pin_memory=False,
    dataloader_prefetch_factor=2,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,  # Updated from deprecated 'tokenizer' parameter
)
print("[OK] Trainer initialized successfully!", flush=True)

# ----------------------------------
# 5) 启动训练
# ----------------------------------
print("\n" + "=" * 60, flush=True)
print("[INFO] Starting fine-tuning...", flush=True)
print("=" * 60, flush=True)
trainer.train()

# ----------------------------------
# 6) 保存 checkpoint
# ----------------------------------
print("\n[INFO] Saving trained model to disk...", flush=True)
trainer.save_model(output_dir)
print("[DONE] Fine-tuning complete!", flush=True)
print("=" * 60, flush=True)
