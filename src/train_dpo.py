#!/usr/bin/env python3
"""
train_dpo.py

DPO (Direct Preference Optimization) training on top of an SFT model.
Stage 2 training to observe how preference optimization affects privacy risk.

Usage (ablation experiment):
  # DPO-no-canary
  python src/train_dpo.py \
    --preference-data data/preference_data_no_canary.jsonl \
    --output-dir models/stage2_dpo_no_canary

  # DPO-with-canary
  python src/train_dpo.py \
    --preference-data data/preference_data_with_canary.jsonl \
    --output-dir models/stage2_dpo_with_canary
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

# Fix OpenMP library conflict on macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# ---------- CLI ----------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DPO training with parameterized data path and output directory"
    )
    parser.add_argument(
        "--preference-data", type=str, required=True,
        help="Path to preference data JSONL file (required)"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for trained model (required)"
    )
    parser.add_argument(
        "--sft-model", type=str, default="models/stage1_sft",
        help="Path to SFT model directory (default: models/stage1_sft)"
    )
    parser.add_argument(
        "--base-model", type=str, default="models/Qwen2.5-0.5B-Instruct",
        help="Path to base model directory (default: models/Qwen2.5-0.5B-Instruct)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    return parser.parse_args(argv)


def validate_inputs(args: argparse.Namespace) -> None:
    """Validate that required input files and directories exist."""
    if not Path(args.preference_data).exists():
        print(f"[ERROR] Preference data file not found: {args.preference_data}", file=sys.stderr)
        sys.exit(1)
    if not Path(args.sft_model).is_dir():
        print(f"[ERROR] SFT model directory not found: {args.sft_model}", file=sys.stderr)
        sys.exit(1)
    # Allow HuggingFace model IDs (e.g. "Qwen/Qwen2.5-0.5B-Instruct")
    # Only validate as local directory if it looks like a local path
    if not args.base_model.startswith((".", "/", "~")) and "/" in args.base_model:
        print(f"[INFO] Treating --base-model as HuggingFace model ID: {args.base_model}")
    elif not Path(args.base_model).is_dir():
        print(f"[ERROR] Base model directory not found: {args.base_model}", file=sys.stderr)
        sys.exit(1)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    print("=" * 60)
    print("Privacy Audit - DPO Training (Stage 2)")
    print("=" * 60)

    # Validate inputs before importing heavy dependencies
    validate_inputs(args)

    # Set random seed for reproducibility
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)

    import torch
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    from trl import DPOTrainer, DPOConfig

    def count_params(m):
        total = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        return total, trainable

    # ----------------------------------
    # 1) Load Tokenizer
    # ----------------------------------
    print("\n[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[OK] Tokenizer loaded. Vocab size: {len(tokenizer)}")

    # ----------------------------------
    # 2) Load SFT model (Stage 1 output)
    # ----------------------------------
    print("\n[INFO] Loading SFT model (Stage 1)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    # Important: load adapter in trainable mode for Stage-2 optimization.
    model = PeftModel.from_pretrained(base_model, args.sft_model, is_trainable=True)
    # Disable gradient checkpointing to avoid tensor metadata mismatch
    # during DPO training (known PEFT + DPO + checkpoint incompatibility)
    model.gradient_checkpointing_disable()
    # Re-enable input require grads for PEFT LoRA layers
    model.enable_input_require_grads()
    total_params, trainable_params = count_params(model)
    print(
        f"[INFO] Model params: total={total_params:,}, "
        f"trainable={trainable_params:,} "
        f"({trainable_params / total_params * 100:.4f}%)"
    )
    if trainable_params == 0:
        print(
            "[ERROR] No trainable parameters detected. "
            "Expected LoRA adapter params to be trainable for DPO.",
            file=sys.stderr,
        )
        sys.exit(1)
    print("[OK] SFT model loaded!")

    # ----------------------------------
    # 3) Load preference data
    # ----------------------------------
    print(f"\n[INFO] Loading preference dataset from {args.preference_data}...")
    dataset = load_dataset("json", data_files=args.preference_data, split="train")
    print(f"[OK] Dataset loaded. Number of examples: {len(dataset)}")
    print(f"[INFO] Sample data: {dataset[0]}")

    # ----------------------------------
    # 4) DPO config (hyperparams are fixed for both variants)
    # ----------------------------------
    print("\n[INFO] Configuring DPO Trainer...")
    dpo_config = DPOConfig(
        learning_rate=5e-5,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        output_dir=args.output_dir,
        logging_steps=10,
        save_steps=100,
        beta=0.1,
        max_length=512,
        max_prompt_length=256,
        remove_unused_columns=False,
        gradient_checkpointing=False,
        bf16=True,
    )

    # ----------------------------------
    # 5) Create DPO Trainer
    # ----------------------------------
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # TRL auto-creates frozen copy from model's initial state
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    print("[OK] DPO Trainer initialized!")

    # ----------------------------------
    # 6) Train
    # ----------------------------------
    print("\n" + "=" * 60)
    print(f"[INFO] Starting DPO training...")
    print(f"  preference-data: {args.preference_data}")
    print(f"  output-dir:      {args.output_dir}")
    print(f"  sft-model:       {args.sft_model}")
    print(f"  base-model:      {args.base_model}")
    print(f"  seed:            {args.seed}")
    print("=" * 60)
    trainer.train()

    # ----------------------------------
    # 7) Save model
    # ----------------------------------
    print(f"\n[INFO] Saving DPO model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[DONE] DPO model saved to {args.output_dir}")

    # Record run metadata
    from run_metadata import append_metadata
    append_metadata({
        "type": "dpo_training",
        "seed": args.seed,
        "model_path": args.output_dir,
        "preference_data": args.preference_data,
        "sft_model": args.sft_model,
        "base_model": args.base_model,
        "hyperparams": {
            "learning_rate": 5e-5,
            "num_train_epochs": 1,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 8,
            "beta": 0.1,
            "max_length": 512,
            "max_prompt_length": 256,
        },
    })
    print("[INFO] Run metadata recorded to reports/run_metadata.jsonl")
    print("=" * 60)


if __name__ == "__main__":
    main()
