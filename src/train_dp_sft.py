"""DP-SFT training script with differential privacy support.

Supports two training modes:
- epsilon=inf: Standard SFT training (non-DP control), reusing existing
  LoRA configuration and training hyperparameters.
- epsilon=finite: DP-SGD training via Opacus (Task 3.2) or manual fallback
  (Task 3.3).

Usage:
    # Non-DP control (epsilon=inf)
    python src/train_dp_sft.py --epsilon inf --delta 0.0001 \\
        --output-dir models/dp_sft_eps_inf_seed42 \\
        --training-data data/wiki_trimmed_with_canary_50.jsonl \\
        --base-model models/Qwen2.5-0.5B-Instruct

    # DP-SGD training via Opacus
    python src/train_dp_sft.py --epsilon 8 --delta 0.0001 \\
        --clipping-norm 1.0 --output-dir models/dp_sft_eps8_seed42 \\
        --training-data data/wiki_trimmed_with_canary_50.jsonl \\
        --base-model models/Qwen2.5-0.5B-Instruct
"""

import argparse
import datetime
import json
import math
import os
import sys
from typing import Optional

import numpy as np


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    """Parse CLI arguments for DP-SFT training.

    Returns:
        Namespace with epsilon (str), delta (float), clipping_norm (float),
        seed (int), output_dir (str), training_data (str), base_model (str),
        accountant_type (str), fallback (bool), calibrate_clipping (bool).
    """
    parser = argparse.ArgumentParser(
        description="DP-SFT training with differential privacy support"
    )
    parser.add_argument(
        "--epsilon", type=str, required=True,
        help="Target epsilon (use 'inf' for non-DP control)"
    )
    parser.add_argument(
        "--delta", type=float, required=True,
        help="Privacy failure probability (typically 1/N)"
    )
    parser.add_argument(
        "--clipping-norm", type=float, default=1.0,
        help="Per-sample gradient clipping norm (default: 1.0)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory to save trained model"
    )
    parser.add_argument(
        "--training-data", type=str,
        default="data/wiki_trimmed_with_canary_50.jsonl",
        help="Path to training data JSONL file"
    )
    parser.add_argument(
        "--base-model", type=str,
        default="models/Qwen2.5-0.5B-Instruct",
        help="Path to base model"
    )
    parser.add_argument(
        "--accountant-type", type=str, default="rdp",
        choices=["rdp", "prv"],
        help="Privacy accountant type (default: rdp)"
    )
    parser.add_argument(
        "--fallback", action="store_true",
        help="Use manual DP if Opacus validation fails"
    )
    parser.add_argument(
        "--calibrate-clipping", action="store_true",
        help="Run gradient norm calibration then exit"
    )
    return parser.parse_args(argv)


def parse_epsilon(epsilon_str: str) -> float:
    """Convert epsilon string to float, supporting 'inf'.

    Returns:
        float: The epsilon value (math.inf for 'inf').

    Raises:
        SystemExit: If epsilon is not a valid positive number or 'inf'.
    """
    if epsilon_str.lower() == "inf":
        return math.inf
    try:
        val = float(epsilon_str)
    except ValueError:
        print(
            f"Error: --epsilon must be a positive number or 'inf', "
            f"got '{epsilon_str}'",
            file=sys.stderr,
        )
        sys.exit(1)
    if val <= 0:
        print(
            f"Error: --epsilon must be positive, got {val}",
            file=sys.stderr,
        )
        sys.exit(1)
    return val


def validate_inputs(args: argparse.Namespace) -> None:
    """Validate that input files and directories exist.

    Checks:
    - training_data file exists
    - base_model directory exists

    Raises:
        SystemExit: If any required input is missing.
    """
    if not os.path.isfile(args.training_data):
        print(
            f"Error: Training data file not found: {args.training_data}",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.path.isdir(args.base_model):
        print(
            f"Error: Base model directory not found: {args.base_model}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Check that model weights exist in the base model directory
    weight_files = ["model.safetensors", "pytorch_model.bin"]
    if not any(os.path.isfile(os.path.join(args.base_model, w)) for w in weight_files):
        print(
            f"Error: No model weights found in {args.base_model}. "
            f"Expected one of: {', '.join(weight_files)}. "
            f"Download the model first, e.g.: "
            f"huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir {args.base_model}",
            file=sys.stderr,
        )
        sys.exit(1)


def append_metadata(metadata: dict, filepath: str = "reports/run_metadata.jsonl") -> None:
    """Append one JSON line of run metadata to a JSONL file.

    Creates the parent directory if it doesn't exist.

    Args:
        metadata: Dictionary of run metadata fields.
        filepath: Path to the JSONL file (default: reports/run_metadata.jsonl).
    """
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(metadata, ensure_ascii=False) + "\n")


def _build_metadata(
    args: argparse.Namespace,
    *,
    dp_method: str,
    target_epsilon,
    actual_epsilon,
    noise_multiplier: float,
    n_for_delta: int,
    status: str = "completed",
    error: Optional[str] = None,
) -> dict:
    """Build a metadata dict with DP-specific and standard fields.

    Args:
        args: Parsed CLI arguments.
        dp_method: "opacus", "manual_lora_dp", or "none".
        target_epsilon: Target epsilon (float or string "inf").
        actual_epsilon: Actual epsilon (float or string "inf").
        noise_multiplier: Noise multiplier used.
        n_for_delta: Number of training samples (for delta=1/N).
        status: "completed" or "failed".
        error: Error message string (when status="failed").

    Returns:
        dict ready for append_metadata().
    """
    # Handle infinity for JSON serialization
    if isinstance(target_epsilon, float) and math.isinf(target_epsilon):
        target_epsilon = "inf"
    if isinstance(actual_epsilon, float) and math.isinf(actual_epsilon):
        actual_epsilon = "inf"

    meta = {
        "type": "dp_sft_training",
        "seed": args.seed,
        "model_path": args.output_dir,
        "training_data": args.training_data,
        "base_model": args.base_model,
        "target_epsilon": target_epsilon,
        "actual_epsilon": actual_epsilon,
        "delta": args.delta,
        "N_for_delta": n_for_delta,
        "clipping_norm": args.clipping_norm,
        "noise_multiplier": noise_multiplier,
        "accountant_type": args.accountant_type,
        "dp_method": dp_method,
        "status": status,
        "hyperparams": {
            "learning_rate": 2e-4,
            "num_train_epochs": 1,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "lora_r": 32,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
        },
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    if status == "failed" and error is not None:
        meta["error"] = error
    return meta


def train_standard_sft(args: argparse.Namespace) -> None:
    """Run standard SFT training (epsilon=inf, no DP noise).

    Reuses the same LoRA configuration and training hyperparameters
    as the existing train_sft.py:
    - LoRA: r=32, lora_alpha=16, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM"
    - Training: learning_rate=2e-4, num_train_epochs=1,
                per_device_train_batch_size=4,
                gradient_accumulation_steps=4 (effective batch=16)
    """
    # Defer heavy imports to avoid slow startup for --help / validation errors
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    # Reproducibility
    torch.manual_seed(args.seed)

    print(f"[INFO] Loading tokenizer from {args.base_model}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print(f"[INFO] Loading base model from {args.base_model}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, device_map="auto"
    )

    print(f"[INFO] Loading training data from {args.training_data}...",
          flush=True)
    train_dataset = load_dataset(
        "json", data_files=args.training_data, split="train"
    )
    print(f"[INFO] Dataset loaded: {len(train_dataset)} examples", flush=True)

    # LoRA configuration (same as existing SFT)
    lora_config = LoraConfig(
        r=32,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training configuration (same as existing SFT)
    training_args = SFTConfig(
        learning_rate=2e-4,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,  # effective batch size = 16
        output_dir=args.output_dir,
        logging_steps=50,
        seed=args.seed,
        dataloader_num_workers=4,
        dataloader_pin_memory=False,
        dataloader_prefetch_factor=2,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    print("=" * 60, flush=True)
    print(
        f"[INFO] Starting standard SFT training (epsilon=inf, seed={args.seed})",
        flush=True,
    )
    print("=" * 60, flush=True)
    trainer.train()

    print(f"[INFO] Saving model to {args.output_dir}...", flush=True)
    trainer.save_model(args.output_dir)

    # Record run metadata (Requirement 7.1, 7.2, 7.3)
    metadata = _build_metadata(
        args,
        dp_method="none",
        target_epsilon="inf",
        actual_epsilon="inf",
        noise_multiplier=0.0,
        n_for_delta=len(train_dataset),
    )
    append_metadata(metadata)
    print("[INFO] Metadata appended to reports/run_metadata.jsonl", flush=True)

    print("[DONE] Standard SFT training complete.", flush=True)

def validate_opacus_compatibility(model) -> list:
    """Validate that the model is compatible with Opacus DP-SGD.

    Calls ModuleValidator.validate(model, strict=False) to check for
    incompatible modules (e.g. BatchNorm, certain custom layers).

    Args:
        model: The PyTorch model (with LoRA applied) to validate.

    Returns:
        List of error strings. Empty list means compatible.
    """
    from opacus.validators import ModuleValidator

    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        print(
            f"[WARN] Opacus validation found {len(errors)} issue(s).",
            flush=True,
        )
    else:
        print("[INFO] Opacus compatibility check passed.", flush=True)
    return errors

def manual_dp_step(model, optimizer, clipping_norm, noise_multiplier):
    """Manual batch-level clipping + Gaussian noise for LoRA params only.

    WARNING: This is NOT per-sample DP-SGD. The reported epsilon is an
    optimistic upper bound, not a formal guarantee. Use the Opacus path
    for rigorous DP guarantees.

    Args:
        model: The model (with LoRA applied). Only LoRA parameters are clipped/noised.
        optimizer: The optimizer to step after noise injection.
        clipping_norm: Max gradient norm for batch-level clipping.
        noise_multiplier: Noise scale factor (noise_std = noise_multiplier * clipping_norm).
    """
    import torch

    lora_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and "lora" in n.lower()
    ]

    # Batch-level gradient clipping (not per-sample)
    torch.nn.utils.clip_grad_norm_(lora_params, clipping_norm)

    # Add Gaussian noise calibrated to clipping_norm
    for p in lora_params:
        if p.grad is not None:
            noise = torch.randn_like(p.grad) * noise_multiplier * clipping_norm
            p.grad += noise

    optimizer.step()
    optimizer.zero_grad()


def _create_accountant(accountant_type):
    """Create a privacy accountant based on --accountant-type.

    Args:
        accountant_type: "rdp" or "prv".

    Returns:
        An Opacus accountant instance (RDPAccountant or PRVAccountant).
    """
    if accountant_type == "prv":
        try:
            from opacus.accountants import PRVAccountant
            return PRVAccountant()
        except ImportError:
            print(
                "[WARNING] PRVAccountant not available in this Opacus version. "
                "Falling back to RDPAccountant.",
                flush=True,
            )
            from opacus.accountants import RDPAccountant
            return RDPAccountant()
    else:
        from opacus.accountants import RDPAccountant
        return RDPAccountant()


def _compute_noise_multiplier(target_epsilon, delta, sample_rate, num_steps, accountant_type="rdp"):
    """Binary-search for the noise_multiplier that yields target_epsilon.

    Uses the same accountant logic as Opacus to find the noise_multiplier
    that achieves the desired (epsilon, delta) budget over the given number
    of composition steps.

    Args:
        target_epsilon: Desired epsilon budget.
        delta: Privacy failure probability.
        sample_rate: Batch size / dataset size (Poisson sampling rate).
        num_steps: Total number of training steps.
        accountant_type: "rdp" or "prv" (must match training accountant).

    Returns:
        float: The noise_multiplier to use.
    """
    accountant_cls = None
    if accountant_type == "prv":
        try:
            from opacus.accountants import PRVAccountant
            accountant_cls = PRVAccountant
        except ImportError:
            from opacus.accountants import RDPAccountant
            accountant_cls = RDPAccountant
    else:
        from opacus.accountants import RDPAccountant
        accountant_cls = RDPAccountant

    low, high = 0.01, 100.0
    # Binary search for noise_multiplier
    for _ in range(64):
        mid = (low + high) / 2.0
        accountant = accountant_cls()
        accountant.history = [(mid, sample_rate, num_steps)]
        try:
            eps = accountant.get_epsilon(delta)
        except Exception:
            # If epsilon computation fails, noise is too low
            low = mid
            continue
        if eps > target_epsilon:
            low = mid
        else:
            high = mid
    return high


def _is_opacus_torch_dtype_compat_error(exc: Exception) -> bool:
    """Detect known opacus/torch dtype compatibility failures.

    These failures often look like:
    "zeros() received an invalid combination of arguments ... dtype=type".
    """
    if not isinstance(exc, TypeError):
        return False
    msg = str(exc)
    return "zeros()" in msg or "dtype" in msg


def train_manual_dp(args, epsilon):
    """Run manual DP training (fallback when Opacus validation fails).

    Uses batch-level gradient clipping + Gaussian noise on LoRA parameters.
    This does NOT provide strict (epsilon, delta)-DP guarantees.
    The reported epsilon is diagnostic only.

    Args:
        args: Parsed CLI arguments.
        epsilon: Target epsilon (finite, positive float).
    """
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 60, flush=True)
    print(
        "[WARNING] Using manual DP fallback (batch-level clipping). "
        "Reported epsilon is diagnostic only, NOT a formal "
        "(ε,δ)-DP guarantee.",
        flush=True,
    )
    print("=" * 60, flush=True)

    # --- Reproducibility ---
    torch.manual_seed(args.seed)

    # --- Load tokenizer ---
    print(f"[INFO] Loading tokenizer from {args.base_model}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Load base model ---
    print(f"[INFO] Loading base model from {args.base_model}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float32,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # --- Apply LoRA (same config as standard SFT) ---
    lora_config = LoraConfig(
        r=32,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Load and tokenize dataset ---
    print(
        f"[INFO] Loading training data from {args.training_data}...",
        flush=True,
    )
    raw_dataset = load_dataset(
        "json", data_files=args.training_data, split="train"
    )
    print(f"[INFO] Dataset loaded: {len(raw_dataset)} examples", flush=True)

    max_length = 512

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenized = raw_dataset.map(
        tokenize_fn, batched=True, remove_columns=raw_dataset.column_names
    )
    tokenized.set_format("torch")

    # --- Build DataLoader ---
    batch_size = 4

    def collate_fn(batch):
        input_ids = torch.stack([b["input_ids"] for b in batch])
        attention_mask = torch.stack([b["attention_mask"] for b in batch])
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    dataloader = DataLoader(
        tokenized,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    # --- Compute noise_multiplier via binary search ---
    num_epochs = 1
    dataset_size = len(tokenized)
    sample_rate = batch_size / dataset_size
    num_steps = (dataset_size // batch_size) * num_epochs

    noise_multiplier = _compute_noise_multiplier(
        target_epsilon=epsilon,
        delta=args.delta,
        sample_rate=sample_rate,
        num_steps=num_steps,
        accountant_type=args.accountant_type,
    )
    print(
        f"[INFO] Manual DP: noise_multiplier={noise_multiplier:.4f} "
        f"(target_ε={epsilon}, δ={args.delta}, "
        f"sample_rate={sample_rate:.6f}, steps={num_steps})",
        flush=True,
    )

    # --- Create privacy accountant (matches --accountant-type) ---
    accountant = _create_accountant(args.accountant_type)

    # --- Optimizer (only LoRA params) ---
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=2e-4,
    )

    # --- Training loop ---
    print("=" * 60, flush=True)
    print(
        f"[INFO] Starting manual DP training "
        f"(dp_method=manual_lora_dp, target_ε={epsilon}, δ={args.delta}, "
        f"C={args.clipping_norm}, seed={args.seed})",
        flush=True,
    )
    print("=" * 60, flush=True)

    model.train()
    total_steps = 0
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad()

            # Forward + backward
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            loss = outputs.loss
            loss.backward()

            # Manual DP step: clip + noise + optimizer.step
            manual_dp_step(model, optimizer, args.clipping_norm, noise_multiplier)

            # Record step in accountant
            accountant.history.append(
                (noise_multiplier, sample_rate, 1)
            )

            epoch_loss += loss.item()
            total_steps += 1

            if total_steps % 50 == 0:
                current_eps = accountant.get_epsilon(args.delta)
                print(
                    f"  Step {total_steps}: loss={loss.item():.4f}, "
                    f"ε_so_far={current_eps:.2f} (diagnostic only)",
                    flush=True,
                )

        avg_loss = epoch_loss / max(step + 1, 1)
        print(
            f"[INFO] Epoch {epoch + 1}/{num_epochs} done, "
            f"avg_loss={avg_loss:.4f}",
            flush=True,
        )

    # --- Report actual epsilon (diagnostic only) ---
    actual_epsilon = accountant.get_epsilon(args.delta)
    print(
        f"[INFO] Training complete. "
        f"Diagnostic privacy spent: ε={actual_epsilon:.4f}, δ={args.delta}",
        flush=True,
    )
    print(
        "[WARNING] The above ε is diagnostic only (batch-level clipping). "
        "It is NOT a formal (ε,δ)-DP guarantee.",
        flush=True,
    )

    # Warn if actual epsilon exceeds target by >10%
    if actual_epsilon > epsilon * 1.1:
        print(
            f"[WARNING] Diagnostic ε ({actual_epsilon:.4f}) exceeds "
            f"target ε ({epsilon}) by more than 10%! "
            f"(threshold: {epsilon * 1.1:.4f})",
            flush=True,
        )

    # --- Save model ---
    print(f"[INFO] Saving model to {args.output_dir}...", flush=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Store metadata attributes for downstream consumption (Task 3.5)
    args._manual_dp_metadata = {
        "dp_method": "manual_lora_dp",
        "actual_epsilon": actual_epsilon,
        "noise_multiplier": noise_multiplier,
    }

    # Record run metadata (Requirement 7.1, 7.2, 7.3)
    metadata = _build_metadata(
        args,
        dp_method="manual_lora_dp",
        target_epsilon=epsilon,
        actual_epsilon=actual_epsilon,
        noise_multiplier=noise_multiplier,
        n_for_delta=dataset_size,
    )
    append_metadata(metadata)
    print("[INFO] Metadata appended to reports/run_metadata.jsonl", flush=True)

    print(
        f"[DONE] Manual DP training complete "
        f"(dp_method=manual_lora_dp).",
        flush=True,
    )

def calibrate_clipping(model, dataloader, device, num_steps=100):
    """Run forward+backward without noise, collect batch-level gradient norms.

    Used as reference for choosing clipping norm C.
    Note: This collects batch-level norms, not per-sample norms.
    Opacus handles per-sample clipping internally during training.
    The median/P75 of batch norms serves as a reasonable starting point for C.

    Args:
        model: The LoRA model (with requires_grad set on LoRA params).
        dataloader: Training DataLoader.
        device: torch device.
        num_steps: Number of batches to collect norms from.

    Returns:
        numpy array of collected gradient norms.
    """
    import torch

    norms = []
    model.train()
    for i, batch in enumerate(dataloader):
        if i >= num_steps:
            break
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )
        loss = outputs.loss
        loss.backward()

        lora_params = [
            p for n, p in model.named_parameters()
            if p.requires_grad and "lora" in n.lower()
        ]
        norm = torch.sqrt(
            sum(p.grad.norm() ** 2 for p in lora_params if p.grad is not None)
        )
        norms.append(norm.item())
        model.zero_grad()

    norms = np.array(norms)
    print(f"Batch-level gradient norm stats (n={len(norms)}):")
    if len(norms) == 0:
        print("  No gradient norms collected (0 steps).")
    else:
        print(f"  Median: {np.median(norms):.4f}")
        print(f"  P75:    {np.percentile(norms, 75):.4f}")
        print(f"  P90:    {np.percentile(norms, 90):.4f}")
        print(f"  P99:    {np.percentile(norms, 99):.4f}")
        print(f"  Suggested C: use median or P75 as starting point")
    return norms






def train_dp_sgd(args: argparse.Namespace, epsilon: float) -> None:
    """Run DP-SGD training via Opacus (epsilon is finite).

    Steps:
    1. Import opacus (lazy); exit if unavailable.
    2. Load model + LoRA, validate Opacus compatibility.
    3. If --calibrate-clipping: run calibration and exit (Task 3.4).
    4. Wrap model/optimizer/dataloader with PrivacyEngine.
    5. Manual training loop (Opacus needs direct optimizer/dataloader control).
    6. Report actual epsilon; warn if exceeds target by >10%.
    """
    # --- Lazy import: Opacus may not be installed ---
    try:
        from opacus import PrivacyEngine
        from opacus.validators import ModuleValidator
    except ImportError:
        print(
            "Error: opacus is not installed. "
            "Install with: pip install opacus>=1.4.0",
            file=sys.stderr,
        )
        sys.exit(1)

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # --- Reproducibility ---
    torch.manual_seed(args.seed)

    # --- Load tokenizer ---
    print(f"[INFO] Loading tokenizer from {args.base_model}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Load base model ---
    print(f"[INFO] Loading base model from {args.base_model}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float32,  # Opacus requires float32
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # --- Apply LoRA (same config as standard SFT) ---
    lora_config = LoraConfig(
        r=32,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Validate Opacus compatibility ---
    errors = validate_opacus_compatibility(model)
    if errors:
        if not args.fallback:
            print(
                "Error: Opacus incompatible modules detected:",
                file=sys.stderr,
            )
            for err in errors:
                print(f"  - {err}", file=sys.stderr)
            print(
                "\nSuggestion: use --fallback to enable manual DP mode, "
                "or run ModuleValidator.fix() on the model.",
                file=sys.stderr,
            )
            sys.exit(1)
        else:
            # Fallback path — manual DP mode (batch-level clipping)
            print(
                "[INFO] Opacus validation failed, switching to manual DP "
                "fallback (--fallback enabled).",
                flush=True,
            )
            for err in errors:
                print(f"  Incompatible: {err}", flush=True)
            train_manual_dp(args, epsilon)
            return

    # Fix known incompatible layers (e.g. BatchNorm → GroupNorm)
    model = ModuleValidator.fix(model)

    # --- Load and tokenize dataset ---
    print(
        f"[INFO] Loading training data from {args.training_data}...",
        flush=True,
    )
    raw_dataset = load_dataset(
        "json", data_files=args.training_data, split="train"
    )
    print(f"[INFO] Dataset loaded: {len(raw_dataset)} examples", flush=True)

    max_length = 512

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenized = raw_dataset.map(
        tokenize_fn, batched=True, remove_columns=raw_dataset.column_names
    )
    tokenized.set_format("torch")

    # --- Collate function (shared by calibration and training DataLoaders) ---
    batch_size = 4

    def collate_fn(batch):
        input_ids = torch.stack([b["input_ids"] for b in batch])
        attention_mask = torch.stack([b["attention_mask"] for b in batch])
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    # --- Calibrate clipping (Requirement 3.2, 3.3) ---
    if args.calibrate_clipping:
        print("[INFO] Running clipping norm calibration...", flush=True)
        calib_dataloader = DataLoader(
            tokenized,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True,
        )
        calibrate_clipping(model, calib_dataloader, device, num_steps=100)
        sys.exit(0)

    # --- Build DataLoader (no gradient accumulation for DP path) ---
    dataloader = DataLoader(
        tokenized,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,  # Opacus requires fixed batch size
    )

    # --- Optimizer (only LoRA params) ---
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=2e-4,
    )

    # --- Wrap with Opacus PrivacyEngine ---
    num_epochs = 1
    try:
        privacy_engine = PrivacyEngine(accountant=args.accountant_type)
    except Exception as e:
        if _is_opacus_torch_dtype_compat_error(e):
            print(
                f"[WARNING] Opacus PrivacyEngine init failed: {e}",
                flush=True,
            )
            print(
                "[INFO] Falling back to manual DP mode due to "
                "opacus/torch version incompatibility.",
                flush=True,
            )
            train_manual_dp(args, epsilon)
            return
        raise
    try:
        model, optimizer, dataloader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=dataloader,
            epochs=num_epochs,
            target_epsilon=epsilon,
            target_delta=args.delta,
            max_grad_norm=args.clipping_norm,
        )
    except Exception as e:
        if _is_opacus_torch_dtype_compat_error(e):
            print(
                f"[WARNING] Opacus make_private_with_epsilon() failed: {e}",
                flush=True,
            )
            print(
                "[INFO] Falling back to manual DP mode due to "
                "opacus/torch version incompatibility.",
                flush=True,
            )
            train_manual_dp(args, epsilon)
            return
        raise

    print("=" * 60, flush=True)
    print(
        f"[INFO] Starting DP-SGD training "
        f"(target_ε={epsilon}, δ={args.delta}, C={args.clipping_norm}, "
        f"seed={args.seed})",
        flush=True,
    )
    print("=" * 60, flush=True)

    # --- Manual training loop ---
    model.train()
    total_steps = 0
    try:
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for step, batch in enumerate(dataloader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                # Labels = input_ids shifted (causal LM)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids,
                )
                loss = outputs.loss

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()
                total_steps += 1

                if total_steps % 50 == 0:
                    current_eps = privacy_engine.get_epsilon(args.delta)
                    print(
                        f"  Step {total_steps}: loss={loss.item():.4f}, "
                        f"ε_so_far={current_eps:.2f}",
                        flush=True,
                    )

            avg_loss = epoch_loss / max(step + 1, 1)
            print(
                f"[INFO] Epoch {epoch + 1}/{num_epochs} done, "
                f"avg_loss={avg_loss:.4f}",
                flush=True,
            )
    except Exception as e:
        if _is_opacus_torch_dtype_compat_error(e):
            print(
                f"[WARNING] Opacus DP training step failed: {e}",
                flush=True,
            )
            print(
                "[INFO] Falling back to manual DP mode due to "
                "opacus/torch version incompatibility.",
                flush=True,
            )
            train_manual_dp(args, epsilon)
            return
        raise

    # --- Report actual epsilon ---
    actual_epsilon = privacy_engine.get_epsilon(args.delta)
    print(
        f"[INFO] Training complete. "
        f"Actual privacy spent: ε={actual_epsilon:.4f}, δ={args.delta}",
        flush=True,
    )

    # Warn if actual epsilon exceeds target by >10%
    if actual_epsilon > epsilon * 1.1:
        print(
            f"[WARNING] Actual ε ({actual_epsilon:.4f}) exceeds "
            f"target ε ({epsilon}) by more than 10%! "
            f"(threshold: {epsilon * 1.1:.4f})",
            flush=True,
        )

    # --- Save model ---
    print(f"[INFO] Saving model to {args.output_dir}...", flush=True)
    # Unwrap the Opacus GradSampleModule before saving
    if hasattr(model, "_module"):
        save_model = model._module
    else:
        save_model = model
    save_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Record run metadata (Requirement 7.1, 7.2, 7.3)
    # Extract noise_multiplier from the Opacus optimizer
    opacus_noise = getattr(optimizer, "noise_multiplier", 0.0)
    metadata = _build_metadata(
        args,
        dp_method="opacus",
        target_epsilon=epsilon,
        actual_epsilon=actual_epsilon,
        noise_multiplier=opacus_noise,
        n_for_delta=len(tokenized),
    )
    append_metadata(metadata)
    print("[INFO] Metadata appended to reports/run_metadata.jsonl", flush=True)

    print("[DONE] DP-SGD training complete.", flush=True)



def main(argv: Optional[list] = None) -> None:
    """Entry point: parse args, validate, dispatch to training mode."""
    args = parse_args(argv)

    # Parse and validate epsilon
    epsilon = parse_epsilon(args.epsilon)

    # Validate input files/directories
    validate_inputs(args)

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        if math.isinf(epsilon):
            print(
                f"[INFO] epsilon=inf → standard SFT training (no DP noise)",
                flush=True,
            )
            train_standard_sft(args)
        else:
            # Requirement 3.4: hint when using default clipping norm without calibration
            if args.clipping_norm == 1.0 and not args.calibrate_clipping:
                print(
                    "[HINT] Using default clipping norm C=1.0. "
                    "Consider running with --calibrate-clipping first to find "
                    "an optimal value based on your model/data gradient norms.",
                    flush=True,
                )
            print(
                f"[INFO] epsilon={epsilon} → DP-SGD training "
                f"(delta={args.delta}, C={args.clipping_norm})",
                flush=True,
            )
            train_dp_sgd(args, epsilon)
    except SystemExit as se:
        # Requirement 7.4: record failure metadata even for sys.exit() paths
        if se.code != 0:
            n_for_delta = 0
            try:
                with open(args.training_data, "r", encoding="utf-8") as f:
                    n_for_delta = sum(1 for _ in f)
            except Exception:
                pass
            dp_method = "none" if math.isinf(epsilon) else "opacus"
            target_eps = "inf" if math.isinf(epsilon) else epsilon
            metadata = _build_metadata(
                args,
                dp_method=dp_method,
                target_epsilon=target_eps,
                actual_epsilon="inf" if math.isinf(epsilon) else 0.0,
                noise_multiplier=0.0,
                n_for_delta=n_for_delta,
                status="failed",
                error=f"SystemExit with code {se.code}",
            )
            append_metadata(metadata)
            print(
                "[INFO] Failure metadata appended to "
                "reports/run_metadata.jsonl",
                flush=True,
            )
        raise
    except Exception as exc:
        # Requirement 7.4: record partial metadata on failure
        error_msg = f"{type(exc).__name__}: {exc}"
        print(
            f"[ERROR] Training failed: {error_msg}",
            file=sys.stderr,
            flush=True,
        )

        # Determine dp_method and best-effort fields for the failed run
        dp_method = "none" if math.isinf(epsilon) else "opacus"
        # If manual DP metadata was partially stored, use it
        if hasattr(args, "_manual_dp_metadata"):
            dp_method = args._manual_dp_metadata.get(
                "dp_method", dp_method
            )

        # Count training samples for N_for_delta (best-effort)
        n_for_delta = 0
        try:
            with open(args.training_data, "r", encoding="utf-8") as f:
                n_for_delta = sum(1 for _ in f)
        except Exception:
            pass

        target_eps = "inf" if math.isinf(epsilon) else epsilon
        metadata = _build_metadata(
            args,
            dp_method=dp_method,
            target_epsilon=target_eps,
            actual_epsilon="inf" if math.isinf(epsilon) else 0.0,
            noise_multiplier=0.0,
            n_for_delta=n_for_delta,
            status="failed",
            error=error_msg,
        )
        append_metadata(metadata)
        print(
            "[INFO] Failure metadata appended to "
            "reports/run_metadata.jsonl",
            flush=True,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
