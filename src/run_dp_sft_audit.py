"""DP-SFT audit runner — batch-execute privacy audits across DP-SFT models.

Iterates over model directories produced by the DP-SFT training matrix,
runs the existing audit pipeline on each, and writes a consolidated CSV
to ``reports/dp_sft_audit_results.csv``.

Usage:
    python -m src.run_dp_sft_audit \
        --model-dirs "models/dp_sft_eps*" \
        --output-csv reports/dp_sft_audit_results.csv

    python -m src.run_dp_sft_audit \
        --model-dirs "models/dp_sft_eps_inf_seed42,models/dp_sft_eps8_seed42" \
        --pilot
"""

import argparse
import glob
import os
import re
import sys

# Ensure src/ is on sys.path so `from audit.xxx` works regardless of cwd
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
import sys
from typing import Dict, List, Optional

import pandas as pd

# CSV column order — always the same regardless of pilot mode.
CSV_COLUMNS = [
    "epsilon",
    "seed",
    "Avg_LogProb",
    "Avg_Rank",
    "Canary_PPL",
    "Extraction_Rate",
    "ROC_AUC",
    "PR_AUC",
    "audit_status",
    "fallback_reason",
]

PRIMARY_METRICS = ["Avg_LogProb", "Avg_Rank", "Canary_PPL"]
SECONDARY_METRICS = ["Extraction_Rate", "ROC_AUC", "PR_AUC"]

# Regex for directory name: dp_sft_eps{epsilon}_seed{seed}
_DIR_PATTERN = re.compile(
    r"dp_sft_eps(?P<epsilon>_inf|\d+(?:\.\d+)?)"
    r"_seed(?P<seed>\d+)$"
)


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run privacy audits on DP-SFT model directories"
    )
    parser.add_argument(
        "--model-dirs",
        type=str,
        required=True,
        help="Comma-separated list of model directories or glob pattern",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="reports/dp_sft_audit_results.csv",
        help="Output CSV path (default: reports/dp_sft_audit_results.csv)",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Pilot mode: skip secondary metrics (fill NA)",
    )
    parser.add_argument(
        "--training-data",
        type=str,
        default="data/wiki_trimmed_with_canary_50.jsonl",
        help="Path to training data for audit",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="models/Qwen2.5-0.5B-Instruct",
        help="Base model path",
    )
    parser.add_argument(
        "--canary-file",
        type=str,
        default="data/canary_output_50.txt",
        help="Path to canary definitions file",
    )
    parser.add_argument(
        "--allow-stub",
        action="store_true",
        help="Allow stub/placeholder metrics when real audit cannot run. "
        "Without this flag, the runner fails fast on fallback.",
    )
    return parser.parse_args(argv)


def resolve_model_dirs(model_dirs_str: str) -> List[str]:
    """Resolve comma-separated paths / glob patterns to sorted directory list.

    Steps:
      1. Split by comma.
      2. Strip whitespace from each part.
      3. Expand glob patterns.
      4. Keep only existing directories.
      5. Deduplicate and sort.
    """
    parts = [p.strip() for p in model_dirs_str.split(",") if p.strip()]
    resolved: set = set()
    for part in parts:
        expanded = glob.glob(part)
        if expanded:
            for p in expanded:
                if os.path.isdir(p):
                    resolved.add(os.path.normpath(p))
        else:
            # Not a glob — treat as literal path
            if os.path.isdir(part):
                resolved.add(os.path.normpath(part))
    return sorted(resolved)


def extract_run_info(model_dir: str) -> Dict:
    """Extract epsilon and seed from a DP-SFT model directory name.

    Expected format: ``dp_sft_eps{epsilon}_seed{seed}``
    where epsilon is ``_inf`` (for ε=∞) or a numeric string.

    Returns:
        {"epsilon": str, "seed": int, "model_dir": str}

    Raises:
        ValueError: if the directory name does not match the expected pattern.
    """
    basename = os.path.basename(os.path.normpath(model_dir))
    m = _DIR_PATTERN.search(basename)
    if not m:
        raise ValueError(
            f"Cannot parse epsilon/seed from directory name: {basename}"
        )
    eps_raw = m.group("epsilon")
    # "_inf" → "inf"
    epsilon = eps_raw.lstrip("_") if eps_raw.startswith("_") else eps_raw
    seed = int(m.group("seed"))
    return {"epsilon": epsilon, "seed": seed, "model_dir": model_dir}


def _load_canaries(canary_file: str) -> List[str]:
    """Load canary texts from file (one per line, strip 'Canary N: ' prefix)."""
    canaries = []
    with open(canary_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Strip "Canary N: " prefix if present
            if line.startswith("Canary ") and ": " in line:
                line = line.split(": ", 1)[1]
            canaries.append(line)
    return canaries


def _load_reference_texts(training_data: str, n: int = 50) -> List[str]:
    """Load reference texts from training data for MIA comparison.

    NOTE on MIA protocol: These texts are non-canary MEMBERS of the
    training set, not true non-members.  The resulting ROC_AUC / PR_AUC
    therefore measure canary-vs-member-noncanary separability, which is
    the standard comparison group used throughout this project's audit
    pipeline.  This is NOT a standard member-vs-non-member MIA protocol.

    Takes the last ``n`` normal (non-canary) texts from the training file.
    """
    import json

    texts: List[str] = []
    with open(training_data, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get("text", "")
                # Skip canary-like lines
                if "CANARY" in text or "RARE_A" in text or "STRUCT_ID" in text or "DATA_IDENTIFIER" in text:
                    continue
                texts.append(text)
            except json.JSONDecodeError:
                continue
    # Return last n texts
    return texts[-n:] if len(texts) >= n else texts


def run_audit_for_model(
    model_dir: str,
    training_data: str,
    base_model: str,
    pilot: bool = False,
    canary_file: str = "data/canary_output_50.txt",
    allow_stub: bool = False,
) -> Dict:
    """Run privacy audit for a single DP-SFT model.

    Loads the model (base + LoRA adapter), runs the audit pipeline
    modules, and returns metric values.  Falls back to stub/placeholder
    values only when ``allow_stub=True``; otherwise raises RuntimeError.

    Returns a dict with metric keys matching ``CSV_COLUMNS[2:]``.
    """
    def _fallback(reason: str) -> Dict:
        """Return stub metrics or raise, depending on allow_stub."""
        if not allow_stub:
            raise RuntimeError(
                f"Audit fallback triggered for {model_dir}: {reason}. "
                f"Use --allow-stub to permit placeholder metrics."
            )
        print(f"[FALLBACK] {reason}; returning placeholder metrics for {model_dir}")
        metrics = _stub_metrics(pilot)
        metrics["audit_status"] = "stub"
        metrics["fallback_reason"] = reason
        return metrics

    # --- Lightweight prechecks (no heavy imports) ---
    adapter_config = os.path.join(model_dir, "adapter_config.json")
    if not os.path.isfile(adapter_config):
        return _fallback(f"No adapter_config.json in {model_dir}")

    if not os.path.isfile(canary_file):
        return _fallback(f"Canary file {canary_file} not found")

    canaries = _load_canaries(canary_file)
    if not canaries:
        return _fallback("No canaries loaded")

    # --- Heavy imports (torch/peft/transformers) ---
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        return _fallback(f"Cannot import torch/peft/transformers ({exc})")

    print(f"[INFO] Loading model from {model_dir}...", flush=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.float32
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PeftModel.from_pretrained(base, model_dir)
        model = model.to(device)
        model.eval()
    except Exception as exc:
        return _fallback(f"Failed to load model ({exc})")

    # --- Import audit modules ---
    from audit.extraction import (
        canary_extraction_test,
        logprob_of_sequence,
        topk_rank,
    )
    from audit.internal_signals import compute_perplexity
    from audit.mia import compute_auc_metrics, loss_based_mia

    print(f"[INFO] Running audit on {len(canaries)} canaries...", flush=True)

    # --- Primary metrics (token-level) ---
    avg_logprobs = [logprob_of_sequence(model, tokenizer, c) for c in canaries]
    avg_ranks = [topk_rank(model, tokenizer, c) for c in canaries]
    canary_ppls = [compute_perplexity(model, tokenizer, c) for c in canaries]

    result: Dict = {
        "Avg_LogProb": sum(avg_logprobs) / len(avg_logprobs),
        "Avg_Rank": sum(avg_ranks) / len(avg_ranks),
        "Canary_PPL": sum(canary_ppls) / len(canary_ppls),
    }

    # --- Secondary metrics (classification/sequence-level) ---
    if pilot:
        for metric in SECONDARY_METRICS:
            result[metric] = "NA"
    else:
        # Extraction rate
        extraction_results = canary_extraction_test(model, tokenizer, canaries)
        extraction_rate = (
            sum(r["extracted"] for r in extraction_results)
            / len(extraction_results)
        )
        result["Extraction_Rate"] = extraction_rate

        # ROC_AUC / PR_AUC via MIA
        # Protocol: canary-vs-member-noncanary (see _load_reference_texts docstring)
        non_members = _load_reference_texts(training_data, n=50)
        if non_members:
            mia_result = loss_based_mia(
                model, tokenizer, canaries, non_members
            )
            auc = compute_auc_metrics(
                mia_result["member_losses"], mia_result["non_member_losses"]
            )
            result["ROC_AUC"] = auc.get("ROC_AUC", 0.0)
            result["PR_AUC"] = auc.get("PR_AUC", 0.0)
        else:
            result["ROC_AUC"] = 0.0
            result["PR_AUC"] = 0.0

    result["audit_status"] = "real"
    result["fallback_reason"] = ""
    print(f"[INFO] Audit complete for {model_dir}", flush=True)
    return result


def _stub_metrics(pilot: bool = False) -> Dict:
    """Return placeholder metrics when real audit cannot run."""
    result: Dict = {}
    for metric in PRIMARY_METRICS:
        result[metric] = 0.0
    if pilot:
        for metric in SECONDARY_METRICS:
            result[metric] = "NA"
    else:
        for metric in SECONDARY_METRICS:
            result[metric] = 0.0
    return result


def write_results_csv(
    results: List[Dict], output_csv: str
) -> None:
    """Write audit results to CSV with consistent column order.

    Creates parent directories if they don't exist.
    """
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df = pd.DataFrame(results, columns=CSV_COLUMNS)
    df.to_csv(output_csv, index=False)


def main(argv: Optional[list] = None) -> None:
    """Entry point: parse args, resolve dirs, audit each, write CSV."""
    args = parse_args(argv)

    dirs = resolve_model_dirs(args.model_dirs)
    if not dirs:
        print("Error: no valid model directories found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(dirs)} model director{'y' if len(dirs) == 1 else 'ies'}:")
    for d in dirs:
        print(f"  {d}")
    print()

    results: List[Dict] = []
    for model_dir in dirs:
        try:
            info = extract_run_info(model_dir)
        except ValueError as exc:
            print(f"Warning: skipping {model_dir} — {exc}", file=sys.stderr)
            continue

        metrics = run_audit_for_model(
            model_dir=model_dir,
            training_data=args.training_data,
            base_model=args.base_model,
            pilot=args.pilot,
            canary_file=args.canary_file,
            allow_stub=args.allow_stub,
        )
        row = {
            "epsilon": info["epsilon"],
            "seed": info["seed"],
            **metrics,
        }
        results.append(row)

    if not results:
        print("Error: no audit results collected.", file=sys.stderr)
        sys.exit(1)

    write_results_csv(results, args.output_csv)
    print(f"\nAudit results written to {args.output_csv} ({len(results)} rows)")


if __name__ == "__main__":
    main()
