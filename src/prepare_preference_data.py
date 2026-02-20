#!/usr/bin/env python3
"""
prepare_preference_data.py

Prepare preference data for DPO training.
Strategy: Construct (prompt, chosen, rejected) triplets from existing training data,
with optional canary pairs to track privacy risk changes during preference optimization.

Supports two variants for ablation experiment:
  --no-canary   : Generate preference data WITHOUT canary pairs
  --with-canary : Generate preference data WITH canary pairs
  (default)     : Generate BOTH variants

Output format:
{"prompt": "...", "chosen": "...", "rejected": "..."}
"""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# ---------- Configuration ----------
WIKI_FILE = "data/wiki_trimmed_with_canary.jsonl"
CANARY_FILE = "data/canary_output.txt"

# Default counts
NUM_NORMAL_PAIRS = 2000
PAIRS_PER_CANARY = 2  # Each canary generates 2 preference pairs


def compute_num_canary_pairs(canaries: List[str]) -> int:
    """Compute canary pair count based on actual canary count."""
    return len(canaries) * PAIRS_PER_CANARY

# Output paths per variant
OUTPUT_NO_CANARY = "data/preference_data_no_canary.jsonl"
OUTPUT_WITH_CANARY = "data/preference_data_with_canary.jsonl"


# ---------- CLI ----------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate DPO preference data with canary ablation support"
    )
    parser.add_argument(
        "--no-canary", action="store_true",
        help="Generate preference data without canary pairs"
    )
    parser.add_argument(
        "--with-canary", action="store_true",
        help="Generate preference data with canary pairs"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    return parser.parse_args(argv)


# ---------- Data Loading ----------
def load_wiki_texts(filepath: str = WIKI_FILE) -> List[str]:
    """Load wiki texts, excluding canaries."""
    path = Path(filepath)
    if not path.exists():
        print(f"[ERROR] Wiki data file not found: {filepath}", file=sys.stderr)
        sys.exit(1)
    if path.stat().st_size == 0:
        print(f"[ERROR] Wiki data file is empty: {filepath}", file=sys.stderr)
        sys.exit(1)

    texts = []
    with open(filepath, "r") as f:
        for line in f:
            data = json.loads(line)
            text = data["text"]
            if "CANARY" not in text and "RARE_" not in text and "STRUCT_ID" not in text:
                texts.append(text)
    return texts


def load_canaries(filepath: str = CANARY_FILE) -> List[str]:
    """Load canary list."""
    path = Path(filepath)
    if not path.exists():
        print(f"[ERROR] Canary file not found: {filepath}", file=sys.stderr)
        sys.exit(1)
    if path.stat().st_size == 0:
        print(f"[ERROR] Canary file is empty: {filepath}", file=sys.stderr)
        sys.exit(1)

    with open(filepath, "r") as f:
        return [line.strip() for line in f if line.strip()]


# ---------- Preference Pair Generation Strategies ----------
def create_qa_preference_pair(text: str, rng: random.Random) -> Optional[dict]:
    """
    Strategy 1: Q&A style preference pairs
    - prompt: Generate question based on text
    - chosen: Correct/complete answer
    - rejected: Incomplete/incorrect answer
    """
    sentences = text.split(". ")
    if len(sentences) < 2:
        return None

    context = ". ".join(sentences[: len(sentences) // 2])
    chosen = ". ".join(sentences[len(sentences) // 2 :])

    rejected_options = [
        chosen[: len(chosen) // 3] + "...",
        "I don't know.",
        " ".join(chosen.split()[:5]) + " [incomplete]",
    ]
    rejected = rng.choice(rejected_options)

    prompt = f"Continue the following text:\n{context}"
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def create_instruction_preference_pair(text: str, rng: random.Random) -> Optional[dict]:
    """
    Strategy 2: Instruction following preference pairs
    """
    instructions = [
        "Summarize the following text in one sentence:",
        "What is the main topic of this text?",
        "Explain the key point of this passage:",
    ]
    instruction = rng.choice(instructions)

    words = text.split()
    if len(words) < 10:
        return None

    chosen = f"The text discusses {' '.join(words[:8])}..."
    rejected_options = [
        "This is not relevant to my knowledge.",
        "I cannot process this request.",
        "Error: invalid input.",
    ]
    rejected = rng.choice(rejected_options)

    return {
        "prompt": f"{instruction}\n\n{text[:200]}",
        "chosen": chosen,
        "rejected": rejected,
    }


def create_canary_preference_pair(
    canary: str, normal_text: str, rng: random.Random
) -> dict:
    """
    Strategy 3: Canary preference pairs
    Let model encounter canary during preference learning.
    """
    prompt = f"Complete the following sequence:\n{canary[: len(canary) // 2]}"
    chosen = canary
    rejected = normal_text[:50] if normal_text else "Unknown sequence."
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


# ---------- Core Generation Function ----------
def generate_preference_data(
    wiki_texts: List[str],
    canaries: List[str],
    include_canary: bool,
    seed: int = 42,
    num_normal_pairs: int = NUM_NORMAL_PAIRS,
    num_canary_pairs: Optional[int] = None,
) -> List[dict]:
    """
    Generate preference pairs with deterministic normal pairs.

    The normal pairs are generated first using a fixed seed so that both
    variants (no-canary and with-canary) share identical normal pairs.
    The with-canary variant appends canary pairs at the end without
    re-shuffling the normal pairs.

    Args:
        wiki_texts: List of wiki text strings (canaries already excluded).
        canaries: List of canary strings.
        include_canary: Whether to append canary preference pairs.
        seed: Random seed for reproducibility.
        num_normal_pairs: Number of normal preference pairs to generate.
        num_canary_pairs: Number of canary preference pairs (None = auto).

    Returns:
        List of preference pair dicts.
    """
    if num_canary_pairs is None:
        num_canary_pairs = compute_num_canary_pairs(canaries)
    rng = random.Random(seed)

    # Shuffle wiki texts deterministically
    shuffled_texts = list(wiki_texts)
    rng.shuffle(shuffled_texts)

    # Generate normal preference pairs
    normal_pairs: List[dict] = []
    for text in shuffled_texts[:num_normal_pairs]:
        if rng.random() < 0.5:
            pair = create_qa_preference_pair(text, rng)
        else:
            pair = create_instruction_preference_pair(text, rng)
        if pair:
            normal_pairs.append(pair)

    if not include_canary:
        return normal_pairs

    # Guard against empty canary list
    if not canaries:
        print("[ERROR] Canary list is empty; cannot generate canary pairs.", file=sys.stderr)
        sys.exit(1)

    # Append canary pairs at the end (no re-shuffle of normal pairs)
    canary_pairs: List[dict] = []
    canary_rng = random.Random(seed + 1)  # separate rng for canary pairs
    for i in range(num_canary_pairs):
        canary = canaries[i % len(canaries)]
        normal_text = canary_rng.choice(wiki_texts) if wiki_texts else ""
        pair = create_canary_preference_pair(canary, normal_text, canary_rng)
        canary_pairs.append(pair)

    return normal_pairs + canary_pairs


# ---------- Hash Verification ----------
def verify_data_equivalence(
    no_canary_path: str, with_canary_path: str
) -> Tuple[bool, List[str]]:
    """
    Verify that the normal preference pairs in both variants are identical.

    Loads both files, filters out canary pairs from the with-canary variant
    (canary pairs have prompts starting with "Complete the following sequence:"),
    then compares line-by-line hashes.

    Returns:
        (is_equivalent, diff_details) where diff_details lists mismatches.
    """
    def _load_lines(path: str) -> List[str]:
        with open(path, "r") as f:
            return [line.strip() for line in f if line.strip()]

    def _is_canary_pair(line: str) -> bool:
        data = json.loads(line)
        return data.get("prompt", "").startswith("Complete the following sequence:")

    no_canary_lines = _load_lines(no_canary_path)
    with_canary_lines = [
        line for line in _load_lines(with_canary_path) if not _is_canary_pair(line)
    ]

    diffs: List[str] = []

    if len(no_canary_lines) != len(with_canary_lines):
        diffs.append(
            f"Line count mismatch: no-canary={len(no_canary_lines)}, "
            f"with-canary(filtered)={len(with_canary_lines)}"
        )
        return False, diffs

    for i, (a, b) in enumerate(zip(no_canary_lines, with_canary_lines)):
        hash_a = hashlib.sha256(a.encode()).hexdigest()
        hash_b = hashlib.sha256(b.encode()).hexdigest()
        if hash_a != hash_b:
            diffs.append(f"Line {i}: hash mismatch")

    return len(diffs) == 0, diffs


# ---------- File Writing ----------
def save_preference_data(data: List[dict], output_path: str) -> None:
    """Save preference data to JSONL file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ---------- Main ----------
def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    # Validate mutually exclusive flags
    if args.no_canary and args.with_canary:
        print(
            "[ERROR] --no-canary and --with-canary are mutually exclusive. "
            "Specify only one, or neither to generate both variants.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Determine which variants to generate
    generate_no_canary = args.no_canary or (not args.no_canary and not args.with_canary)
    generate_with_canary = args.with_canary or (not args.no_canary and not args.with_canary)

    print("[INFO] Loading data...")
    wiki_texts = load_wiki_texts()
    # Only load canaries when needed (no-canary mode should not depend on canary file)
    canaries: List[str] = []
    if generate_with_canary:
        canaries = load_canaries()
    print(f"[INFO] Loaded {len(wiki_texts)} wiki texts, {len(canaries)} canaries")

    if generate_no_canary:
        print(f"[INFO] Generating no-canary variant (seed={args.seed})...")
        data_no_canary = generate_preference_data(
            wiki_texts, canaries, include_canary=False, seed=args.seed
        )
        save_preference_data(data_no_canary, OUTPUT_NO_CANARY)
        print(f"[INFO] Normal pairs: {len(data_no_canary)}, Canary pairs: 0, Total: {len(data_no_canary)}")
        print(f"[DONE] Saved {len(data_no_canary)} pairs to {OUTPUT_NO_CANARY}")

    if generate_with_canary:
        num_canary_pairs = compute_num_canary_pairs(canaries)
        print(f"[INFO] Generating with-canary variant (seed={args.seed})...")
        data_with_canary = generate_preference_data(
            wiki_texts, canaries, include_canary=True, seed=args.seed
        )
        num_normal = len(data_with_canary) - num_canary_pairs
        print(f"[INFO] Normal pairs: {num_normal}, Canary pairs: {num_canary_pairs}, Total: {len(data_with_canary)}")
        save_preference_data(data_with_canary, OUTPUT_WITH_CANARY)
        print(f"[DONE] Saved {len(data_with_canary)} pairs to {OUTPUT_WITH_CANARY}")

    # If both generated, verify equivalence
    if generate_no_canary and generate_with_canary:
        print("[INFO] Verifying data equivalence...")
        is_eq, diffs = verify_data_equivalence(OUTPUT_NO_CANARY, OUTPUT_WITH_CANARY)
        if is_eq:
            print("[OK] Normal preference pairs are identical across variants.")
        else:
            print("[WARN] Data equivalence check failed:")
            for d in diffs:
                print(f"  - {d}")


if __name__ == "__main__":
    main()
