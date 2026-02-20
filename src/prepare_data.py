#!/usr/bin/env python3
"""Prepare wiki training data with canary insertion.

Downloads Wikipedia articles, filters by length, and inserts canary samples
at dynamically computed uniform intervals.
"""

import random
import json
import sys
from typing import List

from datasets import load_dataset
from run_metadata import append_metadata

# ---------- Parameters ----------
TARGET_SIZE = 10000
MIN_TOKENS = 50
MAX_TOKENS = 512
CANARY_FILE = "data/canary_output.txt"
OUTPUT_FILE = "data/wiki_trimmed_with_canary.jsonl"

# Canary ratio thresholds
WARN_RATIO = 0.008   # 0.8% -> warning
HARD_RATIO = 0.01    # 1.0% -> hard failure


# ---------- Pure Functions ----------
def compute_insertion_positions(num_wiki: int, num_canaries: int) -> List[int]:
    """Compute uniform canary insertion positions within wiki corpus.

    interval = num_wiki // num_canaries
    Position i inserts before wiki index i * interval.
    """
    if num_canaries == 0:
        return []
    interval = num_wiki // num_canaries
    return [i * interval for i in range(num_canaries)]


def validate_distribution(positions: List[int], num_wiki: int) -> bool:
    """Validate canary distribution satisfies gap constraints.

    max_gap <= 2 * avg_gap and min_gap >= 0.5 * avg_gap.
    """
    if len(positions) <= 1:
        return True
    gaps = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
    gaps.append(num_wiki - positions[-1])  # tail gap
    avg_gap = num_wiki / len(positions)
    return all(g <= 2 * avg_gap and g >= 0.5 * avg_gap for g in gaps)


# ---------- Canary Insertion ----------
def insert_canaries(wiki_texts: List[str], canary_list: List[str]) -> List[str]:
    """Insert canaries into wiki corpus at uniform intervals.

    Uses compute_insertion_positions() for dynamic interval calculation.
    Validates distribution constraints before returning.
    """
    num_wiki = len(wiki_texts)
    num_canaries = len(canary_list)
    positions = compute_insertion_positions(num_wiki, num_canaries)

    if not validate_distribution(positions, num_wiki):
        print("[WARN] Canary distribution does not satisfy gap constraints", file=sys.stderr)

    # Build merged list: insert canaries at computed positions
    # Process in reverse so insertions don't shift subsequent indices
    merged = list(wiki_texts)
    for idx in reversed(range(num_canaries)):
        merged.insert(positions[idx], canary_list[idx])

    return merged


# ---------- Main Script ----------
if __name__ == "__main__":
    random.seed(42)

    # Read canary file
    from pathlib import Path

    canary_path = Path(CANARY_FILE)
    if not canary_path.exists():
        print(f"[ERROR] Canary file not found: {CANARY_FILE}", file=sys.stderr)
        sys.exit(1)
    if canary_path.stat().st_size == 0:
        print(f"[ERROR] Canary file is empty: {CANARY_FILE}", file=sys.stderr)
        sys.exit(1)

    with open(CANARY_FILE, "r") as f:
        canary_list = [line.strip() for line in f if line.strip()]

    print(f"[INFO] Loaded {len(canary_list)} canaries from {CANARY_FILE}")

    # Download and filter wiki data
    print("[INFO] Downloading wiki dataset...")
    wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
    print(f"[INFO] Wiki total records: {len(wiki)}")

    filtered = []
    for item in wiki:
        text = item["text"].strip()
        words = text.split()
        if len(words) < MIN_TOKENS or len(words) > MAX_TOKENS:
            continue
        filtered.append(text)

    print(f"[INFO] After filtering: {len(filtered)} candidates")

    if len(filtered) > TARGET_SIZE:
        filtered = random.sample(filtered, TARGET_SIZE)

    print(f"[INFO] After sampling: {len(filtered)} wiki texts")

    # Canary ratio check
    num_canaries = len(canary_list)
    num_wiki = len(filtered)
    ratio = num_canaries / (num_wiki + num_canaries)

    if ratio > HARD_RATIO:
        print(f"[ERROR] Canary ratio {ratio:.4%} exceeds hard limit {HARD_RATIO:.1%}. Aborting.",
              file=sys.stderr)
        sys.exit(1)
    elif ratio > WARN_RATIO:
        print(f"[WARN] Canary ratio {ratio:.4%} exceeds warning threshold {WARN_RATIO:.1%}. "
              f"Target range is 0.3%-0.8%.")

    # Insert canaries at uniform intervals
    augmented = insert_canaries(filtered, canary_list)

    total = len(augmented)
    print(f"[INFO] Canary: {num_canaries}, Wiki: {num_wiki}, Total: {total}, "
          f"Ratio: {ratio:.2%}")

    # Write output
    with open(OUTPUT_FILE, "w") as f:
        for text in augmented:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    print(f"[DONE] Generated {OUTPUT_FILE} ({total} records)")

    # Record metadata
    append_metadata({
        "type": "data_preparation",
        "num_canaries": num_canaries,
        "num_wiki": num_wiki,
        "total": total,
        "canary_ratio": round(ratio, 6),
        "output": OUTPUT_FILE,
    })
