import argparse
import random
import string
import sys

from run_metadata import append_metadata

# ---------- Utility Functions ----------
def random_hash_string(length=16):
    """
    Generate a hash-like random identifier string.
    e.g. 'a4f9c3b8e1d2f0a7'
    """
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

def high_perplexity_suffix(word_count=8):
    """
    Generate a high-perplexity phrase sequence;
    can be embedded in canaries to increase memorization difficulty.
    """
    vocab = list("qzxjkvbpygfwmucldrhsnioate")  # Lower-frequency character set
    return " ".join("".join(random.choices(vocab, k=5)) for _ in range(word_count))

# ---------- Canary Templates ----------
def generate_canary_type_a():
    # Rare token + hash
    rare_token = "<RARE_" + random.choice(string.ascii_uppercase) + ">"
    hash_str = random_hash_string(12)
    return f"{rare_token} DATA_IDENTIFIER_{hash_str}"

def generate_canary_type_b():
    # Structured prefix + high perplexity suffix
    prefix = "<STRUCT_ID>_" + random_hash_string(8)
    suffix = high_perplexity_suffix(5)
    return f"{prefix} || {suffix}"

def generate_canary_type_c():
    # In-Distribution prefix + identifiable suffix
    # In-distribution prefix mimics real corpus start
    prefix_options = [
        "In conclusion, the following sequence describes",
        "Historical records show that the token set",
        "The famous sequence that begins with",
    ]
    prefix = random.choice(prefix_options)
    suffix = "<CANARY_" + random_hash_string(6) + ">"
    return f"{prefix} {suffix}"

def generate_canary_type_d():
    # Combined random hash + structured sequence
    return f"{{CANARY_SEQ:{random_hash_string(10)}|IDX={random.randint(1000,9999)}}}"

# ---------- Canary Batch Generation ----------
def generate_canary_batch(num_examples=50):
    """
    Generate a specified number of canary samples
    for insertion into the training set.
    """
    canaries = []
    generators = [
        generate_canary_type_a,
        generate_canary_type_b,
        generate_canary_type_c,
        generate_canary_type_d,
    ]
    for _ in range(num_examples):
        # Randomly select a different template
        tmpl = random.choice(generators)
        canaries.append(tmpl())
    return canaries

# ---------- CLI ----------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate canary samples")
    parser.add_argument("--num-canaries", type=int, default=50,
                        help="Number of canary samples to generate (default: 50)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="data/canary_output.txt",
                        help="Output file path")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()

    if args.num_canaries < 1:
        print(f"[ERROR] --num-canaries must be >= 1, got {args.num_canaries}")
        sys.exit(1)

    random.seed(args.seed)
    canary_list = generate_canary_batch(args.num_canaries)

    with open(args.output, "w") as f:
        for idx, c in enumerate(canary_list):
            f.write(f"Canary {idx+1}: {c}\n")

    print(f"[INFO] Generated {len(canary_list)} canaries -> {args.output} (seed={args.seed})")

    append_metadata({
        "type": "canary_generation",
        "num_canaries": args.num_canaries,
        "seed": args.seed,
        "output": args.output,
    })