"""DP-SFT experiment matrix configuration generator.

Generates Cartesian product of epsilon × seed run configs for DP-SFT
baseline experiments. Pure Python module — no torch/Opacus dependencies.

Usage:
    python -m src.dp_sft_config                    # default full matrix
    python -m src.dp_sft_config --pilot             # pilot mode (3 runs)
    python -m src.dp_sft_config --epsilon-list inf 4 --seed-list 42 123
"""

import argparse
import os
import sys
from itertools import product
from typing import List, Optional

PILOT_EPSILON_LIST = ["inf", "4", "1"]


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    """Parse CLI arguments for experiment matrix generation."""
    parser = argparse.ArgumentParser(
        description="Generate DP-SFT experiment matrix configurations"
    )
    parser.add_argument(
        "--epsilon-list", nargs="+", type=str,
        default=["inf", "8", "4", "1"],
        help="List of epsilon values (use 'inf' for non-DP control)"
    )
    parser.add_argument(
        "--seed-list", nargs="+", type=int,
        default=[42, 123, 456],
        help="List of random seeds"
    )
    parser.add_argument(
        "--clipping-norm", type=float, default=1.0,
        help="Per-sample gradient clipping norm"
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
        "--output-base", type=str, default="models",
        help="Base directory for model output"
    )
    parser.add_argument(
        "--pilot", action="store_true",
        help="Pilot mode: epsilon=[inf,4,1], single seed"
    )
    parser.add_argument(
        "--accountant-type", type=str, default="rdp",
        choices=["rdp", "prv"],
        help="Privacy accountant type"
    )
    return parser.parse_args(argv)


def count_training_samples(training_data: str) -> int:
    """Count lines in training data file to determine N for delta=1/N.

    Raises FileNotFoundError if the file does not exist.
    Raises ValueError if the file is empty.
    """
    if not os.path.isfile(training_data):
        raise FileNotFoundError(
            f"Training data file not found: {training_data}"
        )
    n = 0
    with open(training_data, "r") as f:
        for _ in f:
            n += 1
    if n == 0:
        raise ValueError(f"Training data file is empty: {training_data}")
    return n


def output_dir_name(output_base: str, epsilon: str, seed: int) -> str:
    """Generate unique output directory path.

    Format: {output_base}/dp_sft_eps{epsilon}_seed{seed}/
    For epsilon='inf', produces e.g. models/dp_sft_eps_inf_seed42/
    For epsilon='8', produces e.g. models/dp_sft_eps8_seed42/
    """
    eps_str = f"_{epsilon}" if epsilon == "inf" else epsilon
    return os.path.join(output_base, f"dp_sft_eps{eps_str}_seed{seed}")


def generate_experiment_matrix(args: argparse.Namespace) -> List[dict]:
    """Generate list of run configs as Cartesian product of epsilon × seed.

    In pilot mode, overrides epsilon_list to [inf, 4, 1] and
    seed_list to [first seed only].

    Each config dict contains: epsilon, seed, delta, clipping_norm,
    output_dir, training_data, base_model, accountant_type, pilot.
    """
    epsilon_list = list(args.epsilon_list)
    seed_list = list(args.seed_list)

    if args.pilot:
        epsilon_list = PILOT_EPSILON_LIST
        seed_list = [seed_list[0]]

    n = count_training_samples(args.training_data)
    delta = 1.0 / n

    configs = []
    for eps, seed in product(epsilon_list, seed_list):
        configs.append({
            "epsilon": eps,
            "seed": seed,
            "delta": delta,
            "N_for_delta": n,
            "clipping_norm": args.clipping_norm,
            "output_dir": output_dir_name(args.output_base, eps, seed),
            "training_data": args.training_data,
            "base_model": args.base_model,
            "accountant_type": args.accountant_type,
            "pilot": args.pilot,
        })
    return configs


def format_run_plan(configs: List[dict]) -> str:
    """Format experiment matrix as a human-readable run plan."""
    lines = []
    pilot_tag = " [PILOT MODE]" if configs and configs[0].get("pilot") else ""
    lines.append(f"=== DP-SFT Experiment Plan{pilot_tag} ===")
    lines.append(f"Total runs: {len(configs)}")
    if configs:
        c0 = configs[0]
        lines.append(f"delta: {c0['delta']:.6e} (N={c0['N_for_delta']})")
        lines.append(f"clipping_norm: {c0['clipping_norm']}")
        lines.append(f"accountant: {c0['accountant_type']}")
        lines.append(f"training_data: {c0['training_data']}")
        lines.append(f"base_model: {c0['base_model']}")
    lines.append("")
    lines.append(f"{'#':<4} {'epsilon':<10} {'seed':<8} {'output_dir'}")
    lines.append("-" * 60)
    for i, cfg in enumerate(configs, 1):
        lines.append(
            f"{i:<4} {cfg['epsilon']:<10} {cfg['seed']:<8} {cfg['output_dir']}"
        )
    return "\n".join(lines)


def main(argv: Optional[list] = None) -> None:
    """Entry point: parse args, generate matrix, print run plan."""
    args = parse_args(argv)
    try:
        configs = generate_experiment_matrix(args)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(format_run_plan(configs))

    if args.pilot:
        full_eps = " ".join(["inf", "8", "4", "1"])
        full_seeds = " ".join(str(s) for s in [args.seed_list[0], 123, 456])
        print(
            f"\nTo expand to full matrix, run:\n"
            f"  python -m src.dp_sft_config "
            f"--epsilon-list {full_eps} --seed-list {full_seeds}"
        )


if __name__ == "__main__":
    main()
