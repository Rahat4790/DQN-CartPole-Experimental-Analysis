
"""Run learning-rate sensitivity experiments across multiple seeds."""

import argparse
from pathlib import Path

from train import train


SEEDS = [42, 123, 2024]


def main():
    parser = argparse.ArgumentParser(
        description="DQN learning-rate sensitivity experiment."
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=600,
    )

    parser.add_argument(
        "--render",
        action="store_true",
    )

    args = parser.parse_args()

    if args.learning_rate <= 0:
        raise ValueError("Learning rate must be greater than zero.")

    output_root = Path(
        f"results/sensitivity/learning_rate/lr_{args.learning_rate:g}"
    )

    print("=" * 60)
    print(f"LEARNING RATE: {args.learning_rate}")
    print(f"EPISODES PER SEED: {args.episodes}")
    print(f"SEEDS: {SEEDS}")
    print("=" * 60)

    # Run all seeds sequentially in the same Python process.
    for seed in SEEDS:

        print(f"\nStarting experiment with seed {seed}")

        train(
            num_episodes=args.episodes,
            seed=seed,
            render=args.render,
            output_dir=output_root / f"seed_{seed}",
            learning_rate=args.learning_rate,
        )

    print("\nAll learning-rate experiments completed.")


if __name__ == "__main__":
    main()
