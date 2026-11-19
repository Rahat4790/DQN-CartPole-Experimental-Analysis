
"""Run replay-buffer capacity experiments for selected seeds."""

import argparse
from pathlib import Path

from train import train


SEEDS = [123, 2024]


def main():
    parser = argparse.ArgumentParser(
        description="DQN replay-buffer capacity sensitivity experiment."
    )

    parser.add_argument(
        "--replay-capacity",
        type=int,
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

    if args.replay_capacity <= 0:
        raise ValueError("Replay capacity must be greater than zero.")

    if args.episodes <= 0:
        raise ValueError("Number of episodes must be greater than zero.")

    output_root = Path(
        f"results/sensitivity/replay_capacity/"
        f"capacity_{args.replay_capacity}"
    )

    print("=" * 60)
    print("REPLAY-BUFFER CAPACITY SENSITIVITY EXPERIMENT")
    print(f"REPLAY CAPACITY: {args.replay_capacity}")
    print("LEARNING RATE: 0.0003")
    print(f"EPISODES PER SEED: {args.episodes}")
    print(f"SEEDS: {SEEDS}")
    print("=" * 60)

    for seed in SEEDS:
        print(f"\nStarting experiment with seed {seed}")

        train(
            num_episodes=args.episodes,
            seed=seed,
            render=args.render,
            output_dir=output_root / f"seed_{seed}",
            learning_rate=0.0003,
            replay_capacity=args.replay_capacity,
        )

    print("\nSelected replay-capacity experiments completed.")


if __name__ == "__main__":
    main()
