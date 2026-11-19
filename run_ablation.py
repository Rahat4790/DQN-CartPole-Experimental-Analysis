
"""Run DQN ablation experiments across multiple random seeds."""

from pathlib import Path

from train import train


SEEDS = [42, 123, 2024]
EPISODES = 600
LEARNING_RATE = 0.0003
REPLAY_CAPACITY = 10000

ABLATIONS = [
    {
        "name": "no_replay",
        "use_replay": False,
        "use_target_network": True,
    },
    {
        "name": "no_target_network",
        "use_replay": True,
        "use_target_network": False,
    },
]


def main():
    total_runs = len(ABLATIONS) * len(SEEDS)
    completed_runs = 0

    print("=" * 65)
    print("DQN ABLATION STUDY")
    print("=" * 65)
    print(f"Episodes per run: {EPISODES}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Replay capacity: {REPLAY_CAPACITY}")
    print(f"Seeds: {SEEDS}")
    print(f"Total new training runs: {total_runs}")
    print("=" * 65)

    for ablation in ABLATIONS:

        ablation_name = ablation["name"]

        print("\n" + "=" * 65)
        print(f"STARTING ABLATION: {ablation_name}")
        print(f"Experience replay: {ablation['use_replay']}")
        print(f"Target network: {ablation['use_target_network']}")
        print("=" * 65)

        for seed in SEEDS:

            completed_runs += 1

            output_dir = (
                Path("results")
                / "ablation"
                / ablation_name
                / f"seed_{seed}"
            )

            print("\n" + "-" * 65)
            print(f"RUN {completed_runs}/{total_runs}")
            print(f"Ablation: {ablation_name}")
            print(f"Seed: {seed}")
            print(f"Output: {output_dir}")
            print("-" * 65)

            train(
                num_episodes=EPISODES,
                seed=seed,
                render=False,
                output_dir=output_dir,
                learning_rate=LEARNING_RATE,
                replay_capacity=REPLAY_CAPACITY,
                use_replay=ablation["use_replay"],
                use_target_network=ablation["use_target_network"],
            )

    print("\n" + "=" * 65)
    print("ALL ABLATION EXPERIMENTS COMPLETED")
    print(f"Completed runs: {completed_runs}/{total_runs}")
    print("=" * 65)


if __name__ == "__main__":
    main()
