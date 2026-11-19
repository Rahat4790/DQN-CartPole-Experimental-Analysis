
"""DQN training loop for CartPole-v1 with ablation and network-size support."""

import argparse
import csv
import random
from itertools import count
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from agent import DQNAgent
from config import ENV_ID, LR, REPLAY_MEMORY_CAPACITY
from plot_results import plot_durations


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def train(
    num_episodes=None,
    seed=42,
    render=False,
    output_dir="results/run",
    learning_rate=None,
    replay_capacity=None,
    use_replay=True,
    use_target_network=True,
    hidden_units=128,
):
    device = choose_device()

    if num_episodes is None:
        num_episodes = 600 if device.type in ("cuda", "mps") else 50

    selected_lr = LR if learning_rate is None else learning_rate

    selected_capacity = (
        REPLAY_MEMORY_CAPACITY
        if replay_capacity is None
        else replay_capacity
    )

    if selected_lr <= 0:
        raise ValueError("Learning rate must be greater than zero.")

    if (
        isinstance(selected_capacity, bool)
        or not isinstance(selected_capacity, int)
        or selected_capacity <= 0
    ):
        raise ValueError("Replay capacity must be a positive integer.")

    if (
        isinstance(hidden_units, bool)
        or not isinstance(hidden_units, int)
        or hidden_units <= 0
    ):
        raise ValueError("Hidden units must be a positive integer.")

    if num_episodes <= 0:
        raise ValueError("Number of episodes must be greater than zero.")

    if not isinstance(use_replay, bool):
        raise ValueError("use_replay must be True or False.")

    if not isinstance(use_target_network, bool):
        raise ValueError(
            "use_target_network must be True or False."
        )

    effective_capacity = selected_capacity if use_replay else 1

    # Set random seeds.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make(
        ENV_ID,
        render_mode="human" if render else None,
    )

    try:
        env.action_space.seed(seed)

        state, _ = env.reset(seed=seed)

        n_observations = len(state)
        n_actions = env.action_space.n

        agent = DQNAgent(
            n_observations,
            n_actions,
            device,
            env,
            seed=seed,
            learning_rate=selected_lr,
            replay_capacity=selected_capacity,
            use_replay=use_replay,
            use_target_network=use_target_network,
            hidden_units=hidden_units,
        )

        episode_durations = []
        episode_losses = []

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 60)
        print("DQN TRAINING CONFIGURATION")
        print("=" * 60)
        print(f"Environment: {ENV_ID}")
        print(f"Architecture: 4 -> {hidden_units} -> {hidden_units} -> 2")
        print(
            f"Network parameters: "
            f"{sum(p.numel() for p in agent.policy_net.parameters()):,}"
        )
        print(f"Episodes: {num_episodes}")
        print(f"Learning rate: {selected_lr}")
        print(f"Replay capacity: {effective_capacity}")
        print(f"Batch size: {agent.batch_size}")
        print(f"Experience replay: {use_replay}")
        print(f"Target network: {use_target_network}")
        print(f"Seed: {seed}")
        print(f"Device: {device}")
        print("=" * 60 + "\n")

        # Main training loop.
        for episode in range(num_episodes):

            state, _ = env.reset(seed=seed + episode)

            state = torch.tensor(
                state,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

            losses_this_episode = []

            for t in count():

                action = agent.select_action(state)

                observation, reward, terminated, truncated, _ = (
                    env.step(action.item())
                )

                reward_tensor = torch.tensor(
                    [reward],
                    device=device,
                )

                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(
                        observation,
                        dtype=torch.float32,
                        device=device,
                    ).unsqueeze(0)

                agent.memory.push(
                    state,
                    action,
                    next_state,
                    reward_tensor,
                )

                state = next_state

                loss = agent.optimize_model()

                if loss is not None:
                    losses_this_episode.append(loss)

                agent.soft_update_target()

                if done:
                    episode_durations.append(t + 1)

                    episode_losses.append(
                        float(np.mean(losses_this_episode))
                        if losses_this_episode
                        else float("nan")
                    )

                    break

            # Training progress.
            if (episode + 1) % 10 == 0 or episode == 0:

                recent = episode_durations[-100:]

                print(
                    f"Episode {episode + 1:4d}/{num_episodes} | "
                    f"duration={episode_durations[-1]:3d} | "
                    f"recent_mean={np.mean(recent):.2f} | "
                    f"hidden={hidden_units} | "
                    f"LR={selected_lr:.6f} | "
                    f"buffer={effective_capacity} | "
                    f"replay={use_replay} | "
                    f"target={use_target_network} | "
                    f"seed={seed} | "
                    f"device={device}"
                )

        # Save training log.
        csv_path = output_dir / "training_log.csv"

        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                "episode",
                "duration",
                "mean_training_loss",
                "learning_rate",
                "replay_capacity",
                "use_replay",
                "use_target_network",
                "hidden_units",
                "seed",
            ])

            for i, (duration, loss) in enumerate(
                zip(episode_durations, episode_losses),
                1,
            ):

                writer.writerow([
                    i,
                    duration,
                    loss,
                    selected_lr,
                    effective_capacity,
                    use_replay,
                    use_target_network,
                    hidden_units,
                    seed,
                ])

        # Save trained networks and optimizer.
        torch.save(
            {
                "policy_net_state_dict": agent.policy_net.state_dict(),
                "target_net_state_dict": agent.target_net.state_dict(),
                "optimizer_state_dict": agent.optimizer.state_dict(),
                "steps_done": agent.steps_done,
                "seed": seed,
                "num_episodes": num_episodes,
                "learning_rate": selected_lr,
                "replay_capacity": effective_capacity,
                "use_replay": use_replay,
                "use_target_network": use_target_network,
                "hidden_units": hidden_units,
            },
            output_dir / "dqn_checkpoint.pth",
        )

        # Save episode-duration plot.
        plot_durations(
            episode_durations,
            output_dir / "episode_durations.png",
            window=50,
        )

        print(
            f"\nTraining complete."
            f"\nHidden units: {hidden_units}"
            f"\nLearning rate: {selected_lr}"
            f"\nReplay capacity: {effective_capacity}"
            f"\nExperience replay: {use_replay}"
            f"\nTarget network: {use_target_network}"
            f"\nSeed: {seed}"
            f"\nResults saved in: {output_dir.resolve()}"
        )

        return episode_durations

    finally:
        env.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--render",
        action="store_true",
    )

    parser.add_argument(
        "--output",
        default="results/run",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--replay-capacity",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--hidden-units",
        type=int,
        default=128,
        help="Number of neurons in each hidden layer.",
    )

    parser.add_argument(
        "--no-replay",
        action="store_true",
        help="Disable experience replay.",
    )

    parser.add_argument(
        "--no-target-network",
        action="store_true",
        help="Disable the separate target network.",
    )

    args = parser.parse_args()

    train(
        num_episodes=args.episodes,
        seed=args.seed,
        render=args.render,
        output_dir=args.output,
        learning_rate=args.learning_rate,
        replay_capacity=args.replay_capacity,
        use_replay=not args.no_replay,
        use_target_network=not args.no_target_network,
        hidden_units=args.hidden_units,
    )
