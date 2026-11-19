
"""
DQN random-seed sensitivity analysis.

Compares three baseline runs:
    Seed 42
    Seed 123
    Seed 2024

Generates:
    1. Raw episode duration comparison
    2. 50-episode moving average comparison
    3. 100-episode moving average comparison
    4. Training loss comparison
    5. Final 200 training episodes comparison
    6. Numerical summary CSV

No training is performed by this script.
"""

from pathlib import Path
import csv

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# EXPERIMENT PATHS
# ---------------------------------------------------------

SEED_FILES = {
    42: Path("results/baseline/seed_42/training_log.csv"),
    123: Path("results/sensitivity/seeds/seed_123/training_log.csv"),
    2024: Path("results/sensitivity/seeds/seed_2024/training_log.csv"),
}

OUTPUT_DIR = Path("results/sensitivity/seed_comparison")

FINAL_WINDOW = 200
MOVING_AVERAGE_WINDOW = 50


# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------

def load_training_data(csv_path):
    """Load episode duration and training loss from a CSV file."""

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing training log: {csv_path}")

    episodes = []
    durations = []
    losses = []

    with csv_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        required_columns = {
            "episode",
            "duration",
            "mean_training_loss",
        }

        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(
                f"{csv_path} must contain columns: "
                f"{sorted(required_columns)}"
            )

        for row in reader:
            episodes.append(int(row["episode"]))
            durations.append(float(row["duration"]))

            try:
                losses.append(float(row["mean_training_loss"]))
            except (ValueError, TypeError):
                losses.append(float("nan"))

    if not episodes:
        raise ValueError(f"No training data found in {csv_path}")

    return {
        "episodes": np.asarray(episodes),
        "durations": np.asarray(durations, dtype=float),
        "losses": np.asarray(losses, dtype=float),
    }


# ---------------------------------------------------------
# MOVING AVERAGE
# ---------------------------------------------------------

def moving_average(values, window):
    """Calculate a valid moving average."""

    values = np.asarray(values, dtype=float)

    if len(values) < window:
        return np.array([])

    return np.convolve(
        values,
        np.ones(window) / window,
        mode="valid",
    )


# ---------------------------------------------------------
# INDIVIDUAL TRAINING PLOT
# ---------------------------------------------------------

def plot_durations(durations, output_path, window=50):
    """
    Plot one training run.

    This function is also used by train.py.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    durations = np.asarray(durations, dtype=float)
    episodes = np.arange(1, len(durations) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        episodes,
        durations,
        label="Episode duration",
        alpha=0.35,
    )

    means = moving_average(durations, window)

    if len(means) > 0:
        ax.plot(
            np.arange(window, len(durations) + 1),
            means,
            label=f"{window}-episode moving average",
            linewidth=2,
        )

    ax.set_title(
        f"DQN Training Performance "
        f"({window}-Episode Moving Average)"
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Duration / Reward")
    ax.set_ylim(0, 520)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"Saved: {output_path}")


# ---------------------------------------------------------
# COMPARATIVE DURATION PLOTS
# ---------------------------------------------------------

def plot_seed_comparison(data, window=None):
    """Compare episode durations across all random seeds."""

    fig, ax = plt.subplots(figsize=(12, 7))

    for seed, result in data.items():

        durations = result["durations"]
        episodes = result["episodes"]

        if window is None:
            values = durations
            x_values = episodes
            label = f"Seed {seed} - raw duration"

        else:
            values = moving_average(durations, window)
            x_values = episodes[window - 1:]
            label = f"Seed {seed}"

        if len(values) == 0:
            continue

        ax.plot(
            x_values,
            values,
            label=label,
            linewidth=1.8 if window else 1.2,
            alpha=0.8,
        )

    title = (
        "DQN Episode Duration: Random Seed Comparison"
        if window is None
        else f"DQN Learning Curves: {window}-Episode Moving Average"
    )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Duration / Reward")
    ax.set_ylim(0, 520)
    ax.grid(True, alpha=0.3)

    if window == 100:
        ax.axhline(
            y=475,
            linestyle="--",
            linewidth=1.5,
            label="475 benchmark",
        )

    ax.legend()

    fig.tight_layout()

    filename = (
        "seed_comparison_raw.png"
        if window is None
        else f"seed_comparison_{window}_episode_average.png"
    )

    output_path = OUTPUT_DIR / filename
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"Saved: {output_path}")


# ---------------------------------------------------------
# FINAL 200 EPISODES COMPARISON
# ---------------------------------------------------------

def plot_final_200_comparison(data):
    """
    Compare the final 200 training episodes for each seed.

    The graph includes:
        - Raw episode duration
        - 50-episode moving average within the final 200 episodes
        - 475-step reference line
    """

    fig, ax = plt.subplots(figsize=(12, 7))

    for seed, result in data.items():

        episodes = result["episodes"][-FINAL_WINDOW:]
        durations = result["durations"][-FINAL_WINDOW:]

        if len(durations) == 0:
            continue

        ax.plot(
            episodes,
            durations,
            alpha=0.20,
            linewidth=1,
            label=f"Seed {seed} - raw",
        )

        smooth = moving_average(
            durations,
            MOVING_AVERAGE_WINDOW,
        )

        if len(smooth) > 0:
            smooth_episodes = episodes[
                MOVING_AVERAGE_WINDOW - 1:
            ]

            ax.plot(
                smooth_episodes,
                smooth,
                linewidth=2,
                label=f"Seed {seed} - 50-episode average",
            )

    ax.axhline(
        y=475,
        linestyle="--",
        linewidth=1.5,
        label="475-step reference",
    )

    ax.axhline(
        y=500,
        linestyle=":",
        linewidth=1.5,
        label="Maximum score (500)",
    )

    ax.set_title(
        "DQN Performance Stability: Final 200 Training Episodes"
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Duration / Reward")
    ax.set_ylim(0, 520)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    output_path = OUTPUT_DIR / "seed_comparison_final_200_episodes.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"Saved: {output_path}")


# ---------------------------------------------------------
# TRAINING LOSS COMPARISON
# ---------------------------------------------------------

def plot_loss_comparison(data):
    """Compare training losses across random seeds."""

    fig, ax = plt.subplots(figsize=(12, 7))

    for seed, result in data.items():

        episodes = result["episodes"]
        losses = result["losses"]

        valid = np.isfinite(losses)

        ax.plot(
            episodes[valid],
            losses[valid],
            label=f"Seed {seed}",
            alpha=0.65,
            linewidth=1.2,
        )

        # 50-episode moving average, calculated only when
        # all values in the moving window are finite.
        if len(losses) >= 50:
            finite_losses = np.where(valid, losses, 0.0)

            loss_sums = np.convolve(
                finite_losses,
                np.ones(50),
                mode="valid",
            )

            valid_counts = np.convolve(
                valid.astype(float),
                np.ones(50),
                mode="valid",
            )

            smooth_loss = np.divide(
                loss_sums,
                valid_counts,
                out=np.full_like(loss_sums, np.nan),
                where=valid_counts == 50,
            )

            ax.plot(
                episodes[49:],
                smooth_loss,
                linewidth=2,
                label=f"Seed {seed} - 50-episode average",
            )

    ax.set_title("DQN Training Loss: Random Seed Comparison")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean Training Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    output_path = OUTPUT_DIR / "seed_comparison_training_loss.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"Saved: {output_path}")


# ---------------------------------------------------------
# NUMERICAL SENSITIVITY STATISTICS
# ---------------------------------------------------------

def calculate_statistics(data):
    """Calculate overall, final 100, and final 200 statistics."""

    summary = []

    for seed, result in data.items():

        durations = result["durations"]
        losses = result["losses"]

        final_100 = durations[-100:]
        final_200 = durations[-FINAL_WINDOW:]

        final_100_losses = losses[-100:]
        final_200_losses = losses[-FINAL_WINDOW:]

        valid_losses = losses[np.isfinite(losses)]
        valid_final_100_losses = final_100_losses[
            np.isfinite(final_100_losses)
        ]
        valid_final_200_losses = final_200_losses[
            np.isfinite(final_200_losses)
        ]

        avg_100 = moving_average(durations, 100)

        # First episode at which the 100-episode average
        # reaches or exceeds 475.
        reached_475 = np.where(avg_100 >= 475)[0]

        if len(reached_475) > 0:
            episodes_to_475 = int(
                result["episodes"][reached_475[0] + 99]
            )
        else:
            episodes_to_475 = "Not reached"

        summary.append({
            "seed": seed,
            "episodes": len(durations),

            "overall_mean_reward": np.mean(durations),
            "overall_std_reward": np.std(durations),

            "final_100_mean_reward": np.mean(final_100),
            "final_100_std_reward": np.std(final_100),

            "final_200_mean_reward": np.mean(final_200),
            "final_200_std_reward": np.std(final_200),
            "final_200_min_reward": np.min(final_200),
            "final_200_max_reward": np.max(final_200),

            "final_200_episodes_at_500": int(
                np.sum(final_200 >= 500)
            ),
            "final_200_percentage_at_500": (
                np.mean(final_200 >= 500) * 100
            ),

            "maximum_duration": np.max(durations),

            "episodes_to_100_average_475": episodes_to_475,

            "overall_mean_training_loss": (
                np.mean(valid_losses)
                if len(valid_losses)
                else float("nan")
            ),

            "final_100_mean_training_loss": (
                np.mean(valid_final_100_losses)
                if len(valid_final_100_losses)
                else float("nan")
            ),

            "final_200_mean_training_loss": (
                np.mean(valid_final_200_losses)
                if len(valid_final_200_losses)
                else float("nan")
            ),
        })

    return summary


# ---------------------------------------------------------
# SAVE STATISTICS
# ---------------------------------------------------------

def save_statistics(summary):
    """Save the numerical comparison as a CSV file."""

    output_path = OUTPUT_DIR / "seed_sensitivity_statistics.csv"

    fieldnames = list(summary[0].keys())

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(summary)

    print(f"Saved: {output_path}")

    print("\n========== RANDOM SEED SENSITIVITY RESULTS ==========")

    for result in summary:

        print(f"\nSeed: {result['seed']}")

        print(
            f"Overall mean reward: "
            f"{result['overall_mean_reward']:.2f}"
        )
        print(
            f"Overall reward std: "
            f"{result['overall_std_reward']:.2f}"
        )

        print(
            f"Final 100 episode mean: "
            f"{result['final_100_mean_reward']:.2f}"
        )
        print(
            f"Final 100 episode std: "
            f"{result['final_100_std_reward']:.2f}"
        )

        print(
            f"Final 200 episode mean: "
            f"{result['final_200_mean_reward']:.2f}"
        )
        print(
            f"Final 200 episode std: "
            f"{result['final_200_std_reward']:.2f}"
        )
        print(
            f"Final 200 episode minimum: "
            f"{result['final_200_min_reward']:.0f}"
        )
        print(
            f"Final 200 episode maximum: "
            f"{result['final_200_max_reward']:.0f}"
        )
        print(
            f"Final 200 episodes reaching 500: "
            f"{result['final_200_episodes_at_500']}/200"
        )
        print(
            f"Final 200 percentage reaching 500: "
            f"{result['final_200_percentage_at_500']:.2f}%"
        )

        print(
            f"Maximum duration in training: "
            f"{result['maximum_duration']:.0f}"
        )
        print(
            f"Episodes to 100-episode average >= 475: "
            f"{result['episodes_to_100_average_475']}"
        )

        print(
            f"Final 200 mean training loss: "
            f"{result['final_200_mean_training_loss']:.6f}"
        )


# ---------------------------------------------------------
# CROSS-SEED SUMMARY
# ---------------------------------------------------------

def print_cross_seed_summary(summary):
    """Summarize final-200 mean reward across the three seeds."""

    final_200_means = np.asarray([
        result["final_200_mean_reward"]
        for result in summary
    ])

    print("\n========== CROSS-SEED FINAL 200 SUMMARY ==========")

    print(
        f"Mean of seed-level final-200 means: "
        f"{np.mean(final_200_means):.2f}"
    )
    print(
        f"Standard deviation across seed-level means: "
        f"{np.std(final_200_means, ddof=1):.2f}"
    )
    print(
        f"Lowest seed-level final-200 mean: "
        f"{np.min(final_200_means):.2f}"
    )
    print(
        f"Highest seed-level final-200 mean: "
        f"{np.max(final_200_means):.2f}"
    )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading baseline training data...\n")

    data = {}

    for seed, csv_path in SEED_FILES.items():

        data[seed] = load_training_data(csv_path)

        print(
            f"Seed {seed}: "
            f"{len(data[seed]['durations'])} episodes loaded"
        )

    # Existing comparative graphs
    plot_seed_comparison(data)
    plot_seed_comparison(data, window=50)
    plot_seed_comparison(data, window=100)

    # Existing training loss graph
    plot_loss_comparison(data)

    # New final-200-episode graph
    plot_final_200_comparison(data)

    # Calculate and save numerical statistics
    summary = calculate_statistics(data)
    save_statistics(summary)

    # Cross-seed final-200 comparison
    print_cross_seed_summary(summary)

    print("\nAll random-seed sensitivity analyses completed.")
