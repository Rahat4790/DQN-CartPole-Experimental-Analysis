
import csv
import statistics
from pathlib import Path

import numpy as np


EXPERIMENTS = {
    64: {
        42: "results/network_size/hidden_64/seed_42/training_log.csv",
        123: "results/network_size/hidden_64/seed_123/training_log.csv",
        2024: "results/network_size/hidden_64/seed_2024/training_log.csv",
    },
    128: {
        42: "results/baseline/seed_42/training_log.csv",
        123: "results/sensitivity/seeds/seed_123/training_log.csv",
        2024: "results/sensitivity/seeds/seed_2024/training_log.csv",
    },
    256: {
        42: "results/network_size/hidden_256/seed_42/training_log.csv",
        123: "results/network_size/hidden_256/seed_123/training_log.csv",
        2024: "results/network_size/hidden_256/seed_2024/training_log.csv",
    },
}

PARAMETERS = {
    64: 4610,
    128: 17410,
    256: 67586,
}

OUTPUT_DIR = Path("results/network_size")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_durations(csv_path):
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Missing file: {csv_path}")

    durations = []

    with open(csv_path, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        if "duration" not in reader.fieldnames:
            raise ValueError(f"'duration' column not found in {csv_path}")

        for row in reader:
            durations.append(float(row["duration"]))

    if len(durations) != 600:
        raise ValueError(
            f"Expected 600 episodes in {csv_path}, "
            f"but found {len(durations)}"
        )

    return np.array(durations, dtype=float)


def first_threshold_episode(durations, threshold=475, window=100):
    if len(durations) < window:
        return None

    moving_means = np.convolve(
        durations,
        np.ones(window) / window,
        mode="valid"
    )

    indices = np.where(moving_means >= threshold)[0]

    if len(indices) == 0:
        return None

    # The first window ends at this episode.
    return int(indices[0] + window)


per_seed_results = []

print("\n" + "=" * 85)
print("NETWORK SIZE SENSITIVITY ANALYSIS")
print("=" * 85)

for hidden_units, seeds in EXPERIMENTS.items():

    for seed, csv_path in seeds.items():

        durations = load_durations(csv_path)

        final_100 = durations[-100:]
        final_200 = durations[-200:]

        result = {
            "hidden_units": hidden_units,
            "parameters": PARAMETERS[hidden_units],
            "seed": seed,
            "overall_mean": np.mean(durations),
            "overall_std": np.std(durations, ddof=0),
            "final_100_mean": np.mean(final_100),
            "final_100_std": np.std(final_100, ddof=0),
            "final_200_mean": np.mean(final_200),
            "final_200_std": np.std(final_200, ddof=0),
            "final_200_max_count": int(np.sum(final_200 == 500)),
            "first_100ep_threshold_475": first_threshold_episode(
                durations
            ),
        }

        per_seed_results.append(result)


# Save per-seed statistics.
per_seed_path = OUTPUT_DIR / "network_size_per_seed_statistics.csv"

fieldnames = list(per_seed_results[0].keys())

with open(per_seed_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(per_seed_results)


# Aggregate the three seed-level results for each architecture.
aggregate_results = []

for hidden_units in EXPERIMENTS:

    group = [
        row for row in per_seed_results
        if row["hidden_units"] == hidden_units
    ]

    final_100_means = [row["final_100_mean"] for row in group]
    final_200_means = [row["final_200_mean"] for row in group]
    overall_means = [row["overall_mean"] for row in group]

    aggregate_results.append({
        "hidden_units": hidden_units,
        "architecture": f"4-{hidden_units}-{hidden_units}-2",
        "parameters": PARAMETERS[hidden_units],
        "cross_seed_final_100_mean": np.mean(final_100_means),
        "cross_seed_final_100_sd": statistics.stdev(final_100_means),
        "cross_seed_final_200_mean": np.mean(final_200_means),
        "cross_seed_final_200_sd": statistics.stdev(final_200_means),
        "cross_seed_overall_mean": np.mean(overall_means),
        "seeds_at_500_final_100": sum(
            np.isclose(value, 500.0) for value in final_100_means
        ),
    })


aggregate_path = OUTPUT_DIR / "network_size_aggregate_statistics.csv"

with open(aggregate_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(aggregate_results[0].keys())
    )
    writer.writeheader()
    writer.writerows(aggregate_results)


# Print detailed per-seed results.
print("\nPER-SEED RESULTS")
print("-" * 85)

for row in per_seed_results:
    print(
        f"Network: {row['hidden_units']:3d} | "
        f"Seed: {row['seed']:4d} | "
        f"Final 100 mean: {row['final_100_mean']:7.2f} | "
        f"Final 100 SD: {row['final_100_std']:7.2f} | "
        f"Final 200 mean: {row['final_200_mean']:7.2f} | "
        f"First 475 crossing: {row['first_100ep_threshold_475']}"
    )


# Print architecture-level comparison.
print("\nARCHITECTURE COMPARISON")
print("-" * 105)
print(
    f"{'Hidden':>7} {'Parameters':>12} "
    f"{'Final-100 Mean':>16} {'Across-seed SD':>16} "
    f"{'Final-200 Mean':>16} {'Across-seed SD':>16}"
)

for row in aggregate_results:
    print(
        f"{row['hidden_units']:7d} "
        f"{row['parameters']:12,d} "
        f"{row['cross_seed_final_100_mean']:16.2f} "
        f"{row['cross_seed_final_100_sd']:16.2f} "
        f"{row['cross_seed_final_200_mean']:16.2f} "
        f"{row['cross_seed_final_200_sd']:16.2f}"
    )


print("\nAnalysis completed successfully.")
print(f"Per-seed CSV: {per_seed_path}")
print(f"Aggregate CSV: {aggregate_path}")
