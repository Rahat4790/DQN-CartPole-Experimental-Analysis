# Deep Q-Network (DQN) — CartPole-v1 Experimental Analysis

An individual reinforcement-learning term project implementing the key ideas of **Deep Q-Networks (DQN)** and experimentally investigating the roles of experience replay and target networks, together with sensitivity to learning rate, replay-buffer capacity, and neural-network size.

**Paper:** Mnih et al. (2015), *Human-level control through deep reinforcement learning*, Nature 518, 529–533. https://doi.org/10.1038/nature14236

**Environment:** Gymnasium `CartPole-v1`  
**Framework:** PyTorch  
**Implementation:** Custom DQN agent and training loop; Gymnasium is used for the environment and PyTorch for neural-network operations.

---

## 1. Project objective

The project addresses the following question:

> How do experience replay and target networks affect DQN learning stability, and how sensitive is DQN performance to learning rate, replay-buffer capacity, and network size?

The work follows the course term-project requirements:

1. Implement the core DQN learning procedure.
2. Investigate the paper's central qualitative trend that experience replay and a target network are important for stable DQN learning.
3. Vary multiple training settings and compare at least three random seeds per configuration.
4. Discuss implementation decisions, practical difficulties, limitations, and differences from the original paper.

This is a **small-environment DQN study**, not an exact reproduction of the original Atari experiments in Mnih et al. (2015). CartPole-v1 is intentionally used as a computationally inexpensive environment that makes repeated controlled experiments feasible.

## 2. Environment: CartPole-v1

The agent controls a cart and attempts to keep a pole upright.

- **Observation/state:** four continuous values — cart position, cart velocity, pole angle, and pole angular velocity.
- **Action space:** two discrete actions — push left or push right.
- **Reward:** +1 for each time step the pole remains balanced.
- **Episode limit:** 500 time steps in `CartPole-v1`.
- **Episode return:** because each step gives +1, the episode return equals its duration; the maximum possible episode return is 500.

The state can be written as

\[
s_t = [x_t,\; \dot{x}_t,\; \theta_t,\; \dot{\theta}_t].
\]

## 3. DQN implementation

### 3.1 Q-network architecture

The policy and target networks use the same fully connected multilayer perceptron:

```text
4 input features → 128 → 128 → 2 Q-values
                    ReLU   ReLU
```

The two outputs estimate `Q(state, left)` and `Q(state, right)`. The baseline network contains 17,410 trainable parameters:

| Layer | Parameter calculation | Parameters |
|---|---:|---:|
| Input → hidden 1 | `(4 × 128) + 128` | 640 |
| Hidden 1 → hidden 2 | `(128 × 128) + 128` | 16,512 |
| Hidden 2 → output | `(128 × 2) + 2` | 258 |
| **Total** | | **17,410** |

### 3.2 Main algorithmic components

- **Epsilon-greedy action selection:** chooses a random action with probability epsilon and otherwise chooses the action with the largest policy-network Q-value.
- **Experience replay:** stores transitions in a bounded `deque` and samples random minibatches for learning.
- **Bellman target:** uses the reward plus the discounted maximum next-state Q-value. True terminal transitions do not bootstrap.
- **Target network:** supplies the next-state values used in the Bellman target and is slowly updated toward the policy network.
- **Optimization:** Smooth L1 (Huber) loss, AdamW with AMSGrad, backpropagation, and gradient-value clipping.

The implemented target is

\[
y_t = r_t + \gamma\max_{a'}Q_{\text{target}}(s_{t+1},a'),
\]

with the next-state term omitted for a true terminal transition.

The soft target update is

\[
\theta_{\text{target}} \leftarrow \tau\theta_{\text{policy}}
+ (1-\tau)\theta_{\text{target}}.
\]

### 3.3 Baseline hyperparameters

| Parameter | Baseline value |
|---|---:|
| Environment | `CartPole-v1` |
| Hidden layers | 2 × 128 units |
| Batch size | 128 |
| Discount factor (`GAMMA`) | 0.99 |
| Initial epsilon (`EPS_START`) | 0.90 |
| Final epsilon (`EPS_END`) | 0.01 |
| Epsilon decay (`EPS_DECAY`) | 2,500 environment steps |
| Learning rate (`LR`) | 0.0003 |
| Replay capacity | 10,000 transitions |
| Soft-update coefficient (`TAU`) | 0.005 |
| Gradient-value clipping | 100 |
| Optimizer | AdamW, AMSGrad enabled |
| Loss | Smooth L1 / Huber |
| Training episodes in reported experiments | 600 per run |
| Random seeds | 42, 123, 2024 |

The values are defined in `config.py`; experiment-specific values can be passed through the training function or command-line arguments.

### 3.4 Episode termination and time-limit truncation

The training loop ends an episode when either `terminated` or `truncated` is true. It sets `next_state = None` only for `terminated`. Therefore, when an episode ends because the environment reaches its time limit, the final observation is retained and can be used for bootstrapping. This behaviour follows the implementation convention documented in the project code and should be kept in mind when interpreting the time-limit handling.

## 4. Experimental design

All reported configurations use 600 training episodes per run and three seeds (42, 123, 2024). The baseline configuration is reused as the reference for the learning-rate value 0.0003, replay capacity 10,000, and network width 128; it is not counted as a new run for each comparison.

The archived results contain **30 training runs (18,000 episodes)**:

| Study | Configurations | Seed runs represented | Additional runs beyond the shared baseline |
|---|---|---:|---:|
| Baseline / reference | Baseline DQN | 3 | 3 |
| Ablation | No replay; no target network | 6 | 6 |
| Learning rate | 0.0001, 0.0002, 0.0003 (baseline), 0.001 | 12 | 9 |
| Replay capacity | 2,000; 10,000 (baseline); 20,000 | 9 | 6 |
| Network size | 64; 128 (baseline); 256 hidden units | 9 | 6 |
| **Total unique training runs** | | | **30** |

### 4.1 Ablation study

The full DQN baseline is compared with:

1. **No replay:** experience replay is disabled. In this implementation, the replay memory capacity and minibatch size both become 1, so this is an online single-transition update condition.
2. **No target network:** the policy network is used to calculate next-state values, and target-network updates are disabled. Replay remains enabled with batch size 128.

**Important interpretation:** the no-replay condition changes both replay sampling and minibatch size (128 → 1). Therefore, the result should be described as an online single-transition/no-replay ablation, not as a perfectly isolated test of replay sampling alone. The no-target-network condition retains replay and batch size, so it more directly isolates the removal of the separate target network.

### 4.2 Learning-rate sensitivity

Tested values: `0.0001`, `0.0002`, `0.0003` (baseline), and `0.001`.

### 4.3 Replay-capacity sensitivity

Tested capacities: `2,000`, `10,000` (baseline), and `20,000` transitions.

### 4.4 Network-size sensitivity

Tested hidden widths: 64, 128 (baseline), and 256 units in each of the two hidden layers.

| Hidden width | Architecture | Trainable parameters |
|---:|---|---:|
| 64 | 4–64–64–2 | 4,610 |
| 128 | 4–128–128–2 | 17,410 |
| 256 | 4–256–256–2 | 67,586 |

## 5. Results summary

The following values are calculated from the archived `training_log.csv` files. For each seed, the mean episode duration over its final 200 training episodes is calculated first. The table reports the mean and **sample standard deviation across the three seed-level means**. It is not the within-seed episode standard deviation.

| Experiment | Setting | Final-200 mean ± across-seed SD |
|---|---|---:|
| Ablation | Full DQN | **495.02 ± 7.44** |
| Ablation | No replay (online, batch 1) | 15.63 ± 4.92 |
| Ablation | No target network | 9.86 ± 0.08 |
| Learning rate | 0.0001 | 215.71 ± 253.17 |
| Learning rate | 0.0002 | **496.78 ± 5.58** |
| Learning rate | 0.0003 (baseline) | 495.02 ± 7.44 |
| Learning rate | 0.001 | 478.64 ± 13.90 |
| Replay capacity | 2,000 | 449.91 ± 46.06 |
| Replay capacity | 10,000 (baseline) | **495.02 ± 7.44** |
| Replay capacity | 20,000 | 494.10 ± 5.55 |
| Network size | 64 hidden units | 310.88 ± 209.63 |
| Network size | 128 hidden units (baseline) | **495.02 ± 7.44** |
| Network size | 256 hidden units | 409.88 ± 144.64 |

### 5.1 Baseline checkpoint evaluation

The three baseline checkpoints (training seeds 42, 123, and 2024) were evaluated greedily for 200 episodes each. The saved evaluation summaries report a mean return of 500.00 and standard deviation of 0.00 for each checkpoint (600 evaluation episodes total). Evaluation uses `argmax` actions without epsilon exploration. These are evaluation results, not training returns.

### 5.2 Interpretation

- The baseline achieved near-maximum training performance in the final 200 episodes.
- Both ablations performed very poorly, providing qualitative support for the importance of replay and a target network in this implementation. The no-replay batch-size confound must be acknowledged.
- Learning rate 0.0002 was the strongest of the tested learning rates by final-200 mean and cross-seed consistency. This does not establish a universal optimum.
- Replay capacities 10,000 and 20,000 performed similarly; increasing capacity from 10,000 to 20,000 did not produce a clear gain in mean return.
- The 128-unit architecture performed best and was much more consistent across the three seeds than the 64- and 256-unit architectures. Larger networks did not guarantee better performance.

These are results for CartPole-v1, the tested hyperparameter values, and three seeds. They should not be generalized to all environments or DQN implementations.

## 6. Repository structure

```text
DQN-CartPole-Experimental-Analysis/
├── README.md
├── config.py                  # Baseline hyperparameters
├── dqn_network.py             # Feed-forward Q-network
├── replay_memory.py           # Replay buffer and Transition tuple
├── agent.py                   # Action selection, TD target, loss, optimizer
├── train.py                   # Environment interaction and training loop
├── main.py                    # Convenience baseline entry point
├── evaluate.py                # Greedy checkpoint evaluation
├── plot_results.py            # Baseline seed-comparison plots/statistics
├── run_ablation.py            # No-replay and no-target experiments
├── run_learning_rate.py       # Learning-rate sensitivity runner
├── run_replay_capacity.py     # Replay-capacity sensitivity runner
├── analyze_network_size.py    # Network-size statistics from saved CSVs
├── tests/
│   └── test_components.py     # Basic network and replay-buffer tests
├── requirements.txt
└── results/
    ├── baseline/
    ├── ablation/
    ├── network_size/
    └── sensitivity/
```

Each training run saves a `training_log.csv`, a `dqn_checkpoint.pth`, and an `episode_durations.png` under its output directory. The CSV contains episode number, episode duration, mean training loss, learning rate, replay capacity, ablation flags, hidden width, and seed.

## 7. Setup (Windows PowerShell)

Open PowerShell in the repository root (the folder containing `train.py`) and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, run this once for the current user, then activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

The project requires Python 3.10+ in practice; the recorded development environment used Python 3.12.10. Dependencies are listed in `requirements.txt`:

- Gymnasium with classic-control environments
- PyTorch
- NumPy
- Matplotlib

Install pytest separately to run the included tests:

```powershell
pip install pytest
```

## 8. Run experiments

Run commands from the repository root with the virtual environment activated. Use `--episodes 600` explicitly: the training function defaults to 50 episodes on CPU and 600 on CUDA/MPS, while the reported experiments use 600 episodes on every run.

### 8.1 Baseline runs

```powershell
python train.py --episodes 600 --seed 42 --output results/baseline/seed_42
python train.py --episodes 600 --seed 123 --output results/sensitivity/seeds/seed_123
python train.py --episodes 600 --seed 2024 --output results/sensitivity/seeds/seed_2024
```

### 8.2 Ablation study

The runner executes both ablations for all three seeds (six new runs):

```powershell
python run_ablation.py
```

Outputs are written to `results/ablation/no_replay/` and `results/ablation/no_target_network/`.

### 8.3 Learning-rate sensitivity

Run each learning rate separately. Each invocation runs seeds 42, 123, and 2024:

```powershell
python run_learning_rate.py --learning-rate 0.0001 --episodes 600
python run_learning_rate.py --learning-rate 0.0002 --episodes 600
python run_learning_rate.py --learning-rate 0.001 --episodes 600
```

The baseline runs above provide the `0.0003` comparison. Results are stored under `results/sensitivity/learning_rate/`.

### 8.4 Replay-capacity sensitivity

The archived experiment includes seeds 42, 123, and 2024 for each tested capacity. To regenerate all three seeds for each capacity directly with the common training entry point, use:

```powershell
foreach ($capacity in 2000, 10000, 20000) {
    foreach ($seed in 42, 123, 2024) {
        python train.py --episodes 600 --seed $seed --learning-rate 0.0003 `
          --replay-capacity $capacity `
          --output "results/sensitivity/replay_capacity/capacity_$capacity/seed_$seed"
    }
}
```

`run_replay_capacity.py --replay-capacity <capacity>` is also provided, but its current `SEEDS` list contains only 123 and 2024. For a complete three-seed rerun with that helper, change the list to `[42, 123, 2024]` first, or use the direct commands above. The archived seed-42 result is already included in the reported analysis.

### 8.5 Network-size sensitivity

The archived 128-unit results are the baseline runs. Regenerate the 64- and 256-unit configurations with:

```powershell
foreach ($units in 64, 256) {
    foreach ($seed in 42, 123, 2024) {
        python train.py --episodes 600 --seed $seed --hidden-units $units `
          --output "results/network_size/hidden_$units/seed_$seed"
    }
}
```

Then regenerate the network-size per-seed and aggregate statistics:

```powershell
python analyze_network_size.py
```

The analysis script writes `network_size_per_seed_statistics.csv` and `network_size_aggregate_statistics.csv` under `results/network_size/`.

## 9. Generate baseline plots and statistics

`plot_results.py` loads the three baseline training logs (seed 42 from `results/baseline/`, seeds 123 and 2024 from `results/sensitivity/seeds/`). It does not train a model.

```powershell
python plot_results.py
```

It generates the following files in `results/sensitivity/seed_comparison/`:

- `seed_comparison_raw.png`
- `seed_comparison_50_episode_average.png`
- `seed_comparison_100_episode_average.png`
- `seed_comparison_training_loss.png`
- `seed_comparison_final_200_episodes.png`
- `seed_sensitivity_statistics.csv`

Every individual training run also generates `episode_durations.png` in its output folder. The project currently does not include a single automated plotting script that aggregates all ablation, learning-rate, replay-capacity, and network-size configurations into comparative plots; those comparisons can be produced from the saved CSV logs using a separate analysis script/notebook.

## 10. Evaluate a saved checkpoint

The evaluation helper uses greedy action selection (`argmax`) and does not perform epsilon exploration:

```powershell
python evaluate.py --checkpoint results/baseline/seed_42/dqn_checkpoint.pth --episodes 200
```

Optional rendering:

```powershell
python evaluate.py --checkpoint results/baseline/seed_42/dqn_checkpoint.pth --episodes 20 --render
```

**Current limitation:** `evaluate.py` constructs the default 128-hidden-unit network. It is suitable for the baseline checkpoints, but the helper must be updated to read `hidden_units` from the checkpoint before evaluating 64- or 256-unit checkpoints.

## 11. Run tests

```powershell
pytest -q
```

The included tests check the Q-network output shape and basic replay-memory sampling. They are useful smoke tests, not a complete verification of every DQN update or training behaviour.

## 12. Reproducibility and interpretation notes

- Set the same seed and configuration to make runs more comparable. Exact bit-for-bit reproducibility can still depend on software versions, hardware, and backend nondeterminism.
- Use the CSV logs as the primary numerical record; PNGs are visual summaries.
- Compare configurations using the same episode budget and seed set.
- Report both mean and spread across seeds. With three seeds, the estimates of variability are informative but still uncertain.
- Do not treat the 500-step episode cap as evidence of unlimited performance.
- Distinguish training performance (epsilon-greedy exploration is active) from greedy checkpoint evaluation.
- Do not interpret training loss alone as policy performance. Episode return is the direct environment-performance measure.
- The no-replay ablation also changes batch size to one; report this confound.
- The evaluation helper currently supports the 128-unit baseline architecture only.

## 13. Differences from Mnih et al. (2015) and limitations

This project investigates the central DQN mechanisms in a smaller environment. It is not an exact Atari DQN reproduction. In particular:

- CartPole has a four-dimensional state and two discrete actions; the original paper evaluated high-dimensional visual Atari observations.
- This project uses a small fully connected MLP; the original Atari system used a convolutional neural network.
- This implementation uses AdamW with AMSGrad, Smooth L1 loss, and soft target updates with `tau = 0.005`. These are implementation choices for this project and should not be presented as an exact match to every optimizer, loss, and target-update detail in the original paper.
- The experiments use three random seeds and 600 episodes per run. More seeds and additional environments would strengthen generality.
- Only selected values of learning rate, replay capacity, and network width were tested; the results do not identify global optima.
- The no-replay condition changes both replay use and minibatch size.
- The current automated comparison plotting is strongest for baseline seed sensitivity; cross-configuration aggregate plots require additional post-processing.

A useful next step would be to add a controlled no-random-replay condition that keeps the batch size and update count comparable, extend greedy evaluation to every configuration, and automate plots and aggregate statistics for all experimental groups.

## 14. References

1. Mnih, V., et al. (2015). **Human-level control through deep reinforcement learning.** *Nature*, 518, 529–533. https://doi.org/10.1038/nature14236
2. Gymnasium documentation: https://gymnasium.farama.org/
3. PyTorch documentation: https://pytorch.org/docs/stable/index.html

---

**Project note:** The code, CSV logs, plots, and checkpoints in `results/` are the archived experimental artifacts. If a new run is performed, keep its configuration, seed, command, and output directory recorded so the report can be regenerated and audited.
