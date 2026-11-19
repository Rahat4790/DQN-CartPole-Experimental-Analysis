# DQN Tutorial Baseline — CartPole-v1

This repository reconstructs the DQN implementation shown in the 15 code screenshots
uploaded by the user. It is intended as a clean, runnable baseline before any
research modifications.

## Algorithm retained from the screenshots

- Gymnasium `CartPole-v1`
- PyTorch feed-forward Q-network: observation → 128 → 128 → actions
- ReLU activations
- Replay memory implemented with `deque` and a `Transition` named tuple
- Epsilon-greedy action selection with exponential decay
- Batch size 128; discount factor 0.99
- Epsilon start 0.90, end 0.01, decay 2500
- AdamW optimizer, learning rate 3e-4, AMSGrad enabled
- Smooth L1 (Huber) loss
- Gradient-value clipping at 100
- Soft target-network update with tau 0.005
- Tutorial device-dependent training budget: 600 episodes on CUDA/MPS,
  50 episodes on CPU

The tutorial screenshots show the environment reward being stored directly:
`reward_tensor = torch.tensor([reward], device=device)`. No custom reward shaping
is introduced.

## Important implementation note

The tutorial ends an episode when `terminated or truncated`, but sets
`next_state = None` only when `terminated`. This behaviour is retained from the
screenshots. Consequently, a time-limit truncation keeps its final observation
for bootstrapping, while a true terminal state does not.

The screenshots do not clearly expose every character of every line. Obvious
OCR/transcription issues (such as the class name `DQN`, Python syntax, and
PyTorch API spelling) have been reconstructed into valid Python. The replay
capacity is set to 10,000, following the standard implementation pattern; verify
this against the original video if exact capacity is material to your report.

## 1. Create environment (Windows PowerShell)

```powershell
cd DQN_Tutorial_Baseline
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate the environment again.

## 2. Run the baseline

```powershell
python main.py
```

On CPU, the tutorial's default is 50 episodes. On CUDA/MPS, it is 600.
For a controlled test or longer CPU run:

```powershell
python train.py --episodes 500 --seed 42 --output results/seed_42
```

Optional live environment rendering:

```powershell
python train.py --episodes 50 --seed 42 --render
```

## 3. Evaluate a saved checkpoint

```powershell
python evaluate.py --checkpoint results/seed_42/dqn_checkpoint.pth --episodes 20
```

Add `--render` to watch evaluation episodes.

## 4. Test components

```powershell
pip install pytest
pytest -q
```

## Files

- `config.py`: hyperparameters transcribed from the screenshots
- `dqn_network.py`: Q-network
- `replay_memory.py`: replay buffer
- `agent.py`: epsilon-greedy selection, Bellman target, Huber loss,
  gradient clipping, optimizer, and soft target update
- `train.py`: environment interaction and training loop
- `plot_results.py`: episode-duration and 100-episode moving-average plot
- `evaluate.py`: greedy policy evaluation
- `tests/`: basic component tests

## Scope

This is a small-environment DQN reproduction baseline, not an exact reproduction
of the Atari experiments in Mnih et al. (2015). Keep this baseline unchanged
when conducting later ablation or sensitivity experiments; create separate
configurations or branches for those experiments.
