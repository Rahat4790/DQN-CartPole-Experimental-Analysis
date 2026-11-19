"""Greedy evaluation helper; does not modify the tutorial training algorithm."""
import argparse
import numpy as np
import torch
import gymnasium as gym

from config import ENV_ID
from dqn_network import DQN
from train import choose_device


def evaluate(checkpoint, episodes=20, render=False):
    device = choose_device()
    env = gym.make(ENV_ID, render_mode="human" if render else None)
    state, _ = env.reset()
    model = DQN(len(state), env.action_space.n).to(device)
    data = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(data["policy_net_state_dict"])
    model.eval()

    returns = []
    for ep in range(episodes):
        state, _ = env.reset(seed=10_000 + ep)
        total = 0.0
        done = False
        while not done:
            state_tensor = torch.tensor(
                state, dtype=torch.float32, device=device
            ).unsqueeze(0)
            with torch.no_grad():
                action = model(state_tensor).argmax(dim=1).item()
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        returns.append(total)

    env.close()
    print(f"Evaluation episodes: {episodes}")
    print(f"Mean return: {np.mean(returns):.2f}")
    print(f"Std return:  {np.std(returns, ddof=1) if len(returns)>1 else 0:.2f}")
    print(f"Returns: {returns}")
    return returns


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="results/run/dqn_checkpoint.pth")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.episodes, args.render)
