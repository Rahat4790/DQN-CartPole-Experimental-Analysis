
"""DQN action selection and optimization with configurable ablations."""

import math
import random

import torch
from torch import nn, optim

from config import (
    BATCH_SIZE,
    GAMMA,
    EPS_START,
    EPS_END,
    EPS_DECAY,
    TAU,
    LR,
    REPLAY_MEMORY_CAPACITY,
    GRADIENT_CLIP_VALUE,
)

from dqn_network import DQN
from replay_memory import ReplayMemory, Transition


class DQNAgent:
    def __init__(
        self,
        n_observations: int,
        n_actions: int,
        device: torch.device,
        env,
        seed: int | None = None,
        learning_rate: float | None = None,
        replay_capacity: int | None = None,
        use_replay: bool = True,
        use_target_network: bool = True,
        hidden_units: int = 128,
    ):
        self.device = device
        self.env = env
        self.n_observations = n_observations
        self.n_actions = n_actions

        self.learning_rate = LR if learning_rate is None else learning_rate
        self.use_replay = use_replay
        self.use_target_network = use_target_network
        self.hidden_units = hidden_units

        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be greater than zero.")

        if (
            isinstance(hidden_units, bool)
            or not isinstance(hidden_units, int)
            or hidden_units <= 0
        ):
            raise ValueError("Hidden units must be a positive integer.")

        if not isinstance(use_replay, bool):
            raise ValueError("use_replay must be True or False.")

        if not isinstance(use_target_network, bool):
            raise ValueError(
                "use_target_network must be True or False."
            )

        selected_capacity = (
            REPLAY_MEMORY_CAPACITY
            if replay_capacity is None
            else replay_capacity
        )

        if (
            isinstance(selected_capacity, bool)
            or not isinstance(selected_capacity, int)
            or selected_capacity <= 0
        ):
            raise ValueError("Replay capacity must be a positive integer.")

        # Reproducibility.
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # Policy network.
        self.policy_net = DQN(
            n_observations,
            n_actions,
            hidden_units=hidden_units,
        ).to(device)

        # Target network: exactly the same architecture.
        self.target_net = DQN(
            n_observations,
            n_actions,
            hidden_units=hidden_units,
        ).to(device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer.
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(),
            lr=self.learning_rate,
            amsgrad=True,
        )

        # Experience replay configuration.
        if self.use_replay:
            self.replay_capacity = selected_capacity
            self.batch_size = BATCH_SIZE
        else:
            self.replay_capacity = 1
            self.batch_size = 1

        self.memory = ReplayMemory(self.replay_capacity)

        self.steps_done = 0
        self.loss_history = []

        self.criterion = nn.SmoothL1Loss()

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """Select an action using epsilon-greedy exploration."""

        sample = random.random()

        eps_threshold = EPS_END + (
            EPS_START - EPS_END
        ) * math.exp(-1.0 * self.steps_done / EPS_DECAY)

        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)

        return torch.tensor(
            [[self.env.action_space.sample()]],
            device=self.device,
            dtype=torch.long,
        )

    def optimize_model(self):
        """Perform one Q-learning update."""

        if len(self.memory) < self.batch_size:
            return None

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(state is not None for state in batch.next_state),
            device=self.device,
            dtype=torch.bool,
        )

        non_final_next_states = (
            torch.cat(
                [state for state in batch.next_state if state is not None]
            )
            if non_final_mask.any()
            else torch.empty(
                (0, self.n_observations),
                device=self.device,
                dtype=torch.float32,
            )
        )

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(
            1, action_batch
        )

        next_state_values = torch.zeros(
            self.batch_size,
            device=self.device,
        )

        if self.use_target_network:
            value_network = self.target_net
        else:
            value_network = self.policy_net

        with torch.no_grad():
            if non_final_mask.any():
                next_state_values[non_final_mask] = (
                    value_network(non_final_next_states).max(1).values
                )

        expected_state_action_values = (
            reward_batch + GAMMA * next_state_values
        )

        loss = self.criterion(
            state_action_values,
            expected_state_action_values.unsqueeze(1),
        )

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(
            self.policy_net.parameters(),
            GRADIENT_CLIP_VALUE,
        )

        self.optimizer.step()

        loss_value = float(loss.detach().cpu().item())
        self.loss_history.append(loss_value)

        return loss_value

    def soft_update_target(self):
        """Soft-update target network when enabled."""

        if not self.use_target_network:
            return

        target_state = self.target_net.state_dict()
        policy_state = self.policy_net.state_dict()

        for key in policy_state:
            target_state[key] = (
                TAU * policy_state[key]
                + (1.0 - TAU) * target_state[key]
            )

        self.target_net.load_state_dict(target_state)
