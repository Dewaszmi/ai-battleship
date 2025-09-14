import random
from collections import deque
from typing import cast

import torch
import torch.nn as nn
import torch.optim as optim

from ai_battleship.ai_agent.model import CNN


class Agent:
    def __init__(self, grid_size: int, device: str | torch.device = "cpu"):
        self.grid_size = grid_size
        self.device = torch.device(device)

        self.model = CNN(grid_size=grid_size).to(self.device)
        self.target_model = CNN(grid_size=grid_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer: optim.Optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()

        self.memory: deque[
            tuple[torch.Tensor, tuple[int, int], float, torch.Tensor, bool]
        ] = deque(maxlen=5000)
        self.gamma = 0.99

    def select_action(
        self, state: torch.Tensor, epsilon: float = 0.1
    ) -> tuple[int, int]:
        """epsilon-greedy action selection"""
        if random.random() < epsilon:
            # random action
            return random.randrange(self.grid_size), random.randrange(self.grid_size)

        with torch.no_grad():
            q_values = self.model(state.unsqueeze(0).to(self.device))  # [1, H*W]
            action_idx = q_values.argmax(dim=1).item()
        row, col = divmod(action_idx, self.grid_size)
        return row, col

    def add_to_memory(
        self,
        transition: tuple[torch.Tensor, tuple[int, int], float, torch.Tensor, bool],
    ):
        self.memory.append(transition)

    def train_step(self, batch_size: int = 64):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # flatten action indices
        action_indices = [r * self.grid_size + c for (r, c) in actions]
        action_indices = torch.tensor(action_indices, device=self.device)

        # Q(s, a)
        q_values = self.model(states)  # [B, H*W]
        q_values = q_values.gather(1, action_indices.unsqueeze(1)).squeeze(1)

        # Target Q
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
            targets = cast(
                torch.Tensor, rewards + self.gamma * next_q_values * (1 - dones)
            )

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
