import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from ai_battleship.ai_agent.environment import CNN
from ai_battleship.constants import GRID_SIZE


class Agent:
    def __init__(
        self, grid_size=GRID_SIZE, hidden_dim=128, lr=1e-3, gamma=0.99, tau=0.01
    ):
        self.grid_size = grid_size
        self.gamma = gamma
        self.tau = tau  # soft update rate for target network
        self.memory = deque(maxlen=5000)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNN(self.grid_size, hidden_dim).to(self.device)
        self.target_model = CNN(self.grid_size, hidden_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss

    def select_action(self, state, epsilon=0, invalid_mask=None):
        """
        Epsilon-greedy with optional masking of invalid moves.
        invalid_mask: tensor of shape [grid_size*grid_size], 1=valid, 0=invalid
        """
        if random.random() < epsilon:
            # explore
            valid_indices = (
                torch.nonzero(invalid_mask).flatten()
                if invalid_mask is not None
                else None
            )
            if valid_indices is not None and len(valid_indices) > 0:
                idx = int(valid_indices[torch.randint(len(valid_indices), (1,))])
            else:
                idx = random.randint(0, self.grid_size * self.grid_size - 1)
        else:
            # exploit
            with torch.no_grad():
                state = state.to(self.device)
                q_values = self.model(state).squeeze(0)  # shape [grid_size*grid_size]
                if invalid_mask is not None:
                    q_values = q_values.clone()
                    q_values[invalid_mask == 0] = -float("inf")  # mask invalid
                idx = q_values.argmax().item()

        row, col = divmod(idx, self.grid_size)
        return row, col

    def add_to_memory(self, transition):
        self.memory.append(transition)

    def train_step(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Q-values for current states
        q_values = self.model(states)  # shape [B, grid_size*grid_size]

        # Select actions
        action_indices = torch.tensor(
            [r * self.grid_size + c for (r, c) in actions]
        ).to(self.device)
        q_selected = q_values[range(batch_size), action_indices]

        # Double DQN target
        with torch.no_grad():
            next_q_main = self.model(next_states)
            next_actions = next_q_main.argmax(dim=1)
            next_q_target = self.target_model(next_states)
            q_target = (
                rewards
                + self.gamma
                * (1 - dones)
                * next_q_target[range(batch_size), next_actions]
            )

        # Compute Huber loss
        loss = self.loss_fn(q_selected, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update target network
        for target_param, param in zip(
            self.target_model.parameters(), self.model.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )
