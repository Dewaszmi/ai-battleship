import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from ai_battleship.ai_agent.environment import CNN
from ai_battleship.constants import GRID_SIZE


class Agent:
    def __init__(self, grid_size=GRID_SIZE, hidden_dim=128, lr=1e-3, gamma=0.99):
        self.grid_size = grid_size
        self.gamma = gamma
        self.memory = deque(maxlen=5000)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNN(self.grid_size, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)


    def select_action(self, state, epsilon=0):
        """Select action from the possible input space using epsilon-greedy rule, epsilon is set to 0 by default (always use max Q value)"""
        if random.random() < epsilon:
            # explore
            idx = random.randint(0, self.grid_size * self.grid_size - 1)
        else:
            # exploit
            with torch.no_grad():
                state = state.to(self.device)
                q_values = self.model(state)
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

        q_values = self.model(states)
        next_q_values = self.model(next_states)

        action_indices = [r * self.grid_size + c for (r, c) in actions]
        q_selected = q_values[range(batch_size), action_indices]

        target = rewards + self.gamma * (1 - dones) * next_q_values.max(dim=1)[0]

        loss = nn.MSELoss()(q_selected, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
