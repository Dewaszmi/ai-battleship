from collections import deque

import numpy as np
import torch
import torch.nn as nn

from ai_battleship.constants import GRID_SIZE, SHIPS_DICT
from ai_battleship.grid import Grid
from ai_battleship.utils.grid_utils import generate_random_grid, shoot


class AgentEnvironment:

    def init__(self):
        self.grid_size = GRID_SIZE
        self.reset()

    def reset(self):
        """Generate new target grid for training"""
        ships_queue = deque([k for k, v in SHIPS_DICT.items() for _ in range(v)])
        self.target_grid = generate_random_grid(ships_queue)
        self.done = False
        return self.get_state_from_grid(self.target_grid)

    def step(self, action):  # action = (row, col)
        row, col = action
        valid = shoot(self.target_grid, row, col)
        if not valid:
            # Penalize illegal moves (already targeted)
            reward = -1.0
        else:
            field = self.target_grid[row, col]
            if field.status == "miss":
                reward = -0.1
            elif field.status == "hit":
                reward = 1.0
            elif field.status == "sunk":
                reward = 2.0
            else:
                reward = 0.0

        # Check if all ships sunk
        self.done = all(
            all(f.status == "sunk" for f in ship) for ship in self.target_grid.ships
        )
        return self.get_state_from_grid(self.target_grid), reward, self.done

    @staticmethod
    def get_state_from_grid(grid: Grid):
        """Return grid as a tensor (for the agent)."""
        status_map = {
            "unknown": 0,
            "ship": 0,
            "miss": -1,
            "hit": 1,
            "sunk": 2,
            "empty": -1,
        }
        state = np.array(
            [[status_map.get(f.status, 0) for f in row] for row in grid.fields]
        )
        return torch.tensor(state, dtype=torch.float32).flatten()


class DQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)
