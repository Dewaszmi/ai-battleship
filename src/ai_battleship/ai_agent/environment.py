from collections import deque

import numpy as np
import torch
import torch.nn as nn

from ai_battleship.constants import GRID_SIZE, SHIPS_DICT
from ai_battleship.grid import Grid
from ai_battleship.utils.grid_utils import generate_random_grid, shoot

# Goals for the agent to learn:
# - shooting fields surrounding the "hit" field is valuable --> (high probability of getting another hit)
# - repeated shooting of certain fields: ["hit", "sunk", "miss"] is bad --> (no valuable reward)
# - shooting the neighboring fields of a sunk ship is bad --> (guaranteed to have no ship fields)
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
        """Return the data after performing a chosen action"""
        row, col = action
        valid = shoot(self.target_grid, row, col)
        if not valid:
            reward = -1.0 # Penalize useless moves - shooting already targeted fields / empty fields
        else:
            target = self.target_grid[row, col]
            if target.status == "miss":
                reward = -0.1 # Small penalty for missing
            elif target.status == "hit":
                reward = 1.0 # Reward for hits
            elif target.status == "sunk":
                reward = 2.0 # Large reward for sinking a ship
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
            "unknown": 0, # unshot field
            "ship": 0, # unshot ship field
            "miss": -1, # missed field
            "hit": 1, # hit but not sunk ship field
            "sunk": 2, # sunk ship field
            "empty": 0, # field surrounding a sunk ship, guaranteed to be empty
        }
        state = np.array(
            [[status_map.get(f.status, 0) for f in row] for row in grid.fields]
        )
        # pass a tensor shaped [1, 10, 10] (channel amount, height, width)
        return torch.tensor(state, dtype=torch.float32).unsqueeze(0)


class CNN(nn.Module):
    def __init__(self, grid_size, hidden_dim=128):
        super().__init__()
        self.grid_size = grid_size
        self.output_dim = self.grid_size**2
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # output: 16 x grid x grid
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Flatten and fully connected
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * grid_size * grid_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim)  # Q-value for each cell
        )

    def forward(self, x):
        # x shape: (batch_size, grid_size*grid_size)
        batch_size = x.size(0)
        x = x.view(batch_size, 1, self.grid_size, self.grid_size)  # reshape to (B, C, H, W)
        x = self.conv(x)
        x = self.fc(x)
        return x

