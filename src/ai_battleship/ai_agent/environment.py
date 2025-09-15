from collections import deque

import numpy as np
import torch

from ai_battleship.constants import GRID_SIZE, SHIPS_DICT
from ai_battleship.grid import Grid
from ai_battleship.utils.grid_utils import generate_random_grid, shoot


# Goals for the agent to learn:
# - shooting fields surrounding the "hit" field is valuable --> (high probability of getting another hit)
# - repeated shooting of certain fields: ["hit", "sunk", "miss"] is bad --> (no valuable reward)
# - shooting the neighboring fields of a sunk ship is bad --> (guaranteed to have no ship fields)
class AgentEnvironment:

    def __init__(self):
        self.grid_size: int = GRID_SIZE
        self.reset()

    def reset(self):
        """Generate new target grid for training"""
        ships_queue = deque([k for k, v in SHIPS_DICT.items() for _ in range(v)])
        self.grid = generate_random_grid(ships_queue)
        self.done = False
        return self.get_state_from_grid(self.grid)

    def step(self, action: tuple[int, int]):  # action = (row, col)
        """Return the data after performing a chosen action"""
        row, col = action
        target = self.grid[row, col]
        valid: bool = shoot(self.grid, target)
        if not valid:
            reward = -1.0
        else:
            status = target.status
            if status == "miss":
                reward = -0.1
            elif status == "hit":
                reward = 1.0
            elif status == "sunk":
                reward = 3.0
            else:
                raise ValueError("Invalid target status")

        # Check if all ships sunk
        self.done = all(
            all(f.status == "sunk" for f in ship) for ship in self.grid.ships
        )
        return self.get_state_from_grid(self.grid), reward, self.done

    @staticmethod
    def get_state_from_grid(grid: Grid) -> torch.Tensor:
        """Return grid as a multi-channel tensor for the CNN."""
        status_list = [
            "unknown_or_ship",  # unshot field
            "miss",  # missed field
            "hit",  # hit but not sunk ship field
            "sunk",  # sunk ship field
            "empty",  # field surrounding a sunk ship, guaranteed to be empty
        ]
        H, W = grid.grid_size, grid.grid_size

        # Extract status as an array of strings
        statuses = np.array([[f.status for f in row] for row in grid.fields])

        # Initialize tensor [channels, H, W]
        state = torch.zeros((len(status_list), H, W), dtype=torch.float32)

        state[0] = torch.from_numpy(
            ((statuses == "unknown") | (statuses == "ship")).astype(np.float32)
        )
        state[1] = torch.from_numpy((statuses == "miss").astype(np.float32))
        state[2] = torch.from_numpy((statuses == "hit").astype(np.float32))
        state[3] = torch.from_numpy((statuses == "sunk").astype(np.float32))
        state[4] = torch.from_numpy((statuses == "empty").astype(np.float32))

        return state  # shape: [channels=5, H, W]
