from collections import deque

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from ai_battleship.config import Config
from ai_battleship.constants import GRID_SIZE, SHIPS_DICT
from ai_battleship.grid import Grid
from ai_battleship.utils.grid_utils import generate_random_grid, shoot


class BattleshipEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, config: Config | None = None):
        super().__init__()
        if config is None:
            config = Config()
        self.config = config

        self.grid_size = GRID_SIZE
        self.num_channels = 3

        self.action_space = spaces.Discrete(self.grid_size * self.grid_size)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.num_channels, self.grid_size, self.grid_size), dtype=np.float32
        )

        self.grid = None
        self.done = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        ships_queue = deque([k for k, v in SHIPS_DICT.items() for _ in range(v)])
        self.grid = generate_random_grid(ships_queue)
        self.done = False
        state = self.get_state_from_grid(self.grid)
        info = (
            {"action_mask": self.get_action_mask()} if not self.config.allow_repeated_shots else {}
        )  # get mask of invalid shots if required
        return state, info

    def step(self, action):
        row = action // self.grid_size
        col = action % self.grid_size
        target = self.grid[row, col]
        shoot(self.grid, target, mark_sunk_neighbors=self.config.mark_sunk_neighbors)

        self.done = all(all(f.status == "sunk" for f in ship) for ship in self.grid.ships)
        reward = -0.01

        observation = self.get_state_from_grid(self.grid)
        info = (
            {"action_mask": self.get_action_mask()} if not self.config.allow_repeated_shots else {}
        )  # get mask of invalid shots if required
        return observation, reward, self.done, False, info  # obs, reward, terminated, truncated, info

    @staticmethod
    def get_state_from_grid(grid: Grid) -> torch.Tensor:
        H, W = grid.grid_size, grid.grid_size
        statuses = np.array([[f.status for f in row] for row in grid.fields])
        state = np.zeros((3, H, W), dtype=np.float32)

        state[0] = ((statuses == "unknown") | (statuses == "ship")).astype(np.float32)
        state[1] = (statuses == "hit").astype(np.float32)
        state[2] = ((statuses == "miss") | (statuses == "sunk") | (statuses == "empty")).astype(np.float32)

        return state

    def get_action_mask(self):
        mask = np.zeros(self.grid_size**2, dtype=np.int8)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                field = self.grid.fields[r][c]
                if field.status in ("unknown", "ship"):  # valid if not already shot / confirmed empty
                    mask[r * self.grid_size + c] = 1
        return mask

    def render(self):
        grid_str = "\n".join(" ".join(f.status[0].upper() for f in row) for row in self.grid.fields)
        print(grid_str)

    def close(self):
        pass
