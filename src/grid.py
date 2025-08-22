from typing import Tuple

import numpy as np

from src.field import Field


class Grid:
    """Representation of single player's playing grid (rendered twice for each player during the game)"""

    def __init__(self, player: str, grid_size: int):
        self.player = player
        self.grid_size = grid_size
        self.fields = np.ndarray = np.empty((grid_size, grid_size), dtype=object)
        for row in range(grid_size):
            for col in range(grid_size):
                self.fields[row][col] = Field(row=row, col=col)

    def __getitem__(self, idx: Tuple[int, int]) -> Field:
        """Access fields via grid[row, col]"""
        row, col = idx
        return self.fields[row, col]

    def __setitem__(self, idx: Tuple[int, int], value: Field):
        """Modify fields directly via grid[row, col]"""
        self.fields[idx] = value
