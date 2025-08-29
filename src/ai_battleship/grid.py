from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ai_battleship.field import Field


class Grid:
    """Representation of single player's playing grid (rendered twice for each player during the game)"""

    def __init__(self, player: str, grid_size: int):
        self.player = player
        self.grid_size = grid_size
        self.fields = np.empty((grid_size, grid_size), dtype=object)
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

    def field_exists(self, row, col):
        return row in range(self.grid_size) and col in range(self.grid_size)


@dataclass
class Cursor:
    """Cursor used by the player for navigating the grid"""

    row: int
    col: int

    def move(self, direction: str, grid_size: int):
        new_row, new_col = self.row, self.col

        match direction:
            case "up":
                new_row -= 1
            case "right":
                new_col += 1
            case "down":
                new_row += 1
            case "left":
                new_col -= 1
            case _:
                raise ValueError("Invalid movement direction")

        if new_row in range(grid_size) and new_col in range(grid_size):
            self.row, self.col = new_row, new_col
