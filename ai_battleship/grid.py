from dataclasses import dataclass, field
from typing import cast

import numpy as np

from ai_battleship.field import Field


@dataclass
class Grid:
    """Representation of single player's playing grid (rendered twice for each player during the game)"""

    grid_size: int
    fields: np.ndarray = field(init=False)
    ships: list[list[Field]] = field(default_factory=list)

    def __post_init__(self):
        self.fields = np.empty((self.grid_size, self.grid_size), dtype=object)
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                self.fields[row][col] = Field(row=row, col=col)

    def __getitem__(self, idx: tuple[int, int]) -> Field:
        """Access fields via grid[row, col]"""

        row, col = idx
        return cast(Field, self.fields[row, col])

    def __setitem__(self, idx: tuple[int, int], value: Field):
        """Modify fields directly via grid[row, col]"""
        self.fields[idx] = value

    def field_exists(self, row: int, col: int):
        return row in range(self.grid_size) and col in range(self.grid_size)
