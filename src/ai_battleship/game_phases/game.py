from dataclasses import dataclass, field

from ai_battleship.constants import *
from ai_battleship.game_phases.base import Phase
from ai_battleship.grid import Cursor


@dataclass
class Game(Phase):
    cursor: Cursor = field(default_factory=lambda: Cursor(0, 0))
    done: bool = False

    def draw(self, screen):
        self.draw_grid(screen, grid=self.player_grid, offset_x=0)
        self.draw_grid_with_cursor(
            screen, grid=self.ai_grid, offset_x=20, cursor=self.cursor
        )
