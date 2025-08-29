from dataclasses import dataclass

from ai_battleship.game_phases.base import Phase


@dataclass
class Game(Phase):

    def shoot(self):
        target_field = self.ai_grid[self.cursor.row, self.cursor.col]
        (
            target_field.set_status("hit")
            if target_field.status == "ship"
            else target_field.set_status("empty")
        )

    def draw(self, screen, cursor_pos=1):
        super().draw(screen, cursor_pos=cursor_pos)
