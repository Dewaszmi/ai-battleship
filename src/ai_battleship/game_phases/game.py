from dataclasses import dataclass

from ai_battleship.game_phases.base import Phase

HIGHLIGHT = {"bad": (255, 64, 64)}


@dataclass
class Game(Phase):

    def get_targeted_ship(self, target):
        return next((ship for ship in self.ai_grid.ships if target in ship), None)

    def check_if_sunk(self, target):
        targeted_ship = self.get_targeted_ship(target)
        if targeted_ship and all(field.status == "hit" for field in targeted_ship):
            for field in targeted_ship:
                field.set_status("sunk")
            return len(targeted_ship)
        return False

    def shoot(self):
        target = self.ai_grid[self.cursor.row, self.cursor.col]
        (
            target.set_status("hit")
            if target.status == "ship"
            else target.set_status("empty")
        )
        sink_result = self.check_if_sunk(target)
        if sink_result:
            print(f"Ship sank at AI grid: {sink_result}")

    def move_cursor(self, direction):
        current_target = self.ai_grid[self.cursor.row, self.cursor.col]
        current_target.set_color()

        self.cursor.move(direction, grid_size=self.ai_grid.grid_size)

        new_target = self.ai_grid[self.cursor.row, self.cursor.col]
        if new_target.status == "miss":
            new_target.set_color(HIGHLIGHT["bad"])

    def move(self, direction):
        self.move_cursor(direction)

    def confirm(self):
        self.shoot()

    def draw(self, screen, cursor_pos=1):
        super().draw(screen, cursor_pos=cursor_pos)
