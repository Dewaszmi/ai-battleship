from dataclasses import dataclass, field
from itertools import chain
from random import choice
from time import sleep

from ai_battleship.game_phases.base import Phase
from ai_battleship.grid import Grid

HIGHLIGHT = {"bad": (255, 64, 64)}


@dataclass
class Game(Phase):
    turn: int = 0  # 0 - player, 1 - ai
    target_grid: Grid = field(init=False)

    def __post_init__(self):
        self.turn = choice([0, 1])
        self.target_grid = self.ai_grid if self.turn == 0 else self.player_grid
        starting_player = "player" if self.turn == 0 else "ai"
        print(f"Starting player: {starting_player}")

    def ai_turn(self):
        sleep(0.3)
        # Choose a random target from available
        possible_targets = [
            f
            for f in chain.from_iterable(self.target_grid.fields)
            if self.is_valid_target(f)
        ]
        target = choice(possible_targets)
        player_row, player_col = self.cursor.row, self.cursor.col
        self.cursor.row, self.cursor.col = target.row, target.col
        self.shoot()

        # Bring cursor back to original position
        self.cursor.row, self.cursor.col = player_row, player_col

    def handle_turn(self):
        """Handle turn system, this should run after every player turn"""
        if self.turn == 1:  # Ai turn
            self.target_grid = self.player_grid
            self.check_victory()
            self.ai_turn()

            # Start player turn
            self.turn = 0
            self.target_grid = self.ai_grid
            self.check_victory()

    def check_victory(self):
        """Check if the game has finished"""
        living_ship_fields = [
            f for f in chain.from_iterable(self.target_grid.ships) if f.status == "ship"
        ]
        if not living_ship_fields:
            # No ship tiles left on the target grid
            ending_prompt = "WIN" if self.target_grid is self.ai_grid else "LOSS"
            print(f"Game over, status: {ending_prompt}")
            self.done = True

    def get_targeted_ship(self, target):
        """Return the ship hit by the shot, or None if missed"""
        return next((ship for ship in self.target_grid.ships if target in ship), None)

    def check_if_sunk(self, targeted_ship):
        """Check if the targeted ship was sunk"""
        return all(field.status == "hit" for field in targeted_ship)

    def is_valid_target(self, target):
        """Check if selected field is a valid target"""
        return target.status not in ["hit", "sunk", "miss"]

    def shoot(self):
        """Try to fire at selected field, return False if invalid target"""
        target = self.target_grid[self.cursor.row, self.cursor.col]
        if not self.is_valid_target(target):
            return False

        targeted_ship = self.get_targeted_ship(target)
        if targeted_ship:
            target.set_status("hit")

            if self.check_if_sunk(targeted_ship):
                # Mark neighboring tiles as empty
                adjacent_area = [
                    self.target_grid[field.row + adj_row, field.col + adj_col]
                    for field in targeted_ship
                    for adj_row in [-1, 0, 1]
                    for adj_col in [-1, 0, 1]
                    if self.target_grid.field_exists(
                        field.row + adj_row, field.col + adj_col
                    )
                ]
                for f in adjacent_area:
                    f.set_status("empty")
                # Mark targeted ship as sunk
                for f in targeted_ship:
                    f.set_status("sunk")
        else:
            target.set_status("miss")

        return True

    def move_cursor(self, direction):
        """Move cursor in the specified direction and set highlight accordingly"""
        current_target = self.target_grid[self.cursor.row, self.cursor.col]
        current_target.set_color()

        self.cursor.move(direction, grid_size=self.ai_grid.grid_size)

        new_target = self.ai_grid[self.cursor.row, self.cursor.col]
        if new_target.status == "miss":
            new_target.set_color(HIGHLIGHT["bad"])

    def move(self, direction):
        self.move_cursor(direction)

    def confirm(self):
        # Ai turn if shot valid target
        if self.shoot():
            self.turn = 1

    def handle_events(self, events):
        self.handle_turn()
        super().handle_events(events)

    def handle_extra_events(self, event):  # No additional keybinds for Game phase
        pass

    def draw(self, screen, cursor_pos=1):
        super().draw(screen, cursor_pos=cursor_pos)

    def next_phase(self):
        return None
