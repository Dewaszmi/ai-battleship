from dataclasses import dataclass, field
from itertools import chain
from random import choice
from time import sleep

from ai_battleship.constants import HIGHLIGHT_COLORS
from ai_battleship.game_phases.base import Phase
from ai_battleship.utils.grid_utils import *


@dataclass
class Game(Phase):
    turn: int = field(init=False)  # 0 - player, 1 - ai

    def __post_init__(self):
        self.turn = choice([0, 1])
        starting_player = "player" if self.turn == 0 else "ai"
        print(f"Starting player: {starting_player}")

    def ai_turn(self):
        # TODO: replace with trained agent
        sleep(0.3)
        # Choose a random target from available
        possible_targets = [
            f
            for f in chain.from_iterable(self.player_grid.fields)
            if is_valid_target(f)
        ]
        target = choice(possible_targets)
        shoot(self.player_grid, target.row, target.col)

    def handle_turn(self):
        """Handle turn system, this should run after every player turn"""
        # End of player turn
        self.check_victory()

        # Ai turn
        self.turn = 1
        self.ai_turn()
        self.check_victory()

        self.turn = 0

    def check_victory(self):
        """Check if the game has finished"""
        target_grid = self.ai_grid if self.turn == 0 else self.player_grid
        living_ship_fields = [
            f for f in chain.from_iterable(target_grid.ships) if f.status == "ship"
        ]
        if not living_ship_fields:
            # No ship tiles left on the target grid
            ending_prompt = "WIN" if self.turn is 0 else "LOSS"
            print(f"Game over, status: {ending_prompt}")
            self.done = True

    def move_cursor(self, direction):
        """Move cursor in the specified direction and set highlight accordingly"""
        clear_highlights(self.ai_grid)

        self.cursor.move(direction, grid_size=self.ai_grid.grid_size)
        new_target = self.ai_grid[self.cursor.row, self.cursor.col]
        if not is_valid_target(new_target):
            new_target.set_color(HIGHLIGHT_COLORS["bad"])

    def move(self, direction):
        self.move_cursor(direction)

    def confirm(self):
        # Ai turn if shot valid target
        if shoot(self.ai_grid, self.cursor.row, self.cursor.col):
            self.handle_turn()

    def handle_events(self, events):
        super().handle_events(events)

    def handle_extra_events(self, event):  # No additional keybinds for Game phase
        pass

    def draw(self, screen, cursor_pos=1):
        super().draw(screen, cursor_pos=cursor_pos)

    def next_phase(self):
        return None
