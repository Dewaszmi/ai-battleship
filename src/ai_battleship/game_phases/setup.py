from collections import deque
from dataclasses import dataclass, field
from typing import List

import pygame
from ai_battleship.constants import *
from ai_battleship.field import Field
from ai_battleship.game_phases.base import Phase

SHIPS_DICT = {
    5: 1,
    4: 1,
    3: 2,
    2: 1,
}

HIGHLIGHT = {"good": (0, 255, 0), "bad": (255, 64, 64)}


@dataclass
class Setup(Phase):

    #:TODO create some kind of ship list to use in the game phase

    current_ship: int = 0
    direction: str = "v"  # 'v' - vertical, 'h' - horizontal
    ships_queue: deque[int] = field(
        default_factory=lambda: deque(
            [k for k, v in SHIPS_DICT.items() for _ in range(v)]
        )
    )
    position: List[Field] = field(default_factory=list)

    def init_ai_grid(self):  # for testing
        sample_layout = [
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        ]

        for i in range(len(sample_layout)):
            for j in range(len(sample_layout)):
                self.ai_grid[i, j].set_status(
                    "ship" if sample_layout[i][j] == 1 else "empty"
                )

    def __post_init__(self):
        self.get_next_ship()
        self.get_position()
        self.init_ai_grid()

    def get_next_ship(self):
        """Retrieves the next ship from the queue"""
        if self.ships_queue:
            self.current_ship = self.ships_queue.popleft()
        else:
            self.done = True

    def is_obstructed(self):
        """Check if current selection is valid for ship placement"""
        area = [
            self.player_grid[field.row + adj_row, field.col + adj_col]
            for field in self.position
            for adj_row in [-1, 0, 1]
            for adj_col in [-1, 0, 1]
            if self.player_grid.field_exists(field.row + adj_row, field.col + adj_col)
        ]

        return any(field.status == "ship" for field in area)

    def get_position(self):
        """Updates currently selected position along with highlighting"""
        if not self.current_ship:
            raise ValueError("No current ship to place")

        # Reset highlight for previous selection
        for f in self.position:
            f.set_color()  # type: ignore

        row = self.cursor.row
        col = self.cursor.col

        ship_length = self.current_ship
        position = []

        for _ in range(ship_length):
            if not self.player_grid.field_exists(row, col):
                break

            position.append(self.player_grid[row, col])

            if self.direction == "v":
                row += 1
            else:
                col += 1

        self.position = position

        # Apply highlight
        highlight_color = (
            HIGHLIGHT["good"] if not self.is_obstructed() else HIGHLIGHT["bad"]
        )
        for f in self.position:
            f.set_color(highlight_color)  # type: ignore

    def correct_position(self):
        """Moves the cursor to keep the ship in bounds after movement / rotation"""
        if self.current_ship != len(self.position):
            correction_dir = "left" if self.direction == "h" else "up"
            for _ in range(self.current_ship - len(self.position)):
                self.cursor.move(correction_dir, self.player_grid.grid_size)
            self.get_position()

    def move_ship(self, direction):
        """Moves the ship in the specified direction"""
        self.cursor.move(direction, self.player_grid.grid_size)
        self.get_position()
        self.correct_position()

    def rotate_ship(self):
        """Rotates the ship between vertical and horizontal orientation"""
        self.direction = "h" if self.direction == "v" else "v"
        self.get_position()
        self.correct_position()

    def place_ship(self):
        """Places ship at current position if possible"""
        if self.is_obstructed():
            return

        for field in self.position:
            field.status = "ship"
        self.get_next_ship()
        self.get_position()

    def move(self, direction):
        self.move_ship(direction)

    def confirm(self):
        self.place_ship()

    def handle_extra_events(self, event):
        # Rotation keybind
        if event.key == pygame.K_r:
            self.rotate_ship()

    def draw(self, screen, cursor_pos=0):
        super().draw(screen, cursor_pos=cursor_pos)
