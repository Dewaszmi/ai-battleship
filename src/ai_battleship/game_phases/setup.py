from collections import deque
from dataclasses import dataclass, field
from random import choice, randrange
from typing import List

import pygame

from ai_battleship.constants import *
from ai_battleship.field import Field
from ai_battleship.game_phases.base import Phase
from ai_battleship.game_phases.game import Game

SHIPS_DICT = {
    5: 1,
    4: 1,
    3: 2,
    2: 1,
}

ship_amount = sum(SHIPS_DICT.values())

HIGHLIGHT = {"good": (0, 255, 0), "bad": (255, 64, 64)}


@dataclass
class Setup(Phase):
    current_ship: int = 0
    direction: str = "v"  # 'v' - vertical, 'h' - horizontal
    ships_queue: deque[int] = field(
        default_factory=lambda: deque(
            [k for k, v in SHIPS_DICT.items() for _ in range(v)]
        )
    )
    position: List[Field] = field(default_factory=list)

    def __post_init__(self):
        # Initialize AI grid
        self.current_grid = self.ai_grid
        self.generate_ai_grid()

        # Handle player grid setup
        self.current_grid = self.player_grid
        self.cursor.row, self.cursor.col, self.direction = 0, 0, "v"
        self.get_next_ship()
        self.get_position()

    def generate_ai_grid(self):
        """Generates a random valid grid for the ai"""
        queue_copy = deque(self.ships_queue)
        self.get_next_ship(queue_copy)

        max_attempts = 20
        attempts = 0

        while attempts < max_attempts:
            # Try a random position and direction
            self.cursor.row, self.cursor.col, self.direction = (
                randrange(self.grid_size),
                randrange(self.grid_size),
                choice(["v", "h"]),
            )
            # Update and optionally correct position
            self.get_position()
            self.correct_position()

            # Try to place ship, if successful get next ship
            if self.place_ship():
                if not self.get_next_ship(queue_copy):
                    break

            attempts += 1

        if queue_copy:
            raise RuntimeError("Failed to generate valid AI grid")

    def get_next_ship(self, queue=None):
        """Returns the next ship from the queue, or None if queue is empty"""
        if queue is None:
            queue = self.ships_queue

        if queue:
            self.current_ship = queue.popleft()
            return True
        else:
            return None

    def is_obstructed(self):
        """Check if current selection is valid for ship placement"""
        area = [
            self.current_grid[field.row + adj_row, field.col + adj_col]
            for field in self.position
            for adj_row in [-1, 0, 1]
            for adj_col in [-1, 0, 1]
            if self.current_grid.field_exists(field.row + adj_row, field.col + adj_col)
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
            if not self.current_grid.field_exists(row, col):
                break

            position.append(self.current_grid[row, col])

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
                self.cursor.move(correction_dir, self.current_grid.grid_size)
            self.get_position()

    def move_ship(self, direction):
        """Moves the ship in the specified direction"""
        self.cursor.move(direction, self.current_grid.grid_size)
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
            return False

        for field in self.position:
            field.status = "ship"
        self.current_grid.ships.append(self.position)

        return True

    def move(self, direction):
        self.move_ship(direction)

    def confirm(self):
        # Try to place ship
        if not self.place_ship():
            return

        if not self.get_next_ship():
            # All ships have been placed
            self.done = True
            return

        self.get_position()

    def handle_extra_events(self, event):
        # Rotation keybind
        if event.key == pygame.K_r:
            self.rotate_ship()

    def draw(self, screen, cursor_pos=0):
        super().draw(screen, cursor_pos=cursor_pos)

    def next_phase(self):
        """Return a new Game phase instance"""
        return Game(
            player_grid=self.player_grid,
            ai_grid=self.ai_grid,
            grid_size=self.grid_size,
        )
