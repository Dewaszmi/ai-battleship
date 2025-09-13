from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

import pygame

from ai_battleship.constants import *
from ai_battleship.field import Field
from ai_battleship.game_phases.base import Cursor, Phase
from ai_battleship.game_phases.game import Game
from ai_battleship.utils.grid_utils import *

ship_amount = sum(SHIPS_DICT.values())


@dataclass
class Setup(Phase):
    current_ship: Optional[int] = field(init=False)
    direction: str = field(init=False)  # 'v' - vertical, 'h' - horizontal
    ships_queue: deque[int] = field(
        default_factory=lambda: deque(
            [k for k, v in SHIPS_DICT.items() for _ in range(v)]
        )
    )
    position: List[Field] = field(default_factory=list)
    allowed_fields: list = field(init=False)

    def __post_init__(self):
        self.done = False

        # Initialize AI grid
        self.ai_grid = generate_random_grid(self.ships_queue)

        # Handle player grid setup
        self.player_grid = Grid(GRID_SIZE)
        self.cursor, self.direction = Cursor(0, 0), "v"
        self.current_ship = get_next_ship(self.ships_queue)  # type: ignore
        self.get_ship_position()
        self.draw()

    def get_ship_position(self):
        """Get ship position and update highlights"""
        # Clear previous highlights
        clear_highlights(self.player_grid)

        self.position = get_ship_position(
            self.player_grid,
            self.cursor.row,
            self.cursor.col,
            self.direction,
            self.current_ship,  # type: ignore
        )

        # Add updated highlight
        for f in self.position:
            f.set_color(
                HIGHLIGHT_COLORS["good"]
                if not is_obstructed(self.player_grid, self.position)
                else HIGHLIGHT_COLORS["bad"]
            )

    def correct_cursor_position(self):
        """Move cursor to assert that current ship position fits in the grid"""
        ship_end = self.current_ship + (  # type: ignore
            self.cursor.row if self.direction == "v" else self.cursor.col
        )
        if ship_end > self.player_grid.grid_size:
            if self.direction == "v":
                self.cursor.row -= ship_end - self.player_grid.grid_size
            else:
                self.cursor.col -= ship_end - self.player_grid.grid_size

    def move_ship(self, direction):
        """Moves the ship in the specified direction"""
        self.cursor.move(direction, self.player_grid.grid_size)
        self.correct_cursor_position()
        self.get_ship_position()

    def rotate_ship(self):
        """Rotates the ship between vertical and horizontal orientation"""
        self.direction = "h" if self.direction == "v" else "v"
        self.correct_cursor_position()
        self.get_ship_position()

    def move(self, direction):
        self.move_ship(direction)

    def confirm(self):
        # Try to place ship
        if not place_ship(self.player_grid, self.position):
            return

        self.current_ship = get_next_ship(self.ships_queue)
        if self.current_ship is None:
            # All ships have been placed
            clear_highlights(self.player_grid)
            self.done = True
            return

        self.get_ship_position()

    def handle_extra_events(self, event):
        # Rotation keybind
        if event.key == pygame.K_r:
            self.rotate_ship()

    def draw(self, cursor_pos=0):  # Draw cursor at the player grid (left)
        super().draw(cursor_pos=cursor_pos)

    def next_phase(self):
        """Return a new Game phase instance"""
        return Game(
            screen=self.screen, player_grid=self.player_grid, ai_grid=self.ai_grid
        )
