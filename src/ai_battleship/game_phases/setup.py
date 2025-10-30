from collections import deque
from dataclasses import dataclass, field
from itertools import chain

import pygame
from typing_extensions import override  # for older python

from ai_battleship.constants import *
from ai_battleship.field import Field
from ai_battleship.game_phases.base import Cursor, Phase
from ai_battleship.game_phases.game import Game
from ai_battleship.utils.grid_utils import *

ship_amount = sum(SHIPS_DICT.values())


@dataclass
class Setup(Phase):
    current_ship: int | None = field(init=False)
    direction: str = field(init=False)  # 'v' - vertical, 'h' - horizontal
    ships_queue: deque[int] = field(
        default_factory=lambda: deque([k for k, v in SHIPS_DICT.items() for _ in range(v)])
    )
    position: list[Field] = field(default_factory=list)
    allowed_fields: list[Field] = field(init=False)

    def __post_init__(self):
        self.done: bool = False

        # Initialize AI grid
        self.ai_grid: Grid = generate_random_grid(self.ships_queue)
        for ship_tile in chain.from_iterable(self.ai_grid.ships):
            ship_tile.set_color(FIELD_COLORS["unknown"])

        # Handle player grid setup
        self.player_grid: Grid = Grid(GRID_SIZE)
        self.cursor: Cursor = Cursor(0, 0)
        self.direction = "v"
        self.current_ship = get_next_ship(self.ships_queue)
        self.get_ship_position()
        self.draw()

    def get_ship_position(self):
        """Get ship position and update highlights"""
        assert self.current_ship is not None
        # Clear previous highlights
        clear_highlights(self.player_grid)

        self.position = get_ship_position(
            self.player_grid,
            self.cursor.row,
            self.cursor.col,
            self.direction,
            self.current_ship,
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
        assert self.current_ship is not None
        ship_end = self.current_ship + (self.cursor.row if self.direction == "v" else self.cursor.col)
        if ship_end > self.player_grid.grid_size:
            if self.direction == "v":
                self.cursor.row -= ship_end - self.player_grid.grid_size
            else:
                self.cursor.col -= ship_end - self.player_grid.grid_size

    def move_ship(self, direction: str):
        """Moves the ship in the specified direction"""
        self.cursor.move(direction, self.player_grid.grid_size)
        self.correct_cursor_position()
        self.get_ship_position()

    def rotate_ship(self):
        """Rotates the ship between vertical and horizontal orientation"""
        self.direction = "h" if self.direction == "v" else "v"
        self.correct_cursor_position()
        self.get_ship_position()

    @override
    def move(self, direction: str):
        self.move_ship(direction)

    @override
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

    @override
    def handle_extra_events(self, event: pygame.event.EventType):
        # Rotation keybind
        if event.key == pygame.K_r:
            self.rotate_ship()

    @override
    def draw(self, cursor_grid: int = 0):  # Draw cursor at the player grid (left)
        super().draw(cursor_grid=cursor_grid)

    def next_phase(self):
        """Return a new Game phase instance"""
        return Game(screen=self.screen, player_grid=self.player_grid, ai_grid=self.ai_grid)
