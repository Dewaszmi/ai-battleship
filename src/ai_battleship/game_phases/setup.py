from collections import deque
from dataclasses import dataclass, field
from typing import List

import pygame
from ai_battleship.constants import *
from ai_battleship.field import Field
from ai_battleship.game_phases.base import Phase
from ai_battleship.grid import Cursor

SHIPS_DICT = {
    5: 1,
    4: 1,
    3: 2,
    2: 1,
}

HIGHLIGHT = {"good": (0, 255, 0), "bad": (255, 64, 64)}


@dataclass
class Setup(Phase):
    current_ship: int = 0
    cursor: Cursor = field(default_factory=lambda: Cursor(0, 0))
    direction: str = "v"  # 'v' - vertical, 'h' - horizontal
    ships_queue: deque[int] = field(
        default_factory=lambda: deque(
            [k for k, v in SHIPS_DICT.items() for _ in range(v)]
        )
    )
    position: List[Field] = field(default_factory=list)
    done: bool = False

    def __post_init__(self):
        self.grid = self.player_grid  # for clarity
        self.get_next_ship()
        self.get_position()

    def get_next_ship(self):
        """Retrieves the next ship from the queue"""
        if self.ships_queue:
            self.current_ship = self.ships_queue.popleft()
        else:
            self.done = True

    def is_obstructed(self):
        """Check if current selection is valid for ship placement"""
        area = [
            self.grid[field.row + adj_row, field.col + adj_col]
            for field in self.position
            for adj_row in [-1, 0, 1]
            for adj_col in [-1, 0, 1]
            if self.grid.field_exists(field.row + adj_row, field.col + adj_col)
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
            if not self.grid.field_exists(row, col):
                break

            position.append(self.grid[row, col])

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
                self.cursor.move(correction_dir, self.grid.grid_size)
            self.get_position()

    def move_ship(self, direction):
        """Moves the ship in the specified direction"""
        self.cursor.move(direction, self.grid.grid_size)
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

    def handle_events(self, events):
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()

            elif event.type == pygame.KEYDOWN:
                dir = None

                # Movement keybinds
                if event.key in [pygame.K_k, pygame.K_UP]:
                    dir = "up"
                elif event.key in [pygame.K_l, pygame.K_RIGHT]:
                    dir = "right"
                elif event.key in [pygame.K_j, pygame.K_DOWN]:
                    dir = "down"
                elif event.key in [pygame.K_h, pygame.K_LEFT]:
                    dir = "left"

                if dir:
                    self.move_ship(dir)

                # Rotation keybind
                elif event.key == pygame.K_r:
                    self.rotate_ship()

                # Place ship keybind
                elif event.key == pygame.K_RETURN:
                    self.place_ship()

    def draw(self, screen):
        self.draw_grid_with_cursor(
            screen, grid=self.player_grid, offset_x=0, cursor=self.cursor
        )
        offset = GRID_SIZE * (CELL_SIZE + MARGIN) + MARGIN + 20
        self.draw_grid(screen, self.ai_grid, offset_x=offset)
