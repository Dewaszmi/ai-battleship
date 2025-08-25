from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pygame

from src.constants import *
from src.field import Field
from src.grid import Cursor, Grid
from src.init_grid import SHIPS_DICT, init_grids

# pygame setup
pygame.init()
SCREEN_WIDTH = GRID_SIZE * (CELL_SIZE + MARGIN) * 2 + MARGIN + 20
SCREEN_HEIGHT = GRID_SIZE * (CELL_SIZE + MARGIN) + MARGIN + 20
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("AI Battleship")

player_grid, ai_grid = init_grids(GRID_SIZE)


def draw_grid(grid, offset_x=0):
    """Draw a single grid at horizontal offset offset_x"""
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            field = grid[row, col]
            color = field.color
            rect = pygame.Rect(
                offset_x + (col - 1) * (CELL_SIZE + MARGIN) + MARGIN,
                (row - 1) * (CELL_SIZE + MARGIN) + MARGIN,
                CELL_SIZE,
                CELL_SIZE,
            )
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, (255, 255, 255), rect, 1)  # border


@dataclass
class Setup:
    grid: Grid
    cursor: Cursor = field(default_factory=lambda: Cursor(0, 0))
    direction: str = "v"  # 'v' - vertical, 'h' - horizontal
    ships_queue: deque = field(default_factory=lambda: deque(SHIPS_DICT.items()))
    current_ship: Optional[Tuple[int, int]] = None
    position: List[Field] = field(default_factory=list)
    done: bool = False

    SHIPS_DICT = {
        5: 1,
        4: 1,
        3: 2,
        2: 1,
    }

    def __post_init__(self):
        self.get_next_ship()
        self.position = self.get_ship_position()[0] or []

    def get_next_ship(self):
        """Retrieves the next ship from the queue"""
        if self.ships_queue:
            self.current_ship = self.ships_queue.popleft()
        else:
            self.done = True

    def get_ship_position(self, row=None, col=None):
        """Returns list of Fields occupied by the current ship placement"""
        if not row:
            row = self.cursor.row
        if not col:
            col = self.cursor.col

        if not self.current_ship:
            raise ValueError("No current ship to place")

        ship_length, _ = self.current_ship
        position = []

        for i in range(ship_length):
            if not self.grid[row, col]:
                return False, ship_length - i

            position.append(self.grid[row, col])
            if self.direction == "v":
                row += 1
            else:
                col += 1

        return position, 0

    def is_out_of_bounds(self, position):
        ship_end = position[-1]
        return not (
            ship_end.row in range(self.grid.grid_size)
            and ship_end.col in range(self.grid.grid_size)
        )

    def is_obstructed(self, position):
        return any(field.status == "ship" for field in position)

    def validate_ship_position(self, positions):
        """Checks if the ship is in bounds and unobstructed by another ship"""
        return not (self.is_out_of_bounds(positions) or self.is_obstructed(positions))

    def rotate_ship(self):
        """Rotates the ship between vertical and horizontal orientation"""
        self.direction = "h" if self.direction == "v" else "v"
        position, missing = self.get_ship_position()
        if not position:
            correction_dir = "left" if self.direction == "h" else "up"
            for _ in range(missing):
                self.cursor.move(correction_dir, self.grid.grid_size)

        self.position = self.get_ship_position()[0] or []

    def place_ship(self):
        """Places ship at current position"""
        for field in self.position:
            field.status = "ship"
        self.get_next_ship()

    def handle_events(self, events):
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()

            elif event.type == pygame.KEYDOWN:
                dir = None
                if event.key in [pygame.K_k, pygame.K_UP]:
                    dir = "up"
                elif event.key in [pygame.K_l, pygame.K_RIGHT]:
                    dir = "right"
                elif event.key in [pygame.K_j, pygame.K_DOWN]:
                    dir = "down"
                elif event.key in [pygame.K_h, pygame.K_LEFT]:
                    dir = "left"

                if dir:
                    new_cursor_pos = self.cursor.get_new_position(
                        dir, self.grid.grid_size
                    )
                    if new_cursor_pos:
                        new_row, new_col = new_cursor_pos
                        if not self.is_out_of_bounds(
                            self.get_ship_position(new_row, new_col)
                        ):
                            self.cursor.move(dir, self.grid.grid_size)
                            self.position = self.get_ship_position()[0] or []

                elif event.key == pygame.K_r:
                    self.rotate_ship()

                elif event.key == pygame.K_RETURN:
                    pass


# ==== MAIN LOOP ====

setup_phase = Setup(player_grid)

running = True
while running:
    events = pygame.event.get()
    setup_phase.handle_events(events)

    screen.fill((0, 0, 0))
    draw_grid(player_grid, offset_x=0)
    draw_grid(ai_grid, offset_x=GRID_SIZE * (CELL_SIZE + MARGIN) + 20)

    # Draw cursor highlight
    cursor_rect = pygame.Rect(
        (setup_phase.cursor.col - 1) * (CELL_SIZE + MARGIN) + MARGIN,
        (setup_phase.cursor.row - 1) * (CELL_SIZE + MARGIN) + MARGIN,
        CELL_SIZE,
        CELL_SIZE,
    )
    pygame.draw.rect(screen, (255, 255, 0), cursor_rect, 3)  # yellow border

    pygame.display.flip()
    pygame.time.Clock().tick(60)
