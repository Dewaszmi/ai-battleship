from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pygame

from ai_battleship.constants import *
from ai_battleship.grid import Cursor, Grid


@dataclass
class Phase(ABC):
    player_grid: Grid
    ai_grid: Grid
    grid_size: int
    current_grid: Grid = field(init=False)
    cursor: Cursor = field(default_factory=lambda: Cursor(0, 0))
    done: bool = False

    def draw_grid(self, screen, grid, offset_x, cursor=None):
        """Draw a single grid at horizontal offset offset_x"""
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                field = grid[row, col]
                color = field.color
                rect = pygame.Rect(
                    offset_x + col * (CELL_SIZE + MARGIN) + MARGIN,
                    row * (CELL_SIZE + MARGIN) + MARGIN,
                    CELL_SIZE,
                    CELL_SIZE,
                )
                pygame.draw.rect(screen, color, rect)  # Draw rectangle
                pygame.draw.rect(screen, (255, 255, 255), rect, 1)  # Draw border

        # Draw cursor highlight if required
        if cursor is not None:
            self.draw_grid(screen, grid, offset_x)
            row, col = cursor.row, cursor.col
            rect = pygame.Rect(
                offset_x + col * (CELL_SIZE + MARGIN) + MARGIN,
                row * (CELL_SIZE + MARGIN) + MARGIN,
                CELL_SIZE,
                CELL_SIZE,
            )
            pygame.draw.rect(screen, cursor.color, rect, 3)

    def draw(self, screen, cursor_pos):  # cursor_pos: 0 or 1
        """Draw two grids next to eachother, with cursor placed at the one defined"""
        for i, grid in enumerate([self.player_grid, self.ai_grid]):
            offset = (GRID_SIZE * (CELL_SIZE + MARGIN) + MARGIN + 20) * i
            cursor = self.cursor if i == cursor_pos else None
            self.draw_grid(screen=screen, grid=grid, offset_x=offset, cursor=cursor)

    @abstractmethod
    def move(self, direction):
        pass

    @abstractmethod
    def confirm(self):
        pass

    @abstractmethod
    def handle_extra_events(self, event):
        pass

    def handle_events(self, events):
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()

            elif event.type == pygame.KEYDOWN:
                # Movement keybinds (hjkl or arrow keys)
                direction_map = {
                    pygame.K_k: "up",
                    pygame.K_UP: "up",
                    pygame.K_j: "down",
                    pygame.K_DOWN: "down",
                    pygame.K_h: "left",
                    pygame.K_LEFT: "left",
                    pygame.K_l: "right",
                    pygame.K_RIGHT: "right",
                }
                if event.key in direction_map:
                    self.move(direction_map[event.key])

                # Confirm keybind (Enter)
                elif event.key == pygame.K_RETURN:
                    self.confirm()

                # Phase-specific keybinds
                else:
                    self.handle_extra_events(event)
