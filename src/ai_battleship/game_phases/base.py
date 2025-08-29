from abc import ABC, abstractmethod
from dataclasses import dataclass

import pygame
from ai_battleship.constants import *
from ai_battleship.grid import Grid


@dataclass
class Phase(ABC):
    player_grid: Grid
    ai_grid: Grid

    @abstractmethod
    def handle_events(self, events):
        pass

    def draw_grid(self, screen, grid, offset_x):
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

    def draw_grid_with_cursor(
        self, screen, grid, offset_x, cursor, color=(255, 255, 0)
    ):
        """Draw grid with cursor highlight"""
        self.draw_grid(screen, grid, offset_x)
        row, col = cursor.row, cursor.col
        rect = pygame.Rect(
            offset_x + col * (CELL_SIZE + MARGIN) + MARGIN,
            row * (CELL_SIZE + MARGIN) + MARGIN,
            CELL_SIZE,
            CELL_SIZE,
        )
        pygame.draw.rect(screen, color, rect, 3)
