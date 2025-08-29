from abc import ABC, abstractmethod
from dataclasses import dataclass

import pygame
from ai_battleship.constants import *
from ai_battleship.grid import Grid


@dataclass
class Phase(ABC):
    grid: Grid

    @abstractmethod
    def handle_events(self, events):
        pass

    def draw(self, screen, offset_x=0):
        """Draw a single grid at horizontal offset offset_x"""
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                field = self.grid[row, col]
                color = field.color
                rect = pygame.Rect(
                    offset_x + (col - 1) * (CELL_SIZE + MARGIN) + MARGIN,
                    (row - 1) * (CELL_SIZE + MARGIN) + MARGIN,
                    CELL_SIZE,
                    CELL_SIZE,
                )
                pygame.draw.rect(screen, color, rect)  # Draw rectangle
                pygame.draw.rect(screen, (255, 255, 255), rect, 1)  # Draw border
