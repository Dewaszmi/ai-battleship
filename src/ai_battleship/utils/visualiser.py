import numpy as np
import pygame

from ai_battleship.constants import CELL_SIZE, CURSOR_COLOR, GRID_SIZE, MARGIN


class Visualiser:
    def __init__(self, grid_size=GRID_SIZE):
        pygame.init()
        self.grid_size = grid_size
        width = (CELL_SIZE + MARGIN) * grid_size * 2  # two grids side by side
        height = (CELL_SIZE + MARGIN) * grid_size
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Battleship Training Visualization")

    def draw(self, grid, q_map=None, cursor=False, hitmarker=False):
        """
        Draws two grids side by side:
        - Left: normal grid (ships, hits, misses)
        - Right: heatmap of Q-values if provided
        """
        self.screen.fill((0, 0, 0))

        # --- LEFT GRID ---
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                field = grid[row, col]
                color = field.color
                rect = pygame.Rect(
                    col * (CELL_SIZE + MARGIN) + MARGIN,
                    row * (CELL_SIZE + MARGIN) + MARGIN,
                    CELL_SIZE,
                    CELL_SIZE,
                )
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (255, 255, 255), rect, 1)

        if cursor:
            row, col = cursor
            rect = pygame.Rect(
                col * (CELL_SIZE + MARGIN) + MARGIN,
                row * (CELL_SIZE + MARGIN) + MARGIN,
                CELL_SIZE,
                CELL_SIZE,
            )
            pygame.draw.rect(self.screen, CURSOR_COLOR, rect, 3)

        if hitmarker:
            row, col = hitmarker
            rect = pygame.Rect(
                col * (CELL_SIZE + MARGIN) + MARGIN,
                row * (CELL_SIZE + MARGIN) + MARGIN,
                CELL_SIZE,
                CELL_SIZE,
            )
            pygame.draw.line(
                self.screen,
                (0, 255, 0),
                (rect.left, rect.top),
                (rect.right, rect.bottom),
                4,
            )
            pygame.draw.line(
                self.screen,
                (0, 255, 0),
                (rect.left, rect.bottom),
                (rect.right, rect.top),
                4,
            )

        # --- RIGHT GRID (heatmap) ---
        if q_map is not None:
            q_min, q_max = np.min(q_map), np.max(q_map)
            norm_q = (q_map - q_min) / (q_max - q_min + 1e-8)

            offset_x = (CELL_SIZE + MARGIN) * self.grid_size
            for row in range(self.grid_size):
                for col in range(self.grid_size):
                    intensity = int(255 * norm_q[row, col])
                    color = (intensity, 0, 255 - intensity)  # red=high, blue=low
                    rect = pygame.Rect(
                        offset_x + col * (CELL_SIZE + MARGIN) + MARGIN,
                        row * (CELL_SIZE + MARGIN) + MARGIN,
                        CELL_SIZE,
                        CELL_SIZE,
                    )
                    s = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
                    s.fill((*color, 150))  # semi-transparent
                    self.screen.blit(s, (rect.x, rect.y))

        pygame.display.flip()
