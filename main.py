import pygame

from src.constants import *
from src.init_grid import init_grids

# pygame setup
pygame.init()
SCREEN_WIDTH = GRID_SIZE * (CELL_SIZE + MARGIN) * 2 + MARGIN + 20
SCREEN_HEIGHT = GRID_SIZE * (CELL_SIZE + MARGIN) + MARGIN + 20
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("AI Battleship")

player_grid, ai_grid = init_grids(GRID_SIZE)


def draw_grid(grid, offset_x=0):
    """Draw a single grid at horizontal offset offset_x (1-based indexing)."""
    for row in range(1, GRID_SIZE + 1):
        for col in range(1, GRID_SIZE + 1):
            field = grid[row][col]
            color = field.color
            rect = pygame.Rect(
                offset_x + (col - 1) * (CELL_SIZE + MARGIN) + MARGIN,
                (row - 1) * (CELL_SIZE + MARGIN) + MARGIN,
                CELL_SIZE,
                CELL_SIZE,
            )
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, (255, 255, 255), rect, 1)  # border


# main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((128, 128, 128))

    # draw player grid
    draw_grid(player_grid, offset_x=10)
    # draw AI grid next to it
    draw_grid(ai_grid, offset_x=GRID_SIZE * (CELL_SIZE + MARGIN) + 20)

    pygame.display.flip()

pygame.quit()
