import pygame

from ai_battleship.constants import *
from ai_battleship.game_phases.setup import Setup
from ai_battleship.grid import Grid

# pygame setup
pygame.init()
SCREEN_WIDTH = GRID_SIZE * (CELL_SIZE + MARGIN) * 2 + MARGIN + 20
SCREEN_HEIGHT = GRID_SIZE * (CELL_SIZE + MARGIN) + MARGIN + 20
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("AI Battleship")

player_grid = Grid(player="Player", grid_size=GRID_SIZE)
ai_grid = Grid(player="AI", grid_size=GRID_SIZE)


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
