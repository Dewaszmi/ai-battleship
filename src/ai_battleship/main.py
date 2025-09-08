from os import environ

import pygame

from ai_battleship.constants import *
from ai_battleship.game_phases.setup import Setup
from ai_battleship.grid import Grid

environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"  # Supress pygame welcome prompt


def main():
    # pygame setup
    pygame.init()
    SCREEN_WIDTH = GRID_SIZE * (CELL_SIZE + MARGIN) * 2 + MARGIN + 20
    SCREEN_HEIGHT = GRID_SIZE * (CELL_SIZE + MARGIN) + MARGIN + 20
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("AI Battleship")

    player_grid = Grid(player="Player", grid_size=GRID_SIZE)
    ai_grid = Grid(player="AI", grid_size=GRID_SIZE)

    current_phase = Setup(player_grid, ai_grid, GRID_SIZE)

    clock = pygame.time.Clock()
    running = True

    while running:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                running = False

        if current_phase.done:
            current_phase = current_phase.next_phase()
            if current_phase is None:
                break

        current_phase.handle_events(events)

        screen.fill((0, 0, 0))
        current_phase.draw(screen)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
