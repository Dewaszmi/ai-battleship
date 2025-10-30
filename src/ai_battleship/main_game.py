from os import environ

import pygame

from ai_battleship.config import Config
from ai_battleship.constants import *
from ai_battleship.game_phases.setup import Setup

environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"  # Supress pygame welcome prompt


def game_loop(config):
    if config is None:
        config = Config()

    # pygame setup
    pygame.init()
    SCREEN_WIDTH = GRID_SIZE * (CELL_SIZE + MARGIN) * 2 + MARGIN + 20
    SCREEN_HEIGHT = GRID_SIZE * (CELL_SIZE + MARGIN) + MARGIN + 20
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("AI Battleship")

    current_phase = Setup(screen=screen, config=config)

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
                break  # finished game

        current_phase.handle_events(events)
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    game_loop()
