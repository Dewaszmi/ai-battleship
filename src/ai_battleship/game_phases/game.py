from dataclasses import dataclass, field
from itertools import chain
from random import choice
from time import sleep
import torch
from typing import final
from typing_extensions import override # for older python

import pygame

from ai_battleship.constants import HIGHLIGHT_COLORS
from ai_battleship.game_phases.base import Cursor, Phase
from ai_battleship.ai.ppo import Agent
from ai_battleship.envs.battleship_env_gym import BattleshipEnv
from ai_battleship.utils.grid_utils import *
from gymnasium.vector import SyncVectorEnv

def EnvWrapper():
    """Wrapper to create vectorized battleship environments compatible with ppo code"""
    def make_env():
        return BattleshipEnv()
    return SyncVectorEnv([make_env])



@final
@dataclass
class Game(Phase):
    player_grid: Grid
    ai_grid: Grid
    turn: int = field(init=False)  # 0 - player, 1 - ai
    agent: Agent = field(init=False)
    hitmarker: tuple[int, int] | None = field(default=None)  # shot indicator for the ai

    def __post_init__(self):
        # Prepare the AI agent
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        envs = EnvWrapper()
        self.agent = Agent(envs=envs).to(self.device)
        self.agent.load_state_dict(torch.load("models/battleship_model.pth", map_location=self.device, weights_only=True))
        self.agent.eval()

        # Start game
        self.turn = choice([0, 1])
        starting_player = "player" if self.turn == 0 else "ai"
        print(f"Starting player: {starting_player}")
        self.draw()
        self.handle_turn()

    def ai_turn(self):
        sleep(0.3)
        # Convert player grid into a tensor
        state = BattleshipEnv.get_state_from_grid(self.player_grid)
        state_tensor = torch.from_numpy(state).float().to(self.device)

        action, _, _, _ = self.agent.get_action_and_value(state_tensor)
        action_idx = action.item()
        row = action_idx // self.player_grid.grid_size
        col = action_idx % self.player_grid.grid_size

        target = self.player_grid[row, col]
        self.hitmarker = (row, col)  # add visible indicator for where the ai has shot
        # print(f"ai chose: {row}, {col} with state: {target.status}")
        # if not is_valid_target(target):
        #     print("ai has chosen an invalid target!")

        shoot(self.player_grid, target)

    def check_victory(self):
        """Check if the game has finished"""
        target_grid = self.ai_grid if self.turn == 0 else self.player_grid
        living_ship_fields = [f for f in chain.from_iterable(target_grid.ships) if f.status == "ship"]
        if not living_ship_fields:
            # No ship tiles left on the target grid
            ending_prompt = "WIN" if self.turn == 0 else "LOSS"
            print(f"Game over, status: {ending_prompt}")
            self.done = True

    def move_cursor(self, direction: str):
        """Move cursor in the specified direction and set highlight accordingly"""
        # clear old highlight
        target = self.ai_grid[self.cursor.row, self.cursor.col]
        target.set_color(FIELD_COLORS["unknown"] if target.status == "ship" else None)

        self.cursor.move(direction, grid_size=self.ai_grid.grid_size)
        new_target = self.ai_grid[self.cursor.row, self.cursor.col]
        if not is_valid_target(new_target):
            new_target.set_color(HIGHLIGHT_COLORS["bad"])

    def handle_turn(self):
        """Handle ai turn"""
        if self.turn == 1:
            self.ai_turn()
            self.check_victory()
            self.turn = 0

    @override
    def move(self, direction: str):
        self.move_cursor(direction)

    @override
    def confirm(self):
        """Handle player turn"""
        if not self.done:
            shoot(
                self.ai_grid, self.ai_grid[self.cursor.row, self.cursor.col]
            )  # (allows shooting invalid targets by the player to match ai capabilities)
            self.check_victory()
            self.draw()
            if not self.done:
                self.turn = 1
                self.handle_turn()

    @override
    def handle_extra_events(self, event: pygame.event.EventType):  # No additional keybinds for Game phase
        pass

    @override
    def draw_grid(self, grid: Grid, offset_x: int, cursor: Cursor | None = None):
        """Draw additional hitmarkers for the player grid"""
        super().draw_grid(grid, offset_x, cursor=cursor)

        if (
            grid is self.player_grid and self.hitmarker is not None
        ):  # draw extra hitmarkers for the player grid
            row, col = self.hitmarker
            rect = pygame.Rect(
                offset_x + col * (CELL_SIZE + MARGIN) + MARGIN,
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
        self.hitmarker = None

    @override
    def draw(self, cursor_grid: int = 1):  # Draw cursor at the ai grid (right)
        super().draw(cursor_grid=cursor_grid)

    def next_phase(self):
        """Returns None to signify that the game has ended"""
        return None
