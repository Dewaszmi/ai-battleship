import time

import numpy as np
import pygame
import torch

from ai_battleship.ai_agent.agent import Agent
from ai_battleship.ai_agent.environment import AgentEnvironment
from ai_battleship.constants import *
from ai_battleship.grid import Grid

VISUALISE_INTERVAL = 50


def draw_visualisation(
    screen: pygame.Surface,
    grid: Grid,
    agent: Agent,
    state: torch.Tensor,
    hitmarker: tuple[int, int] | None = None,
):
    """
    Draw the game grid (left) and a Q-value heatmap (right) side by side.
    - agent: your Agent object (used to compute Q-values)
    - state: current state tensor [C, H, W]
    """
    screen.fill((0, 0, 0))
    grid_size: int = grid.grid_size

    # --- LEFT GRID: normal game ---
    for row in range(grid_size):
        for col in range(grid_size):
            field = grid[row, col]
            color = field.color
            rect = pygame.Rect(
                col * (CELL_SIZE + MARGIN) + MARGIN,
                row * (CELL_SIZE + MARGIN) + MARGIN,
                CELL_SIZE,
                CELL_SIZE,
            )
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, (255, 255, 255), rect, 1)

    # Hitmarker on left grid
    if hitmarker:
        row, col = hitmarker
        rect = pygame.Rect(
            col * (CELL_SIZE + MARGIN) + MARGIN,
            row * (CELL_SIZE + MARGIN) + MARGIN,
            CELL_SIZE,
            CELL_SIZE,
        )
        pygame.draw.line(
            screen, (0, 255, 0), (rect.left, rect.top), (rect.right, rect.bottom), 4
        )
        pygame.draw.line(
            screen, (0, 255, 0), (rect.left, rect.bottom), (rect.right, rect.top), 4
        )

    # --- RIGHT GRID: Q-value heatmap ---
    with torch.no_grad():
        q_map = (
            agent.model(state.unsqueeze(0))
            .squeeze(0)
            .view(grid_size, grid_size)
            .cpu()
            .numpy()
        )
    q_min, q_max = np.min(q_map), np.max(q_map)
    norm_q = (q_map - q_min) / (q_max - q_min + 1e-8)  # normalize 0-1

    offset_x = (CELL_SIZE + MARGIN) * grid_size
    for row in range(grid_size):
        for col in range(grid_size):
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
            screen.blit(s, (rect.x, rect.y))

    pygame.display.flip()


def train(
    episodes: int = 2000,  # 500
    batch_size: int = 64,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.997,  # 0.995,
    target_update: int = 10,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = AgentEnvironment()
    agent = Agent(grid_size=env.grid_size, device=device)
    torch.backends.cudnn.benchmark = True  # optimize performance for fixed input sizes

    epsilon = epsilon_start

    pygame.init()
    grid_size = GRID_SIZE
    width = (CELL_SIZE + MARGIN) * grid_size * 2  # two grids side by side
    height = (CELL_SIZE + MARGIN) * grid_size
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Battleship AI Visualisation")

    current_time = time.time()

    for episode in range(episodes):
        state = env.reset().to(device)
        total_reward = 0.0

        while not env.done:
            # select action
            action = agent.select_action(state, epsilon=epsilon)

            # environment step
            next_state, reward, done = env.step(action)
            next_state = next_state.to(device)

            # store transition
            agent.add_to_memory((state, action, reward, next_state, done))

            # train agent
            agent.train_step(batch_size)

            state = next_state
            total_reward += reward
            # if episode % VISUALISE_INTERVAL == 0 and episode >= 1500:
            #     draw_visualisation(screen, env.grid, agent, state, hitmarker=action)
            #     # print(f"Reward: {reward}, total reward: {total_reward}")
            #     pygame.time.delay(50)

        # decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # update target network
        if episode % target_update == 0:
            agent.update_target_network()

        if episode % VISUALISE_INTERVAL == 0:
            print(
                f"Episode {episode+1}/{episodes}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}"
            )
            elapsed = time.time() - current_time
            # print(f"Time: {elapsed:.2f}")
            current_time = time.time()


if __name__ == "__main__":
    torch.manual_seed(42)
    train()
