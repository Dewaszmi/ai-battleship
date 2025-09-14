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
    hitmarker: tuple[int, int],
):
    """Draw the game grid and a heatmap of Q-values side by side."""
    screen.fill((0, 0, 0))
    grid_size = grid.grid_size

    # --- Left: normal game grid ---
    offset_x = 0
    for row in range(grid_size):
        for col in range(grid_size):
            field = grid[row, col]
            color = field.color
            rect = pygame.Rect(
                offset_x + col * (CELL_SIZE + MARGIN) + MARGIN,
                row * (CELL_SIZE + MARGIN) + MARGIN,
                CELL_SIZE,
                CELL_SIZE,
            )
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, (255, 255, 255), rect, 1)

            row, col = hitmarker
            rect = pygame.Rect(
                offset_x + col * (CELL_SIZE + MARGIN) + MARGIN,
                row * (CELL_SIZE + MARGIN) + MARGIN,
                CELL_SIZE,
                CELL_SIZE,
            )
            pygame.draw.line(
                screen,
                (0, 255, 0),
                (rect.left, rect.top),
                (rect.right, rect.bottom),
                4,
            )
            pygame.draw.line(
                screen,
                (0, 255, 0),
                (rect.left, rect.bottom),
                (rect.right, rect.top),
                4,
            )

    # --- Right: heatmap from Q-values ---
    offset_x = grid_size * (CELL_SIZE + MARGIN) + MARGIN + 20
    with torch.no_grad():
        q_values = agent.model(state.unsqueeze(0).to(agent.device))[0]  # [H*W]
    q_map = q_values.view(grid_size, grid_size).cpu().numpy()

    # Normalize for colormap [0..1]
    q_min, q_max = q_map.min(), q_map.max()
    norm = (q_map - q_min) / (q_max - q_min + 1e-8)

    for row in range(grid_size):
        for col in range(grid_size):
            val = norm[row, col]
            # map value → color (blue→red)
            color = (int(255 * val), 0, int(255 * (1 - val)))
            rect = pygame.Rect(
                offset_x + col * (CELL_SIZE + MARGIN) + MARGIN,
                row * (CELL_SIZE + MARGIN) + MARGIN,
                CELL_SIZE,
                CELL_SIZE,
            )
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, (255, 255, 255), rect, 1)

    pygame.display.flip()


def train(
    episodes: int = 500,
    batch_size: int = 64,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.995,
    target_update: int = 10,
    device: str = "cpu",
):
    env = AgentEnvironment()
    agent = Agent(grid_size=env.grid_size, device=device)

    epsilon = epsilon_start

    pygame.init()
    grid_size = GRID_SIZE
    width = (CELL_SIZE + MARGIN) * grid_size * 2  # two grids side by side
    height = (CELL_SIZE + MARGIN) * grid_size
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Battleship AI Visualisation")

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0.0

        while not env.done:
            # select action
            action = agent.select_action(state, epsilon=epsilon)

            # environment step
            next_state, reward, done = env.step(action)

            # store transition
            agent.add_to_memory((state, action, reward, next_state, done))

            # train agent
            agent.train_step(batch_size)

            state = next_state
            total_reward += reward
            if episode % VISUALISE_INTERVAL == 0:
                draw_visualisation(screen, env.grid, agent, state, action)

        # decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # update target network
        if episode % target_update == 0:
            agent.update_target_network()

        if episode % VISUALISE_INTERVAL == 0:
            print(
                f"Episode {episode+1}/{episodes}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}"
            )


if __name__ == "__main__":
    torch.manual_seed(42)
    train()
