import pygame
import torch

from ai_battleship.ai_agent.old.agent import Agent, RolloutBuffer
from ai_battleship.ai_agent.old.environment import AgentEnvironment
from ai_battleship.ai_agent.old.model import ActorCritic
from ai_battleship.constants import *

VISUALISE_INTERVAL = 50


def draw_visualisation(
    screen: pygame.Surface,
    env: AgentEnvironment,
    agent: Agent,
    state: torch.Tensor,
    hitmarker: tuple[int, int] | None = None,
):
    screen.fill((0, 0, 0))
    grid = env.grid
    grid_size: int = grid.grid_size

    # --- LEFT GRID: environment ---
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

    # draw last action marker
    if hitmarker:
        row, col = hitmarker
        rect = pygame.Rect(
            col * (CELL_SIZE + MARGIN) + MARGIN,
            row * (CELL_SIZE + MARGIN) + MARGIN,
            CELL_SIZE,
            CELL_SIZE,
        )
        pygame.draw.line(
            screen, (0, 255, 0), (rect.left, rect.top), (rect.right, rect.bottom), 3
        )
        pygame.draw.line(
            screen, (0, 255, 0), (rect.left, rect.bottom), (rect.right, rect.top), 3
        )

    # --- RIGHT GRID: policy probability heatmap ---
    with torch.no_grad():
        logits, _ = agent.model(state.unsqueeze(0))
        probs = torch.softmax(logits, dim=-1).view(grid_size, grid_size).cpu().numpy()

    q_min, q_max = probs.min(), probs.max()
    norm_q = (probs - q_min) / (q_max - q_min + 1e-8)
    font = pygame.font.SysFont("Arial", max(12, CELL_SIZE // 3))

    offset_x = (CELL_SIZE + MARGIN) * grid_size
    for row in range(grid_size):
        for col in range(grid_size):
            intensity = int(255 * norm_q[row, col])
            color = (intensity, 0, 255 - intensity)
            rect = pygame.Rect(
                offset_x + col * (CELL_SIZE + MARGIN) + MARGIN,
                row * (CELL_SIZE + MARGIN) + MARGIN,
                CELL_SIZE,
                CELL_SIZE,
            )
            s = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
            s.fill((*color, 150))
            screen.blit(s, (rect.x, rect.y))

            # Draw probability text
            prob_text = f"{probs[row, col]:.2f}"
            text_surf = font.render(prob_text, True, (255, 255, 255))
            text_rect = text_surf.get_rect(center=rect.center)
            screen.blit(text_surf, text_rect)

    pygame.display.flip()


def train(
    episodes: int = 10000,
    rollout_steps: int = 128,
    batch_size: int = 64,
    epochs: int = 4,
    gamma: float = 0.95,
    lam: float = 0.95,
    max_steps: int = 100,  # maximum steps per episode
    collapse_penalty: float = -1.0,  # penalty if agent gets stuck
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = AgentEnvironment()
    agent = Agent(
        model=ActorCritic(in_channels=3, grid_size=env.grid_size), device=device
    )
    buffer = RolloutBuffer(
        rollout_steps, (3, env.grid_size, env.grid_size), device=device
    )
    pygame.init()
    grid_size = env.grid_size
    width = (CELL_SIZE + MARGIN) * grid_size * 2
    height = (CELL_SIZE + MARGIN) * grid_size
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Battleship PPO Visualisation")

    for episode in range(episodes):
        state = env.reset().to(device)
        total_reward = 0.0
        step = 0
        done = False

        while not done:
            # --- Sample action from policy ---
            action_idx, logp, value, _ = agent.model.act(state.unsqueeze(0).to(device))
            action_idx = action_idx.item()
            row, col = divmod(action_idx, env.grid_size)
            action_tuple = (int(row), int(col))

            # --- Environment step ---
            next_state, reward, done = env.step(action_tuple)
            next_state = next_state.to(device)

            step += 1
            if step >= max_steps:
                done = True
                reward += collapse_penalty  # penalize collapse onto a loop

            mask = 0.0 if done else 1.0

            # --- Store transition in buffer ---
            buffer.add(
                state,
                torch.tensor(action_idx),
                logp.detach(),
                reward,
                value.detach().squeeze(0),
                mask,
            )

            state = next_state
            total_reward += reward

            # --- Update PPO when buffer is full ---
            if buffer.ptr == rollout_steps:
                with torch.no_grad():
                    _, last_value = agent.model(state.unsqueeze(0).to(device))
                buffer.compute_returns_and_advantages(last_value.squeeze(0), gamma, lam)
                agent.update(buffer, epochs=epochs, batch_size=batch_size)
                buffer.clear()

            # --- Visualization ---
            # if episode % VISUALISE_INTERVAL == 0:
            #     draw_visualisation(screen, env, agent, state, hitmarker=action_tuple)
            #     pygame.time.delay(50)

        # --- Episode stats ---
        if episode % VISUALISE_INTERVAL == 0:
            print(f"Episode {episode+1}/{episodes}, Reward: {total_reward:.2f}")


if __name__ == "__main__":
    torch.manual_seed(42)
    train()
