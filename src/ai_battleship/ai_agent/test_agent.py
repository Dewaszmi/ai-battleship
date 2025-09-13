import random

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim

from ai_battleship.utils.grid_utils import generate_random_grid
from ai_battleship.utils.visualiser import Visualiser

SHIPS_DICT = {4: 1}  # one 4-tile ship
GRID_SIZE = 5
MAX_STEPS = 25
BATCH_SIZE = 32
GAMMA = 0.9
LR = 0.01
REPLAY_CAPACITY = 1000


# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# --- Environment ---
class AgentEnvironment:
    def __init__(self):
        self.grid_size = GRID_SIZE
        self.reset()

    def reset(self):
        self.grid = generate_random_grid(ships_queue=SHIPS_DICT)
        self.done = False
        self.steps = 0
        return self.get_state()

    def get_state(self):
        """Return multi-channel one-hot tensor [C, H, W]"""
        status_map = {"unknown": 0, "ship": 1, "miss": 2, "hit": 3, "sunk": 4}
        c = len(status_map)
        h, w = GRID_SIZE, GRID_SIZE
        state = np.zeros((c, h, w), dtype=np.float32)
        for row in range(h):
            for col in range(w):
                idx = status_map[self.grid[row, col].status]
                state[idx, row, col] = 1.0
        return torch.tensor(state)

    def step(self, action):
        row, col = action
        self.steps += 1
        field = self.grid[row, col]

        if field.status == "ship":
            field.set_status("hit")
            reward = 2.0
        elif field.status == "hit":
            reward = -0.1
        elif field.status == "sunk":
            reward = 5.0
        else:
            field.set_status("miss")
            reward = -1.0

        self.done = (field.status == "sunk") or self.steps >= MAX_STEPS
        return self.get_state(), reward, self.done


# --- CNN ---
class CNN(nn.Module):
    def __init__(self, grid_size=5, in_channels=5):
        super().__init__()
        self.grid_size = grid_size
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2),
            nn.ReLU(),
        )
        self.fc = nn.Linear(
            32 * (grid_size - 2) * (grid_size - 2), grid_size * grid_size
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# --- Training ---
env = AgentEnvironment()
model = CNN(GRID_SIZE)
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()
buffer = ReplayBuffer(REPLAY_CAPACITY)

epsilon_start = 1.0
epsilon_final = 0.05
epsilon_decay = 500  # episodes

visualiser = Visualiser()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for episode in range(1, 1001):
    state = env.reset().to(device)
    done = False
    total_reward = 0

    epsilon = epsilon_final + (epsilon_start - epsilon_final) * np.exp(
        -1.0 * episode / epsilon_decay
    )

    while not done:
        # epsilon-greedy
        if random.random() < epsilon:
            action_idx = random.randint(0, GRID_SIZE * GRID_SIZE - 1)
        else:
            with torch.no_grad():
                q_vals = model(state.unsqueeze(0))  # add batch dim
                action_idx = q_vals.argmax().item()

        row, col = divmod(action_idx, GRID_SIZE)
        next_state, reward, done = env.step((row, col))
        next_state = next_state.to(device)
        total_reward += reward

        buffer.push(state, action_idx, reward, next_state, done)
        state = next_state

        # --- train from replay buffer ---
        if len(buffer) >= BATCH_SIZE:
            batch = buffer.sample(BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*batch)
            states = torch.stack(states).to(device)
            next_states = torch.stack(next_states).to(device)
            actions = torch.tensor(actions, device=device)
            rewards = torch.tensor(rewards, device=device)
            dones = torch.tensor(dones, dtype=torch.float32, device=device)

            q_values = model(states)
            q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_q = model(next_states).max(1)[0]
                target = rewards + GAMMA * next_q * (1 - dones)

            loss = loss_fn(q_values, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # inside your episode loop
        if episode % 50 == 1 and episode >= 751:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

            with torch.no_grad():
                q_map = (
                    model(state.unsqueeze(0))
                    .squeeze(0)
                    .view(GRID_SIZE, GRID_SIZE)
                    .cpu()
                    .numpy()
                )

            visualiser.draw(env.grid, q_map=q_map, hitmarker=(row, col))
            pygame.time.delay(500)  # adjust speed

    if episode % 50 == 1:
        print(
            f"Episode {episode}, Total reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}"
        )
