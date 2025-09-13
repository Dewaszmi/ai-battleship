import numpy as np
import torch

from ai_battleship.ai_agent.agent import Agent
from ai_battleship.ai_agent.environment import AgentEnvironment
from ai_battleship.constants import GRID_SIZE


def train(episodes=500, num_envs=16):
    envs = [AgentEnvironment() for _ in range(num_envs)]
    agent = Agent(grid_size=GRID_SIZE)
    print("Using device:", agent.device)

    torch.backends.cudnn.benchmark = True

    epsilon = 1.0
    epsilon_decay = 0.996
    epsilon_min = 0.05

    reward_window = []

    for episode in range(1, episodes + 1):
        states = [env.reset().to(agent.device) for env in envs]
        dones = [False] * num_envs
        episode_rewards = [0] * num_envs
        transitions = []

        while not all(dones):
            actions = []
            for i, env in enumerate(envs):
                if not dones[i]:
                    action = agent.select_action(states[i], epsilon)
                else:
                    action = (0, 0)
                actions.append(action)

            for i, env in enumerate(envs):
                if not dones[i]:
                    next_state, reward, done = env.step(actions[i])
                    next_state = next_state.to(agent.device)
                    transitions.append(
                        (states[i], actions[i], reward, next_state, done)
                    )
                    states[i] = next_state
                    episode_rewards[i] += reward
                    dones[i] = done

        # Add transitions and train once per step block
        for t in transitions:
            agent.add_to_memory(t)

        agent.train_step(batch_size=64)
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        reward_window.append(sum(episode_rewards) / num_envs)
        if episode % 50 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode}, Avg Reward (last 10): {avg_reward:.2f}")

            with torch.no_grad():
                q = (
                    agent.model(states[0])
                    .squeeze(0)
                    .view(agent.grid_size, agent.grid_size)
                )
                print("Q-values for env 0:\n", q.cpu().numpy())

    torch.save(agent.model.state_dict(), "battleship_model.pth")


if __name__ == "__main__":
    train()
