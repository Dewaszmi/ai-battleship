import torch

from ai_battleship.ai_agent.agent import Agent
from ai_battleship.ai_agent.environment import AgentEnvironment
from ai_battleship.constants import GRID_SIZE


def train(episodes=500):
    env = AgentEnvironment()
    agent = Agent(grid_size=GRID_SIZE)

    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.05

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state, epsilon)
            next_state, reward, done = env.step(action)

            agent.remember((state, action, reward, next_state, done))
            agent.train_step(batch_size=32)

            state = next_state
            total_reward += reward

        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        print(
            f"Episode {episode + 1}/{episodes} - Total reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}"
        )

        torch.save(agent.model.state_dict(), "dqn_battleship.pth")
        print("model saved to dqn_battleship.pth")


if __name__ == "__main__":
    train()
