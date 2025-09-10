import torch
import matplotlib.pyplot as plt

from ai_battleship.ai_agent.agent import Agent
from ai_battleship.ai_agent.environment import AgentEnvironment
from ai_battleship.constants import GRID_SIZE


def train(episodes):
    """Train agent in battleship environment, save trained model plus visualisation graph to separate files."""
    env = AgentEnvironment()
    agent = Agent(grid_size=GRID_SIZE)

    print("Using device:", agent.device)

    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.05
    
    # Store training data for graph visualisation
    window_size = 5
    reward_history = []
    reward_window = []
    epsilon_history = []

    for episode in range(1, episodes + 1):
        state = env.reset().to(agent.device)
        done = False
        total_reward = 0

        while not done: # Shoot until all ships are sunk
            action = agent.select_action(state, epsilon)
            next_state, reward, done = env.step(action)
            next_state = next_state.to(agent.device) # use cuda

            agent.add_to_memory((state, action, reward, next_state, done))
            agent.train_step(batch_size=32)

            state = next_state
            total_reward += reward

        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        reward_window.append(total_reward)

        if episode % window_size == 0:
            avg_reward = sum(reward_window) / len(reward_window)
            reward_history.append(avg_reward)
            epsilon_history.append(epsilon)
            reward_window = []
            print(f"Episode {episode}/{episodes} - Avg reward (per {window_size} window size): {avg_reward:.2f}, Epsilon: {epsilon:.3f}")

    torch.save(agent.model.state_dict(), "battleship_model.pth")

    # Plotting
    x = list(range(10, episodes + 1, 10))
    plt.figure(figsize=(10, 6))
    plt.plot(x, reward_history, label=f"Average Reward per {window_size} episodes", color="blue")
    plt.plot(x, epsilon_history, label="Epsilon", color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Value")
    plt.title("Training Progress")
    plt.legend()
    plt.grid(True)
    plt.savefig("nn_training_graph.png")
    plt.show()

if __name__ == "__main__":
    train(episodes=50)
