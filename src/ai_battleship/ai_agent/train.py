import torch

from ai_battleship.ai_agent.agent import Agent
from ai_battleship.ai_agent.environment import AgentEnvironment
from ai_battleship.constants import GRID_SIZE


def train(episodes=500, num_envs=16, steps_per_update=30):
    envs = [AgentEnvironment() for _ in range(num_envs)]
    agent = Agent(grid_size=GRID_SIZE)
    print("Using device:", agent.device)

    torch.backends.cudnn.benchmark = True

    epsilon = 1.0
    epsilon_decay = 0.997
    epsilon_min = 0.05

    reward_history = []
    reward_window = []

    for episode in range(1, episodes + 1):
        states = [env.reset().to(agent.device) for env in envs]
        dones = [False] * num_envs
        episode_rewards = [0] * num_envs
        transitions = []

        for _ in range(steps_per_update):
            actions = []
            for i, env in enumerate(envs):
                if not dones[i]:
                    # Build invalid action mask
                    state_flat = states[i].flatten()
                    invalid_mask = (
                        state_flat == 0
                    ).float()  # unknown or unhit ship = valid
                    action = agent.select_action(states[i], epsilon, invalid_mask)
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

        agent.train_step(
            batch_size=min(len(agent.memory), 32 * num_envs * steps_per_update)
        )
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        reward_window.append(sum(episode_rewards) / num_envs)
        if episode % 5 == 0:
            avg_reward = sum(reward_window) / len(reward_window)
            reward_history.append(avg_reward)
            reward_window = []
            print(
                f"Episode {episode}/{episodes} - Avg reward: {avg_reward:.2f}, Epsilon: {epsilon:.3f}"
            )

    torch.save(agent.model.state_dict(), "battleship_model.pth")


if __name__ == "__main__":
    train()
