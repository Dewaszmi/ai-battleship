from ai_battleship.ai_agent.environment import reset_vec, step_vec
from ai_battleship.constants import GRID_SIZE


class BattleshipGymnax:
    """Gymnax compatible wrapper for Battleship environment"""

    def __init__(self, num_envs=8):
        self.num_envs = num_envs
        self.num_actions = GRID_SIZE * GRID_SIZE
        self.obs_shape = (3, GRID_SIZE, GRID_SIZE)

    def reset(self, rng, params=None):
        obs, states = reset_vec(rng, self.num_envs)
        return obs, states

    def step(self, rng, states, actions, params=None):
        obs, new_states, rewards, dones = step_vec(states, actions)
        info = {}
        return obs, new_states, rewards, dones, info

    def action_space(self, params=None):
        class ActionSpace:
            def __init__(self, n):
                self.n = n

        return ActionSpace(self.num_actions)

    def observation_space(self, params=None):
        class ObservationSpace:
            def __init__(self, shape):
                self.shape = shape

        return ObservationSpace(self.obs_shape)
