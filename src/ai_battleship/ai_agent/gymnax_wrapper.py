# gymnax_wrapper.py
from typing import NamedTuple

import jax
import jax.numpy as jnp
from gymnax_env import (
    GRID_SIZE,
    BattleshipState,
    get_obs_from_grid,
    reset_vec,
    step_vec,
)


class BattleshipGymnax:
    def __init__(self, num_envs=8):
        self.num_envs = num_envs
        self.num_actions = GRID_SIZE * GRID_SIZE
        self.obs_shape = (3, GRID_SIZE, GRID_SIZE)

    def reset(self, rng, params=None):
        obs, states = reset_vec(rng, self.num_envs)
        return obs, states

    def step(self, rng, states, actions, params=None):
        obs, new_states, rewards, dones = step_vec(states, actions)
        info = {}  # Gymnax expects an info dict
        return obs, new_states, rewards, dones, info
