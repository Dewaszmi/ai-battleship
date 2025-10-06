import jax
import jax.numpy as jnp
from flax import struct

from gymnax.environments import environment, spaces


@struct.dataclass
class EnvState(environment.EnvState):
    pos: int
    time: int
    terminal: int


@struct.dataclass
class EnvParams(environment.EnvParams):
    max_steps_in_episode: int = 100


class LineWorld(environment.Environment[EnvState, EnvParams]):
    def __init__(self):
        super().__init__()
        self.grid_length = 5
        self.obs_shape = (self.grid_length,)

    @property
    def default_params(self):
        return EnvParams()

    def step_env(self, key, state: EnvState, action: int, params=EnvParams):
        pos = jnp.where(action == 0, state.pos - 1, state.pos + 1)
        time = state.time + 1
        terminal = (pos == 0) | (pos == self.grid_length - 1)

        state = EnvState(
            pos=pos,
            time=time,
            terminal=terminal,
        )

        reward = jnp.where(pos == 0, -1.0, 0.0)
        reward = jnp.where(pos == self.grid_length - 1, 1.0, reward)

        done = state.terminal
        info = {
            # "reward": reward,
        }

        return (
            jax.lax.stop_gradient(self.get_obs(state)),
            jax.lax.stop_gradient(state),
            reward.astype(jnp.float32),
            done,
            info,
        )

    def reset_env(self, key, params: EnvParams):
        state = EnvState(
            pos=1,
            time=0,
            terminal=False,
        )
        return self.get_obs(state), state

    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        done_steps = state.time >= params.max_steps_in_episode
        return jnp.logical_or(state.terminal, done_steps)

    def get_obs(self, state: EnvState):
        obs = jax.nn.one_hot(state.pos, self.grid_length, dtype=jnp.float32)
        return obs

    @property
    def name(self):
        return "LineWorld"

    @property
    def num_actions(self):
        return len(self.action_set)

    def action_space(self, params=EnvParams):
        # 0 = left, 1 = right
        return spaces.Discrete(2)

    def observation_space(self, params=EnvParams):
        return spaces.Box(low=0, high=1, shape=self.obs_shape, dtype=jnp.int32)

    def state_space(self, params=EnvParams):
        return spaces.Dict(
            {
                "pos": spaces.Discrete(self.grid_length),
                "time": spaces.Discrete(params.max_steps_in_episode),
                "terminal": spaces.Discrete(2),
            }
        )


def make():
    env = LineWorld()
    return env, env.default_params
