import flax
import jax
import jax.numpy as jnp

from ai_battleship.ai_agent.environment import get_obs_from_grid
from ai_battleship.ai_agent.purejaxrl.ppo import ActorCritic
from ai_battleship.constants import GRID_SIZE


class Agent:
    """Wrapper for the trained model to utilise during the game loop."""

    def __init__(self, model_path: str = "battleship_model.msgpack", seed: int = 42):
        self.rng = jax.random.PRNGKey(seed)
        self.network = ActorCritic(action_dim=GRID_SIZE * GRID_SIZE, activation="tanh")

        dummy_input = jnp.zeros((3, GRID_SIZE, GRID_SIZE))
        self.params = self.network.init(self.rng, dummy_input)

        with open(model_path, "rb") as f:
            self.params = flax.serialization.from_bytes(self.params, f.read())

    def select_action(self, grid) -> tuple[int, int]:
        """Takes the current player grid and returns an action as (row, col)."""
        obs = get_obs_from_grid(grid)
        pi, _ = self.network.apply(self.params, obs)
        self.rng, subkey = jax.random.split(self.rng)
        action = int(pi.sample(seed=subkey))
        row, col = divmod(action, GRID_SIZE)
        return row, col
