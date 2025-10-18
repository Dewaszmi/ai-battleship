# Goals for the agent to learn:
# - shooting fields surrounding the "hit" field is valuable --> (high probability of getting another hit)
# - repeated shooting of certain fields: ["hit", "sunk", "miss"] is bad --> (no valuable reward)
# - shooting the neighboring fields of a sunk ship is bad --> (guaranteed to have no ship fields)

STATUS_MAP = {
    "UNKNOWN": 0,
    "SHIP": 1,
    "HIT": 2,
    "MISS": 3,
    "SUNK": 4,
    "EMPTY": 5,
}

from typing import Any

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces

from ai_battleship.constants import GRID_SIZE


@struct.dataclass
class EnvState(environment.EnvState):
    ships_grid: jnp.ndarray  # (GRID_SIZE, GRID_SIZE, SHIP_AMOUNT)
    obs_grid: jnp.ndarray  # (GRID_SIZE, GRID_SIZE, 1)
    time: int

@struct.dataclass
class EnvParams(environment.EnvParams):
    max_steps_in_episode: int = 100


class BattleshipEnv(environment.Environment[EnvState, EnvParams]):
    """JAX compatbile implementation of Battleship board game gym environment"""
    def __init__(self):
        super().__init__()
        self.grid_size = GRID_SIZE
        self.ships_list: jnp.ndarray = jnp.array([5, 4, 3, 3, 2])
        self.obs_shape = (self.grid_size, self.grid_size, 3)

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def reset_env(self, key, params: EnvParams):
        ship_sizes = self.ships_list
        num_ships = ship_sizes.shape[0]
        grid_size = self.grid_size

        grid = jnp.zeros((grid_size, grid_size), dtype=jnp.int32)
        ships_mask = jnp.zeros((num_ships, grid_size, grid_size), dtype=jnp.int32)
        arange_grid = jnp.arange(grid_size, dtype=jnp.int32)

        def place_ship(carry, size_idx):
            grid, ships_mask, rng = carry
            ship_len = ship_sizes[size_idx]

            def sample_and_check(rng, grid):
                rng, rng_dir, rng_row, rng_col = jax.random.split(rng, 4)
                direction = jax.random.randint(rng_dir, (), 0, 2)  # 0=H, 1=V

                max_row = grid_size - ship_len * (direction == 1)
                max_col = grid_size - ship_len * (direction == 0)
                row = jax.random.randint(rng_row, (), 0, max_row)
                col = jax.random.randint(rng_col, (), 0, max_col)

                def place_h(_):
                    row_mask = arange_grid == row
                    col_mask = (arange_grid >= col) & (arange_grid < col + ship_len)
                    mask = (row_mask[:, None] & col_mask[None, :]).astype(jnp.int32)
                    return mask

                def place_v(_):
                    col_mask = arange_grid == col
                    row_mask = (arange_grid >= row) & (arange_grid < row + ship_len)
                    mask = (row_mask[:, None] & col_mask[None, :]).astype(jnp.int32)
                    return mask

                mask = jax.lax.cond(direction == 0, place_h, place_v, operand=None)
                overlap = jnp.any((grid * mask) == 1)
                return mask, overlap, rng

            def cond_fun(val):
                _, overlap, _ = val
                return overlap

            def body_fun(val):
                _, _, rng = val
                return sample_and_check(rng, grid)

            init_val = sample_and_check(rng, grid)
            mask, _, rng = jax.lax.while_loop(cond_fun, body_fun, init_val)

            grid = grid + mask
            ships_mask = ships_mask.at[size_idx].set(mask)
            return (grid, ships_mask, rng), None

        (grid, ships_mask, key), _ = jax.lax.scan(
            place_ship, (grid, ships_mask, key), jnp.arange(num_ships, dtype=jnp.int32)
        )

        # Combine individual ship channels → single binary grid (for debugging)
        ships_grid = ships_mask  # (grid_size, grid_size, num_ships)
        obs_grid = jnp.zeros((grid_size, grid_size, 1), dtype=jnp.int32)

        state = EnvState(ships_grid=ships_grid, obs_grid=obs_grid, time=0)

        return jax.lax.stop_gradient(obs_grid.astype(jnp.float32)), jax.lax.stop_gradient(state)

    def step_env(self, key: jax.Array, state: EnvState, action: int | float | jax.Array, params=EnvParams) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        grid_size = self.grid_size
        ships_grid = state.ships_grid # (num_ships, H, W)
        obs_grid = state.obs_grid # (H, W, 1)

        row = action // grid_size
        col = action % grid_size
        is_hit = jnp.any(ships_grid[:, row, col] == STATUS_MAP["SHIP"])

        def hit_fn(data):
            ships_grid, obs_grid = data
            ships_with_tile = ships_grid[:, row, col] == STATUS_MAP["SHIP"]
            ship_idx = jnp.argmax(ships_with_tile)
            ship_mask = ships_grid[ship_idx] == STATUS_MAP["SHIP"]

            # Mark the hit
            obs_grid = obs_grid.at[row, col, 0].set(STATUS_MAP["HIT"])
            ships_grid = ships_grid.at[ship_idx, row, col].set(STATUS_MAP["HIT"])

            # Check if that ship is now sunk
            ship_hits = jnp.sum((ships_grid[ship_idx] == STATUS_MAP["HIT"]).astype(jnp.int32))
            ship_size = jnp.sum(ship_mask)
            is_sunk = ship_hits == ship_size

            def mark_sunk(data):
                ships_grid, obs_grid = data

                # Mark ship tiles as SUNK
                obs_grid = obs_grid.at[..., 0].set(
                    jnp.where(ship_mask == 1, STATUS_MAP["SUNK"], obs_grid[..., 0])
                )

                # Generate neighbor mask via broadcasting
                r_idx = jnp.arange(GRID_SIZE)[:, None]  # (GRID_SIZE,1)
                c_idx = jnp.arange(GRID_SIZE)[None, :]  # (1, GRID_SIZE)

                # offsets for 3x3 neighborhood
                dr = jnp.arange(-1, 2)[:, None]
                dc = jnp.arange(-1, 2)[None, :]

                # Broadcasted neighbor indices
                r_neighbors = jnp.clip(r_idx + dr, 0, GRID_SIZE - 1)
                c_neighbors = jnp.clip(c_idx + dc, 0, GRID_SIZE - 1)

                # neighbor_mask: True if any neighbor belongs to the sunk ship
                neighbor_mask = jnp.any(ship_mask[r_neighbors, c_neighbors], axis=(-2, -1))

                # Only mark UNKNOWN neighbors as EMPTY
                unknown_mask = obs_grid[..., 0] == STATUS_MAP["UNKNOWN"]
                empty_mask = neighbor_mask & unknown_mask
                obs_grid = obs_grid.at[..., 0].set(jnp.where(empty_mask, STATUS_MAP["EMPTY"], obs_grid[..., 0]))

                return ships_grid, obs_grid


            ships_grid, obs_grid = jax.lax.cond(is_sunk, mark_sunk, lambda x: x, (ships_grid, obs_grid))
            return ships_grid, obs_grid

        def miss_fn(data):
            ships_grid, obs_grid = data
            obs_grid = obs_grid.at[row, col, 0].set(STATUS_MAP["MISS"])
            return ships_grid, obs_grid

        ships_grid, obs_grid = jax.lax.cond(is_hit, hit_fn, miss_fn, (ships_grid, obs_grid))

        reward = -0.01
        time = state.time + 1

        state = EnvState(
            ships_grid=ships_grid,
            obs_grid=obs_grid,
            time=time
        )
        done = self.is_terminal(state, params)

        return jax.lax.stop_gradient(self.get_obs(state)), jax.lax.stop_gradient(state), jnp.array(reward), done, {} # TODO: finish

    def get_obs(self, state: EnvState) -> jax.Array: # type: ignore
        """Each field gets assigned a state based on it's status:
        
        0 - unshot field (valid shot, what the agent should target)
        
        1 - hit but not sunk ship (invalid shot, used for the agent to determine the location of nearby ship fields, shouldn't be target
        
        2 - everything else - misses, sunk ship fields, empty ships surrounding a sunk ship (invalid shots, shouldn't be targeted)
        """
        # Combine all ship channels into a single 2D grid
        combined_grid = jnp.max(state.ships_grid, axis=0)  # shape: (GRID_SIZE, GRID_SIZE)

        unknown = ((combined_grid == STATUS_MAP["UNKNOWN"]) | (combined_grid == STATUS_MAP["SHIP"])).astype(jnp.int32)
        hit = (combined_grid == STATUS_MAP["HIT"]).astype(jnp.int32)
        other = (combined_grid >= STATUS_MAP["MISS"]).astype(jnp.int32)  # MISS, SUNK, EMPTY

        obs = jnp.stack([unknown, hit, other], axis=0)  # (GRID_SIZE, GRID_SIZE, 3)
        return obs.astype(jnp.float32)
    
    def is_terminal(self, state: EnvState, params: EnvParams):
        all_ships_sunk = jnp.all(state.ships_grid != STATUS_MAP["SHIP"])
        return jnp.logical_or(all_ships_sunk, jnp.array(state.time >= params.max_steps_in_episode))

    @property
    def num_actions(self) -> int:
        return self.grid_size**2

    def action_space(self, params=EnvParams):
        return spaces.Discrete(self.grid_size**2)

    def observation_space(self, params=EnvParams):
        return spaces.Box(low=0, high=1, shape=self.obs_shape, dtype=jnp.float32)

    def state_space(self, params=EnvParams):
        return spaces.Dict(
            {
                "ships_grid": spaces.Box(low=0, high=2, shape=(10, 10, 5), dtype=jnp.int32),
                "obs_grid": spaces.Box(low=0, high=2, shape=(10, 10, 1), dtype=jnp.int32),
                "time": spaces.Discrete(params.max_steps_in_episode),
                "terminal": spaces.Discrete(2),
            }
        )