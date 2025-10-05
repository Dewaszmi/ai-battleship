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

from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces

from ai_battleship.constants import GRID_SIZE, SHIPS_DICT


@struct.dataclass
class EnvState(environment.EnvState):
    ships_grid: jnp.ndarray  # (GRID_SIZE, GRID_SIZE, SHIP_AMOUNT)
    obs_grid: jnp.ndarray  # (GRID_SIZE, GRID_SIZE, 1)
    time: int
    terminal: bool


@struct.dataclass
class EnvParams(environment.EnvParams):
    max_steps_in_episode: int = 100


class Battleship(environment.Environment[EnvState, EnvParams]):
    def __init__(self):
        super().__init__()
        self.grid_size = GRID_SIZE
        self.ships_list: jnp.ndarray = jnp.array([5, 4, 3, 3, 2])
        self.obs_shape = (10, 10, 1)

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
        ships_grid = jnp.moveaxis(ships_mask, 0, -1)  # (grid_size, grid_size, num_ships)
        obs_grid = jnp.zeros((grid_size, grid_size, 1), dtype=jnp.int32)

        state = EnvState(ships_grid=ships_grid, obs_grid=obs_grid, time=0, terminal=False)

        return obs_grid.astype(jnp.float32), state

    def step_env(self, key, state: EnvState, action: int, params=EnvParams):
        grid_size = self.grid_size
        ships_grid = state.ships_grid
        obs_grid = state.obs_grid

        row = action // grid_size
        col = action % grid_size
        is_hit = (ships_grid[:, row, col] == STATUS_MAP["SHIP"]).any()

        def hit_fn(data):
            ships_grid, obs_grid = data
            ships_with_tile = ships_grid[:, row, col] == 1
            ship_idx = jnp.argmax(ships_with_tile)
            ship_mask = ships_grid[ship_idx]

            # Mark the hit
            obs_grid = obs_grid.at[row, col, 0].set(STATUS_MAP["HIT"])
            ships_grid = ships_grid.at[ship_idx, row, col].set(STATUS_MAP["HIT"])

            # Check if that ship is now sunk
            ship_hits = jnp.sum((ships_grid[ship_idx] == STATUS_MAP["HIT"]).astype(jnp.int32))
            ship_size = jnp.sum(ship_mask)
            is_sunk = ship_hits == ship_size

            def mark_sunk(data):
                ships_grid, obs_grid = data

                # Mark the ship tiles as SUNK
                ships_grid = ships_grid.at[ship_idx].set(
                    jnp.where(ship_mask == 1, STATUS_MAP["SUNK"], ships_grid[ship_idx])
                )

                # Mark neighbors as EMPTY if UNKNOWN
                # Create 2D boolean mask of neighbors
                rows = jnp.arange(GRID_SIZE)[:, None]  # (GRID_SIZE,1)
                cols = jnp.arange(GRID_SIZE)[None, :]  # (1, GRID_SIZE)
                neighbor_mask = jnp.zeros((GRID_SIZE, GRID_SIZE), dtype=jnp.bool_)

                def update_neighbor(neighbor_mask, r_offset, c_offset):
                    r_idx = jnp.clip(rows + r_offset, 0, GRID_SIZE - 1)
                    c_idx = jnp.clip(cols + c_offset, 0, GRID_SIZE - 1)
                    mask = ship_mask[r_idx, c_idx]
                    return neighbor_mask | mask

                # Consider 8 neighbors + self (3x3)
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        neighbor_mask = update_neighbor(neighbor_mask, dr, dc)

                # Only mark unknown cells
                unknown_mask = obs_grid[..., 0] == STATUS_MAP["UNKNOWN"]
                empty_mask = neighbor_mask & unknown_mask
                obs_grid = obs_grid.at[..., 0].set(
                    jnp.where(empty_mask, STATUS_MAP["EMPTY"], obs_grid[..., 0])
                )

                return ships_grid, obs_grid

            ships_grid, obs_grid = jax.lax.cond(is_sunk, mark_sunk, lambda x: x, (ships_grid, obs_grid))
            return ships_grid, obs_grid

        def miss_fn(data):
            ships_grid, obs_grid = data
            obs_grid = obs_grid.at[row, col, 0].set(1)
            return ships_grid, obs_grid

        ships_grid, obs_grid = jax.lax.cond(is_hit, hit_fn, miss_fn, (ships_grid, obs_grid))

        reward = -0.01
        time = state.time + 1
        done = self.is_terminal(ships_grid=ships_grid, time=time)

        state = EnvState(
            ships_grid=ships_grid,
            obs_grid=obs_grid,
            time=time,
            terminal=done,
        )

        return state, reward, done, {}

    def is_terminal(self, ships_grid, time: int, params: EnvParams):
        all_ships_sunk = jnp.all(ships_grid != 2, axis=(0, 1))
        return jnp.logical_or(all_ships_sunk, jnp.array(time >= params.max_steps_in_episode))

    def get_obs(self, ships_grid):
        # Combine all ship channels into a single 2D grid
        combined_grid = jnp.max(ships_grid, axis=0)  # shape: (GRID_SIZE, GRID_SIZE)

        unknown = (combined_grid == STATUS_MAP["UNKNOWN"]) | (combined_grid == STATUS_MAP["SHIP"])
        hit = combined_grid == STATUS_MAP["HIT"]
        other = combined_grid >= STATUS_MAP["MISS"]  # MISS, SUNK, EMPTY

        obs = jnp.stack([unknown, hit, other], axis=-1).astype(jnp.int32)  # (GRID_SIZE, GRID_SIZE, 3)
        return obs

    def num_actions(self) -> int:
        return len(self.action_set)

    def action_space(self, params=EnvParams):
        return spaces.Discrete(self.grid_size**2)

    def observation_space(self, params=EnvParams):
        return spaces.Box(low=0, high=2, shape=self.obs_shape, dtype=jnp.int32)

    def state_space(self, params=EnvParams):
        return spaces.Dict(
            {
                "ships_grid": spaces.Box(low=0, high=1, shape=(10, 10, 5), dtype=jnp.int32),
                "obs_grid": spaces.Box(low=0, high=2, shape=(10, 10, 1), dtype=jnp.int32),
                "time": spaces.Discrete(params.max_steps_in_episode),
                "terminal": spaces.Discrete(2),
            }
        )


class BattleshipState(NamedTuple):
    grid: jnp.ndarray  # shape [GRID_SIZE, GRID_SIZE]
    ships_mask: jnp.ndarray  # shape [num_ships, GRID_SIZE, GRID_SIZE]


# Each field gets assigned a state based on it's status:
# 0 - unshot field (valid shot, what the agent should target)
# 1 - hit but not sunk ship (invalid shot, used for the agent to determine the location of nearby ship fields, shouldn't be target
# 2 - everything else - misses, sunk ship fields, empty ships surrounding a sunk ship (invalid shots, shouldn't be targeted)
def get_obs_from_grid(grid):
    """Convert current grid state to agent input."""
    unknown = (grid == 0) | (grid == 1)
    hit = grid == 2
    other = grid >= 3
    obs = jnp.stack([unknown, hit, other], axis=0).astype(jnp.int32)
    return obs


# === RESET ===


# really complex JAX function to generate a valid grid (no overlapping ships)
def generate_random_grid_jax(rng, ship_sizes=SHIP_SIZES):
    num_ships = ship_sizes.shape[0]
    grid = jnp.zeros((GRID_SIZE, GRID_SIZE), dtype=jnp.int32)
    ships_mask = jnp.zeros((num_ships, GRID_SIZE, GRID_SIZE), dtype=jnp.int32)

    arange_grid = jnp.arange(GRID_SIZE, dtype=jnp.int32)  # static-length helper

    def place_ship(carry, size_idx):
        grid, ships_mask, rng = carry
        ship_len = ship_sizes[size_idx]  # jnp.int32 tracer OK

        def sample_and_check(rng, grid):
            rng, rng_dir, rng_row, rng_col = jax.random.split(rng, 4)
            direction = jax.random.randint(rng_dir, (), 0, 2)  # 0 = H, 1 = V

            # compute row/col max using tracer-safe arithmetic
            max_row = GRID_SIZE - ship_len * (direction == 1)
            max_col = GRID_SIZE - ship_len * (direction == 0)
            row = jax.random.randint(rng_row, (), 0, max_row)
            col = jax.random.randint(rng_col, (), 0, max_col)

            # Build masks using comparisons over fixed-length arange vectors
            def place_h(_):
                # row mask: True only at 'row'
                row_mask = arange_grid == row  # shape (GRID_SIZE,)
                # col mask: True where arange in [col, col + ship_len)
                col_start = col
                col_end = col + ship_len
                col_mask = (arange_grid >= col_start) & (arange_grid < col_end)
                # outer product -> grid mask
                mask = (row_mask[:, None] & col_mask[None, :]).astype(jnp.int32)
                return mask

            def place_v(_):
                col_mask = arange_grid == col
                row_start = row
                row_end = row + ship_len
                row_mask = (arange_grid >= row_start) & (arange_grid < row_end)
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

        # update grid and ships_mask
        grid = grid + mask
        ships_mask = ships_mask.at[size_idx].set(mask)

        return (grid, ships_mask, rng), None

    (grid, ships_mask, _), _ = jax.lax.scan(
        place_ship, (grid, ships_mask, rng), jnp.arange(num_ships, dtype=jnp.int32)
    )
    return BattleshipState(grid, ships_mask)


def reset_vec(rng, num_envs):
    rngs = jax.random.split(rng, num_envs)
    states = jax.vmap(generate_random_grid_jax)(rngs)
    obs = jax.vmap(lambda s: get_obs_from_grid(s.grid))(states)
    return obs, states


# === STEP ===


def step_jax(state: BattleshipState, action: int):
    """Single environment step - fully JAX-traceable, no Python loops over tracers."""
    grid, ships_mask = state.grid, state.ships_mask
    row = action // GRID_SIZE
    col = action % GRID_SIZE

    # hit or miss
    cell = grid[row, col]
    hit = jnp.equal(cell, 1)
    grid = jax.lax.cond(hit, lambda g: g.at[row, col].set(2), lambda g: g.at[row, col].set(3), grid)

    # only compute sunk logic if it was a hit
    def hit_ship_logic(g):
        # get the ship that contains this tile
        ships_with_tile = ships_mask[:, row, col] == 1
        ship_idx = jnp.argmax(ships_with_tile)
        ship_mask = ships_mask[ship_idx]

        # check if sunk
        ship_hits = jnp.sum((g == 2).astype(jnp.int32) * ship_mask)
        ship_size = jnp.sum(ship_mask)
        is_sunk = ship_hits == ship_size

        # mark sunk cells + neighbors if sunk
        def mark_ship_fn(g):
            g = jnp.where(ship_mask == 1, 4, g)

            # neighbors
            row_idx = jnp.arange(GRID_SIZE)[:, None, None] + jnp.arange(-1, 2)[None, None, :]
            col_idx = jnp.arange(GRID_SIZE)[None, :, None] + jnp.arange(-1, 2)[None, None, :]
            row_idx = jnp.clip(row_idx, 0, GRID_SIZE - 1)
            col_idx = jnp.clip(col_idx, 0, GRID_SIZE - 1)

            neighbor_mask = ship_mask[row_idx, col_idx]
            neighbor_mask = jnp.any(neighbor_mask, axis=(-2, -1))

            g = jnp.where(neighbor_mask & (g == 0), 5, g)
            return g

        g = jax.lax.cond(is_sunk, mark_ship_fn, lambda g: g, g)
        return g

    grid = jax.lax.cond(hit, hit_ship_logic, lambda g: g, grid)

    # reward and done
    reward = -0.01
    done = jnp.all(grid != 1)

    return BattleshipState(grid, ships_mask), reward, done


def step_vec(states: BattleshipState, actions: jnp.ndarray):
    """Batched vectorized step using vmap."""
    new_states, rewards, dones = jax.vmap(step_jax)(states, actions)
    obs = jax.vmap(lambda s: get_obs_from_grid(s.grid))(new_states)
    return obs, new_states, rewards, dones
