# Goals for the agent to learn:
# - shooting fields surrounding the "hit" field is valuable --> (high probability of getting another hit)
# - repeated shooting of certain fields: ["hit", "sunk", "miss"] is bad --> (no valuable reward)
# - shooting the neighboring fields of a sunk ship is bad --> (guaranteed to have no ship fields)

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ai_battleship.constants import GRID_SIZE

SHIP_SIZES = [5, 4, 3, 3, 2]


# Human readable status mapping:
# 0: unknown
# 1: ship
# 2: hit
# 3: miss
# 4: sunk
# 5: empty


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


# really complex JAX function to generate a valid grid (no overlapping ships)
def generate_random_grid_jax(rng, ship_sizes=SHIP_SIZES):
    num_ships = len(ship_sizes)
    grid = jnp.zeros((GRID_SIZE, GRID_SIZE), dtype=jnp.int32)
    ships_mask = jnp.zeros((num_ships, GRID_SIZE, GRID_SIZE), dtype=jnp.int32)

    MAX_SHIP_LEN = 5  # largest ship in Battleship

    def place_ship(carry, size_idx):
        grid, ships_mask, rng = carry
        ship_sizes = jnp.array([5, 4, 3, 3, 2])
        ship_len = ship_sizes[size_idx]

        # place a ship in a random position within grid bounds
        def sample_and_check(rng, grid):
            rng, rng_dir, rng_row, rng_col = jax.random.split(rng, 4)
            direction = jax.random.randint(rng_dir, (), 0, 2)  # 0=H, 1=V

            # row/col limits
            max_row = GRID_SIZE - ship_len * (direction == 1)
            max_col = GRID_SIZE - ship_len * (direction == 0)
            row = jax.random.randint(rng_row, (), 0, max_row)
            col = jax.random.randint(rng_col, (), 0, max_col)

            # horizontal ship
            def place_h(_):
                mask = jnp.zeros((GRID_SIZE, GRID_SIZE), dtype=jnp.int32)
                updates = jnp.concatenate(
                    [
                        jnp.ones(ship_len, dtype=jnp.int32),
                        jnp.zeros(MAX_SHIP_LEN - ship_len, dtype=jnp.int32),
                    ]
                )
                updates = updates[:ship_len].reshape(1, ship_len)
                mask = jax.lax.dynamic_update_slice(mask, updates, (row, col))
                return mask

            # vertical ship
            def place_v(_):
                mask = jnp.zeros((GRID_SIZE, GRID_SIZE), dtype=jnp.int32)
                updates = jnp.concatenate(
                    [
                        jnp.ones(ship_len, dtype=jnp.int32),
                        jnp.zeros(MAX_SHIP_LEN - ship_len, dtype=jnp.int32),
                    ]
                )
                updates = updates[:ship_len].reshape(ship_len, 1)
                mask = jax.lax.dynamic_update_slice(mask, updates, (row, col))
                return mask

            mask = jax.lax.cond(direction == 0, place_h, place_v, operand=None)
            overlap = jnp.any((grid * mask) == 1)
            return mask, overlap, rng

        # check if ships overlap
        def cond_fun(val):
            _, overlap, _ = val
            return overlap

        def body_fun(val):
            _, _, rng = val
            return sample_and_check(rng, grid)

        init_val = sample_and_check(rng, grid)
        mask, _, rng = jax.lax.while_loop(cond_fun, body_fun, init_val)

        # add new mask to grid
        grid = grid + mask
        ships_mask = ships_mask.at[size_idx].set(mask)

        return (grid, ships_mask, rng), None

    (grid, ships_mask, _), _ = jax.lax.scan(
        place_ship, (grid, ships_mask, rng), jnp.arange(num_ships)
    )
    return BattleshipState(grid, ships_mask)


def reset_vec(rng, num_envs):
    rngs = jax.random.split(rng, num_envs)
    states = jax.vmap(generate_random_grid_jax)(rngs)
    obs = jax.vmap(lambda s: get_obs_from_grid(s.grid))(states)
    return obs, states


def step_jax(state: BattleshipState, action: int):
    grid, ships_mask = state

    row = action // GRID_SIZE
    col = action % GRID_SIZE
    cell = grid[row, col]

    hit = cell == 1
    reward = -0.01
    # reward = jnp.where(hit, 1.0, -0.01)
    grid = jax.lax.cond(
        hit, lambda g: g.at[row, col].set(2), lambda g: g.at[row, col].set(3), grid
    )

    # Mark sunk ships and empty-around-sunk
    def mark_ship_sunk(g, ship_mask):
        """Check if targeted ship got sunk, mark it's fields as 'sunk' and surrounding fields as 'empty' if True."""
        ship_hits = jnp.sum(jnp.where(g == 2, 1, 0) * ship_mask)
        ship_size = jnp.count_nonzero(ship_mask)
        is_sunk = (
            ship_hits == ship_size
        )  # check if amount of hits is equal to ship size
        g = jnp.where(
            is_sunk, jnp.where(ship_mask == 1, 4, g), g
        )  # mark ship tiles as sunk

        indices = jnp.argwhere(ship_mask == 1)
        offsets = jnp.array(
            [
                [-1, -1],
                [-1, 0],
                [-1, 1],
                [0, -1],
                [0, 1],
                [1, -1],
                [1, 0],
                [1, 1],
            ]
        )
        neighbors = indices[:, None, :] + offsets[None, :, :]
        neighbors = neighbors.reshape(-1, 2)

        r, c = neighbors[:, 0], neighbors[:, 1]
        valid = (r >= 0) & (r < GRID_SIZE) & (c >= 0) & (c < GRID_SIZE)
        r, c = r[valid], c[valid]

        g = g.at[r, c].set(jnp.where(g[r, c] == 0, 5, g[r, c]))

        return g

    grid = jax.lax.cond(
        hit,
        lambda g: jax.lax.fori_loop(
            0, ships_mask.shape[0], lambda i, g: mark_ship_sunk(g, ships_mask[i]), g
        ),
        lambda g: g,
        grid,
    )

    done = jnp.all(grid != 1)
    return BattleshipState(grid, ships_mask), reward, done


def step_vec(states, actions):
    new_states, rewards, dones = jax.vmap(step_jax)(states, actions)
    obs = jax.vmap(lambda s: get_obs_from_grid(s.grid))(new_states)
    return obs, new_states, rewards, dones
