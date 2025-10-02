from typing import NamedTuple

import jax
import jax.numpy as jnp

GRID_SIZE = 10
SHIP_SIZES = [5, 4, 3, 3, 2]


class BattleshipState(NamedTuple):
    grid: jnp.ndarray  # shape [GRID_SIZE, GRID_SIZE]
    ships_mask: jnp.ndarray  # shape [num_ships, GRID_SIZE, GRID_SIZE]


# ------------------ RESET ------------------
def generate_random_grid_jax(rng, ship_sizes=SHIP_SIZES):
    num_ships = len(ship_sizes)
    grid = jnp.zeros((GRID_SIZE, GRID_SIZE), dtype=jnp.int32)
    ships_mask = jnp.zeros((num_ships, GRID_SIZE, GRID_SIZE), dtype=jnp.int32)

    def place_ship(carry, size_idx):
        grid, ships_mask, rng = carry
        ship_len = ship_sizes[size_idx]

        rng, rng_dir, rng_row, rng_col = jax.random.split(rng, 4)
        direction = jax.random.randint(rng_dir, (), 0, 2)  # 0=H,1=V
        row = jax.random.randint(
            rng_row, (), 0, GRID_SIZE - (ship_len if direction == 0 else 0)
        )
        col = jax.random.randint(
            rng_col, (), 0, GRID_SIZE - (ship_len if direction == 1 else 0)
        )

        def place_h(g):
            g = g.at[row, col : col + ship_len].set(1)
            mask = jnp.zeros((GRID_SIZE, GRID_SIZE), dtype=jnp.int32)
            mask = mask.at[row, col : col + ship_len].set(1)
            return g, mask

        def place_v(g):
            g = g.at[row : row + ship_len, col].set(1)
            mask = jnp.zeros((GRID_SIZE, GRID_SIZE), dtype=jnp.int32)
            mask = mask.at[row : row + ship_len, col].set(1)
            return g, mask

        new_grid, mask = jax.lax.cond(direction == 0, place_h, place_v, grid)
        ships_mask = ships_mask.at[size_idx].set(mask)
        return (new_grid, ships_mask, rng), None

    (grid, ships_mask, _), _ = jax.lax.scan(
        place_ship, (grid, ships_mask, rng), jnp.arange(num_ships)
    )
    return BattleshipState(grid, ships_mask)


def get_obs_from_grid(grid):
    unknown = (grid == 0) | (grid == 1)
    hit = grid == 2
    others = grid >= 3
    obs = jnp.stack([unknown, hit, others], axis=0).astype(jnp.float32)
    return obs


def reset_vec(rng, num_envs):
    rngs = jax.random.split(rng, num_envs)
    states = jax.vmap(generate_random_grid_jax)(rngs)
    obs = jax.vmap(lambda s: get_obs_from_grid(s.grid))(states)
    return obs, states


# ------------------ STEP ------------------
def step_jax(state: BattleshipState, action: int):
    grid, ships_mask = state

    row = action // GRID_SIZE
    col = action % GRID_SIZE
    cell = grid[row, col]

    hit = cell == 1
    reward = jnp.where(hit, 1.0, -0.01)
    grid = jax.lax.cond(
        hit, lambda g: g.at[row, col].set(2), lambda g: g.at[row, col].set(3), grid
    )

    # Mark sunk ships and empty-around-sunk
    def mark_ship_sunk(g, ship_mask):
        ship_hits = jnp.sum(jnp.where(g == 2, 1, 0) * ship_mask)
        ship_size = jnp.sum(ship_mask)
        sunk = ship_hits == ship_size
        g = jnp.where(sunk, jnp.where(ship_mask == 1, 4, g), g)

        indices = jnp.argwhere(ship_mask == 1)
        for i in range(indices.shape[0]):
            r, c = indices[i]
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                        g = g.at[nr, nc].set(jnp.where(g[nr, nc] == 0, 4, g))
        return g

    grid = jax.lax.fori_loop(
        0, ships_mask.shape[0], lambda i, g: mark_ship_sunk(g, ships_mask[i]), grid
    )
    done = jnp.all(jnp.logical_or(grid != 1, grid == 2))
    return BattleshipState(grid, ships_mask), reward, done


def step_vec(states, actions):
    new_states, rewards, dones = jax.vmap(step_jax)(states, actions)
    obs = jax.vmap(lambda s: get_obs_from_grid(s.grid))(new_states)
    return obs, new_states, rewards, dones
