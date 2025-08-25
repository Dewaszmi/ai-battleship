from src.constants import *
from src.grid import Grid

SHIPS_DICT = {
    5: 1,
    4: 1,
    3: 2,
    2: 1,
}


def init_grids(grid_size):
    player_grid = Grid(player="Player", grid_size=grid_size)
    ai_grid = Grid(player="AI", grid_size=grid_size)

    # Create sample for testing
    create_sample_grid(player_grid)
    create_sample_grid(ai_grid)
    return player_grid, ai_grid


def create_sample_grid(grid):
    sample = [
        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    for i in range(len(sample)):
        for j in range(len(sample[i])):
            if sample[i][j] == 1:
                status = "ship"
            else:
                status = "empty"
            grid[i, j].set_status(status)
