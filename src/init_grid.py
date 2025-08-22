from src.constants import *
from src.grid import Grid


def init_grids(grid_size):
    player_grid = Grid(player="Player", grid_size=grid_size)
    ai_grid = Grid(player="AI", grid_size=grid_size)
    return player_grid, ai_grid
