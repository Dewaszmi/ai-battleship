# Util methods used in agent training and game loop

from random import choice

from ai_battleship.constants import *
from ai_battleship.field import Field
from ai_battleship.grid import Grid


def generate_random_grid(ships_queue):
    """Returns a randomly generated valid grid"""
    max_attempts = 30
    attempts = 0

    grid = Grid(GRID_SIZE)
    current_ship = get_next_ship(ships_queue)

    while current_ship and attempts < max_attempts:
        # Try a random position and direction
        random_dir = choice(["v", "h"])
        random_field = choice(get_valid_placements(grid, random_dir, current_ship))
        random_position = get_ship_position(
            grid, random_field.row, random_field.col, random_dir, current_ship
        )

        # Try to place ship, if successful get next ship
        if place_ship(grid, random_position):
            current_ship = get_next_ship(ships_queue)

        attempts += 1

    if current_ship:
        raise RuntimeError("Failed to generate valid AI grid")

    return grid


def get_valid_placements(grid: Grid, direction, ship_length) -> list[Field]:
    """Return a list of valid placements for the current ship length"""
    max_row = grid.grid_size
    max_col = grid.grid_size
    if direction == "v":
        max_row = grid.grid_size - ship_length
    else:
        max_col = grid.grid_size - ship_length

    return [
        f for row in grid.fields for f in row if f.row <= max_row and f.col <= max_col
    ]


def get_ship_position(
    grid: Grid, _row: int, _col: int, direction: str, ship_length: int
):
    """Gets position occupied by ship with set coordinates and direction"""
    row, col = _row, _col
    position = []

    for _ in range(ship_length):
        position.append(grid[row, col])
        if direction == "v":
            row += 1
        else:
            col += 1

    return position


def is_obstructed(grid: Grid, position: list[Field]):
    """Check if current selection is valid for ship placement"""
    area = [
        grid[field.row + adj_row, field.col + adj_col]
        for field in position
        for adj_row in [-1, 0, 1]
        for adj_col in [-1, 0, 1]
        if grid.field_exists(field.row + adj_row, field.col + adj_col)
    ]

    return any(field.status == "ship" for field in area)


def place_ship(grid: Grid, position: list[Field]):
    """Places ship at current position if possible"""
    if is_obstructed(grid, position):
        return False

    for field in position:
        field.status = "ship"
    grid.ships.append(position)
    return True


def get_next_ship(queue):
    """Returns the next ship from the queue, or None if queue is empty"""
    return queue.popleft() if queue else None


def shoot(target_grid, row, col):
    """Try to fire at selected field, return False if invalid target"""
    target = target_grid[row, col]

    # Check if valid target
    if not is_valid_target(target):
        return False

    targeted_ship = get_targeted_ship(target_grid, target)
    if targeted_ship:
        target.set_status("hit")

        if check_if_sunk(targeted_ship):
            # Mark neighboring tiles as empty
            adjacent_area = [
                target_grid[f.row + adj_row, f.col + adj_col]
                for f in targeted_ship
                for adj_row in [-1, 0, 1]
                for adj_col in [-1, 0, 1]
                if target_grid.field_exists(f.row + adj_row, f.col + adj_col)
            ]
            for field in adjacent_area:
                field.set_status("empty")
            # Mark targeted ship as sunk
            for field in targeted_ship:
                field.set_status("sunk")
    else:
        target.set_status("miss")

    return True


def is_valid_target(target):
    """Check if selected field is a valid target"""
    return target.status not in ["hit", "sunk", "miss"]


def get_targeted_ship(target_grid, target):
    """Return the ship hit by the shot, or None if missed"""
    return next((ship for ship in target_grid.ships if target in ship), None)


def check_if_sunk(target_ship):
    """Check if the targeted ship was sunk"""
    return all(f.status == "hit" for f in target_ship)


def clear_highlights(grid):
    for f in grid.fields.flat:
        f.set_color()
