# ====================
# GRID STATS
# ====================

# Amount of tiles (squares) in a row
# MODIFY THIS IF YOU WANT TO CHECK DIFFERENT SETUPS (DEFAULT VALUE: 10)
GRID_SIZE = 10

# Dictionary of ships (ship length: ship amount)
# MODIFY THIS IS YOU WANT TO TRY DIFFERENT SETUPS (DEFAULT VALUES BELOW)
# SHIPS_DICT = {
#     5: 1,
#     4: 1,
#     3: 2,
#     2: 1,
# }
SHIPS_DICT = {
    5: 1,
    4: 1,
    3: 2,
    2: 1,
}


# For Pygame display
CELL_SIZE = 40
MARGIN = 3
GRID_WIDTH = CELL_SIZE * GRID_SIZE
GRID_HEIGHT = CELL_SIZE * GRID_SIZE

# ====================
# COLORS
# ====================

# Field colors
FIELD_COLORS = {
    "unknown": (128, 128, 255),  # blue
    "ship": (128, 128, 128),  # gray
    "hit": (255, 165, 0),  # orange
    "miss": (192, 192, 192),  # light gray
    "sunk": (64, 64, 64),  # dark gray
    "empty": (
        192,
        192,
        192,
    ),  # light gray (same as "miss")
}

# Field highlight colors
HIGHLIGHT_COLORS = {"good": (0, 255, 0), "bad": (255, 64, 64)}

# Color of the displayed cursor
CURSOR_COLOR = (255, 255, 0)  # yellow
