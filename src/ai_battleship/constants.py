# Grid stats
CELL_SIZE = 20
GRID_SIZE = 10
MARGIN = 3
GRID_WIDTH = CELL_SIZE * GRID_SIZE
GRID_HEIGHT = CELL_SIZE * GRID_SIZE

# Dictionary of ships (ship length: ship amount)
SHIPS_DICT = {
    5: 1,
    4: 1,
    3: 2,
    2: 1,
}

# Field colors
FIELD_COLORS = {
    "empty": (128, 128, 255),
    "ship": (128, 128, 128),
    "hit": (255, 0, 0),
    "miss": (200, 200, 200),
    "sunk": (64, 64, 64),
    "unknown": (200, 200, 200),
}

# Field highlight colors
HIGHLIGHT_COLORS = {"good": (0, 255, 0), "bad": (255, 64, 64)}

# Color of the displayed cursor
CURSOR_COLOR = (255, 255, 0)  # yellow
