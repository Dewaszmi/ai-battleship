# Grid stats
GRID_SIZE = 10
CELL_SIZE = 20
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
    "empty": (128, 128, 255), # blue
    "ship": (128, 128, 128), # gray
    "hit": (255, 165, 0), # orange
    "miss": (192, 192, 192), # light gray
    "sunk": (64, 64, 64), # dark gray
    "unknown": (128, 128, 255), # blue (same as "empty")
}

# Field highlight colors
HIGHLIGHT_COLORS = {"good": (0, 255, 0), "bad": (255, 64, 64)}

# Color of the displayed cursor
CURSOR_COLOR = (255, 255, 0)  # yellow
