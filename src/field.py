from dataclasses import dataclass

FIELD_COLORS = {
    "empty": (128, 128, 255),
    "ship": (128, 128, 128),
    "hit": (255, 0, 0),
    "sunk": (64, 64, 64),
    "unknown": (255, 255, 255),
}


@dataclass
class Field:
    """Representation of a single square on a grid"""

    row: int
    col: int
    status: str = "empty"  # possible statuses = "empty, ship, hit, sunk, unknown"
    color: tuple = (128, 128, 255)

    def assign_color(self):
        self.color = FIELD_COLORS[self.status]
