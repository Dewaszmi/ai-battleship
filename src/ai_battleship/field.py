from dataclasses import dataclass, field

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
    color: tuple = field(
        init=False
    )  # colors dependent on status, optionally overriden by highlights

    def __post_init__(self):
        self.set_color()

    def set_color(self, color=None):
        """Set default color unless specifically passed"""
        self.color = color if color else FIELD_COLORS[self.status]

    def set_status(self, status: str):
        """Update status and color"""
        self.status = status
        self.set_color()
