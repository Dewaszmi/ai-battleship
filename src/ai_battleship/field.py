from dataclasses import dataclass, field

from ai_battleship.constants import FIELD_COLORS


@dataclass
class Field:
    """Representation of a single square on a grid"""

    row: int
    col: int
    status: str = "unknown"  # possible statuses = "unknown, ship, hit, miss, sunk, empty"
    color: tuple[int, int, int] = field(
        init=False
    )  # colors dependent on status, optionally overriden by highlights

    def __post_init__(self):
        self.set_color()

    def set_color(self, color: tuple[int, int, int] | None = None):
        """Set default color unless specifically passed"""
        self.color = color if color is not None else FIELD_COLORS[self.status]

    def set_status(self, status: str):
        """Update status and color"""
        self.status = status
        self.set_color()

    def is_valid_target(self):
        """Check if the field is a valid target"""
        return self.status in [
            "unknown",
            "ship",
        ]
