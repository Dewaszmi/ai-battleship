def generate_random_grid():
    """Generates a random valid grid"""
    queue_copy = deque(self.ships_queue)
    self.get_next_ship(queue_copy)

    max_attempts = 20
    attempts = 0

    while attempts < max_attempts:
        # Try a random position and direction
        self.cursor.row, self.cursor.col, self.direction = (
            randrange(self.grid_size),
            randrange(self.grid_size),
            choice(["v", "h"]),
        )
        # Update and optionally correct position
        self.get_position()
        self.correct_position()

        # Try to place ship, if successful get next ship
        if self.place_ship():
            if not self.get_next_ship(queue_copy):
                break

        attempts += 1

    if queue_copy:
        raise RuntimeError("Failed to generate valid AI grid")
