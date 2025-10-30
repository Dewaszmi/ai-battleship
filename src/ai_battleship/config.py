import argparse
from collections import deque

from ai_battleship.constants import *
from ai_battleship.utils.grid_utils import generate_random_grid


class Config:
    """Utility class for storing data about the trained agent and path to the stored agent file"""

    def __init__(
        self,
        episode_count: int = 3000000,
        allow_repeated_shots: bool = 0,
        mark_sunk_neighbors: bool = 0,
    ):
        self.episode_count = episode_count
        self.allow_repeated_shots = allow_repeated_shots
        self.mark_sunk_neighbors = mark_sunk_neighbors

        ships_queue = deque([k for k, v in SHIPS_DICT.items() for _ in range(v)])

        try:
            generate_random_grid(ships_queue, max_attempts=99999)
        except RuntimeError as e:
            raise AssertionError("Unable to generate a valid grid with specified values:") from e

        self.model_name = f"Battleship_EPISODES-{self.episode_count}_REPEATSHOTS-{self.allow_repeated_shots}_MARKSUNK-{self.mark_sunk_neighbors}.pth"
        self.model_path = f"models/{self.model_name}"


def generate_config_cli() -> Config:
    """CLI argument parser to generate config"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3000000, help="Number of episodes to run")
    parser.add_argument(
        "--allow-repeated-shots",
        type=bool,
        choices=[0, 1],
        default=0,
        help="Allow the agent to shoot the same tile repeatedly (default: 0)",
    )
    parser.add_argument(
        "--mark-sunk-neighbors",
        type=bool,
        choices=[0, 1],
        default=0,
        help="Automatically mark tiles surrounding sunk ship as guaranteed misses (default: 0)",
    )
    args = parser.parse_args()

    return Config(
        episode_count=args.episodes,
        allow_repeated_shots=args.allow_repeated_shots,
        mark_sunk_neighbors=args.mark_sunk_neighbors,
    )
