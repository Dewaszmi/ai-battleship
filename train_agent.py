import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument("episodes", type=int, required=True)
parser.add_argument("block_repeated_shots", type=bool, default=True)
parser.add_argument("mark_sunk_neighbors", type=bool, default=False)
args = parser.parse_args()
episodes = args.episodes

# to finish
