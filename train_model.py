from pathlib import Path

from ai_battleship.ai.ppo import train_loop
from ai_battleship.config import generate_config_cli


def main():
    config = generate_config_cli()
    print(
        f"""
Attempting to train model with specified rules:
episodes: {config.episode_count}
allow-repeated-shots: {config.allow_repeated_shots}
mark-sunk-neighbors: {config.mark_sunk_neighbors}
"""
    )
    model_path = Path(config.model_path)
    if model_path.exists():
        print(
            f"""
Found existing model matching the specified rules in the models/ directory under {config.model_path}, aborting.
"""
        )
        return

    print(
        "Model with specified rules not found under models/ directory, training a new model with specified rules."
    )
    train_loop(config)


if __name__ == "__main__":
    main()
