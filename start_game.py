from pathlib import Path

from ai_battleship.config import generate_config_cli
from ai_battleship.main_game import game_loop


def main():
    config = generate_config_cli()
    model_path = Path(config.model_path)
    if not model_path.exists():
        print(
            f"""
    Couldn't find a model matching specified rules in the models/ directory.
    Please run:
    python train_agent.py --episodes {config.episode_count} --block-repeated-shots {config.block_repeated_shots} --mark-sunk-neighbors {config.mark_sunk_neighbors}
    in order to train a new agent with set rules, then repeat the command.
    """
        )
        return

    print(f"Using model: {model_path}")
    print("Launching main game loop with specified agent rules.")
    game_loop(config)


if __name__ == "__main__":
    main()
