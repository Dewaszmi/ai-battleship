from pathlib import Path

from ai_battleship.ai.ppo import train_loop
from ai_battleship.config import generate_config_cli


def main():
    config = generate_config_cli()
    print(
        f"""
          Attemtping to train agent with specified rules:
          episode_count={config.episode_count}, block_repeated_shots={config.block_repeated_shots}, mark_sunk_neighbors={config.mark_sunk_neighbors}
          """
    )
    model_path = Path(config.model_path)
    if model_path.exists():
        print(
            f"""
                Found existing agent matching the specified rules in the models/ directory under name {config.model_name}, aborting.
              """
        )
        return

    print("Training agent with specified rules.")
    train_loop(config)


if __name__ == "__main__":
    main()
