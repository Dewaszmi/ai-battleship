from gymnasium.envs.registration import register

register(
    id="Battleship-v0",
    entry_point="ai_battleship.ai.envs.battleship_env_gym:BattleshipEnv",
)
