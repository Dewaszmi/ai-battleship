from gymnasium.envs.registration import register, registry

if "Battleship-v0" not in registry:
    register(
        id="Battleship-v0",
        entry_point="ai_battleship.ai.envs.battleship_env:BattleshipEnv",
    )
