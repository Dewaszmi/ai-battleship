import jax

from ai_battleship.ai_agent.environment import BattleshipEnv
from ai_battleship.ai_agent.purejaxrl.ppo import make_train

rng = jax.random.PRNGKey(42)
battleship_env = BattleshipEnv()


config = {
    "LR": 2.5e-4,
    "NUM_ENVS": 4,
    "NUM_STEPS": 128,
    "TOTAL_TIMESTEPS": 5e5,
    "UPDATE_EPOCHS": 4,
    "NUM_MINIBATCHES": 4,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.01,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "ACTIVATION": "tanh",
    "ENV_NAME": "test",
    "ANNEAL_LR": True,
    "DEBUG": True,
}
train_jit = jax.jit(make_train(config, custom_env=battleship_env))
out = train_jit(rng)