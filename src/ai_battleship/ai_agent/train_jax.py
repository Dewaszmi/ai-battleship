# train_battleship.py
import jax

from ai_battleship.ai_agent.purejaxrl.ppo import make_train  # your PPO implementation
from src.ai_battleship.ai_agent.gymnax_wrapper import BattleshipGymnax

rng = jax.random.PRNGKey(42)
env = BattleshipGymnax(num_envs=8)

config = {
    "LR": 2.5e-4,
    "NUM_ENVS": env.num_envs,
    "NUM_STEPS": 128,
    "TOTAL_TIMESTEPS": 1e6,
    "UPDATE_EPOCHS": 4,
    "NUM_MINIBATCHES": 4,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.01,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "ACTIVATION": "tanh",
    "ENV_NAME": "Battleship-v0",  # dummy, only used for logging
    "ANNEAL_LR": True,
    "DEBUG": True,
}

train_jit = jax.jit(make_train(config))
out = train_jit(rng)
