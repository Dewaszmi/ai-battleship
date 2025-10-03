import flax
import jax

from ai_battleship.ai_agent.gymnax_wrapper import BattleshipGymnax

# PureJaxRL PPO implementation
from ai_battleship.ai_agent.purejaxrl.ppo import make_train

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
    "ENV_NAME": "Battleship-v0",
    "ANNEAL_LR": True,
    "DEBUG": True,
}

train_jit = jax.jit(make_train(config, env=env))
out = train_jit(rng)

train_state = out["runner_state"][0]
params = train_state.params

with open("battleship_model.msgpack", "wb") as f:
    f.write(flax.serialization.to_bytes(params))
print("Model saved to battleship_model.msgpack")
