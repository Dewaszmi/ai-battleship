# AI battleship game

### Implementation of the Battleship board game which places the player against an AI model trained via reinforcement learning methods.

The project consists of the code to train an agent to play Battleship as well as a graphical port of the game for the human player to use.

The algorithm used for training is a modified version of the [classic PPO implementation](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py) from the [CleanRL library](https://github.com/vwxyzjn/cleanrl). It utilises a custom Battleship Gymnasium environment, which takes additional arguments in the training script to set special "difficulty" variations for the agent to learn (or not ^\_^).

The project also includes the ability to log and monitor training processes via Tensorboard (also borrowed from the CleanRL's project repository).

## Setup

As the main training loop is taken from CleanRL's repository, most of the dependencies are also shared.

Main requirement is **Python version >=3.7.1,<3.11**, after that the easiest way to setup is via uv:

```
uv venv
uv pip install -e .
```

should prepare the virtual environment with all the dependencies ready.

(in the future i might try to update the code to make it compatible with newer Python, for now stick to the older versions)

## Usage

**Training the agent:**

```
python src/ai_battleship/ai/ppo.py --seed 1 --env-id Battleship-v0 --total-timesteps 1000000
```

**Start the game:**

```
python src/ai_battleship/main.py
```

Controls:

- arrow keys / hjkl to move cursor
- space to place ship / shoot target
- rotate ship during setup with r

**Access Tensorboard training log data:**

```
tensorboard --logdir runs
```

## Training results

Before digging into the results of training I recommend looking into this [short blog post](https://mattfife.com/?p=5252) about the expected results for a Battleship game played by a human player utilising various strategies. The most optimal strategy using a probability map + hunting method + parity searching seems to require on average 42 moves for the player to win.

Here are the results I received for a standard 10x10 grid with 5 ships (default Battleship setup)

**Relationship between amount of episodes used during training and average shot count observed during the ending of the training process:**

- 1 million - 70
- 2 million - 55
- 3 million - 48

**Crude screenshot from the Tensorboard panel:**

![Tensorboard screenshot](./pictures/tensorboard_ss.jpg)

This was tested with both shot masking (no repeated shots), and empty tile highlight (tiles neighboring sunk ship are highlighted for the agent as guaranteed misses) enabled, so with all the extra "difficulty" settings disabled, which means the agent had the least amount of rules to learn by itself. Enabling any of these rules would result in much worse performance at the start and slighlty worse results at the ending, but overall the agent seems to learn all scenarios pretty well.

I didn't try using episode count higher than 3 million due to quite long time taken by the training process, but I think the average result of 48 shots is pretty close to the optimum amount of 42 measured in the linked post, so I'm pretty happy with the outcome.

## TODO:

I might try adding a JAX-compatible variant of the environment and make it work with [PureJaxRL's PPO implementation](https://github.com/luchris429/purejaxrl/blob/main/purejaxrl/ppo.py) to try to get better training speeds (currently I think it's kind of close to working)

## Citing

[CleanRL's](https://github.com/vwxyzjn/cleanrl) algorithm library and Tensorboard logging implementation:

```
@article{huang2022cleanrl,
  author  = {Shengyi Huang and Rousslan Fernand Julien Dossa and Chang Ye and Jeff Braga and Dipam Chakraborty and Kinal Mehta and João G.M. Araújo},
  title   = {CleanRL: High-quality Single-file Implementations of Deep Reinforcement Learning Algorithms},
  journal = {Journal of Machine Learning Research},
  year    = {2022},
  volume  = {23},
  number  = {274},
  pages   = {1--18},
  url     = {http://jmlr.org/papers/v23/21-1342.html}
}
```

[PureJaxRL's](https://github.com/luchris429/purejaxrl) PPO implementation (currently unused, however some borrowed code is present)

```
@article{lu2022discovered,
    title={Discovered policy optimisation},
    author={Lu, Chris and Kuba, Jakub and Letcher, Alistair and Metz, Luke and Schroeder de Witt, Christian and Foerster, Jakob},
    journal={Advances in Neural Information Processing Systems},
    volume={35},
    pages={16455--16468},
    year={2022}
}
```
