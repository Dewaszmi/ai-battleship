# AI battleship game

## Implementation of the Battleship board game which places the player against an AI model trained via reinforcement learning methods.



### Agent training
Utilises [JAX](https://github.com/jax-ml/jax), specifically [PureJaxRL fork's PPO implementation](https://github.com/luchris429/purejaxrl)

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

I originally wrote the RL code in PyTorch, but the training process took a very long time for bigger grid sizes, it has a pretty visualisation though so for legacy reasons it's accessible at the latest commit of "agent" branch.


uv venv
uv pip install -e . / uv sync
python src/ai_battleship/ai/ppo.py --seed 1 --env-id Battleship-v0 --total-timesteps 500000

alternatively
pip install -r requirements.txt
