# AI battleship game

## Implementation of the Battleship board game which places the player against an AI model trained via reinforcement learning methods.

### Agent training
Utilises [JAX](https://github.com/jax-ml/jax), specifically [PureJaxRL fork's PPO implementation](https://github.com/luchris429/purejaxrl)

I originally wrote the RL code in PyTorch, but the training process took a very long time for bigger grid sizes, it has a pretty visualisation though so for legacy reasons it's accessible at the latest commit of "agent" branch.

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