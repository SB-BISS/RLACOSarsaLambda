# RLACOSarsaLambda
GYM based Reinforcement learning library that connects Ant colony optimization and Sarsa Lambda.
This library includes SARSA lambda agents programmed with tabular and tile coding approximations.

The main purpose of this library is to present a heuristically accelerated approximated SARSA lambda
model in which the heuristic is dynamically calculated using an ACO algoritm. 

For the moment, the library has examples for the Mountain Car and Mountain Car 3D examples.

To install the Mountain car 3D environment and the Maze environment:

```bash
cd gym-maze
pip install -e .
```

and

```bash
cd gym_m3d
pip install -e .

This library is written in Python 2.72 and depends on:

Numpy
Scipy
Gym
Pygame


If you happen to use this library, please cite it as follows: 

@misc{rlacosl,
    author = {{Stefano Bromuri}},
    title = {{RLACOSarsaLambda: A heuristic accelaration and potential based reward shaping library for reinfrocement learning written in Python}},
    howpublished = {\url{https://github.com/SB-BISS/RLACOSarsaLambda}},
    note = {Software {L}ibrary}
}
