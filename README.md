# Snake Game with Reinforcement Learning

This repository contains an implementation of the classic Snake game using Python and Pygame, along with a reinforcement learning agent that learns to play the game using Q-learning and genetic algorithms.

Given a static environment, the agent learns to navigate the grid, collect apples, and avoid collisions with walls and itself. The agent's performance improves over time through training episodes.

The idea is to collect the apples as quickly as possible, that is, minimize the trajectory. Mathematically it should be equaivalent to the traveling salesman problem, ie. we can compare the agent's performance to an optimal solution given be A* search. Obviously, for small grids, we can bruteforce the optimal solution.

We shall compare: 
- A* search (optimal)
- DQN agent (learned) (we shall penalize each step to encourage shorter paths)
- Genetic Algorithm agent (learned)
- Simulated Annealing agent (learned)
- Kohonen map agent (learned)
