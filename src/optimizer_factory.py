from __future__ import annotations

import sys
from pathlib import Path

import torch

OPTIMIZERS_SRC = Path(__file__).resolve().parents[1] / "optimizers" / "src"
if str(OPTIMIZERS_SRC) not in sys.path:
    sys.path.insert(0, str(OPTIMIZERS_SRC))

from optimizers import Annealing
from optimizers import ExtendedKalmanFilter
from optimizers import Genetic
from optimizers import LevenbergMarquardt
from optimizers import Metropolis
from optimizers import Newton


def dqn_optimizer(name: str, params) -> object:
    if name == "adam":
        return torch.optim.Adam(params, lr=5e-2)
    if name == "newton":
        return Newton(params, line_search_method="armijo")
    if name == "extended_kalman_filter":
        return ExtendedKalmanFilter(params, q=1e-5, tau=0.99)
    if name == "levenberg_marquardt":
        return LevenbergMarquardt(params, mu=10.0, strategy="line search", line_search_method="armijo")
    if name == "annealing":
        optimizer = Annealing(params, cooling_rate=5e-3)
        optimizer.temperature = 3.0
        return optimizer
    if name == "metropolis":
        optimizer = Metropolis(params, cooling_rate=5e-3)
        optimizer.temperature = 3.0
        return optimizer
    if name == "genetic":
        optimizer = Genetic(params, noise_scale=0.25)
        optimizer.pop_size = 24
        optimizer.population = optimizer.params.unsqueeze(0).repeat(optimizer.pop_size, 1)
        optimizer.population += torch.randn_like(optimizer.population) * optimizer.noise_scale
        return optimizer
    raise ValueError(f"Unknown DQN optimizer: {name}")


def route_optimizer(name: str, params) -> object:
    if name == "genetic":
        optimizer = Genetic(params, noise_scale=0.3)
        optimizer.pop_size = 24
        optimizer.population = optimizer.params.unsqueeze(0).repeat(optimizer.pop_size, 1)
        optimizer.population += torch.randn_like(optimizer.population) * optimizer.noise_scale
        optimizer.mutation_strength = 0.2
        return optimizer
    if name == "annealing":
        optimizer = Annealing(params, cooling_rate=2e-3)
        optimizer.temperature = 6.0
        return optimizer
    if name == "metropolis":
        optimizer = Metropolis(params, cooling_rate=2e-3)
        optimizer.temperature = 6.0
        return optimizer
    raise ValueError(f"Unknown route optimizer: {name}")


def kohonen_optimizer(name: str, params) -> object:
    if name == "adam":
        return torch.optim.Adam(params, lr=2e-2)
    if name == "extended_kalman_filter":
        return ExtendedKalmanFilter(params, q=1e-5, tau=0.995)
    if name == "levenberg_marquardt":
        return LevenbergMarquardt(params, mu=5.0, strategy="line search", line_search_method="armijo")
    raise ValueError(f"Unknown Kohonen optimizer: {name}")
