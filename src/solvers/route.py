from __future__ import annotations

import torch

from domain import SnakeProblem
from domain import SnakeResult
from grid import path_from_order
from optimizer_factory import route_optimizer


def route_loss(keys: torch.nn.Parameter, apples: torch.Tensor, start: torch.Tensor, grid_size: int) -> torch.Tensor:
    order = torch.argsort(keys)
    tour = torch.vstack((start, apples[order]))
    delta = tour.diff(dim=0).abs()
    wrapped = torch.minimum(delta, grid_size - delta)
    return wrapped.sum()


def solve_route_optimizer(
    problem: SnakeProblem,
    optimizer_name: str,
    steps: int = 250,
    seed: int = 0,
) -> SnakeResult:
    torch.manual_seed(seed)
    apples = torch.tensor(problem.apples, dtype=torch.float32)
    start = torch.tensor(problem.start, dtype=torch.float32).view(1, 2)
    keys = torch.nn.Parameter(torch.randn(len(problem.apples)))
    optimizer = route_optimizer(optimizer_name, [keys])

    def closure() -> torch.Tensor:
        return route_loss(keys, apples, start, problem.grid_size)

    for _ in range(steps):
        optimizer.step(closure)

    order = [problem.apples[index] for index in torch.argsort(keys).tolist()]
    trajectory = path_from_order(problem, order)
    approach = "simulated_annealing" if optimizer_name == "annealing" else optimizer_name
    return SnakeResult(
        approach=approach,
        optimizer=optimizer_name,
        trajectory=trajectory,
        order=order,
        completed=len(order) == len(problem.apples),
        steps=len(trajectory) - 1,
        score=float(route_loss(keys, apples, start, problem.grid_size).item()),
    )
