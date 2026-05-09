from __future__ import annotations

import torch

from domain import SnakeProblem
from domain import SnakeResult
from grid import path_from_order
from optimizer_factory import kohonen_optimizer


def solve_kohonen(
    problem: SnakeProblem,
    optimizer_name: str,
    train_steps: int = 40,
    seed: int = 0,
) -> SnakeResult:
    torch.manual_seed(seed)
    apples = torch.tensor(problem.apples, dtype=torch.float32)
    start = torch.tensor(problem.start, dtype=torch.float32)
    node_count = max(2 * len(problem.apples), len(problem.apples) + 3)
    base = torch.linspace(0.0, 1.0, node_count).unsqueeze(1)
    center = apples.mean(dim=0, keepdim=True)
    init = start.view(1, 2) * (1.0 - base) + center * base
    nodes = torch.nn.Parameter(init)
    optimizer = kohonen_optimizer(optimizer_name, [nodes])

    def residuals() -> torch.Tensor:
        distances = torch.cdist(apples, nodes)
        winners = distances.argmin(dim=1)
        fit = (nodes[winners] - apples).reshape(-1)
        smooth = 0.35 * (nodes[1:] - nodes[:-1]).reshape(-1)
        anchor = 0.75 * (nodes[0] - start)
        return torch.cat((fit, smooth, anchor))

    for _ in range(train_steps):
        optimizer.step(residuals)

    with torch.no_grad():
        distances = torch.cdist(apples, nodes)
        winners = distances.argmin(dim=1)
        ranking = sorted(
            range(len(problem.apples)),
            key=lambda index: (int(winners[index].item()), float(distances[index, winners[index]].item())),
        )
    order = [problem.apples[index] for index in ranking]
    trajectory = path_from_order(problem, order)
    return SnakeResult(
        approach="kohonen",
        optimizer=optimizer_name,
        trajectory=trajectory,
        order=order,
        completed=len(order) == len(problem.apples),
        steps=len(trajectory) - 1,
        score=float(len(trajectory) - 1),
    )
