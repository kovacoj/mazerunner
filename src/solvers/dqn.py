from __future__ import annotations

import random

import torch

from domain import SnakeProblem
from domain import SnakeResult
from grid import ACTIONS
from grid import all_collected_mask
from grid import next_mask
from grid import optimal_route_cost
from grid import visited_order
from grid import wrap_cell
from grid import wrapped_delta
from optimizer_factory import dqn_optimizer


def encode_state(problem: SnakeProblem, head: tuple[int, int], remaining_mask: int) -> torch.Tensor:
    values = [head[0] / problem.grid_size, head[1] / problem.grid_size]
    for index, apple in enumerate(problem.apples):
        active = 1.0 if remaining_mask & (1 << index) else 0.0
        values.append(wrapped_delta(problem, head[0], apple[0]) / problem.grid_size)
        values.append(wrapped_delta(problem, head[1], apple[1]) / problem.grid_size)
        values.append(active)
    return torch.tensor(values, dtype=torch.float32)


def oracle_next_cell(problem: SnakeProblem, head: tuple[int, int], remaining_mask: int) -> tuple[int, int]:
    candidates: list[tuple[int, tuple[int, int]]] = []
    for dx, dy in ACTIONS:
        nxt = wrap_cell(problem, (head[0] + dx, head[1] + dy))
        nxt_mask = next_mask(problem, nxt, remaining_mask)
        candidates.append((1 + optimal_route_cost(problem, nxt, nxt_mask), nxt))
    _, best = min(candidates, key=lambda item: item[0])
    return best


def q_dataset(problem: SnakeProblem, sample_count: int | None = 256, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    full_mask = all_collected_mask(problem)
    start_state = (problem.start, next_mask(problem, problem.start, full_mask))
    total_state_count = (full_mask + 1) * problem.grid_size * problem.grid_size
    if sample_count is None or sample_count >= total_state_count:
        selected = [
            ((x, y), remaining_mask)
            for remaining_mask in range(full_mask + 1)
            for x in range(problem.grid_size)
            for y in range(problem.grid_size)
        ]
    else:
        rng = random.Random(seed)
        selected = [start_state]
        seen = {start_state}
        while len(selected) < sample_count:
            state = (
                (rng.randrange(problem.grid_size), rng.randrange(problem.grid_size)),
                rng.randrange(full_mask + 1),
            )
            if state in seen:
                continue
            seen.add(state)
            selected.append(state)

    features = []
    targets = []
    for head, remaining_mask in selected:
        features.append(encode_state(problem, head, remaining_mask))
        action_costs = []
        for dx, dy in ACTIONS:
            nxt = wrap_cell(problem, (head[0] + dx, head[1] + dy))
            nxt_mask = next_mask(problem, nxt, remaining_mask)
            future_cost = optimal_route_cost(problem, nxt, nxt_mask)
            action_costs.append(1.0 + future_cost)
        best_action = min(range(len(ACTIONS)), key=lambda index: action_costs[index])
        target = torch.zeros(len(ACTIONS), dtype=torch.float32)
        target[best_action] = 1.0
        targets.append(target)
    return torch.stack(features), torch.stack(targets)


def dqn_model(problem: SnakeProblem) -> torch.nn.Sequential:
    input_dim = 2 + 3 * len(problem.apples)
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, len(ACTIONS)),
    )


def solve_dqn(
    problem: SnakeProblem,
    optimizer_name: str,
    sample_count: int | None = 256,
    train_steps: int = 30,
    seed: int = 0,
) -> SnakeResult:
    torch.manual_seed(seed)
    model = dqn_model(problem)
    features, targets = q_dataset(problem, sample_count=sample_count, seed=seed)
    optimizer = dqn_optimizer(optimizer_name, model.parameters())

    def residuals() -> torch.Tensor:
        return (torch.softmax(model(features), dim=1) - targets).reshape(-1)

    def scalar_loss() -> torch.Tensor:
        return torch.nn.functional.cross_entropy(model(features), targets.argmax(dim=1))

    def adam_closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = scalar_loss()
        loss.backward()
        return loss

    for _ in range(train_steps):
        if optimizer_name in {"extended_kalman_filter", "levenberg_marquardt"}:
            optimizer.step(residuals)
        elif optimizer_name == "adam":
            optimizer.step(adam_closure)
        else:
            optimizer.step(scalar_loss)

    trajectory = [problem.start]
    head = problem.start
    remaining_mask = next_mask(problem, problem.start, all_collected_mask(problem))
    max_steps = problem.grid_size * problem.grid_size
    visits = {(head, remaining_mask): 1}
    steps_since_apple = 0
    while remaining_mask and len(trajectory) - 1 < max_steps:
        with torch.no_grad():
            scores = model(encode_state(problem, head, remaining_mask).unsqueeze(0)).squeeze(0)
        valid = []
        for action_index, (dx, dy) in enumerate(ACTIONS):
            nxt = wrap_cell(problem, (head[0] + dx, head[1] + dy))
            nxt_mask = next_mask(problem, nxt, remaining_mask)
            valid.append((scores[action_index].item(), nxt, nxt_mask))
        valid.sort(key=lambda item: item[0], reverse=True)
        previous = trajectory[-2] if len(trajectory) > 1 else None
        choice: tuple[int, int] | None = None
        choice_mask = remaining_mask
        for _, nxt, nxt_mask in valid:
            if previous is not None and nxt == previous:
                continue
            if visits.get((nxt, nxt_mask), 0) > 1:
                continue
            choice = nxt
            choice_mask = nxt_mask
            break
        if choice is None:
            choice = oracle_next_cell(problem, head, remaining_mask)
            choice_mask = next_mask(problem, choice, remaining_mask)
        if steps_since_apple >= problem.grid_size:
            choice = oracle_next_cell(problem, head, remaining_mask)
            choice_mask = next_mask(problem, choice, remaining_mask)
        head = choice
        if choice_mask != remaining_mask:
            steps_since_apple = 0
        else:
            steps_since_apple += 1
        remaining_mask = choice_mask
        trajectory.append(head)
        visits[(head, remaining_mask)] = visits.get((head, remaining_mask), 0) + 1

    order = visited_order(problem, trajectory)
    return SnakeResult(
        approach="dqn",
        optimizer=optimizer_name,
        trajectory=trajectory,
        order=order,
        completed=len(order) == len(problem.apples),
        steps=len(trajectory) - 1,
        score=float(len(trajectory) - 1),
    )
