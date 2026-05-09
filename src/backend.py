from __future__ import annotations

import argparse
import heapq
import json
import random
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
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
from trajectory import Point
from trajectory import build_trajectory_from_path

Position = tuple[int, int]
ACTIONS: tuple[Position, ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))
DEFAULT_APPLES: tuple[Position, ...] = (
    (5, 3),
    (9, 6),
    (13, 6),
    (13, 11),
    (16, 14),
)
DEFAULT_FRAME_DURATION_MS = 160
DEFAULT_WALL_MODE = "wrap"


@dataclass(frozen=True)
class SnakeProblem:
    grid_size: int = 20
    start: Position = (2, 3)
    apples: tuple[Position, ...] = DEFAULT_APPLES


@dataclass
class SnakeResult:
    approach: str
    optimizer: str
    trajectory: list[Position]
    order: list[Position]
    completed: bool
    steps: int
    score: float

    def to_dict(self) -> dict[str, object]:
        return {
            "approach": self.approach,
            "optimizer": self.optimizer,
            "trajectory": [list(cell) for cell in self.trajectory],
            "order": [list(cell) for cell in self.order],
            "completed": self.completed,
            "steps": self.steps,
            "score": self.score,
        }


DEFAULT_PROBLEM = SnakeProblem()


def wrap_cell(problem: SnakeProblem, cell: Position) -> Position:
    return (cell[0] % problem.grid_size, cell[1] % problem.grid_size)


def wrapped_delta(problem: SnakeProblem, start: int, goal: int) -> int:
    direct = goal - start
    wrapped = direct - problem.grid_size if direct > 0 else direct + problem.grid_size
    if abs(direct) <= abs(wrapped):
        return direct
    return wrapped


def wrapped_manhattan(problem: SnakeProblem, a: Position, b: Position) -> int:
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return min(dx, problem.grid_size - dx) + min(dy, problem.grid_size - dy)


def manhattan_path(problem: SnakeProblem, start: Position, goal: Position) -> list[Position]:
    x, y = start
    path: list[Position] = []
    dx = wrapped_delta(problem, x, goal[0])
    dy = wrapped_delta(problem, y, goal[1])
    for _ in range(abs(dx)):
        x = (x + (1 if dx > 0 else -1)) % problem.grid_size
        path.append((x, y))
    for _ in range(abs(dy)):
        y = (y + (1 if dy > 0 else -1)) % problem.grid_size
        path.append((x, y))
    return path


def path_from_order(problem: SnakeProblem, order: list[Position]) -> list[Position]:
    trajectory = [problem.start]
    head = problem.start
    for apple in order:
        for step in manhattan_path(problem, head, apple):
            trajectory.append(step)
        head = apple
    return trajectory


def initial_snake(problem: SnakeProblem, length: int = 4) -> list[Position]:
    snake = [problem.start]
    while len(snake) < length:
        tail = snake[-1]
        for dx, dy in ((-1, 0), (0, -1), (1, 0), (0, 1)):
            nxt = wrap_cell(problem, (tail[0] + dx, tail[1] + dy))
            if nxt not in snake:
                snake.append(nxt)
                break
    return snake


def visited_order(problem: SnakeProblem, trajectory: list[Position]) -> list[Position]:
    remaining = set(problem.apples)
    order: list[Position] = []
    for cell in trajectory:
        if cell in remaining:
            remaining.remove(cell)
            order.append(cell)
    return order


def trajectory_payload(
    problem: SnakeProblem,
    result: SnakeResult,
    frame_duration_ms: int = DEFAULT_FRAME_DURATION_MS,
) -> dict[str, object]:
    replay = build_trajectory_from_path(
        title=f"{result.approach} trajectory ({result.optimizer})",
        grid_size=problem.grid_size,
        frame_duration_ms=frame_duration_ms,
        apples=[Point(*apple) for apple in problem.apples],
        head_path=[Point(*cell) for cell in result.trajectory],
        initial_snake=[Point(*cell) for cell in initial_snake(problem)],
        wall_mode=DEFAULT_WALL_MODE,
    )
    return replay.as_dict()


def all_collected_mask(problem: SnakeProblem) -> int:
    return (1 << len(problem.apples)) - 1


def next_mask(problem: SnakeProblem, cell: Position, remaining_mask: int) -> int:
    for index, apple in enumerate(problem.apples):
        if apple == cell:
            remaining_mask &= ~(1 << index)
    return remaining_mask


def mask_points(problem: SnakeProblem, remaining_mask: int) -> list[Position]:
    return [apple for index, apple in enumerate(problem.apples) if remaining_mask & (1 << index)]


def mst_cost(problem: SnakeProblem, points: list[Position]) -> int:
    if len(points) < 2:
        return 0
    used = [False] * len(points)
    best = [10**9] * len(points)
    best[0] = 0
    total = 0
    for _ in range(len(points)):
        choice = -1
        choice_cost = 10**9
        for index, value in enumerate(best):
            if not used[index] and value < choice_cost:
                choice = index
                choice_cost = value
        used[choice] = True
        total += choice_cost
        for index, point in enumerate(points):
            if used[index]:
                continue
            candidate = wrapped_manhattan(problem, points[choice], point)
            if candidate < best[index]:
                best[index] = candidate
    return total


def heuristic(problem: SnakeProblem, head: Position, remaining_mask: int) -> int:
    remaining = mask_points(problem, remaining_mask)
    if not remaining:
        return 0
    return min(wrapped_manhattan(problem, head, apple) for apple in remaining) + mst_cost(problem, remaining)


def solve_astar(problem: SnakeProblem = DEFAULT_PROBLEM) -> SnakeResult:
    start_mask = next_mask(problem, problem.start, all_collected_mask(problem))
    start_state = (problem.start, start_mask)
    queue: list[tuple[int, int, tuple[Position, int]]] = []
    heapq.heappush(queue, (heuristic(problem, problem.start, start_mask), 0, start_state))
    costs = {start_state: 0}
    parents: dict[tuple[Position, int], tuple[Position, int] | None] = {start_state: None}

    goal_state = start_state
    while queue:
        _, steps, state = heapq.heappop(queue)
        if steps != costs[state]:
            continue
        head, remaining_mask = state
        if remaining_mask == 0:
            goal_state = state
            break
        for dx, dy in ACTIONS:
            nxt = wrap_cell(problem, (head[0] + dx, head[1] + dy))
            nxt_mask = next_mask(problem, nxt, remaining_mask)
            nxt_state = (nxt, nxt_mask)
            nxt_steps = steps + 1
            if nxt_steps >= costs.get(nxt_state, 10**9):
                continue
            costs[nxt_state] = nxt_steps
            parents[nxt_state] = state
            priority = nxt_steps + heuristic(problem, nxt, nxt_mask)
            heapq.heappush(queue, (priority, nxt_steps, nxt_state))

    trajectory: list[Position] = []
    cursor: tuple[Position, int] | None = goal_state
    while cursor is not None:
        trajectory.append(cursor[0])
        cursor = parents[cursor]
    trajectory.reverse()
    order = visited_order(problem, trajectory)
    return SnakeResult(
        approach="astar",
        optimizer="a_star",
        trajectory=trajectory,
        order=order,
        completed=len(order) == len(problem.apples),
        steps=len(trajectory) - 1,
        score=float(len(trajectory) - 1),
    )


@lru_cache(maxsize=None)
def optimal_route_cost(problem: SnakeProblem, head: Position, remaining_mask: int) -> int:
    if remaining_mask == 0:
        return 0
    best = 10**9
    for index, apple in enumerate(problem.apples):
        bit = 1 << index
        if not remaining_mask & bit:
            continue
        candidate = wrapped_manhattan(problem, head, apple) + optimal_route_cost(problem, apple, remaining_mask ^ bit)
        if candidate < best:
            best = candidate
    return best


def encode_state(problem: SnakeProblem, head: Position, remaining_mask: int) -> torch.Tensor:
    values = [head[0] / problem.grid_size, head[1] / problem.grid_size]
    for index, apple in enumerate(problem.apples):
        active = 1.0 if remaining_mask & (1 << index) else 0.0
        values.append(wrapped_delta(problem, head[0], apple[0]) / problem.grid_size)
        values.append(wrapped_delta(problem, head[1], apple[1]) / problem.grid_size)
        values.append(active)
    return torch.tensor(values, dtype=torch.float32)


def oracle_next_cell(problem: SnakeProblem, head: Position, remaining_mask: int) -> Position:
    candidates: list[tuple[int, Position]] = []
    for dx, dy in ACTIONS:
        nxt = wrap_cell(problem, (head[0] + dx, head[1] + dy))
        nxt_mask = next_mask(problem, nxt, remaining_mask)
        candidates.append((1 + optimal_route_cost(problem, nxt, nxt_mask), nxt))
    _, best = min(candidates, key=lambda item: item[0])
    return best


def q_dataset(problem: SnakeProblem, sample_count: int | None = 256, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    states = [
        ((x, y), remaining_mask)
        for remaining_mask in range(all_collected_mask(problem) + 1)
        for x in range(problem.grid_size)
        for y in range(problem.grid_size)
    ]
    start_state = (problem.start, next_mask(problem, problem.start, all_collected_mask(problem)))
    if sample_count is None or sample_count >= len(states):
        selected = states
    else:
        rng = random.Random(seed)
        pool = [state for state in states if state != start_state]
        selected = [start_state, *rng.sample(pool, sample_count - 1)]

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


def dqn_optimizer(name: str, params) -> object:
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


def solve_dqn(
    problem: SnakeProblem = DEFAULT_PROBLEM,
    optimizer_name: str = "extended_kalman_filter",
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

    for _ in range(train_steps):
        if optimizer_name in {"extended_kalman_filter", "levenberg_marquardt"}:
            optimizer.step(residuals)
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
        choice: Position | None = None
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


def route_loss(keys: torch.nn.Parameter, apples: torch.Tensor, start: torch.Tensor, grid_size: int) -> torch.Tensor:
    order = torch.argsort(keys)
    tour = torch.vstack((start, apples[order]))
    delta = tour.diff(dim=0).abs()
    wrapped = torch.minimum(delta, grid_size - delta)
    return wrapped.sum()


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


def solve_route_optimizer(
    problem: SnakeProblem = DEFAULT_PROBLEM,
    optimizer_name: str = "genetic",
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


def kohonen_optimizer(name: str, params) -> object:
    if name == "extended_kalman_filter":
        return ExtendedKalmanFilter(params, q=1e-5, tau=0.995)
    if name == "levenberg_marquardt":
        return LevenbergMarquardt(params, mu=5.0, strategy="line search", line_search_method="armijo")
    raise ValueError(f"Unknown Kohonen optimizer: {name}")


def solve_kohonen(
    problem: SnakeProblem = DEFAULT_PROBLEM,
    optimizer_name: str = "levenberg_marquardt",
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


def run_all_approaches(
    problem: SnakeProblem = DEFAULT_PROBLEM,
    dqn_optimizer_name: str = "extended_kalman_filter",
) -> dict[str, SnakeResult]:
    return {
        "astar": solve_astar(problem),
        "dqn": solve_dqn(problem, optimizer_name=dqn_optimizer_name),
        "genetic": solve_route_optimizer(problem, optimizer_name="genetic"),
        "simulated_annealing": solve_route_optimizer(problem, optimizer_name="annealing"),
        "kohonen": solve_kohonen(problem),
    }


def run_backend(
    problem: SnakeProblem = DEFAULT_PROBLEM,
    approach: str = "astar",
    dqn_optimizer_name: str = "extended_kalman_filter",
    frame_duration_ms: int = DEFAULT_FRAME_DURATION_MS,
) -> dict[str, object]:
    if approach == "astar":
        result = solve_astar(problem)
    elif approach == "dqn":
        result = solve_dqn(problem, optimizer_name=dqn_optimizer_name)
    elif approach == "genetic":
        result = solve_route_optimizer(problem, optimizer_name="genetic")
    elif approach == "simulated_annealing":
        result = solve_route_optimizer(problem, optimizer_name="annealing")
    elif approach == "kohonen":
        result = solve_kohonen(problem)
    elif approach == "metropolis":
        result = solve_route_optimizer(problem, optimizer_name="metropolis")
    else:
        raise ValueError(f"Unknown approach: {approach}")
    return trajectory_payload(problem, result, frame_duration_ms=frame_duration_ms)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the snake backend solvers.")
    parser.add_argument(
        "--approach",
        choices=["astar", "dqn", "genetic", "simulated_annealing", "kohonen", "metropolis"],
        default="astar",
    )
    parser.add_argument(
        "--dqn-optimizer",
        choices=["newton", "extended_kalman_filter", "levenberg_marquardt", "annealing", "metropolis", "genetic"],
        default="extended_kalman_filter",
    )
    parser.add_argument("--output", default="")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_backend(
        DEFAULT_PROBLEM,
        approach=args.approach,
        dqn_optimizer_name=args.dqn_optimizer,
    )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
