from __future__ import annotations

from functools import lru_cache

from domain import Position
from domain import SnakeProblem

ACTIONS: tuple[Position, ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))


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
