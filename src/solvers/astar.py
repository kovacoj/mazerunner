from __future__ import annotations

import heapq

from domain import SnakeProblem
from domain import SnakeResult
from grid import ACTIONS
from grid import all_collected_mask
from grid import heuristic
from grid import next_mask
from grid import visited_order
from grid import wrap_cell


def solve_astar(problem: SnakeProblem) -> SnakeResult:
    start_mask = next_mask(problem, problem.start, all_collected_mask(problem))
    start_state = (problem.start, start_mask)
    queue: list[tuple[int, int, tuple[tuple[int, int], int]]] = []
    heapq.heappush(queue, (heuristic(problem, problem.start, start_mask), 0, start_state))
    costs = {start_state: 0}
    parents: dict[tuple[tuple[int, int], int], tuple[tuple[int, int], int] | None] = {start_state: None}

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

    trajectory: list[tuple[int, int]] = []
    cursor: tuple[tuple[int, int], int] | None = goal_state
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
