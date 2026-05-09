from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from domain import APPROACH_CHOICES
from domain import DEFAULT_FRAME_DURATION_MS
from domain import DEFAULT_PROBLEM
from domain import DEFAULT_WALL_MODE
from domain import DQN_OPTIMIZER_CHOICES
from domain import SnakeProblem
from domain import SnakeResult
from grid import initial_snake
from solvers import solve_astar
from solvers import solve_dqn
from solvers import solve_kohonen
from solvers import solve_route_optimizer
from trajectory import Point
from trajectory import build_trajectory_from_path


def trajectory_payload(
    problem: SnakeProblem,
    result: SnakeResult,
    frame_duration_ms: int = DEFAULT_FRAME_DURATION_MS,
) -> dict[str, object]:
    try:
        replay = build_trajectory_from_path(
            title=f"{result.approach} trajectory ({result.optimizer})",
            grid_size=problem.grid_size,
            frame_duration_ms=frame_duration_ms,
            apples=[Point(*apple) for apple in problem.apples],
            head_path=[Point(*cell) for cell in result.trajectory],
            initial_snake=[Point(*cell) for cell in initial_snake(problem)],
            wall_mode=DEFAULT_WALL_MODE,
        )
    except ValueError as exc:
        raise ValueError(f"{result.approach} produced an invalid snake trajectory for replay: {exc}") from exc
    return replay.as_dict()


def solve_approach(
    problem: SnakeProblem,
    approach: str,
    dqn_optimizer_name: str,
) -> SnakeResult:
    if approach == "astar":
        return solve_astar(problem)
    if approach == "dqn":
        return solve_dqn(problem, optimizer_name=dqn_optimizer_name)
    if approach == "genetic":
        return solve_route_optimizer(problem, optimizer_name="genetic")
    if approach == "simulated_annealing":
        return solve_route_optimizer(problem, optimizer_name="annealing")
    if approach == "kohonen":
        return solve_kohonen(problem, optimizer_name="levenberg_marquardt")
    if approach == "metropolis":
        return solve_route_optimizer(problem, optimizer_name="metropolis")
    raise ValueError(f"Unknown approach: {approach}")


def run_all_approaches(
    problem: SnakeProblem = DEFAULT_PROBLEM,
    dqn_optimizer_name: str = "extended_kalman_filter",
) -> dict[str, SnakeResult]:
    return {
        approach: solve_approach(problem, approach, dqn_optimizer_name)
        for approach in ("astar", "dqn", "genetic", "simulated_annealing", "kohonen")
    }


def run_backend(
    problem: SnakeProblem = DEFAULT_PROBLEM,
    approach: str = "astar",
    dqn_optimizer_name: str = "extended_kalman_filter",
    frame_duration_ms: int = DEFAULT_FRAME_DURATION_MS,
) -> dict[str, object]:
    result = solve_approach(problem, approach, dqn_optimizer_name)
    return trajectory_payload(problem, result, frame_duration_ms=frame_duration_ms)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the snake backend solvers.")
    parser.add_argument("--approach", choices=APPROACH_CHOICES, default="astar")
    parser.add_argument("--dqn-optimizer", choices=DQN_OPTIMIZER_CHOICES, default="extended_kalman_filter")
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
