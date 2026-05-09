from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from domain import APPROACH_CHOICES
from domain import DEFAULT_FRAME_DURATION_MS
from domain import DEFAULT_PROBLEM
from domain import DEFAULT_WALL_MODE
from domain import DQN_OPTIMIZER_CHOICES
from domain import KOHONEN_OPTIMIZER_CHOICES
from domain import SnakeProblem
from domain import SnakeResult
from grid import initial_snake
from solvers import solve_astar
from solvers import solve_dqn
from solvers import solve_kohonen
from solvers import solve_route_optimizer
from trajectory import Point
from trajectory import build_trajectory
from trajectory import build_trajectory_from_path

TRAJECTORY_INDEX_NAME = "trajectories.json"


@dataclass(frozen=True, slots=True)
class ReplayExportSpec:
    approach: str
    label: str
    file_name: str
    group: str
    dqn_optimizer_name: str = "extended_kalman_filter"
    kohonen_optimizer_name: str = "levenberg_marquardt"
    dqn_sample_count: int | None = 256
    dqn_train_steps: int = 30

    def manifest_entry(self) -> dict[str, str]:
        return {
            "label": self.label,
            "file": self.file_name,
            "group": self.group,
        }


DEFAULT_REPLAY_EXPORT_SPECS: tuple[ReplayExportSpec, ...] = (
    ReplayExportSpec(
        approach="astar",
        label="A*",
        file_name="trajectory.json",
        group="Search",
    ),
    ReplayExportSpec(
        approach="dqn",
        label="DQN (Adam)",
        file_name="dqn_adam.json",
        group="Learned",
        dqn_optimizer_name="adam",
        dqn_sample_count=128,
        dqn_train_steps=12,
    ),
    ReplayExportSpec(
        approach="kohonen",
        label="Kohonen",
        file_name="kohonen.json",
        group="Learned",
    ),
    ReplayExportSpec(
        approach="genetic",
        label="Genetic",
        file_name="genetic.json",
        group="Route",
    ),
    ReplayExportSpec(
        approach="simulated_annealing",
        label="Simulated Annealing",
        file_name="simulated_annealing.json",
        group="Route",
    ),
    ReplayExportSpec(
        approach="metropolis",
        label="Metropolis",
        file_name="metropolis.json",
        group="Route",
    ),
)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def trajectory_payload(
    problem: SnakeProblem,
    result: SnakeResult,
    frame_duration_ms: int = DEFAULT_FRAME_DURATION_MS,
) -> dict[str, object]:
    visit_order = [problem.apples.index(apple) for apple in result.order]
    replay = build_trajectory(
        title=f"{result.approach} trajectory ({result.optimizer})",
        grid_size=problem.grid_size,
        frame_duration_ms=frame_duration_ms,
        apples=[Point(*apple) for apple in problem.apples],
        visit_order=visit_order,
        initial_snake=[Point(*cell) for cell in initial_snake(problem)],
        wall_mode=DEFAULT_WALL_MODE,
    )
    return replay.as_dict()


def solve_approach(
    problem: SnakeProblem,
    approach: str,
    dqn_optimizer_name: str,
    kohonen_optimizer_name: str,
    dqn_sample_count: int | None = 256,
    dqn_train_steps: int = 30,
) -> SnakeResult:
    if approach == "astar":
        return solve_astar(problem)
    if approach == "dqn":
        return solve_dqn(
            problem,
            optimizer_name=dqn_optimizer_name,
            sample_count=dqn_sample_count,
            train_steps=dqn_train_steps,
        )
    if approach == "genetic":
        return solve_route_optimizer(problem, optimizer_name="genetic")
    if approach == "simulated_annealing":
        return solve_route_optimizer(problem, optimizer_name="annealing")
    if approach == "kohonen":
        return solve_kohonen(problem, optimizer_name=kohonen_optimizer_name)
    if approach == "metropolis":
        return solve_route_optimizer(problem, optimizer_name="metropolis")
    raise ValueError(f"Unknown approach: {approach}")


def run_all_approaches(
    problem: SnakeProblem = DEFAULT_PROBLEM,
    dqn_optimizer_name: str = "extended_kalman_filter",
    kohonen_optimizer_name: str = "levenberg_marquardt",
) -> dict[str, SnakeResult]:
    return {
        approach: solve_approach(problem, approach, dqn_optimizer_name, kohonen_optimizer_name)
        for approach in ("astar", "dqn", "genetic", "simulated_annealing", "kohonen", "metropolis")
    }


def run_backend(
    problem: SnakeProblem = DEFAULT_PROBLEM,
    approach: str = "astar",
    dqn_optimizer_name: str = "extended_kalman_filter",
    kohonen_optimizer_name: str = "levenberg_marquardt",
    frame_duration_ms: int = DEFAULT_FRAME_DURATION_MS,
    dqn_sample_count: int | None = 256,
    dqn_train_steps: int = 30,
) -> dict[str, object]:
    result = solve_approach(
        problem,
        approach,
        dqn_optimizer_name,
        kohonen_optimizer_name,
        dqn_sample_count=dqn_sample_count,
        dqn_train_steps=dqn_train_steps,
    )
    return trajectory_payload(problem, result, frame_duration_ms=frame_duration_ms)


def export_replay_library(
    output_dir: Path,
    problem: SnakeProblem = DEFAULT_PROBLEM,
    frame_duration_ms: int = DEFAULT_FRAME_DURATION_MS,
    specs: Sequence[ReplayExportSpec] = DEFAULT_REPLAY_EXPORT_SPECS,
) -> list[dict[str, str]]:
    manifest: list[dict[str, str]] = []
    for spec in specs:
        payload = run_backend(
            problem,
            approach=spec.approach,
            dqn_optimizer_name=spec.dqn_optimizer_name,
            kohonen_optimizer_name=spec.kohonen_optimizer_name,
            frame_duration_ms=frame_duration_ms,
            dqn_sample_count=spec.dqn_sample_count,
            dqn_train_steps=spec.dqn_train_steps,
        )
        write_json(output_dir / spec.file_name, payload)
        manifest.append(spec.manifest_entry())
    write_json(output_dir / TRAJECTORY_INDEX_NAME, manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the snake backend solvers.")
    parser.add_argument("--approach", choices=APPROACH_CHOICES, default="astar")
    parser.add_argument("--dqn-optimizer", choices=DQN_OPTIMIZER_CHOICES, default="extended_kalman_filter")
    parser.add_argument("--kohonen-optimizer", choices=KOHONEN_OPTIMIZER_CHOICES, default="levenberg_marquardt")
    parser.add_argument("--output", default="")
    parser.add_argument("--export-library", default="")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.export_library:
        manifest = export_replay_library(Path(args.export_library))
        print(json.dumps(manifest, indent=2))
        return
    payload = run_backend(
        DEFAULT_PROBLEM,
        approach=args.approach,
        dqn_optimizer_name=args.dqn_optimizer,
        kohonen_optimizer_name=args.kohonen_optimizer,
    )
    if args.output:
        write_json(Path(args.output), payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
