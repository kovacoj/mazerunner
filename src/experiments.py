from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from time import process_time

import torch

from backend import trajectory_payload
from domain import DEFAULT_FRAME_DURATION_MS
from domain import DEFAULT_PROBLEM
from domain import SnakeProblem
from solvers import solve_astar
from solvers import solve_dqn
from solvers import solve_kohonen
from solvers import solve_route_optimizer


def cpu_count() -> int:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    count = os.cpu_count()
    if count is not None:
        return count
    return 1


def configure_cpu_threads() -> int:
    threads = cpu_count()
    torch.set_num_threads(threads)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(threads)
    return threads


@dataclass(frozen=True)
class ExperimentSpec:
    slug: str
    label: str
    approach: str
    optimizer: str
    sample_count: int | None = None
    train_steps: int | None = None
    route_steps: int | None = None


def experiment_specs() -> list[ExperimentSpec]:
    return [
        ExperimentSpec("astar", "A*", "astar", "a_star"),
        ExperimentSpec("genetic", "Genetic", "genetic", "genetic", route_steps=400),
        ExperimentSpec("simulated_annealing", "Simulated Annealing", "simulated_annealing", "annealing", route_steps=400),
        ExperimentSpec("metropolis", "Metropolis", "metropolis", "metropolis", route_steps=400),
        ExperimentSpec("tiny_dqn_extended_kalman_filter", "Tiny DQN MLP + EKF", "dqn", "extended_kalman_filter", sample_count=96, train_steps=10),
        ExperimentSpec("tiny_dqn_levenberg_marquardt", "Tiny DQN MLP + LM", "dqn", "levenberg_marquardt", sample_count=96, train_steps=8),
        ExperimentSpec("tiny_dqn_adam", "Tiny DQN MLP + Adam", "dqn", "adam", sample_count=128, train_steps=60),
        ExperimentSpec("kohonen_extended_kalman_filter", "Kohonen + EKF", "kohonen", "extended_kalman_filter", train_steps=40),
        ExperimentSpec("kohonen_levenberg_marquardt", "Kohonen + LM", "kohonen", "levenberg_marquardt", train_steps=40),
        ExperimentSpec("kohonen_adam", "Kohonen + Adam", "kohonen", "adam", train_steps=120),
    ]


def run_spec(spec: ExperimentSpec, problem: SnakeProblem) -> tuple[dict[str, object], dict[str, object]]:
    start_cpu = process_time()
    start_wall = perf_counter()
    if spec.approach == "astar":
        result = solve_astar(problem)
    elif spec.approach == "dqn":
        result = solve_dqn(
            problem,
            optimizer_name=spec.optimizer,
            sample_count=spec.sample_count,
            train_steps=spec.train_steps or 30,
            seed=0,
        )
    elif spec.approach == "kohonen":
        result = solve_kohonen(
            problem,
            optimizer_name=spec.optimizer,
            train_steps=spec.train_steps or 40,
            seed=0,
        )
    else:
        result = solve_route_optimizer(
            problem,
            optimizer_name=spec.optimizer,
            steps=spec.route_steps or 250,
            seed=0,
        )
    cpu_time = process_time() - start_cpu
    wall_time = perf_counter() - start_wall
    payload = trajectory_payload(problem, result, frame_duration_ms=DEFAULT_FRAME_DURATION_MS)
    replay_steps = len(payload["frames"]) - 1
    metrics = {
        "slug": spec.slug,
        "label": spec.label,
        "approach": spec.approach,
        "optimizer": spec.optimizer,
        "completed": result.completed,
        "solver_steps": result.steps,
        "replay_steps": replay_steps,
        "score": result.score,
        "apples_collected": len(result.order),
        "cpu_time_seconds": cpu_time,
        "wall_time_seconds": wall_time,
    }
    return payload, metrics


def write_outputs(output_dir: Path = Path("data"), problem: SnakeProblem = DEFAULT_PROBLEM) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    threads = configure_cpu_threads()
    specs = experiment_specs()
    results = []
    manifest = []
    optimal_replay_steps = None
    for spec in specs:
        payload, metrics = run_spec(spec, problem)
        file_name = "trajectory.json" if spec.slug == "astar" else f"{spec.slug}.json"
        (output_dir / file_name).write_text(json.dumps(payload, indent=2) + "\n")
        manifest.append({"label": spec.label, "file": file_name})
        if spec.approach == "astar":
            optimal_replay_steps = metrics["replay_steps"]
        results.append(metrics | {"file": file_name})
    for metrics in results:
        if optimal_replay_steps is None:
            metrics["optimal_replay_steps"] = None
            metrics["replay_steps_over_optimal"] = None
            metrics["optimality_ratio"] = None
        else:
            metrics["optimal_replay_steps"] = optimal_replay_steps
            metrics["replay_steps_over_optimal"] = metrics["replay_steps"] - optimal_replay_steps
            metrics["optimality_ratio"] = metrics["replay_steps"] / optimal_replay_steps if optimal_replay_steps else None
    summary = {
        "problem": {
            "grid_size": problem.grid_size,
            "start": list(problem.start),
            "apples": [list(apple) for apple in problem.apples],
            "wall_mode": "wrap",
        },
        "cpu_threads": threads,
        "experiments": results,
    }
    (output_dir / "trajectories.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output_dir / "experiment_results.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    summary = write_outputs()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
