from __future__ import annotations

from dataclasses import dataclass

Position = tuple[int, int]

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

DQN_OPTIMIZER_CHOICES = (
    "newton",
    "extended_kalman_filter",
    "levenberg_marquardt",
    "annealing",
    "metropolis",
    "genetic",
)

APPROACH_CHOICES = (
    "astar",
    "dqn",
    "genetic",
    "simulated_annealing",
    "kohonen",
    "metropolis",
)
