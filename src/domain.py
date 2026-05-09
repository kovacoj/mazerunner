from __future__ import annotations

from dataclasses import dataclass

Position = tuple[int, int]

DEFAULT_APPLES: tuple[Position, ...] = (
    (1, 2),
    (2, 4),
    (4, 3),
    (5, 8),
    (6, 15),
    (8, 18),
    (10, 1),
    (11, 17),
    (13, 6),
    (15, 12),
    (16, 2),
    (17, 14),
    (18, 16),
)
DEFAULT_FRAME_DURATION_MS = 160
DEFAULT_WALL_MODE = "wrap"


@dataclass(frozen=True)
class SnakeProblem:
    grid_size: int = 20
    start: Position = (10, 10)
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
    "adam",
    "newton",
    "extended_kalman_filter",
    "levenberg_marquardt",
    "annealing",
    "metropolis",
    "genetic",
)

KOHONEN_OPTIMIZER_CHOICES = (
    "adam",
    "extended_kalman_filter",
    "levenberg_marquardt",
)

APPROACH_CHOICES = (
    "astar",
    "dqn",
    "genetic",
    "simulated_annealing",
    "kohonen",
    "metropolis",
)
