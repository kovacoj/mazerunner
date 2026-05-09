from __future__ import annotations

import json
from argparse import ArgumentParser
from argparse import Namespace
from pathlib import Path

from .trajectory import Point
from .trajectory import build_trajectory


DEFAULT_OUTPUT = Path("data/trajectory.json")


def sample_trajectory():
    apples = [
        Point(5, 3),
        Point(9, 6),
        Point(13, 6),
        Point(13, 11),
        Point(16, 14),
    ]
    initial_snake = [
        Point(2, 3),
        Point(1, 3),
        Point(0, 3),
        Point(0, 2),
    ]

    return build_trajectory(
        title="Sample orchard run",
        grid_size=20,
        frame_duration_ms=160,
        apples=apples,
        visit_order=[0, 1, 2, 3, 4],
        initial_snake=initial_snake,
    )


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Export a snake trajectory for the frontend player.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to write the frontend trajectory JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trajectory = sample_trajectory()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(trajectory.as_dict(), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
