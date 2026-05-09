from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Point:
    x: int
    y: int

    def as_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y}


@dataclass(frozen=True, slots=True)
class Frame:
    snake: tuple[Point, ...]
    eaten: tuple[int, ...]

    def as_dict(self) -> dict[str, list[dict[str, int]] | list[int]]:
        return {
            "snake": [segment.as_dict() for segment in self.snake],
            "eaten": list(self.eaten),
        }


@dataclass(frozen=True, slots=True)
class Trajectory:
    title: str
    grid_size: int
    frame_duration_ms: int
    wall_mode: str
    apples: tuple[Point, ...]
    frames: tuple[Frame, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "gridSize": self.grid_size,
            "frameDurationMs": self.frame_duration_ms,
            "wallMode": self.wall_mode,
            "apples": [apple.as_dict() for apple in self.apples],
            "frames": [frame.as_dict() for frame in self.frames],
        }


def _is_in_bounds(point: Point, grid_size: int) -> bool:
    return 0 <= point.x < grid_size and 0 <= point.y < grid_size


def _validate_wall_mode(wall_mode: str) -> None:
    if wall_mode not in {"bounded", "wrap"}:
        raise ValueError("Wall mode must be 'bounded' or 'wrap'.")


def _axis_distance(a: int, b: int, grid_size: int, wall_mode: str) -> int:
    diff = abs(a - b)
    if wall_mode == "wrap":
        return min(diff, grid_size - diff)
    return diff


def _is_adjacent(a: Point, b: Point, grid_size: int, wall_mode: str) -> bool:
    return _axis_distance(a.x, b.x, grid_size, wall_mode) + _axis_distance(a.y, b.y, grid_size, wall_mode) == 1


def _validate_snake(snake: tuple[Point, ...], grid_size: int, wall_mode: str) -> None:
    if not snake:
        raise ValueError("Initial snake must contain at least one segment.")

    if len(set(snake)) != len(snake):
        raise ValueError("Initial snake cannot overlap itself.")

    for segment in snake:
        if not _is_in_bounds(segment, grid_size):
            raise ValueError(f"Snake segment {segment} is out of bounds.")

    for head, tail in zip(snake, snake[1:]):
        if not _is_adjacent(head, tail, grid_size, wall_mode):
            raise ValueError("Initial snake must be a contiguous orthogonal chain.")


def _validate_apples(apples: tuple[Point, ...], grid_size: int) -> None:
    if len(set(apples)) != len(apples):
        raise ValueError("Apple positions must be unique.")

    for apple in apples:
        if not _is_in_bounds(apple, grid_size):
            raise ValueError(f"Apple {apple} is out of bounds.")


def _validate_visit_order(visit_order: tuple[int, ...], apple_count: int) -> None:
    if len(set(visit_order)) != len(visit_order):
        raise ValueError("Visit order cannot contain duplicate apple indices.")

    for index in visit_order:
        if index < 0 or index >= apple_count:
            raise ValueError(f"Apple index {index} is out of range.")


def _step_axis(current: int, target: int, grid_size: int, wall_mode: str) -> int:
    if wall_mode == "wrap":
        forward = (target - current) % grid_size
        backward = (current - target) % grid_size
        if forward <= backward:
            return (current + 1) % grid_size
        return (current - 1) % grid_size

    return current + (1 if target > current else -1)


def manhattan_path(start: Point, goal: Point, grid_size: int, wall_mode: str) -> list[Point]:
    x = start.x
    y = start.y
    path: list[Point] = []

    while x != goal.x:
        x = _step_axis(x, goal.x, grid_size, wall_mode)
        path.append(Point(x, y))

    while y != goal.y:
        y = _step_axis(y, goal.y, grid_size, wall_mode)
        path.append(Point(x, y))

    return path


def build_trajectory(
    *,
    title: str,
    grid_size: int,
    frame_duration_ms: int,
    apples: list[Point],
    visit_order: list[int],
    initial_snake: list[Point],
    wall_mode: str = "bounded",
) -> Trajectory:
    if grid_size < 1:
        raise ValueError("Grid size must be at least 1.")

    if frame_duration_ms < 1:
        raise ValueError("Frame duration must be at least 1 ms.")

    _validate_wall_mode(wall_mode)

    apples_tuple = tuple(apples)
    snake = tuple(initial_snake)
    visit_order_tuple = tuple(visit_order)

    _validate_apples(apples_tuple, grid_size)
    _validate_snake(snake, grid_size, wall_mode)
    _validate_visit_order(visit_order_tuple, len(apples_tuple))

    if set(apples_tuple) & set(snake):
        raise ValueError("Initial snake cannot overlap an apple.")

    apple_lookup = {(apple.x, apple.y): index for index, apple in enumerate(apples_tuple)}
    eaten: set[int] = set()
    current_snake = list(snake)
    frames = [Frame(tuple(current_snake), tuple())]

    for apple_index in visit_order_tuple:
        target = apples_tuple[apple_index]

        for next_head in manhattan_path(current_snake[0], target, grid_size, wall_mode):
            eaten_index = apple_lookup.get((next_head.x, next_head.y))
            grows = eaten_index is not None and eaten_index not in eaten
            occupied = current_snake if grows else current_snake[:-1]

            if next_head in occupied:
                raise ValueError(f"Trajectory self-collides at {next_head}.")

            current_snake.insert(0, next_head)

            if grows:
                eaten.add(eaten_index)
            else:
                current_snake.pop()

            frames.append(Frame(tuple(current_snake), tuple(sorted(eaten))))

    return Trajectory(
        title=title,
        grid_size=grid_size,
        frame_duration_ms=frame_duration_ms,
        wall_mode=wall_mode,
        apples=apples_tuple,
        frames=tuple(frames),
    )
