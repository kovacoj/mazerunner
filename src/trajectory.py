from __future__ import annotations

from collections import deque
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


def _validate_apples(apples: tuple[Point, ...], grid_size: int) -> None:
    if len(set(apples)) != len(apples):
        raise ValueError("Apple positions must be unique.")
    for apple in apples:
        if not _is_in_bounds(apple, grid_size):
            raise ValueError(f"Apple {apple} is out of bounds.")


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


def _validate_visit_order(visit_order: tuple[int, ...], apple_count: int) -> None:
    if len(set(visit_order)) != len(visit_order):
        raise ValueError("Visit order cannot contain duplicate apple indices.")
    for index in visit_order:
        if index < 0 or index >= apple_count:
            raise ValueError(f"Apple index {index} is out of range.")


def _validate_head_path(head_path: tuple[Point, ...], initial_head: Point, grid_size: int, wall_mode: str) -> None:
    if not head_path:
        raise ValueError("Head path must contain at least one position.")
    if head_path[0] != initial_head:
        raise ValueError("Head path must start at the initial snake head.")
    for point in head_path:
        if not _is_in_bounds(point, grid_size):
            raise ValueError(f"Head path point {point} is out of bounds.")
    for previous, current in zip(head_path, head_path[1:]):
        if not _is_adjacent(previous, current, grid_size, wall_mode):
            raise ValueError("Head path must move one orthogonal cell at a time.")


def _step_axis(current: int, target: int, grid_size: int, wall_mode: str) -> int:
    if wall_mode == "wrap":
        forward = (target - current) % grid_size
        backward = (current - target) % grid_size
        if forward <= backward:
            return (current + 1) % grid_size
        return (current - 1) % grid_size
    return current + (1 if target > current else -1)


def _neighbors(point: Point, grid_size: int, wall_mode: str) -> tuple[Point, ...]:
    candidates = (
        Point(point.x + 1, point.y),
        Point(point.x - 1, point.y),
        Point(point.x, point.y + 1),
        Point(point.x, point.y - 1),
    )
    if wall_mode == "wrap":
        return tuple(Point(candidate.x % grid_size, candidate.y % grid_size) for candidate in candidates)
    return tuple(candidate for candidate in candidates if _is_in_bounds(candidate, grid_size))


def _shortest_path(
    start: Point,
    goal: Point,
    grid_size: int,
    wall_mode: str,
    blocked: set[Point],
) -> list[Point] | None:
    if start == goal:
        return []

    queue: deque[Point] = deque([start])
    parents: dict[Point, Point | None] = {start: None}

    while queue:
        point = queue.popleft()
        for candidate in _neighbors(point, grid_size, wall_mode):
            if candidate in parents:
                continue
            if candidate in blocked and candidate != goal:
                continue
            parents[candidate] = point
            if candidate == goal:
                path = [candidate]
                while parents[path[-1]] != start:
                    path.append(parents[path[-1]])
                path.reverse()
                return path
            queue.append(candidate)

    return None


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


def _build_frames(
    apples: tuple[Point, ...],
    head_path: tuple[Point, ...],
    initial_snake: tuple[Point, ...],
) -> tuple[Frame, ...]:
    apple_lookup = {(apple.x, apple.y): index for index, apple in enumerate(apples)}
    eaten: set[int] = set()
    current_snake = list(initial_snake)
    frames = [Frame(tuple(current_snake), tuple())]
    for next_head in head_path[1:]:
        eaten_index = apple_lookup.get((next_head.x, next_head.y))
        grows = eaten_index is not None and eaten_index not in eaten
        current_snake.insert(0, next_head)
        if grows:
            eaten.add(eaten_index)
        else:
            current_snake.pop()
        frames.append(Frame(tuple(current_snake), tuple(sorted(eaten))))
    return tuple(frames)


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
    visit_order_tuple = tuple(visit_order)
    snake_tuple = tuple(initial_snake)

    _validate_apples(apples_tuple, grid_size)
    _validate_snake(snake_tuple, grid_size, wall_mode)
    _validate_visit_order(visit_order_tuple, len(apples_tuple))

    if set(apples_tuple) & set(snake_tuple):
        raise ValueError("Initial snake cannot overlap an apple.")

    apple_lookup = {(apple.x, apple.y): index for index, apple in enumerate(apples_tuple)}
    eaten: set[int] = set()
    current_snake = list(snake_tuple)
    frames = [Frame(tuple(current_snake), tuple())]

    for apple_index in visit_order_tuple:
        if apple_index in eaten:
            continue
        target = apples_tuple[apple_index]

        while current_snake[0] != target:
            relaxed_blocked = set(current_snake[:-1])
            segment = _shortest_path(current_snake[0], target, grid_size, wall_mode, relaxed_blocked)
            if segment is None:
                segment = _shortest_path(current_snake[0], target, grid_size, wall_mode, set(current_snake))
            if not segment:
                raise ValueError(f"No replay path from {current_snake[0]} to {target}.")

            next_head = segment[0]
            eaten_index = apple_lookup.get((next_head.x, next_head.y))
            grows = eaten_index is not None and eaten_index not in eaten
            occupied = current_snake if grows else current_snake[:-1]
            if next_head in occupied:
                strict_segment = _shortest_path(current_snake[0], target, grid_size, wall_mode, set(current_snake))
                if not strict_segment:
                    raise ValueError(f"No snake-safe replay path from {current_snake[0]} to {target}.")
                next_head = strict_segment[0]
                eaten_index = apple_lookup.get((next_head.x, next_head.y))
                grows = eaten_index is not None and eaten_index not in eaten

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


def build_trajectory_from_path(
    *,
    title: str,
    grid_size: int,
    frame_duration_ms: int,
    apples: list[Point],
    head_path: list[Point],
    initial_snake: list[Point],
    wall_mode: str = "bounded",
) -> Trajectory:
    if grid_size < 1:
        raise ValueError("Grid size must be at least 1.")
    if frame_duration_ms < 1:
        raise ValueError("Frame duration must be at least 1 ms.")

    _validate_wall_mode(wall_mode)

    apples_tuple = tuple(apples)
    snake_tuple = tuple(initial_snake)
    head_path_tuple = tuple(head_path)

    _validate_apples(apples_tuple, grid_size)
    _validate_snake(snake_tuple, grid_size, wall_mode)
    _validate_head_path(head_path_tuple, snake_tuple[0], grid_size, wall_mode)

    if set(apples_tuple) & set(snake_tuple):
        raise ValueError("Initial snake cannot overlap an apple.")

    return Trajectory(
        title=title,
        grid_size=grid_size,
        frame_duration_ms=frame_duration_ms,
        wall_mode=wall_mode,
        apples=apples_tuple,
        frames=_build_frames(apples_tuple, head_path_tuple, snake_tuple),
    )
