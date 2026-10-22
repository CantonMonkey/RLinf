"""Small deterministic grid used to verify navigation semantics before integration.

The image is a top-down diagnostic map, not a realistic camera observation. A
blocked forward move never changes pose or geodesic distance. All reward progress
comes from BFS on traversable cells, so a wall cannot be crossed for reward.
"""

from __future__ import annotations

import math
import random
from collections import deque
from collections.abc import Sequence

from navrl.contracts import NavAction, NavInfo, Observation
from navrl.envs.base import NavEnvAdapter
from navrl.episode import Cell, GridEpisode
from navrl.metrics import spl
from navrl.rewards import RewardConfig, navigation_reward

DEFAULT_GRID = (
    "#########",
    "#...#...#",
    "#...#...#",
    "#.......#",
    "#...#...#",
    "#########",
)

# East, north, west, south. World pose uses y-up; grid row index uses y-down.
DIRECTIONS: tuple[Cell, ...] = ((1, 0), (0, -1), (-1, 0), (0, 1))
RgbFrame = tuple[tuple[tuple[int, int, int], ...], ...]


def shortest_path(grid: Sequence[str], start: Cell, goal: Cell) -> tuple[Cell, ...] | None:
    """Return a deterministic BFS path, including both endpoints, or None."""
    if start == goal:
        return (start,)
    width, height = len(grid[0]), len(grid)
    queue = deque([start])
    parent: dict[Cell, Cell | None] = {start: None}
    while queue:
        x, y = queue.popleft()
        for dx, dy in DIRECTIONS:
            neighbor = (x + dx, y + dy)
            nx, ny = neighbor
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if grid[ny][nx] == "#" or neighbor in parent:
                continue
            parent[neighbor] = (x, y)
            if neighbor == goal:
                path: list[Cell] = [goal]
                current = goal
                previous = parent[current]
                while previous is not None:
                    current = previous
                    path.append(current)
                    previous = parent[current]
                path.reverse()
                return tuple(path)
            queue.append(neighbor)
    return None


class GridNavEnv(NavEnvAdapter):
    """Single episode at a time, with discrete actions and no external dependencies."""

    def __init__(
        self,
        *,
        grid: Sequence[str] = DEFAULT_GRID,
        seed: int = 0,
        max_steps: int = 64,
        success_radius: float = 0.2,
        max_consecutive_collisions: int | None = 3,
        history_size: int = 4,
        expose_pose: bool = False,
        reward_config: RewardConfig | None = None,
    ) -> None:
        self.grid = tuple(grid)
        if not self.grid or not self.grid[0] or any(len(row) != len(self.grid[0]) for row in self.grid):
            raise ValueError("grid must be nonempty and rectangular")
        if set("".join(self.grid)) - {"#", "."}:
            raise ValueError("grid may contain only '#' and '.'")
        self._open_cells = tuple(
            (x, y)
            for y, row in enumerate(self.grid)
            for x, char in enumerate(row)
            if char == "."
        )
        if len(self._open_cells) < 2:
            raise ValueError("grid needs at least two traversable cells")
        first = self._open_cells[0]
        if any(shortest_path(self.grid, first, cell) is None for cell in self._open_cells[1:]):
            raise ValueError("grid traversable cells must form one connected component")
        if max_steps < 1 or history_size < 0:
            raise ValueError("max_steps must be positive and history_size nonnegative")
        if not math.isfinite(success_radius) or success_radius < 0:
            raise ValueError("success_radius must be finite and nonnegative")
        if max_consecutive_collisions is not None and max_consecutive_collisions < 1:
            raise ValueError("max_consecutive_collisions must be positive or None")
        self.max_steps = max_steps
        self.success_radius = success_radius
        self.max_consecutive_collisions = max_consecutive_collisions
        self.history_size = history_size
        self.expose_pose = expose_pose
        self.reward_config = reward_config or RewardConfig()
        self._seed = seed
        self._rng = random.Random(seed)
        self._sample_index = 0
        self._episode: GridEpisode | None = None
        self._closed = False
        self._done = False
        self._history: list[RgbFrame] = []
        self._frame: RgbFrame = ()

    @property
    def current_episode(self) -> GridEpisode:
        if self._episode is None:
            raise RuntimeError("call reset before reading the episode")
        return self._episode

    def reset(self, episode: object | None = None, *, seed: int | None = None) -> Observation:
        if self._closed:
            raise RuntimeError("environment is closed")
        if seed is not None:
            self._seed = seed
            self._rng.seed(seed)
            self._sample_index = 0
        if episode is None:
            start, goal = self._rng.sample(self._open_cells, 2)
            episode = GridEpisode(
                episode_id=f"mock-{self._seed}-{self._sample_index:04d}",
                start=start,
                goal=goal,
                heading=self._rng.randrange(4),
            )
            self._sample_index += 1
        if not isinstance(episode, GridEpisode):
            raise TypeError("episode must be a GridEpisode or None")
        if episode.start not in self._open_cells or episode.goal not in self._open_cells:
            raise ValueError("episode start and goal must be traversable cells")
        path = shortest_path(self.grid, episode.start, episode.goal)
        if path is None:
            raise ValueError("episode goal is unreachable")
        self._episode = episode
        self._position = episode.start
        self._heading = episode.heading
        self._start_geodesic = float(len(path) - 1)
        self._geodesic = self._start_geodesic
        self._path_length = 0.0
        self._step_id = 0
        self._collision_count = 0
        self._consecutive_collisions = 0
        self._last_collision = False
        self._last_invalid = False
        self._stop_called = False
        self._success = False
        self._timeout = False
        self._termination_reason: str | None = None
        self._done = False
        self._history = []
        self._frame = self._render_rgb()
        return self._observation()

    def step(self, action: object) -> tuple[Observation, float, bool, bool, NavInfo]:
        if self._closed:
            raise RuntimeError("environment is closed")
        self.current_episode
        if self._done:
            raise RuntimeError("episode ended; call reset before step")
        previous_geodesic = self._geodesic
        collision = False
        invalid_action = False
        try:
            parsed_action = NavAction(action)
        except (ValueError, TypeError):
            parsed_action = None
            invalid_action = True
        if parsed_action == NavAction.MOVE_FORWARD:
            dx, dy = DIRECTIONS[self._heading]
            candidate = (self._position[0] + dx, self._position[1] + dy)
            if candidate in self._open_cells:
                self._position = candidate
                self._path_length += 1.0
            else:
                collision = True
        elif parsed_action == NavAction.TURN_LEFT:
            self._heading = (self._heading + 1) % 4
        elif parsed_action == NavAction.TURN_RIGHT:
            self._heading = (self._heading - 1) % 4
        elif parsed_action == NavAction.STOP:
            self._stop_called = True

        self._step_id += 1
        path = shortest_path(self.grid, self._position, self.current_episode.goal)
        assert path is not None  # movement never leaves the connected open-cell set
        self._geodesic = float(len(path) - 1)
        self._collision_count += int(collision)
        self._consecutive_collisions = self._consecutive_collisions + 1 if collision else 0
        self._last_collision = collision
        self._last_invalid = invalid_action

        terminated = False
        truncated = False
        if self._stop_called:
            # Match Habitat PointNav's strict distance < success_distance rule.
            self._success = self._geodesic < self.success_radius
            terminated = True
            self._termination_reason = "success" if self._success else "premature_stop"
        elif (
            self.max_consecutive_collisions is not None
            and self._consecutive_collisions >= self.max_consecutive_collisions
        ):
            terminated = True
            self._termination_reason = "collision_limit"
        elif self._step_id >= self.max_steps:
            truncated = True
            self._timeout = True
            self._termination_reason = "timeout"
        self._done = terminated or truncated

        reward = navigation_reward(
            previous_geodesic,
            self._geodesic,
            success=self._success,
            collision=collision,
            invalid_action=invalid_action,
            config=self.reward_config,
        )
        if self.history_size:
            self._history.append(self._frame)
            del self._history[:-self.history_size]
        self._frame = self._render_rgb()
        return self._observation(), reward, terminated, truncated, self.get_metrics()

    def get_metrics(self) -> NavInfo:
        episode = self.current_episode
        dx = self._position[0] - episode.goal[0]
        dy = self._position[1] - episode.goal[1]
        return {
            "success": self._success,
            "distance_to_goal": math.hypot(dx, dy),
            "geodesic_distance": self._geodesic,
            "path_length": self._path_length,
            "spl": spl(self._success, self._start_geodesic, self._path_length),
            "collision": self._last_collision,
            "collision_count": self._collision_count,
            "stop_called": self._stop_called,
            "timeout": self._timeout,
            "invalid_action": self._last_invalid,
            "termination_reason": self._termination_reason,
        }

    def close(self) -> None:
        self._closed = True
        self._history.clear()

    def _observation(self) -> Observation:
        episode = self.current_episode
        x, y = self._position
        pose = (float(x), float(-y), self._heading * math.pi / 2) if self.expose_pose else None
        return {
            "rgb": self._frame,
            "depth": None,
            "instruction": episode.instruction,
            "history": list(self._history),
            "pose": pose,
            "episode_id": episode.episode_id,
            "step_id": self._step_id,
        }

    def _render_rgb(self) -> RgbFrame:
        episode = self.current_episode
        rows: list[tuple[tuple[int, int, int], ...]] = []
        for y, grid_row in enumerate(self.grid):
            for sub_y in range(3):
                pixels: list[tuple[int, int, int]] = []
                for x, char in enumerate(grid_row):
                    cell = (x, y)
                    base = (25, 30, 38) if char == "#" else (225, 230, 235)
                    if cell == episode.goal:
                        base = (56, 174, 93)
                    if cell == self._position:
                        base = (47, 101, 214)
                    for sub_x in range(3):
                        color = base
                        if cell == self._position:
                            dx, dy = DIRECTIONS[self._heading]
                            front = (1 + dx, 1 + dy)
                            if (sub_x, sub_y) == front:
                                color = (252, 213, 58)
                        pixels.append(color)
                rows.append(tuple(pixels))
        return tuple(rows)
