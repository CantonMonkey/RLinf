"""Serializable episode specification for deterministic mock replay."""

from __future__ import annotations

from dataclasses import dataclass

Cell = tuple[int, int]


@dataclass(frozen=True)
class GridEpisode:
    episode_id: str
    start: Cell
    goal: Cell
    heading: int = 0  # east, north, west, south; grid y increases downward
    instruction: str = "Navigate to the highlighted goal."

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id must not be empty")
        if self.heading not in range(4):
            raise ValueError("heading must be 0..3")
