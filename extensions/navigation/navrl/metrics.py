"""Navigation outcome metrics kept separate from shaping reward."""

from __future__ import annotations


def spl(success: bool, shortest_path: float, path_length: float) -> float:
    if not success:
        return 0.0
    if shortest_path == 0:
        return 1.0 if path_length == 0 else 0.0
    return shortest_path / max(shortest_path, path_length)
