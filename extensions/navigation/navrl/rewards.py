"""Pure reward calculation using traversable-path progress."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    progress_weight: float = 1.0
    success_bonus: float = 10.0
    step_penalty: float = 0.01
    collision_penalty: float = 0.2
    invalid_action_penalty: float = 0.2

    def __post_init__(self) -> None:
        if any(not math.isfinite(v) or v < 0 for v in vars(self).values()):
            raise ValueError("reward weights must be finite and nonnegative")


def navigation_reward(
    previous_geodesic: float,
    current_geodesic: float,
    *,
    success: bool,
    collision: bool,
    invalid_action: bool,
    config: RewardConfig,
) -> float:
    if not all(math.isfinite(v) for v in (previous_geodesic, current_geodesic)):
        raise ValueError("reward requires finite geodesic distances")
    return (
        config.progress_weight * (previous_geodesic - current_geodesic)
        + (config.success_bonus if success else 0.0)
        - config.step_penalty
        - (config.collision_penalty if collision else 0.0)
        - (config.invalid_action_penalty if invalid_action else 0.0)
    )
