"""Minimal simulator interface consumed by a future RL trainer."""

from __future__ import annotations

from abc import ABC, abstractmethod

from navrl.contracts import NavInfo, Observation


class NavEnvAdapter(ABC):
    @abstractmethod
    def reset(self, episode: object | None = None, *, seed: int | None = None) -> Observation:
        """Start an episode. A seed restarts deterministic sampling."""

    @abstractmethod
    def step(self, action: object) -> tuple[Observation, float, bool, bool, NavInfo]:
        """Return observation, reward, terminated, truncated, info."""

    @abstractmethod
    def get_metrics(self) -> NavInfo:
        """Return an independent snapshot of current episode metrics."""

    @abstractmethod
    def close(self) -> None:
        """Release environment resources; safe to call more than once."""
