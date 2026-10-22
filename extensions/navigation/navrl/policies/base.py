"""Policy interface shared by discrete, continuous and waypoint policies."""

from __future__ import annotations

from abc import ABC, abstractmethod

from navrl.contracts import Observation, PolicyOutput


class PolicyAdapter(ABC):
    @abstractmethod
    def reset(self) -> None:
        """Clear recurrent/history state at an episode boundary."""

    @abstractmethod
    def act(self, observation: Observation) -> PolicyOutput:
        """Return an action plus optional training values and policy state."""
