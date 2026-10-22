"""Angular-sector collision memory for G7.5."""

from __future__ import annotations

try:
    from .streamvln_candidates_g74 import WorldHeadingShield
except ImportError:
    from streamvln_candidates_g74 import WorldHeadingShield


class AngularSectorShield(WorldHeadingShield):
    def __init__(
        self,
        turn_radians: float,
        *,
        sector_radius_bins: int,
        cooldown_decisions: int = 2,
    ) -> None:
        if sector_radius_bins < 1:
            raise ValueError("sector radius must be positive")
        self.sector_radius_bins = int(sector_radius_bins)
        super().__init__(
            turn_radians=turn_radians,
            cooldown_decisions=cooldown_decisions,
        )

    def block_world_heading(self, heading_radians: float) -> None:
        center = self._bin(heading_radians)
        remaining = self.cooldown_decisions + 1
        for offset in range(-self.sector_radius_bins, self.sector_radius_bins + 1):
            key = (center + offset) % self.bin_count
            self.blocked_bins[key] = max(self.blocked_bins.get(key, 0), remaining)

    def summary(self) -> dict:
        result = super().summary()
        result["sector_radius_bins"] = self.sector_radius_bins
        result["sector_radius_degrees"] = self.sector_radius_bins * 15.0
        return result
