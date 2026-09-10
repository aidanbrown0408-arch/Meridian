"""FLUX — fast trend-follower (MA 10/50)."""

from __future__ import annotations

from strategies.orbit import OrbitStrategy


class FluxStrategy(OrbitStrategy):
    """Identical crossover mechanic to ORBIT on shorter windows. Reacts faster,
    whipsaws more."""

    callsign = "FLUX"
    style = "Fast trend-follower"
    family = "trending"
    best_regimes = ("trending",)

    @classmethod
    def default_params(cls) -> dict:
        return {"fast": 10, "slow": 50}
