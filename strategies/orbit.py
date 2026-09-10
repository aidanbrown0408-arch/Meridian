"""ORBIT — slow trend-follower (MA 20/100)."""

from __future__ import annotations

import pandas as pd

from strategies.base import Strategy


class OrbitStrategy(Strategy):
    callsign = "ORBIT"
    style = "Slow trend-follower"
    family = "trending"
    best_regimes = ("trending",)

    @classmethod
    def default_params(cls) -> dict:
        return {"fast": 20, "slow": 100}

    def _validate_params(self) -> None:
        if self.params["fast"] >= self.params["slow"]:
            raise ValueError(f"{self.callsign}: fast window must be shorter than slow")
        if self.params["fast"] < 2:
            raise ValueError(f"{self.callsign}: fast window must be at least 2")

    @property
    def warmup(self) -> int:
        return int(self.params["slow"]) + 1

    def generate_signals(self, bars: pd.DataFrame) -> pd.Series:
        close = bars["close"]
        fast = close.rolling(int(self.params["fast"]), min_periods=int(self.params["fast"])).mean()
        slow = close.rolling(int(self.params["slow"]), min_periods=int(self.params["slow"])).mean()
        # Long while the fast average is above the slow one, flat otherwise.
        return (fast > slow).astype("float64").where(slow.notna(), 0.0)
