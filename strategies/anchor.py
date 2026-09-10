"""ANCHOR — Bollinger mean-reversion (20-day, 2.0 std dev)."""

from __future__ import annotations

import pandas as pd

from strategies.base import Strategy


class AnchorStrategy(Strategy):
    callsign = "ANCHOR"
    style = "Bollinger mean-reversion"
    family = "mean_reversion"
    best_regimes = ("choppy",)

    @classmethod
    def default_params(cls) -> dict:
        return {"window": 20, "num_std": 2.0}

    def _validate_params(self) -> None:
        if self.params["window"] < 2:
            raise ValueError(f"{self.callsign}: window must be at least 2")
        if self.params["num_std"] <= 0:
            raise ValueError(f"{self.callsign}: num_std must be positive")

    @property
    def warmup(self) -> int:
        return int(self.params["window"]) + 1

    def generate_signals(self, bars: pd.DataFrame) -> pd.Series:
        window = int(self.params["window"])
        close = bars["close"]
        middle = close.rolling(window, min_periods=window).mean()
        std = close.rolling(window, min_periods=window).std(ddof=0)
        lower = middle - float(self.params["num_std"]) * std
        # Enter below the lower band, take profit back at the middle band.
        return self._hold(close < lower, close >= middle)
