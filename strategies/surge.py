"""SURGE — slow Donchian breakout (70/30, upgraded from the 55/20 default)."""

from __future__ import annotations

import pandas as pd

from strategies.base import Strategy


class SurgeStrategy(Strategy):
    callsign = "SURGE"
    style = "Slow breakout"
    family = "breakout"
    best_regimes = ("trending",)

    @classmethod
    def default_params(cls) -> dict:
        return {"entry_window": 70, "exit_window": 30}

    def _validate_params(self) -> None:
        if self.params["entry_window"] < 2 or self.params["exit_window"] < 2:
            raise ValueError(f"{self.callsign}: windows must be at least 2")

    @property
    def warmup(self) -> int:
        return int(max(self.params["entry_window"], self.params["exit_window"])) + 1

    def generate_signals(self, bars: pd.DataFrame) -> pd.Series:
        entry_n = int(self.params["entry_window"])
        exit_n = int(self.params["exit_window"])
        close = bars["close"]
        high = bars["high"].fillna(close)
        low = bars["low"].fillna(close)

        # Prior-window extremes: exclude today so a new high is a genuine break.
        upper = high.rolling(entry_n, min_periods=entry_n).max().shift(1)
        lower = low.rolling(exit_n, min_periods=exit_n).min().shift(1)
        return self._hold(close > upper, close < lower)
