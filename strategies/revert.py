"""REVERT — RSI mean-reversion (period 14, buy <30, exit >55)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.base import Strategy


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI. Wilder smoothing is an EMA with alpha = 1/period."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # An unbroken run of gains gives zero average loss: that is RSI 100, not NaN.
    # Warmup bars (avg_gain still NaN) must stay NaN.
    out = out.where(avg_loss.ne(0.0) | avg_gain.isna(), 100.0)
    return out.astype("float64")


class RevertStrategy(Strategy):
    callsign = "REVERT"
    style = "RSI mean-reversion"
    family = "mean_reversion"
    best_regimes = ("choppy",)

    @classmethod
    def default_params(cls) -> dict:
        return {"period": 14, "entry": 30, "exit": 55}

    def _validate_params(self) -> None:
        if not 0 < self.params["entry"] < self.params["exit"] < 100:
            raise ValueError(f"{self.callsign}: need 0 < entry < exit < 100")
        if self.params["period"] < 2:
            raise ValueError(f"{self.callsign}: period must be at least 2")

    @property
    def warmup(self) -> int:
        return int(self.params["period"]) * 3

    def generate_signals(self, bars: pd.DataFrame) -> pd.Series:
        values = rsi(bars["close"], int(self.params["period"]))
        # Buy oversold, hold through the recovery, release above the exit level.
        return self._hold(values < float(self.params["entry"]),
                          values > float(self.params["exit"]))
