"""SPARK — fast Donchian breakout (20/10)."""

from __future__ import annotations

from strategies.surge import SurgeStrategy


class SparkStrategy(SurgeStrategy):
    """Same mechanic as SURGE on shorter windows: more trades, smaller moves,
    more false breakouts."""

    callsign = "SPARK"
    style = "Fast breakout"
    family = "breakout"
    best_regimes = ("trending",)

    @classmethod
    def default_params(cls) -> dict:
        return {"entry_window": 20, "exit_window": 10}
