"""Strategy base class.

Every trader is one subclass with fixed parameters. The contract is deliberately
tiny:

    generate_signals(bars) -> Series of desired exposure in [0, 1]

`Strategy.positions()` then shifts that by one bar, which is where the spec's
"signals applied on next bar" rule is enforced once, centrally, so no individual
strategy can accidentally look ahead.

Long-or-flat only: exposure is clipped to [0, 1]. No shorts, no leverage.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

_REGISTRY: dict[str, type["Strategy"]] = {}


class Strategy(ABC):
    #: Trader callsign, e.g. "ORBIT"
    callsign: str = "UNNAMED"
    #: Human-readable style, used in reports
    style: str = ""
    #: "trending" | "mean_reversion" | "breakout" — Greg maps this to regimes
    family: str = ""
    #: Regimes this trader is naturally suited to
    best_regimes: tuple[str, ...] = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.callsign != "UNNAMED":
            _REGISTRY[cls.callsign] = cls

    def __init__(self, **params):
        self.params = {**self.default_params(), **params}
        self._validate_params()

    # -------------------------------------------------------------- overridable

    @classmethod
    def default_params(cls) -> dict:
        return {}

    def _validate_params(self) -> None:
        """Subclasses raise ValueError on nonsense parameters."""

    @property
    def warmup(self) -> int:
        """Bars needed before the strategy can emit a meaningful signal.
        The backtester refuses to evaluate a series shorter than this."""
        return 1

    @abstractmethod
    def generate_signals(self, bars: pd.DataFrame) -> pd.Series:
        """Desired exposure per bar, in [0, 1], indexed like `bars`.

        This is the *decision at the close of that bar*. Shifting to the next
        bar happens in `positions()` — do not shift here.
        """

    # -------------------------------------------------------------------- fixed

    def positions(self, bars: pd.DataFrame) -> pd.Series:
        """Actual held exposure per bar: yesterday's decision, applied today."""
        signals = self.generate_signals(bars)
        signals = pd.Series(signals, index=bars.index).astype("float64")
        signals = signals.fillna(0.0).clip(0.0, 1.0)
        return signals.shift(1).fillna(0.0)

    @staticmethod
    def _hold(entry: pd.Series, exit_: pd.Series) -> pd.Series:
        """State machine for entry/exit rules that are not a simple comparison.

        Enter on `entry`, stay in until `exit_`. Used by the breakout and
        mean-reversion traders, where 'not an exit signal' is not the same as
        'an entry signal'.
        """
        enter = entry.fillna(False).to_numpy(dtype=bool)
        leave = exit_.fillna(False).to_numpy(dtype=bool)
        state = np.zeros(len(enter), dtype="float64")
        holding = False
        for i in range(len(enter)):
            if holding:
                if leave[i]:
                    holding = False
            elif enter[i]:
                holding = True
            state[i] = 1.0 if holding else 0.0
        return pd.Series(state, index=entry.index)

    def describe(self) -> dict:
        return {
            "callsign": self.callsign,
            "style": self.style,
            "family": self.family,
            "params": dict(self.params),
            "best_regimes": list(self.best_regimes),
            "warmup": self.warmup,
        }

    def __repr__(self) -> str:
        joined = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.callsign}({joined})"


def registry() -> dict[str, type[Strategy]]:
    """All registered strategy classes, keyed by callsign."""
    import strategies  # noqa: F401  (triggers imports that populate the registry)
    return dict(_REGISTRY)


def build_strategies(config) -> dict[str, Strategy]:
    """Instantiate every enabled strategy from config."""
    available = registry()
    built: dict[str, Strategy] = {}
    for callsign, spec in config.section("strategies").items():
        if not spec.get("enabled", True):
            continue
        cls = available.get(callsign)
        if cls is None:
            raise KeyError(f"No strategy class registered for callsign {callsign!r}")
        built[callsign] = cls(**(spec.get("params") or {}))
    return built
