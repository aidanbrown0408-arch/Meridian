"""Vectorized backtest engine with cost modeling.

Design notes worth keeping in mind when reading this:

* Exposure is long-or-flat, in [0, 1]. `Strategy.positions()` has already
  shifted the signal by one bar, so `position[t]` is genuinely knowable at the
  open of bar t. The engine does not shift again.
* Costs are charged on *position change*, not on trade count: moving from 0 to
  1 costs one-way, 1 to 0 costs one-way, so a full round trip pays twice. This
  is why the spec's table lists a one-way figure and a round-trip total.
* Cost is charged in the same bar as the change, subtracted from that bar's
  return. At daily granularity that is the right approximation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from utils.config import Config
from utils.logging_setup import get_logger
from utils.metrics import PerformanceSummary, drawdown_series, equity_curve, summarize
from strategies.base import Strategy

log = get_logger("engine", agent="Leo")

BPS = 1e-4


@dataclass
class CostModel:
    """One-way costs in basis points, split into the two components so a report
    can attribute them separately."""
    commission_bps: float
    slippage_bps: float

    @property
    def one_way_bps(self) -> float:
        return self.commission_bps + self.slippage_bps

    @property
    def round_trip_bps(self) -> float:
        return 2.0 * self.one_way_bps

    @classmethod
    def from_config(cls, config: Config, asset_class: str) -> "CostModel":
        section = config.section(f"costs.{asset_class}")
        return cls(float(section["commission_bps"]), float(section["slippage_bps"]))

    def charge(self, position: pd.Series) -> pd.Series:
        """Cost as a fraction of capital, per bar."""
        turnover = position.diff().abs().fillna(position.abs())
        return turnover * self.one_way_bps * BPS


@dataclass
class BacktestResult:
    """Everything one strategy/ticker run produces."""
    strategy: str
    symbol: str
    asset_class: str
    data_source: str
    summary: PerformanceSummary
    returns: pd.Series = field(repr=False, default_factory=pd.Series)
    gross_returns: pd.Series = field(repr=False, default_factory=pd.Series)
    position: pd.Series = field(repr=False, default_factory=pd.Series)
    costs: pd.Series = field(repr=False, default_factory=pd.Series)
    benchmark_returns: pd.Series = field(repr=False, default_factory=pd.Series)
    blocked: bool = False
    block_reason: str = ""
    params: dict = field(default_factory=dict)

    @property
    def is_trusted(self) -> bool:
        return self.data_source != "synthetic" and not self.blocked

    @property
    def equity(self) -> pd.Series:
        return equity_curve(self.returns)

    @property
    def drawdown(self) -> pd.Series:
        return drawdown_series(self.returns)

    def to_row(self) -> dict:
        """Flat dict for the results table George eventually renders."""
        row = {
            "strategy": self.strategy,
            "symbol": self.symbol,
            "asset_class": self.asset_class,
            "data_source": self.data_source,
            "trusted": self.is_trusted,
            "blocked": self.blocked,
        }
        row.update(self.summary.to_dict())
        return row


class BacktestEngine:
    """The mechanical core. Leo drives it; it holds no agent logic itself."""

    def __init__(self, config: Config):
        self.config = config
        self.risk_free_rate = float(config.get("risk.risk_free_rate", 0.0))

    def run(self, strategy: Strategy, bars: pd.DataFrame, symbol: str,
            asset_class: str = "stocks", data_source: str = "unknown") -> BacktestResult:
        cost_model = CostModel.from_config(self.config, asset_class)

        if len(bars) <= strategy.warmup:
            reason = (f"insufficient history: {len(bars)} bars, "
                      f"{strategy.callsign} needs more than {strategy.warmup}")
            log.warning("%s/%s skipped — %s", strategy.callsign, symbol, reason)
            return BacktestResult(
                strategy=strategy.callsign, symbol=symbol, asset_class=asset_class,
                data_source=data_source, summary=PerformanceSummary(),
                blocked=True, block_reason=reason, params=dict(strategy.params),
            )

        close = bars["close"].astype("float64")
        asset_returns = close.pct_change().fillna(0.0)

        position = strategy.positions(bars).reindex(bars.index).fillna(0.0)
        # Nothing is tradable during warmup, whatever the indicator says.
        position.iloc[: strategy.warmup] = 0.0

        gross = position * asset_returns
        costs = cost_model.charge(position)
        net = gross - costs

        summary = summarize(
            net, position=position, asset_class=asset_class,
            risk_free_rate=self.risk_free_rate, total_cost=float(costs.sum()),
        )

        log.debug("%s/%s: Sharpe %.2f, MaxDD %.1f%%, %d trades, %.0f bps costs",
                  strategy.callsign, symbol, summary.sharpe,
                  summary.max_drawdown * 100, summary.trades, costs.sum() / BPS)

        return BacktestResult(
            strategy=strategy.callsign, symbol=symbol, asset_class=asset_class,
            data_source=data_source, summary=summary, returns=net,
            gross_returns=gross, position=position, costs=costs,
            benchmark_returns=asset_returns, params=dict(strategy.params),
        )

    def buy_and_hold(self, bars: pd.DataFrame, symbol: str,
                     asset_class: str = "stocks") -> PerformanceSummary:
        """Benchmark leg. Charged one one-way cost for the initial purchase, so
        it is compared on the same footing as a strategy that enters once."""
        close = bars["close"].astype("float64")
        returns = close.pct_change().fillna(0.0)
        cost_model = CostModel.from_config(self.config, asset_class)
        if len(returns) > 0:
            returns.iloc[0] -= cost_model.one_way_bps * BPS
        position = pd.Series(1.0, index=bars.index)
        return summarize(returns, position=position, asset_class=asset_class,
                         risk_free_rate=self.risk_free_rate)


def results_frame(results: list[BacktestResult]) -> pd.DataFrame:
    """Sortable table of every strategy/ticker run."""
    if not results:
        return pd.DataFrame()
    frame = pd.DataFrame([r.to_row() for r in results])
    return frame.sort_values(["trusted", "sharpe"], ascending=[False, False]).reset_index(drop=True)
