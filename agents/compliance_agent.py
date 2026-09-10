"""David — Compliance Agent.

Runs sanity checks on fetched data before any strategy touches it (spec §4).
A symbol that fails any check is blocked for the day: no strategy runs
against it, and the report says why. A problem with SPY never stops QQQ or
BTC/USDT from trading — blocking is per symbol, not global.

Four checks, each independent and each able to block on its own:
  * no multi-day gaps in the trading-day series (crypto uses calendar days
    instead of business days — it has no weekend to speak of)
  * no zero or negative prices
  * no obviously stale feed (last bar too many trading days old)
  * no implausible day-over-day move (>50%, suggesting an unadjusted
    corporate action rather than a real price move)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from utils.config import Config
from utils.logging_setup import get_logger
from agents.data_agent import MarketData

log = get_logger("compliance", agent="David")


@dataclass
class ComplianceCheck:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class ComplianceReport:
    """Every check run for one symbol."""
    symbol: str
    checks: list[ComplianceCheck] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return any(not c.passed for c in self.checks)

    @property
    def reason(self) -> str:
        return "; ".join(c.detail for c in self.checks if not c.passed)


class ComplianceAgent:
    """David."""

    name = "David"
    role = "Compliance"

    def __init__(self, config: Config):
        self.config = config
        self.max_gap_days = int(config.get("compliance.max_gap_trading_days", 3))
        self.stale_after_days = int(config.get("compliance.stale_after_trading_days", 3))
        self.max_daily_move = float(config.get("compliance.max_daily_move_pct", 0.50))

    def review(self, market: dict[str, MarketData]) -> dict[str, ComplianceReport]:
        """One report per symbol, whether or not it ends up blocked."""
        return {symbol: self.check(data) for symbol, data in market.items()}

    def blocked_symbols(self, market: dict[str, MarketData]) -> dict[str, str]:
        """Symbol -> reason, ready for `BacktestAgent.run_all(blocked=...)`."""
        reports = self.review(market)
        blocked = {s: r.reason for s, r in reports.items() if r.blocked}
        for symbol, reason in blocked.items():
            log.warning("%s blocked: %s", symbol, reason)
        if not blocked:
            log.info("No blocks.")
        return blocked

    def check(self, data: MarketData) -> ComplianceReport:
        bars = data.bars
        report = ComplianceReport(symbol=data.symbol)
        report.checks.append(self._check_prices(bars))
        report.checks.append(self._check_gaps(bars, data.asset_class))
        report.checks.append(self._check_staleness(bars, data.asset_class))
        report.checks.append(self._check_moves(bars))
        return report

    # ------------------------------------------------------------------ checks

    def _check_prices(self, bars: pd.DataFrame) -> ComplianceCheck:
        if bars.empty:
            return ComplianceCheck("prices", False, "no bars at all")
        bad = (bars[["open", "high", "low", "close"]] <= 0).any(axis=1)
        if bad.any():
            first = bars.index[bad][0].date()
            return ComplianceCheck("prices", False,
                                   f"zero/negative price on {int(bad.sum())} bar(s), first {first}")
        return ComplianceCheck("prices", True)

    def _check_gaps(self, bars: pd.DataFrame, asset_class: str) -> ComplianceCheck:
        if len(bars) < 2:
            return ComplianceCheck("gaps", True)
        dates = bars.index.normalize()

        if asset_class == "crypto":
            gap = (dates[1:] - dates[:-1]).days.to_numpy()
        else:
            # Business-day distance between consecutive bars: 1 is a normal
            # trading day (weekends already excluded by the calendar), so
            # only a run of >1 skipped business days counts as a real gap.
            prev = dates[:-1].to_numpy(dtype="datetime64[D]")
            curr = dates[1:].to_numpy(dtype="datetime64[D]")
            gap = np.busday_count(prev, curr)

        worst = int(gap.max()) if len(gap) else 0
        if worst > self.max_gap_days:
            where = dates[1:][int(np.argmax(gap))]
            return ComplianceCheck("gaps", False,
                                   f"{worst}-day gap in the trading-day series ending {where.date()}")
        return ComplianceCheck("gaps", True)

    def _check_staleness(self, bars: pd.DataFrame, asset_class: str) -> ComplianceCheck:
        if bars.empty:
            return ComplianceCheck("staleness", False, "no bars at all")
        last = bars.index[-1].normalize()
        today = pd.Timestamp.utcnow().tz_localize(None).normalize()

        if asset_class == "crypto":
            age = int((today - last).days)
        else:
            age = int(np.busday_count(np.datetime64(last, "D"), np.datetime64(today, "D")))

        if age > self.stale_after_days:
            return ComplianceCheck("staleness", False,
                                   f"last bar {last.date()} is {age} trading day(s) old")
        return ComplianceCheck("staleness", True)

    def _check_moves(self, bars: pd.DataFrame) -> ComplianceCheck:
        if len(bars) < 2:
            return ComplianceCheck("moves", True)
        moves = bars["close"].pct_change().abs().dropna()
        if moves.empty:
            return ComplianceCheck("moves", True)
        worst = float(moves.max())
        if worst > self.max_daily_move:
            where = moves.idxmax()
            return ComplianceCheck("moves", False,
                                   f"{worst:.0%} day-over-day move on {where.date()} "
                                   "(possible unadjusted corporate action)")
        return ComplianceCheck("moves", True)
