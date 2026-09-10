"""Leo — Backtest Agent.

Runs every enabled strategy against every symbol and produces raw performance
metrics. Leo makes no judgements: filtering, validation, and sizing all belong
to Charles in Phase 2. Leo's only opinion is that a run which cannot produce a
meaningful result should be marked blocked rather than silently reported as
zero.
"""

from __future__ import annotations

from utils.config import Config
from utils.logging_setup import get_logger
from utils.metrics import PerformanceSummary
from agents.data_agent import MarketData
from backtester.engine import BacktestEngine, BacktestResult
from strategies.base import Strategy, build_strategies

log = get_logger("backtest", agent="Leo")


class BacktestAgent:
    """Leo."""

    name = "Leo"
    role = "Backtest"

    def __init__(self, config: Config, strategies: dict[str, Strategy] | None = None):
        self.config = config
        self.engine = BacktestEngine(config)
        self.strategies = strategies if strategies is not None else build_strategies(config)

    def run_all(self, market: dict[str, MarketData],
                blocked: dict[str, str] | None = None) -> list[BacktestResult]:
        """Cross product of strategies and symbols.

        `blocked` maps symbol -> reason, supplied by David in Phase 2. Blocked
        symbols still appear in the results so the report can show them as
        blocked rather than silently missing.
        """
        blocked = blocked or {}
        results: list[BacktestResult] = []

        for symbol, data in market.items():
            if symbol in blocked:
                reason = blocked[symbol]
                log.warning("%s blocked by compliance: %s", symbol, reason)
                for callsign, strategy in self.strategies.items():
                    results.append(BacktestResult(
                        strategy=callsign, symbol=symbol, asset_class=data.asset_class,
                        data_source=data.data_source, summary=PerformanceSummary(),
                        blocked=True, block_reason=reason, params=dict(strategy.params),
                    ))
                continue

            for callsign, strategy in self.strategies.items():
                results.append(self.engine.run(
                    strategy, data.bars, symbol,
                    asset_class=data.asset_class, data_source=data.data_source,
                ))

        ran = sum(1 for r in results if not r.blocked)
        log.info("Ran %d/%d strategy-ticker combinations (%d blocked)",
                 ran, len(results), len(results) - ran)
        return results

    def benchmarks(self, market: dict[str, MarketData]) -> dict[str, PerformanceSummary]:
        """Buy-and-hold for each symbol. SPY is the portfolio benchmark; the
        others are useful context for whether a strategy beat simply owning
        the thing."""
        return {
            symbol: self.engine.buy_and_hold(data.bars, symbol, data.asset_class)
            for symbol, data in market.items() if len(data.bars) > 1
        }
