"""Walk-forward validation.

The single most important quality-control mechanism in the system (spec §6).
Split each ticker's history into 5 sequential folds; score the strategy on
each fold's out-of-sample bars only; call it a pass if at least 3 of 5 folds
clear the bar. Strict (5/5) is too demanding — nothing ever trades. Lenient
(average) lets one great fold mask four losing ones. Majority is the bar the
spec settled on: the strategy has to work more often than not across
different chunks of history, without demanding perfection.

Meridian's traders have fixed parameters — there is no fitting step. "Training
window" therefore does not mean parameter search; it means the strategy's
indicators (moving averages, RSI, Donchian channels...) warm up over history
that predates the fold, and only the fold's own bars are ever scored. That is
the sense in which each fold is genuinely out-of-sample: the strategy has
never been evaluated on those bars before.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from utils.config import Config
from utils.logging_setup import get_logger
from utils.metrics import PerformanceSummary, summarize
from backtester.engine import BacktestEngine, CostModel
from strategies.base import Strategy

log = get_logger("walkforward", agent="Charles")


@dataclass
class WalkForwardFold:
    """One fold's out-of-sample scorecard."""
    index: int
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    bars: int
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    trades: int = 0
    passed: bool = False
    gradeable: bool = True
    reason: str = ""


@dataclass
class WalkForwardResult:
    """Everything Charles needs to decide whether a strategy/symbol
    combination has earned capital."""
    strategy: str
    symbol: str
    folds_total: int
    min_folds_passing: int
    folds: list[WalkForwardFold] = field(default_factory=list)
    pass_fraction: float = 0.6
    min_gradeable_folds: int = 2

    @property
    def folds_passed(self) -> int:
        return sum(1 for f in self.folds if f.passed)

    @property
    def gradeable_folds(self) -> int:
        """Folds that saw enough trades to constitute evidence either way."""
        return sum(1 for f in self.folds if f.gradeable)

    @property
    def undecidable(self) -> bool:
        """Too few gradeable folds to form a verdict at all. Distinct from
        failing: the strategy did not perform badly, it simply did not act
        often enough over this history to be measured."""
        return self.gradeable_folds < self.min_gradeable_folds

    @property
    def passed(self) -> bool:
        """A majority of the GRADEABLE folds must clear the bar.

        Folds with too few trades are excluded from both numerator and
        denominator rather than counted as failures: a strategy that
        correctly sat flat through a fold has produced no evidence, and
        scoring that as a loss silently penalises every low-frequency
        trader. If too few folds are gradeable, the combination is
        undecidable and does not pass."""
        if self.folds_total <= 0 or self.undecidable:
            return False
        needed = max(1, int(round(self.gradeable_folds * self.pass_fraction)))
        return self.folds_passed >= needed

    def to_row(self) -> dict:
        return {
            "strategy": self.strategy,
            "symbol": self.symbol,
            "folds_passed": self.folds_passed,
            "folds_total": self.folds_total,
            "gradeable_folds": self.gradeable_folds,
            "undecidable": self.undecidable,
            "passed": self.passed,
        }


class WalkForwardValidator:
    """Runs the N-fold, majority-of-N walk-forward check for one
    strategy/symbol combination."""

    def __init__(self, config: Config, engine: BacktestEngine | None = None):
        self.config = config
        self.engine = engine or BacktestEngine(config)
        self.folds_n = int(config.get("validation.folds"))
        self.min_folds_passing = int(config.get("validation.min_folds_passing"))
        self.min_sharpe = float(config.get("validation.min_sharpe"))
        self.min_trades = int(config.get("validation.min_trades_per_fold"))
        # A fold below `min_trades` is treated as NO EVIDENCE and excluded
        # from the verdict, not as a failed fold. These two keys say how
        # much evidence is needed before a verdict is possible at all.
        self.min_gradeable_folds = int(config.get("validation.min_gradeable_folds", 2))
        self.pass_fraction = float(config.get("validation.pass_fraction", 0.6))
        self.max_drawdown_limit = float(config.get("risk.max_strategy_drawdown"))
        self.risk_free_rate = float(config.get("risk.risk_free_rate", 0.0))

    def validate(self, strategy: Strategy, bars: pd.DataFrame, symbol: str,
                asset_class: str = "stocks") -> WalkForwardResult:
        result = WalkForwardResult(
            strategy=strategy.callsign, symbol=symbol,
            folds_total=self.folds_n, min_folds_passing=self.min_folds_passing,
            pass_fraction=self.pass_fraction,
            min_gradeable_folds=self.min_gradeable_folds,
        )
        if len(bars) < self.folds_n:
            log.warning("%s/%s: too few bars (%d) to split into %d folds",
                       strategy.callsign, symbol, len(bars), self.folds_n)
            return result

        boundaries = np.array_split(np.arange(len(bars)), self.folds_n)
        cost_model = CostModel.from_config(self.config, asset_class)
        close = bars["close"].astype("float64")
        asset_returns = close.pct_change().fillna(0.0)

        for i, idx in enumerate(boundaries):
            if len(idx) == 0:
                continue
            test_start, test_end = bars.index[idx[0]], bars.index[idx[-1]]

            if idx[-1] < strategy.warmup:
                result.folds.append(WalkForwardFold(
                    index=i, test_start=test_start, test_end=test_end,
                    bars=len(idx), passed=False, gradeable=False,
                    reason=f"fold ends before warmup ({strategy.warmup} bars)",
                ))
                continue

            # Everything up through this fold's last bar: earlier bars are the
            # "training" window the indicators warm up over. Only this fold's
            # own rows are ever scored below.
            window = bars.iloc[: idx[-1] + 1]
            position = strategy.positions(window).reindex(window.index).fillna(0.0)
            position.iloc[: strategy.warmup] = 0.0

            gross = position * asset_returns.reindex(window.index)
            costs = cost_model.charge(position)
            net = (gross - costs).iloc[idx[0]: idx[-1] + 1]
            fold_position = position.iloc[idx[0]: idx[-1] + 1]
            fold_costs = costs.iloc[idx[0]: idx[-1] + 1]

            summary = summarize(
                net, position=fold_position, asset_class=asset_class,
                risk_free_rate=self.risk_free_rate, total_cost=float(fold_costs.sum()),
            )
            gradeable = summary.trades >= self.min_trades
            passed, reason = self._grade(summary)
            result.folds.append(WalkForwardFold(
                index=i, test_start=test_start, test_end=test_end, bars=len(idx),
                sharpe=summary.sharpe, max_drawdown=summary.max_drawdown,
                trades=summary.trades, passed=passed, gradeable=gradeable,
                reason=reason,
            ))

        log.debug("%s/%s: %d/%d folds passed", strategy.callsign, symbol,
                  result.folds_passed, result.folds_total)
        return result

    def _grade(self, summary: PerformanceSummary) -> tuple[bool, str]:
        if summary.trades < self.min_trades:
            # Not a failure — no evidence. `gradeable=False` keeps this fold
            # out of the verdict entirely (see WalkForwardResult.passed).
            return False, (f"not graded: only {summary.trades} trade(s), "
                           f"need {self.min_trades} for evidence")
        if summary.sharpe < self.min_sharpe:
            return False, f"Sharpe {summary.sharpe:.2f} below {self.min_sharpe:.2f}"
        if summary.max_drawdown > self.max_drawdown_limit:
            return False, (f"drawdown {summary.max_drawdown:.1%} exceeds "
                           f"{self.max_drawdown_limit:.1%}")
        return True, "ok"


def results_frame(results: list[WalkForwardResult]) -> pd.DataFrame:
    """Sortable summary table: one row per strategy/symbol combination."""
    if not results:
        return pd.DataFrame()
    frame = pd.DataFrame([r.to_row() for r in results])
    return frame.sort_values(["passed", "folds_passed"], ascending=[False, False]).reset_index(drop=True)
