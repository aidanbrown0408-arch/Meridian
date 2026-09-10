"""Cornelius — Execution Agent.

Three implementations, per spec §2:
  * ResearchExecutor -- no-op. Research mode never places an order.
  * PaperBroker -- simulated fills against a persisted $5,000 virtual
    ledger. This is what builds the real forward-tested track record the
    live-trading gate (spec §13) cares about.
  * LiveBroker -- a gated stub. It ships in Phase 5.

The paper ledger is a JSON file on disk (`reports/paper_ledger.json` by
default) so the account survives between daily runs. Every rebalance is
costed with the same `CostModel` the backtester uses, so paper P&L and
backtest P&L are directly comparable -- that comparability is the whole
point of paper mode.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from agents.data_agent import MarketData
from agents.risk_agent import NettedPosition
from backtester.engine import BPS, CostModel
from utils.config import Config
from utils.logging_setup import get_logger

log = get_logger("execution", agent="Cornelius")


@dataclass
class Position:
    symbol: str
    asset_class: str
    shares: float
    entry_price: float
    entry_date: str  # ISO date of the first fill that opened this position

    def days_held(self, as_of: datetime | None = None) -> int:
        as_of = as_of or datetime.now(timezone.utc)
        entered = datetime.fromisoformat(self.entry_date)
        return max(0, (as_of.date() - entered.date()).days)

    def market_value(self, price: float) -> float:
        return self.shares * price

    def unrealized_pnl(self, price: float) -> float:
        return self.shares * (price - self.entry_price)


@dataclass
class Trade:
    date: str
    symbol: str
    side: str          # "buy" | "sell"
    shares: float
    price: float
    cost: float         # dollars, commission + slippage
    reason: str


@dataclass
class PaperLedger:
    starting_capital: float
    cash: float
    positions: dict = field(default_factory=dict)       # symbol -> Position
    trades: list = field(default_factory=list)            # Trade, most recent last
    equity_history: list = field(default_factory=list)    # [{"date":..., "equity":...}]
    halted: bool = False
    halt_reason: str = ""
    # Per-trader shadow equity, used only to detect live-drift (spec §10) --
    # independent of the netted positions above, so one trader's drift never
    # gets masked by another sharing the same ticker.
    trader_shadow: dict = field(default_factory=dict)     # trader -> [{"date","return"}]
    created_at: str = ""
    updated_at: str = ""

    @property
    def equity(self) -> float:
        """Cash-only equity; call `mark_to_market` for the version that
        includes open positions at current prices."""
        return self.cash

    def mark_to_market(self, prices: dict) -> float:
        total = self.cash
        for symbol, pos in self.positions.items():
            price = prices.get(symbol)
            if price is not None:
                total += pos.market_value(price)
        return total

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PaperLedger":
        positions = {s: Position(**p) for s, p in d.get("positions", {}).items()}
        trades = [Trade(**t) for t in d.get("trades", [])]
        return cls(
            starting_capital=d["starting_capital"], cash=d["cash"],
            positions=positions, trades=trades,
            equity_history=d.get("equity_history", []),
            halted=d.get("halted", False), halt_reason=d.get("halt_reason", ""),
            trader_shadow=d.get("trader_shadow", {}),
            created_at=d.get("created_at", ""), updated_at=d.get("updated_at", ""),
        )


class ResearchExecutor:
    """No-op. Research mode is backtest + report only -- nothing here ever
    places an order."""

    name = "Cornelius"
    role = "Execution (Research)"

    def __init__(self, config: Config):
        self.config = config

    def execute(self, *args, **kwargs) -> None:
        log.debug("Research mode -- no execution.")
        return None


class PaperBroker:
    """Cornelius, in paper mode."""

    name = "Cornelius"
    role = "Execution (Paper)"

    def __init__(self, config: Config):
        self.config = config
        self.ledger_path: Path = config.repo_path(
            config.get("execution.paper_ledger_path", "reports/paper_ledger.json"))
        self.starting_capital = float(config.get("capital.starting_paper_capital"))
        self.min_trade_dollars = float(config.get("execution.min_trade_dollars", 25.0))

    # ------------------------------------------------------------- persistence

    def load_ledger(self) -> PaperLedger:
        if not self.ledger_path.exists():
            now = datetime.now(timezone.utc).isoformat()
            return PaperLedger(starting_capital=self.starting_capital,
                               cash=self.starting_capital, created_at=now, updated_at=now)
        try:
            data = json.loads(self.ledger_path.read_text())
            return PaperLedger.from_dict(data)
        except Exception as exc:
            log.error("Ledger at %s is unreadable (%s) -- refusing to guess at state. "
                     "Move or delete it to start a fresh paper account.", self.ledger_path, exc)
            raise

    def save_ledger(self, ledger: PaperLedger) -> None:
        ledger.updated_at = datetime.now(timezone.utc).isoformat()
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.ledger_path.write_text(json.dumps(ledger.to_dict(), indent=2, default=str))

    # ------------------------------------------------------------------ trading

    def execute(self, market: dict[str, MarketData],
               netted_positions: dict[str, NettedPosition],
               regime_adjustments: list | None = None) -> PaperLedger:
        """Rebalance the paper ledger toward today's target weights."""
        ledger = self.load_ledger()
        if ledger.halted:
            log.warning("HALTED (%s) -- refusing to trade until the operator clears it "
                       "(python main.py clear-halt).", ledger.halt_reason)
            return ledger

        today = datetime.now(timezone.utc)
        prices = {s: float(d.bars["close"].iloc[-1]) for s, d in market.items() if not d.bars.empty}
        equity = ledger.mark_to_market(prices)

        for symbol, price in prices.items():
            asset_class = market[symbol].asset_class
            target_weight = netted_positions.get(
                symbol, NettedPosition(symbol, 0.0, False, [])).target_weight
            self._rebalance_one(ledger, symbol, asset_class, price, target_weight, equity, today)

        self._record_shadow(ledger, market, regime_adjustments or [], today)

        equity_after = ledger.mark_to_market(prices)
        ledger.equity_history.append({"date": today.date().isoformat(), "equity": equity_after})
        # Keep the history from growing without bound; ~2 years of daily marks is plenty.
        ledger.equity_history = ledger.equity_history[-800:]

        self.save_ledger(ledger)
        log.info("Paper equity: $%.2f (cash $%.2f, %d open position(s))",
                 equity_after, ledger.cash, len(ledger.positions))
        return ledger

    def _rebalance_one(self, ledger: PaperLedger, symbol: str, asset_class: str, price: float,
                       target_weight: float, equity: float, today: datetime) -> None:
        cost_model = CostModel.from_config(self.config, asset_class)
        current = ledger.positions.get(symbol)
        current_shares = current.shares if current else 0.0
        target_shares = (target_weight * equity) / price if price > 0 else 0.0
        delta_shares = target_shares - current_shares
        delta_value = abs(delta_shares) * price
        if delta_value < self.min_trade_dollars:
            return

        cost = delta_value * cost_model.one_way_bps * BPS

        if delta_shares > 0:
            needed = delta_shares * price + cost
            if needed > ledger.cash:
                # Size the buy down to what cash actually allows rather than
                # refuse the whole rebalance.
                affordable_value = max(0.0, ledger.cash / (1.0 + cost_model.one_way_bps * BPS))
                delta_shares = affordable_value / price
                delta_value = delta_shares * price
                cost = delta_value * cost_model.one_way_bps * BPS
                if delta_shares * price < self.min_trade_dollars:
                    return
            ledger.cash -= delta_shares * price + cost
            new_shares = current_shares + delta_shares
            if current is None or current_shares == 0:
                entry_price, entry_date = price, today.date().isoformat()
            else:
                # Weighted-average entry price across the combined position.
                entry_price = ((current.entry_price * current_shares + price * delta_shares)
                               / new_shares)
                entry_date = current.entry_date
            ledger.positions[symbol] = Position(symbol, asset_class, new_shares,
                                                entry_price, entry_date)
            ledger.trades.append(Trade(today.date().isoformat(), symbol, "buy",
                                       delta_shares, price, cost, "rebalance to target"))
        else:
            sell_shares = min(abs(delta_shares), current_shares)
            proceeds = sell_shares * price - cost
            ledger.cash += proceeds
            remaining = current_shares - sell_shares
            if remaining * price < self.min_trade_dollars:
                ledger.positions.pop(symbol, None)
            else:
                ledger.positions[symbol] = Position(symbol, asset_class, remaining,
                                                    current.entry_price, current.entry_date)
            ledger.trades.append(Trade(today.date().isoformat(), symbol, "sell",
                                       sell_shares, price, cost, "rebalance to target"))

    def _record_shadow(self, ledger: PaperLedger, market: dict[str, MarketData],
                       regime_adjustments: list, today: datetime) -> None:
        """One daily return per live trader, independent of ticker netting,
        so live-drift can be attributed to a specific trader rather than a
        blended position it happens to share with others."""
        by_trader: dict[str, float] = {}
        counts: dict[str, int] = {}
        for adj in regime_adjustments:
            data = market.get(adj.symbol)
            if data is None or len(data.bars) < 2 or not adj.active:
                continue
            asset_return = float(data.bars["close"].pct_change().iloc[-1])
            weighted = adj.adjusted_weight * asset_return
            by_trader[adj.strategy] = by_trader.get(adj.strategy, 0.0) + weighted
            counts[adj.strategy] = counts.get(adj.strategy, 0) + 1

        date_str = today.date().isoformat()
        for trader, ret in by_trader.items():
            history = ledger.trader_shadow.setdefault(trader, [])
            history.append({"date": date_str, "return": ret})
            ledger.trader_shadow[trader] = history[-800:]

    # ------------------------------------------------------------------ safety

    def killswitch(self, market: dict[str, MarketData] | None = None,
                  reason: str = "operator killswitch") -> PaperLedger:
        """Flatten every open position and halt. The system refuses to trade
        again until `clear_halt()` is called explicitly -- never automatic."""
        ledger = self.load_ledger()
        prices = {s: float(d.bars["close"].iloc[-1]) for s, d in (market or {}).items()
                  if not d.bars.empty}
        today = datetime.now(timezone.utc)

        for symbol, pos in list(ledger.positions.items()):
            price = prices.get(symbol, pos.entry_price)
            if symbol not in prices:
                log.warning("No live price for %s at killswitch -- flattening at last "
                           "known entry price %.2f instead.", symbol, price)
            cost_model = CostModel.from_config(self.config, pos.asset_class)
            cost = pos.shares * price * cost_model.one_way_bps * BPS
            proceeds = pos.shares * price - cost
            ledger.cash += proceeds
            ledger.trades.append(Trade(today.date().isoformat(), symbol, "sell",
                                       pos.shares, price, cost, "killswitch flatten"))
            del ledger.positions[symbol]

        ledger.halted = True
        ledger.halt_reason = reason
        self.save_ledger(ledger)
        log.warning("KILLSWITCH: flattened %d position(s), halted. Clear with "
                   "'python main.py clear-halt' when ready.", len(prices))
        return ledger

    def clear_halt(self) -> PaperLedger:
        ledger = self.load_ledger()
        ledger.halted = False
        ledger.halt_reason = ""
        self.save_ledger(ledger)
        log.info("Halt cleared -- paper trading resumes on the next run.")
        return ledger


class LiveBroker:
    """Gated stub (spec §13). Ships as a stub that raises NotImplementedError
    -- implementing this against a real broker API is the operator's
    responsibility, deliberately, per the three-gate live-trading design."""

    name = "Cornelius"
    role = "Execution (Live)"

    def __init__(self, config: Config):
        self.config = config

    def execute(self, *args, **kwargs):
        raise NotImplementedError(
            "LiveBroker has no real broker adapter. Implement one (e.g. Alpaca for "
            "equities, ccxt with real exchange keys for crypto) before live mode can run."
        )
