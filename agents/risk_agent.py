"""Charles — Risk Agent.

Turns Leo's raw backtest results into capital (spec §7). Three jobs, in order:

1. Walk-forward validate every strategy/symbol combination David didn't
   block, and reject anything whose full-period max drawdown breaches the
   per-strategy limit even if it cleared the folds — the drawdown filter is
   a hard reject applied "before any capital is assigned," independent of
   the walk-forward pass/fail.
2. Cap the roster: at most `risk.max_live_traders` traders (callsigns, not
   strategy/symbol pairs — a live trader can still hold positions across
   every symbol it individually cleared) get capital. Ties are broken by
   the best walk-forward Sharpe among a trader's passing symbols.
3. Size the survivors with risk parity — inversely to realized volatility —
   with one twist: ANCHOR and REVERT are structurally the same bet (both
   mean-reversion, both trigger on the same conditions), so they share a
   single risk-parity slot instead of one each. Treating them as
   independent slots would silently double true exposure to mean-reversion.
   Same-ticker signals from different live traders then net into one
   blended ticker-level target, capped at `risk.max_position_pct`.

Regime-mismatch capital cuts are Greg's signal (Phase 3) and get layered on
top of these base weights; this module does not know about regimes yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from utils.config import Config
from utils.logging_setup import get_logger
from agents.data_agent import MarketData
from backtester.engine import BacktestResult
from backtester.walkforward import WalkForwardResult, WalkForwardValidator
from strategies.base import Strategy

log = get_logger("risk", agent="Charles")

# Traders that trigger on the same market conditions and must not be
# double-counted as independent diversification slots.
CORRELATED_SLOTS: tuple[frozenset, ...] = (frozenset({"ANCHOR", "REVERT"}),)


@dataclass
class TraderCandidate:
    """One strategy/symbol combination's full validation verdict."""
    strategy: str
    symbol: str
    asset_class: str
    walkforward: WalkForwardResult
    backtest_max_drawdown: float
    annual_vol: float
    sharpe: float
    eligible: bool = False
    reject_reason: str = ""

    def to_row(self) -> dict:
        return {
            "strategy": self.strategy,
            "symbol": self.symbol,
            "folds_passed": self.walkforward.folds_passed,
            "folds_total": self.walkforward.folds_total,
            "max_drawdown": self.backtest_max_drawdown,
            "eligible": self.eligible,
            "reject_reason": self.reject_reason,
        }


@dataclass
class SlotAllocation:
    """One risk-parity slot: a trader alone, or a correlated group sharing
    one slot's worth of capital."""
    members: tuple[str, ...]
    weight: float = 0.0
    member_weights: dict = field(default_factory=dict)


@dataclass
class NettedPosition:
    """Ticker-level target after blending every live trader's current
    signal on that symbol."""
    symbol: str
    target_weight: float
    capped: bool
    contributors: list = field(default_factory=list)


@dataclass
class RiskReport:
    candidates: list = field(default_factory=list)
    live_traders: list = field(default_factory=list)
    benched_traders: dict = field(default_factory=dict)
    slots: list = field(default_factory=list)
    capital_weights: dict = field(default_factory=dict)
    netted_positions: dict = field(default_factory=dict)
    starting_capital: float = 0.0

    def dollars(self, callsign: str) -> float:
        return self.starting_capital * self.capital_weights.get(callsign, 0.0)


class RiskAgent:
    """Charles."""

    name = "Charles"
    role = "Risk"

    def __init__(self, config: Config):
        self.config = config
        self.validator = WalkForwardValidator(config)
        self.max_live_traders = int(config.get("risk.max_live_traders"))
        self.max_strategy_drawdown = float(config.get("risk.max_strategy_drawdown"))
        self.max_position_pct = float(config.get("risk.max_position_pct"))
        self.starting_capital = float(config.get("capital.starting_paper_capital"))

    def run(self, strategies: dict[str, Strategy], market: dict[str, MarketData],
            backtest_results: list[BacktestResult]) -> RiskReport:
        by_key = {(r.strategy, r.symbol): r for r in backtest_results}
        report = RiskReport(starting_capital=self.starting_capital)

        for callsign, strategy in strategies.items():
            for symbol, data in market.items():
                bt = by_key.get((callsign, symbol))
                if bt is None or bt.blocked:
                    continue
                wf = self.validator.validate(strategy, data.bars, symbol, data.asset_class)
                candidate = TraderCandidate(
                    strategy=callsign, symbol=symbol, asset_class=data.asset_class,
                    walkforward=wf, backtest_max_drawdown=bt.summary.max_drawdown,
                    annual_vol=bt.summary.annual_vol, sharpe=bt.summary.sharpe,
                )
                candidate.eligible, candidate.reject_reason = self._grade(candidate)
                report.candidates.append(candidate)

        report.live_traders, report.benched_traders = self._pick_roster(report.candidates)
        report.slots = self._build_slots(report.live_traders)
        report.capital_weights = self._risk_parity(report.slots, report.candidates)
        eligible_pairs = {(c.strategy, c.symbol) for c in report.candidates if c.eligible}
        report.netted_positions = self._net_positions(
            report.live_traders, eligible_pairs, report.capital_weights, market, by_key,
        )

        log.info("%d/%d candidates eligible; live: %s",
                 sum(1 for c in report.candidates if c.eligible), len(report.candidates),
                 ", ".join(report.live_traders) or "none")
        return report

    # ------------------------------------------------------------------ steps

    def _grade(self, candidate: TraderCandidate) -> tuple[bool, str]:
        if not candidate.walkforward.passed:
            return False, (f"walk-forward {candidate.walkforward.folds_passed}/"
                           f"{candidate.walkforward.folds_total} folds")
        if candidate.backtest_max_drawdown > self.max_strategy_drawdown:
            return False, (f"max drawdown {candidate.backtest_max_drawdown:.1%} exceeds "
                           f"{self.max_strategy_drawdown:.1%}")
        return True, ""

    def _pick_roster(self, candidates: list[TraderCandidate]) -> tuple[list, dict]:
        """A trader (callsign) is eligible for the roster if at least one of
        its symbols passed. Rank traders by the best Sharpe among their
        passing symbols and keep the top `max_live_traders`."""
        best_sharpe: dict[str, float] = {}
        for c in candidates:
            if c.eligible:
                best_sharpe[c.strategy] = max(best_sharpe.get(c.strategy, float("-inf")), c.sharpe)

        all_traders = {c.strategy for c in candidates}
        ranked = sorted(best_sharpe, key=lambda t: best_sharpe[t], reverse=True)
        live = ranked[: self.max_live_traders]
        live_set = set(live)

        benched: dict[str, str] = {}
        for trader in sorted(all_traders - live_set):
            if trader in best_sharpe:
                benched[trader] = (f"passed validation but ranked outside the top "
                                   f"{self.max_live_traders} by Sharpe")
            else:
                reasons = sorted({c.reject_reason for c in candidates
                                  if c.strategy == trader and c.reject_reason})
                benched[trader] = "; ".join(reasons) or "no symbol passed validation"
        return live, benched

    def _build_slots(self, live_traders: list[str]) -> list[SlotAllocation]:
        live_set = set(live_traders)
        grouped: set[str] = set()
        slots: list[SlotAllocation] = []
        for correlated in CORRELATED_SLOTS:
            members = tuple(sorted(correlated & live_set))
            if len(members) >= 2:
                slots.append(SlotAllocation(members=members))
                grouped.update(members)
        for trader in live_traders:
            if trader not in grouped:
                slots.append(SlotAllocation(members=(trader,)))
        return slots

    def _trader_vol(self, trader: str, candidates: list[TraderCandidate]) -> float:
        """A trader's own realized volatility, averaged across the symbols it
        actually cleared. Only eligible rows count — a rejected symbol's vol
        shouldn't move the sizing of one the trader is actually trading."""
        vols = [c.annual_vol for c in candidates
               if c.strategy == trader and c.eligible and c.annual_vol > 0]
        return sum(vols) / len(vols) if vols else 0.0

    def _risk_parity(self, slots: list[SlotAllocation],
                     candidates: list[TraderCandidate]) -> dict[str, float]:
        """Inverse-volatility weighting across slots. A correlated slot is one
        risk unit no matter how many traders share it; its share then splits
        among members by their own inverse vol, so they still diverge at the
        margins without ever summing to more than one slot's capital."""
        if not slots:
            return {}

        trader_vol = {m: self._trader_vol(m, candidates)
                      for slot in slots for m in slot.members}
        slot_vol: dict[tuple, float] = {}
        for slot in slots:
            member_vols = [trader_vol[m] for m in slot.members if trader_vol[m] > 0]
            slot_vol[slot.members] = sum(member_vols) / len(member_vols) if member_vols else 0.0

        inv_vol = {members: (1.0 / v if v > 0 else 0.0) for members, v in slot_vol.items()}
        total_inv = sum(inv_vol.values())

        weights: dict[str, float] = {}
        for slot in slots:
            slot.weight = (inv_vol[slot.members] / total_inv if total_inv > 0
                           else 1.0 / len(slots))
            member_inv = {m: (1.0 / trader_vol[m] if trader_vol[m] > 0 else 0.0)
                         for m in slot.members}
            total_member_inv = sum(member_inv.values())
            for m in slot.members:
                share = (member_inv[m] / total_member_inv if total_member_inv > 0
                        else 1.0 / len(slot.members))
                slot.member_weights[m] = slot.weight * share
                weights[m] = slot.weight * share
        return weights

    def _net_positions(self, live_traders: list[str], eligible_pairs: set,
                       capital_weights: dict, market: dict[str, MarketData],
                       by_key: dict) -> dict[str, NettedPosition]:
        """Blend every live, eligible trader's *current* signal (the last bar
        of its already-shifted position series) into one ticker-level target,
        capped at the configured position-size ceiling. A trader only sizes a
        ticker it individually cleared walk-forward on — being live overall
        does not license it to trade a symbol it failed."""
        netted: dict[str, NettedPosition] = {}
        for symbol in market:
            raw = 0.0
            contributors: list[str] = []
            for trader in live_traders:
                if (trader, symbol) not in eligible_pairs:
                    continue
                bt = by_key.get((trader, symbol))
                if bt is None or bt.position.empty:
                    continue
                current = float(bt.position.iloc[-1])
                if current <= 0:
                    continue
                raw += capital_weights.get(trader, 0.0) * current
                contributors.append(trader)
            netted[symbol] = NettedPosition(
                symbol=symbol, target_weight=min(raw, self.max_position_pct),
                capped=raw > self.max_position_pct, contributors=contributors,
            )
        return netted
