"""Greg — Regime Agent.

A pure technical-analysis system with no awareness of current market
conditions is brittle: it treats every day the same, whether the market is
trending or in chop (spec §8). Greg adds an explicit, per-ticker read of
today's regime and adjusts allocations before trading — SPY can be trending
while BTC/USDT is choppy on the same day.

Classification (all sub-conditions must hold together, or it's undecided):
  * trending: ADX above its threshold, MA slope consistently signed, realized
    vol at or above a "normal-to-high" percentile of its own trailing range.
  * choppy: ADX below its threshold, MA slope oscillating, price bouncing
    inside its recent range rather than making new highs/lows.

Regime-mismatch adjustment: when a trader's `best_regimes` doesn't include
today's regime on a ticker, two things apply together — its risk-parity
weight on that ticker is halved, and its position only counts once it has
held steady for `confirmation_days` consecutive bars (filters single-day
noise, raises the bar for trading against the regime). A trader whose
`best_regimes` matches, or a ticker that's undecided, trades at full weight
with normal timing. No trader is ever fully benched by the regime call —
only tilted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from utils.config import Config
from utils.logging_setup import get_logger
from agents.data_agent import MarketData
from agents.risk_agent import NettedPosition, RiskReport
from backtester.engine import BacktestResult
from strategies.base import Strategy

log = get_logger("regime", agent="Greg")


# ------------------------------------------------------------------ indicators

def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def adx(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's Average Directional Index."""
    high, low, close = bars["high"], bars["low"], bars["close"]
    prev_close, prev_high, prev_low = close.shift(1), high.shift(1), low.shift(1)

    true_range = pd.concat([
        high - low, (high - prev_close).abs(), (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    up_move = high - prev_high
    down_move = prev_low - low
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
                        index=bars.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
                         index=bars.index)

    smoothed_tr = _wilder_smooth(true_range, period)
    plus_di = 100.0 * _wilder_smooth(plus_dm, period) / smoothed_tr.replace(0.0, np.nan)
    minus_di = 100.0 * _wilder_smooth(minus_dm, period) / smoothed_tr.replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return _wilder_smooth(dx.fillna(0.0), period)


def _slope_consistent(close: pd.Series, ma_period: int, window: int, threshold: float) -> bool:
    """True if at least `threshold` of the last `window` daily MA moves share
    the same sign -- a consistently-signed slope rather than an oscillating
    one."""
    ma = close.rolling(ma_period, min_periods=ma_period).mean()
    diffs = ma.diff().dropna().tail(window)
    if len(diffs) < window:
        return False
    same_sign = max(int((diffs > 0).sum()), int((diffs < 0).sum()))
    return same_sign / len(diffs) >= threshold


def _vol_percentile(close: pd.Series, vol_window: int, lookback: int) -> float:
    """Today's realized vol, ranked against its own trailing distribution.
    1.0 = the highest-vol day in the lookback window, 0.0 = the lowest."""
    realized = close.pct_change().rolling(vol_window).std().dropna().tail(lookback)
    if realized.empty:
        return 0.0
    current = realized.iloc[-1]
    return float((realized <= current).mean())


def _range_bound_fraction(close: pd.Series, window: int) -> float:
    """Fraction of the last `window` bars that were NOT a new window-high or
    window-low -- how much of the recent tape has just been bouncing."""
    rolling_high = close.rolling(window, min_periods=window).max()
    rolling_low = close.rolling(window, min_periods=window).min()
    is_new_extreme = (close >= rolling_high) | (close <= rolling_low)
    recent = is_new_extreme.dropna().tail(window)
    if recent.empty:
        return 0.0
    return float((~recent).mean())


# ------------------------------------------------------------------ data types

@dataclass
class RegimeClassification:
    symbol: str
    regime: str                 # "trending" | "choppy" | "undecided"
    adx: float = 0.0
    slope_consistent: bool = False
    vol_percentile: float = 0.0
    range_bound_fraction: float = 0.0
    detail: str = ""


@dataclass
class TraderRegimeAdjustment:
    """One live, eligible (trader, symbol) pair's regime-adjusted sizing."""
    strategy: str
    symbol: str
    regime: str
    matched: bool
    confirmed: bool             # only meaningful when mismatched
    base_weight: float
    adjusted_weight: float
    active: bool                # contributes to today's netted position?
    note: str = ""


@dataclass
class RegimeReport:
    regimes: dict = field(default_factory=dict)          # symbol -> RegimeClassification
    adjustments: list = field(default_factory=list)       # TraderRegimeAdjustment
    netted_positions: dict = field(default_factory=dict)  # symbol -> NettedPosition


class RegimeAgent:
    """Greg."""

    name = "Greg"
    role = "Regime"

    def __init__(self, config: Config):
        self.config = config
        self.adx_period = int(config.get("regime.adx_period"))
        self.adx_trending = float(config.get("regime.adx_trending_threshold"))
        self.adx_choppy = float(config.get("regime.adx_choppy_threshold"))
        self.ma_period = int(config.get("regime.ma_period"))
        self.slope_window = int(config.get("regime.slope_window"))
        self.slope_threshold = float(config.get("regime.slope_consistency_threshold"))
        self.vol_window = int(config.get("regime.vol_window"))
        self.vol_lookback = int(config.get("regime.vol_percentile_lookback"))
        self.vol_normal_high = float(config.get("regime.vol_normal_high_percentile"))
        self.range_window = int(config.get("regime.range_window"))
        self.range_threshold = float(config.get("regime.range_bound_threshold"))
        self.capital_cut = float(config.get("regime.capital_cut"))
        self.confirmation_days = int(config.get("regime.confirmation_days"))
        self.max_position_pct = float(config.get("risk.max_position_pct"))

    # ------------------------------------------------------------- classify

    def classify(self, bars: pd.DataFrame, symbol: str = "") -> RegimeClassification:
        min_history = max(self.adx_period * 2, self.ma_period + self.slope_window,
                          self.vol_window + 1, self.range_window)
        if len(bars) < min_history:
            return RegimeClassification(symbol=symbol, regime="undecided",
                                        detail=f"insufficient history ({len(bars)} bars)")

        close = bars["close"]
        adx_series = adx(bars, self.adx_period)
        current_adx = float(adx_series.iloc[-1]) if pd.notna(adx_series.iloc[-1]) else 0.0
        slope_ok = _slope_consistent(close, self.ma_period, self.slope_window, self.slope_threshold)
        vol_pct = _vol_percentile(close, self.vol_window, self.vol_lookback)
        range_frac = _range_bound_fraction(close, self.range_window)

        trending = current_adx > self.adx_trending and slope_ok and vol_pct >= self.vol_normal_high
        choppy = current_adx < self.adx_choppy and not slope_ok and range_frac >= self.range_threshold

        if trending:
            regime, detail = "trending", f"ADX {current_adx:.1f}, consistent slope"
        elif choppy:
            regime, detail = "choppy", f"ADX {current_adx:.1f}, range-bound {range_frac:.0%}"
        else:
            regime, detail = "undecided", f"ADX {current_adx:.1f}, neither condition held"

        return RegimeClassification(symbol=symbol, regime=regime, adx=current_adx,
                                    slope_consistent=slope_ok, vol_percentile=vol_pct,
                                    range_bound_fraction=range_frac, detail=detail)

    def classify_universe(self, market: dict[str, MarketData]) -> dict[str, RegimeClassification]:
        regimes = {symbol: self.classify(data.bars, symbol) for symbol, data in market.items()}
        log.info(", ".join(f"{s} {r.regime}" for s, r in regimes.items()) or "no symbols")
        return regimes

    # ------------------------------------------------------------- adjust

    def run(self, strategies: dict[str, Strategy], market: dict[str, MarketData],
            backtest_results: list[BacktestResult], risk_report: RiskReport) -> RegimeReport:
        regimes = self.classify_universe(market)
        by_key = {(r.strategy, r.symbol): r for r in backtest_results}
        eligible_pairs = {(c.strategy, c.symbol) for c in risk_report.candidates if c.eligible}

        adjustments: list[TraderRegimeAdjustment] = []
        for trader in risk_report.live_traders:
            strategy = strategies[trader]
            base_weight = risk_report.capital_weights.get(trader, 0.0)
            for symbol in market:
                if (trader, symbol) not in eligible_pairs:
                    continue
                bt = by_key.get((trader, symbol))
                if bt is None or bt.position.empty:
                    continue
                classification = regimes[symbol]
                adjustments.append(self._adjust_one(strategy, symbol, classification,
                                                     base_weight, bt.position))

        netted = self._net_positions(adjustments, market)
        return RegimeReport(regimes=regimes, adjustments=adjustments, netted_positions=netted)

    def _adjust_one(self, strategy: Strategy, symbol: str, classification: RegimeClassification,
                    base_weight: float, position: pd.Series) -> TraderRegimeAdjustment:
        matched = classification.regime == "undecided" or classification.regime in strategy.best_regimes
        current = float(position.iloc[-1])

        if matched:
            return TraderRegimeAdjustment(
                strategy=strategy.callsign, symbol=symbol, regime=classification.regime,
                matched=True, confirmed=True, base_weight=base_weight,
                adjusted_weight=base_weight, active=current > 0,
                note="regime-matched, full weight",
            )

        # Mismatched: capital cut always applies; the smaller position only
        # counts once it has held steady for `confirmation_days` bars.
        adjusted_weight = base_weight * self.capital_cut
        window = position.tail(self.confirmation_days)
        confirmed = (len(window) == self.confirmation_days and current > 0
                    and (window == current).all())
        return TraderRegimeAdjustment(
            strategy=strategy.callsign, symbol=symbol, regime=classification.regime,
            matched=False, confirmed=confirmed, base_weight=base_weight,
            adjusted_weight=adjusted_weight, active=confirmed,
            note=("mismatched, halved weight, confirmed" if confirmed else
                  "mismatched, halved weight, awaiting confirmation" if current > 0 else
                  "mismatched, halved weight, flat"),
        )

    def _net_positions(self, adjustments: list[TraderRegimeAdjustment],
                       market: dict[str, MarketData]) -> dict[str, NettedPosition]:
        netted: dict[str, NettedPosition] = {}
        for symbol in market:
            raw = 0.0
            contributors: list[str] = []
            for adj in adjustments:
                if adj.symbol != symbol or not adj.active:
                    continue
                raw += adj.adjusted_weight
                contributors.append(adj.strategy)
            netted[symbol] = NettedPosition(
                symbol=symbol, target_weight=min(raw, self.max_position_pct),
                capped=raw > self.max_position_pct, contributors=contributors,
            )
        return netted
