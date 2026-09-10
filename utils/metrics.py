"""Performance metrics.

All functions take a pandas Series of *periodic* returns (daily by default) and
are defensive about short or degenerate inputs — a strategy that never trades
should report zeros, not raise or emit NaN.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252
CRYPTO_DAYS = 365


def periods_per_year(asset_class: str = "stocks") -> int:
    return CRYPTO_DAYS if asset_class == "crypto" else TRADING_DAYS


def _clean(returns: pd.Series) -> pd.Series:
    if returns is None or len(returns) == 0:
        return pd.Series(dtype="float64")
    return pd.Series(returns).astype("float64").replace([np.inf, -np.inf], np.nan).dropna()


def total_return(returns: pd.Series) -> float:
    r = _clean(returns)
    if r.empty:
        return 0.0
    return float((1.0 + r).prod() - 1.0)


def equity_curve(returns: pd.Series, starting_value: float = 1.0) -> pd.Series:
    r = _clean(returns)
    if r.empty:
        return pd.Series(dtype="float64")
    return starting_value * (1.0 + r).cumprod()


def cagr(returns: pd.Series, ppy: int = TRADING_DAYS) -> float:
    r = _clean(returns)
    if len(r) < 2:
        return 0.0
    growth = float((1.0 + r).prod())
    years = len(r) / ppy
    if years <= 0 or growth <= 0:
        return 0.0
    return float(growth ** (1.0 / years) - 1.0)


def annual_volatility(returns: pd.Series, ppy: int = TRADING_DAYS) -> float:
    r = _clean(returns)
    if len(r) < 2:
        return 0.0
    return float(r.std(ddof=1) * np.sqrt(ppy))


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0,
                 ppy: int = TRADING_DAYS) -> float:
    r = _clean(returns)
    if len(r) < 2:
        return 0.0
    excess = r - (risk_free_rate / ppy)
    sd = excess.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(excess.mean() / sd * np.sqrt(ppy))


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0,
                  ppy: int = TRADING_DAYS) -> float:
    """Sharpe's downside-only cousin. Upside volatility is not a risk."""
    r = _clean(returns)
    if len(r) < 2:
        return 0.0
    excess = r - (risk_free_rate / ppy)
    downside = excess[excess < 0]
    if downside.empty:
        # No losing periods at all — meaningful only if there was upside.
        return float("inf") if excess.mean() > 0 else 0.0
    dd = np.sqrt((downside ** 2).mean())
    if dd == 0:
        return 0.0
    return float(excess.mean() / dd * np.sqrt(ppy))


def drawdown_series(returns: pd.Series) -> pd.Series:
    curve = equity_curve(returns)
    if curve.empty:
        return pd.Series(dtype="float64")
    return curve / curve.cummax() - 1.0


def max_drawdown(returns: pd.Series) -> float:
    """Returned as a positive fraction: 0.15 means a 15% peak-to-trough loss."""
    dd = drawdown_series(returns)
    if dd.empty:
        return 0.0
    return float(abs(dd.min()))


def calmar_ratio(returns: pd.Series, ppy: int = TRADING_DAYS) -> float:
    mdd = max_drawdown(returns)
    if mdd == 0:
        return 0.0
    return float(cagr(returns, ppy) / mdd)


def trade_stats(position: pd.Series, returns: pd.Series) -> dict:
    """Collapse a bar-level position series into per-trade statistics.

    A trade is one contiguous run of non-zero position. Win rate is measured on
    compounded per-trade return, not per-bar, which is what an operator actually
    experiences.
    """
    pos = pd.Series(position).fillna(0.0).astype("float64")
    r = pd.Series(returns).reindex(pos.index).fillna(0.0).astype("float64")
    if pos.empty:
        return {"trades": 0, "win_rate": 0.0, "avg_trade_return": 0.0,
                "best_trade": 0.0, "worst_trade": 0.0, "avg_holding_days": 0.0}

    in_market = pos != 0
    # New trade whenever we move from flat to non-flat.
    trade_id = (in_market & ~in_market.shift(1, fill_value=False)).cumsum()
    trade_returns: list[float] = []
    holding: list[int] = []
    for _, chunk in r[in_market].groupby(trade_id[in_market]):
        if len(chunk) == 0:
            continue
        trade_returns.append(float((1.0 + chunk).prod() - 1.0))
        holding.append(int(len(chunk)))

    if not trade_returns:
        return {"trades": 0, "win_rate": 0.0, "avg_trade_return": 0.0,
                "best_trade": 0.0, "worst_trade": 0.0, "avg_holding_days": 0.0}

    arr = np.array(trade_returns, dtype="float64")
    return {
        "trades": int(len(arr)),
        "win_rate": float((arr > 0).mean()),
        "avg_trade_return": float(arr.mean()),
        "best_trade": float(arr.max()),
        "worst_trade": float(arr.min()),
        "avg_holding_days": float(np.mean(holding)),
    }


@dataclass
class PerformanceSummary:
    """Everything Leo hands downstream for one strategy/ticker combination."""
    total_return: float = 0.0
    cagr: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0
    calmar: float = 0.0
    annual_vol: float = 0.0
    trades: int = 0
    win_rate: float = 0.0
    avg_trade_return: float = 0.0
    avg_holding_days: float = 0.0
    bars: int = 0
    exposure: float = 0.0
    total_cost: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def summarize(returns: pd.Series, position: pd.Series | None = None,
              asset_class: str = "stocks", risk_free_rate: float = 0.0,
              total_cost: float = 0.0) -> PerformanceSummary:
    ppy = periods_per_year(asset_class)
    r = _clean(returns)
    summary = PerformanceSummary(
        total_return=total_return(r),
        cagr=cagr(r, ppy),
        sharpe=sharpe_ratio(r, risk_free_rate, ppy),
        sortino=sortino_ratio(r, risk_free_rate, ppy),
        max_drawdown=max_drawdown(r),
        calmar=calmar_ratio(r, ppy),
        annual_vol=annual_volatility(r, ppy),
        bars=int(len(r)),
        total_cost=float(total_cost),
    )
    if position is not None and len(position) > 0:
        stats = trade_stats(position, returns)
        summary.trades = stats["trades"]
        summary.win_rate = stats["win_rate"]
        summary.avg_trade_return = stats["avg_trade_return"]
        summary.avg_holding_days = stats["avg_holding_days"]
        pos = pd.Series(position).fillna(0.0)
        summary.exposure = float((pos != 0).mean())
    return summary
