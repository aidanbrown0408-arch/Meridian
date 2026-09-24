"""Phase 1 smoke tests.

Run with:  python -m pytest tests -q      (or)  python tests/test_phase1.py

These target the invariants that would silently corrupt every downstream number
if broken: lookahead bias, cost accounting, and metric definitions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.backtest_agent import BacktestAgent  # noqa: E402
from agents.data_agent import DataAgent  # noqa: E402
from backtester.engine import BacktestEngine, CostModel  # noqa: E402
from strategies.base import build_strategies  # noqa: E402
from utils import metrics  # noqa: E402
from utils.config import load_config  # noqa: E402

CONFIG = load_config()


def _ramp(n: int = 400, start: float = 100.0, step: float = 0.5) -> pd.DataFrame:
    """A monotonically rising price series. Every trend-follower should be long
    on it and every mean-reversion trader should be flat."""
    index = pd.bdate_range(end=pd.Timestamp("2026-01-01"), periods=n)
    close = start + step * np.arange(n)
    return pd.DataFrame({"open": close, "high": close * 1.001,
                         "low": close * 0.999, "close": close,
                         "volume": 1e6}, index=index)


def _noise(n: int = 500, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    index = pd.bdate_range(end=pd.Timestamp("2026-01-01"), periods=n)
    return pd.DataFrame({"open": close, "high": close * 1.005,
                         "low": close * 0.995, "close": close,
                         "volume": 1e6}, index=index)


def test_no_lookahead():
    """Changing the final bar must not alter any earlier position. If it does,
    a strategy is peeking at data it could not have had."""
    bars = _noise()
    tampered = bars.copy()
    tampered.iloc[-1] = tampered.iloc[-1] * 3.0

    for callsign, strategy in build_strategies(CONFIG).items():
        base = strategy.positions(bars)
        after = strategy.positions(tampered)
        pd.testing.assert_series_equal(
            base.iloc[:-1], after.iloc[:-1],
            check_names=False,
            obj=f"{callsign} positions changed retroactively",
        )


def test_signals_applied_next_bar():
    """Position must lag the signal by exactly one bar."""
    bars = _noise()
    for callsign, strategy in build_strategies(CONFIG).items():
        signals = strategy.generate_signals(bars).fillna(0.0).clip(0, 1)
        positions = strategy.positions(bars)
        assert positions.iloc[0] == 0.0, f"{callsign} traded on bar 0"
        pd.testing.assert_series_equal(
            signals.shift(1).fillna(0.0), positions,
            check_names=False, obj=f"{callsign} shift",
        )


def test_long_or_flat_only():
    bars = _noise()
    for callsign, strategy in build_strategies(CONFIG).items():
        pos = strategy.positions(bars)
        assert pos.between(0.0, 1.0).all(), f"{callsign} left the [0,1] range"


def test_warmup_is_respected():
    """No strategy may hold a position before it has enough history."""
    bars = _noise()
    engine = BacktestEngine(CONFIG)
    for callsign, strategy in build_strategies(CONFIG).items():
        result = engine.run(strategy, bars, "TEST", "stocks", "test")
        assert (result.position.iloc[: strategy.warmup] == 0).all(), \
            f"{callsign} traded during warmup"


def test_trend_followers_ride_a_ramp():
    """Sanity check on direction: on a straight-line rally the trend traders
    should be fully long and the mean-reversion traders should be flat."""
    bars = _ramp()
    built = build_strategies(CONFIG)
    for callsign in ("ORBIT", "FLUX", "SURGE", "SPARK"):
        assert built[callsign].positions(bars).iloc[-1] == 1.0, f"{callsign} missed the trend"
    for callsign in ("REVERT", "ANCHOR"):
        assert built[callsign].positions(bars).iloc[-1] == 0.0, \
            f"{callsign} bought into a straight-line rally"


def test_round_trip_cost_matches_spec():
    """Stocks: 20 bps round trip. Crypto: 40 bps."""
    stock = CostModel.from_config(CONFIG, "stocks")
    crypto = CostModel.from_config(CONFIG, "crypto")
    assert abs(stock.round_trip_bps - 20.0) < 1e-9
    assert abs(crypto.round_trip_bps - 40.0) < 1e-9

    # One in-and-out on a flat price should cost exactly one round trip.
    position = pd.Series([0.0, 1.0, 1.0, 0.0, 0.0])
    assert abs(stock.charge(position).sum() - 20e-4) < 1e-12


def test_costs_reduce_returns():
    bars = _noise()
    engine = BacktestEngine(CONFIG)
    strategy = build_strategies(CONFIG)["SPARK"]
    result = engine.run(strategy, bars, "TEST", "stocks", "test")
    assert result.summary.total_cost > 0, "an active strategy paid nothing"
    assert result.returns.sum() < result.gross_returns.sum(), "costs were not charged"


def test_metrics_definitions():
    # A 20% peak-to-trough fall must report as 0.20.
    returns = pd.Series([0.10, -0.20, 0.05])
    assert abs(metrics.max_drawdown(returns) - 0.20) < 1e-9

    # Constant positive returns: infinite-ish Sharpe guarded to a finite number.
    flat = pd.Series([0.001] * 50)
    assert metrics.sharpe_ratio(flat) == 0.0  # zero stdev -> no signal, not a crash

    # Empty and single-value inputs must not raise.
    assert metrics.sharpe_ratio(pd.Series(dtype=float)) == 0.0
    assert metrics.cagr(pd.Series([0.01])) == 0.0

    # Trade counting: two separate holds = two trades.
    position = pd.Series([0, 1, 1, 0, 1, 0], dtype=float)
    rets = pd.Series([0, 0.01, 0.01, 0, -0.02, 0], dtype=float)
    stats = metrics.trade_stats(position, rets)
    assert stats["trades"] == 2
    assert abs(stats["win_rate"] - 0.5) < 1e-9


def test_synthetic_fallback_is_flagged_and_deterministic():
    wong = DataAgent(CONFIG)
    first = wong._synthetic("SPY", reason="test")
    second = wong._synthetic("SPY", reason="test")
    assert first.is_synthetic and not first.is_trusted
    assert len(first.bars) > 250
    np.testing.assert_allclose(first.bars["close"].to_numpy(),
                               second.bars["close"].to_numpy())
    # Different symbols must not produce identical series.
    other = wong._synthetic("QQQ", reason="test")
    assert not np.allclose(first.bars["close"].to_numpy()[:100],
                           other.bars["close"].to_numpy()[:100])
    # OHLC envelope must be coherent.
    bars = first.bars
    assert (bars["high"] >= bars["close"]).all()
    assert (bars["low"] <= bars["close"]).all()
    assert (bars["close"] > 0).all()


def test_short_history_is_blocked_not_zeroed():
    engine = BacktestEngine(CONFIG)
    strategy = build_strategies(CONFIG)["ORBIT"]  # needs 101 bars
    result = engine.run(strategy, _noise(n=50), "TEST", "stocks", "test")
    assert result.blocked and "insufficient history" in result.block_reason
    assert not result.is_trusted


def test_full_pipeline_shape():
    """Leo produces one row per strategy/ticker, and synthetic rows are untrusted."""
    wong = DataAgent(CONFIG)
    market = {s: wong._synthetic(s) for s in CONFIG.universe}
    leo = BacktestAgent(CONFIG)
    results = leo.run_all(market)
    assert len(results) == len(CONFIG.universe) * len(leo.strategies)
    assert all(not r.is_trusted for r in results)
    assert len(leo.benchmarks(market)) == len(CONFIG.universe)


def test_compliance_block_propagates():
    wong = DataAgent(CONFIG)
    market = {s: wong._synthetic(s) for s in CONFIG.universe}
    leo = BacktestAgent(CONFIG)
    results = leo.run_all(market, blocked={"QQQ": "stale feed"})
    qqq = [r for r in results if r.symbol == "QQQ"]
    assert qqq and all(r.blocked and r.block_reason == "stale feed" for r in qqq)


def test_crypto_exchange_is_reachable_from_the_us():
    """api.binance.com returns HTTP 451 to US users, which silently put
    BTC/USDT on synthetic data for every run until 2026-09-23. Guard against
    switching back."""
    exchange = CONFIG.get("data.crypto_exchange")
    assert exchange != "binance", "Binance.com blocks US users (HTTP 451)"
    try:
        import ccxt
    except ImportError:
        return
    assert hasattr(ccxt, exchange), f"ccxt has no exchange named {exchange!r}"


def test_open_crypto_candle_is_dropped():
    """At the 8:15 PM Eastern run it's already 00:15 UTC: the newest candle is
    15 minutes old. Only fully closed UTC days may reach the strategies."""
    idx = pd.date_range(end=pd.Timestamp("2026-09-24"), periods=5, freq="D")
    bars = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                         "volume": 1.0}, index=idx)
    now = pd.Timestamp("2026-09-24 00:15", tz="UTC")
    out = DataAgent._drop_open_crypto_bar("BTC/USDT", bars, now=now)
    assert out.index[-1] == pd.Timestamp("2026-09-23")
    assert len(out) == 4
    # Mid-day UTC: same rule, today's candle still forming.
    out = DataAgent._drop_open_crypto_bar("BTC/USDT", bars,
                                          now=pd.Timestamp("2026-09-24 20:15", tz="UTC"))
    assert out.index[-1] == pd.Timestamp("2026-09-23")
    # A frame that already ends yesterday is untouched.
    assert len(DataAgent._drop_open_crypto_bar("BTC/USDT", out, now=now)) == 4


def test_crypto_fetch_never_returns_the_open_candle(monkeypatch):
    wong = DataAgent(CONFIG)
    wong.use_cache = False
    today = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
    idx = pd.date_range(end=today, periods=300, freq="D")
    raw = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
                        "volume": 1.0}, index=idx)
    monkeypatch.setattr(DataAgent, "_fetch_crypto", lambda self, s: raw)
    data = wong.fetch("BTC/USDT")
    assert data.data_source != "synthetic"
    assert data.bars.index[-1] < today


def test_ledger_labels_use_eastern_dates():
    from datetime import datetime, timezone
    from utils.dates import local_date
    # 8:15 PM EDT on Sep 23 == 00:15 UTC on Sep 24 -> labelled Sep 23.
    assert local_date(CONFIG, datetime(2026, 9, 24, 0, 15, tzinfo=timezone.utc)) == "2026-09-23"
    # 8:15 PM EST on Nov 2 == 01:15 UTC Nov 3 -> Nov 2.
    assert local_date(CONFIG, datetime(2026, 11, 3, 1, 15, tzinfo=timezone.utc)) == "2026-11-02"
    assert local_date(None, datetime(2026, 9, 23, 16, 0, tzinfo=timezone.utc)) == "2026-09-23"


def test_install_script_schedules_after_the_utc_close():
    script = (Path(__file__).resolve().parent.parent / "scripts"
              / "install_daily_options.sh").read_text()
    assert "<key>Hour</key><integer>20</integer><key>Minute</key><integer>15</integer>" in script


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL  {name}: {exc}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
