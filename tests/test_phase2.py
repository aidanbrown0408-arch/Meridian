"""Phase 2 smoke tests: walk-forward validation, compliance checks, risk
allocation.

Run with:  python -m pytest tests -q      (or)  python tests/test_phase2.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.compliance_agent import ComplianceAgent  # noqa: E402
from agents.data_agent import MarketData  # noqa: E402
from agents.risk_agent import RiskAgent, TraderCandidate  # noqa: E402
from backtester.engine import BacktestResult  # noqa: E402
from backtester.walkforward import WalkForwardValidator  # noqa: E402
from strategies.base import build_strategies  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.metrics import PerformanceSummary  # noqa: E402

CONFIG = load_config()


def _ramp(n: int = 400, start: float = 100.0, step: float = 0.5) -> pd.DataFrame:
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


def _clean_bars(n: int = 60, end: pd.Timestamp | None = None) -> pd.DataFrame:
    """Gap-free, positive-price, calm-moves bars -- should clear every
    compliance check as-is."""
    end = end or pd.Timestamp.utcnow().normalize().tz_localize(None)
    index = pd.bdate_range(end=end, periods=n)
    close = np.linspace(100.0, 105.0, n)
    return pd.DataFrame({"open": close, "high": close * 1.001,
                         "low": close * 0.999, "close": close,
                         "volume": 1e6}, index=index)


def _market_data(symbol: str, bars: pd.DataFrame, asset_class: str = "stocks") -> MarketData:
    return MarketData(symbol, asset_class, bars, "test", datetime.now(timezone.utc))


def _fake_result(position_last: float) -> BacktestResult:
    return BacktestResult(
        strategy="X", symbol="Y", asset_class="stocks", data_source="test",
        summary=PerformanceSummary(), position=pd.Series([position_last]),
    )


# ------------------------------------------------------------------ walk-forward

def test_walkforward_splits_into_configured_folds():
    validator = WalkForwardValidator(CONFIG)
    strategy = build_strategies(CONFIG)["FLUX"]
    result = validator.validate(strategy, _noise(), "TEST", "stocks")
    assert result.folds_total == int(CONFIG.get("validation.folds"))
    assert len(result.folds) == result.folds_total


def test_walkforward_majority_rule_is_3_of_5():
    validator = WalkForwardValidator(CONFIG)
    assert validator.min_folds_passing == 3
    assert validator.folds_n == 5


def test_walkforward_too_short_history_fails_cleanly():
    validator = WalkForwardValidator(CONFIG)
    strategy = build_strategies(CONFIG)["ORBIT"]
    result = validator.validate(strategy, _noise(n=3), "TEST", "stocks")
    assert not result.passed
    assert result.folds_passed == 0


def test_walkforward_no_lookahead_across_folds():
    """A fold's grade must not change if only *later* folds' data changes."""
    validator = WalkForwardValidator(CONFIG)
    strategy = build_strategies(CONFIG)["FLUX"]
    bars = _noise()
    tampered = bars.copy()
    tampered.iloc[-1] = tampered.iloc[-1] * 3.0

    base = validator.validate(strategy, bars, "TEST", "stocks")
    after = validator.validate(strategy, tampered, "TEST", "stocks")
    for b, a in zip(base.folds[:-1], after.folds[:-1]):
        assert b.passed == a.passed and abs(b.sharpe - a.sharpe) < 1e-9


def _oscillation(n: int = 600, base_step: float = 0.05, amp: float = 15.0,
                 period: int = 30) -> pd.DataFrame:
    """A repeating bounce, tailor-made for a mean-reversion trader: enough
    round trips per fold to clear the minimum trade count, with edges sharp
    enough to be genuinely profitable rather than noise."""
    index = pd.bdate_range(end=pd.Timestamp("2026-01-01"), periods=n)
    t = np.arange(n)
    close = 100 + base_step * t + amp * np.sin(2 * np.pi * t / period)
    return pd.DataFrame({"open": close, "high": close * 1.002,
                         "low": close * 0.998, "close": close,
                         "volume": 1e6}, index=index)


def test_walkforward_a_real_edge_clears_the_bar():
    """On data with a genuine, repeatable edge, the majority rule must
    actually be satisfiable -- otherwise the 3-of-5 bar would be
    miscalibrated and nothing could ever pass."""
    validator = WalkForwardValidator(CONFIG)
    strategy = build_strategies(CONFIG)["REVERT"]
    result = validator.validate(strategy, _oscillation(), "TEST", "stocks")
    assert result.folds_passed >= 3
    assert result.passed


# ------------------------------------------------------------------ compliance

def test_compliance_clean_data_passes():
    david = ComplianceAgent(CONFIG)
    report = david.check(_market_data("TEST", _clean_bars()))
    assert not report.blocked


def test_compliance_flags_negative_price():
    bars = _clean_bars()
    bars.iloc[10, bars.columns.get_loc("close")] = -1.0
    david = ComplianceAgent(CONFIG)
    report = david.check(_market_data("TEST", bars))
    assert report.blocked
    assert "price" in report.reason.lower()


def test_compliance_flags_multiday_gap():
    bars = _clean_bars(n=80)
    # Drop five consecutive business days from the middle.
    bars = bars.drop(bars.index[40:45])
    david = ComplianceAgent(CONFIG)
    report = david.check(_market_data("TEST", bars))
    assert report.blocked
    assert "gap" in report.reason.lower()


def test_compliance_weekend_is_not_a_gap():
    """A normal weekend (already absent from a business-day index) must never
    trip the gap check on its own."""
    bars = _clean_bars(n=60)
    david = ComplianceAgent(CONFIG)
    report = david.check(_market_data("TEST", bars))
    assert not report.blocked


def test_compliance_flags_stale_feed():
    stale_end = pd.Timestamp.utcnow().normalize().tz_localize(None) - pd.Timedelta(days=20)
    bars = _clean_bars(n=60, end=stale_end)
    david = ComplianceAgent(CONFIG)
    report = david.check(_market_data("TEST", bars))
    assert report.blocked
    assert "old" in report.reason.lower()


def test_compliance_flags_implausible_move():
    bars = _clean_bars()
    bars.iloc[15, bars.columns.get_loc("close")] *= 3.0
    david = ComplianceAgent(CONFIG)
    report = david.check(_market_data("TEST", bars))
    assert report.blocked
    assert "move" in report.reason.lower()


def test_compliance_blocked_symbols_maps_reason():
    market = {
        "GOOD": _market_data("GOOD", _clean_bars()),
        "BAD": _market_data("BAD", _clean_bars(n=60,
            end=pd.Timestamp.utcnow().normalize().tz_localize(None) - pd.Timedelta(days=30))),
    }
    david = ComplianceAgent(CONFIG)
    blocked = david.blocked_symbols(market)
    assert "BAD" in blocked and "GOOD" not in blocked


# ------------------------------------------------------------------ risk agent

def _candidate(strategy: str, symbol: str, sharpe: float, vol: float = 0.10,
               eligible: bool = True, drawdown: float = 0.05) -> TraderCandidate:
    return TraderCandidate(
        strategy=strategy, symbol=symbol, asset_class="stocks", walkforward=None,
        backtest_max_drawdown=drawdown, annual_vol=vol, sharpe=sharpe, eligible=eligible,
    )


def test_risk_agent_caps_live_roster():
    charles = RiskAgent(CONFIG)
    candidates = [
        _candidate("ORBIT", "SPY", 2.0), _candidate("FLUX", "SPY", 1.8),
        _candidate("SURGE", "SPY", 1.5), _candidate("SPARK", "SPY", 1.2),
    ]
    live, benched = charles._pick_roster(candidates)
    assert len(live) == charles.max_live_traders == 3
    assert live == ["ORBIT", "FLUX", "SURGE"]
    assert "SPARK" in benched


def test_risk_agent_correlated_pair_shares_one_slot():
    """ANCHOR and REVERT must never sum to more than one solo trader's
    risk-parity share (spec §7's correlated-pair budgeting rule)."""
    charles = RiskAgent(CONFIG)
    live = ["ANCHOR", "REVERT", "ORBIT"]
    candidates = [
        _candidate("ANCHOR", "SPY", 1.0, vol=0.10),
        _candidate("REVERT", "SPY", 1.0, vol=0.10),
        _candidate("ORBIT", "SPY", 1.0, vol=0.10),
    ]
    slots = charles._build_slots(live)
    weights = charles._risk_parity(slots, candidates)

    combined = weights["ANCHOR"] + weights["REVERT"]
    assert abs(combined - weights["ORBIT"]) < 1e-9, \
        "ANCHOR+REVERT combined must equal one solo trader's slot, not two"
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_risk_agent_risk_parity_favors_lower_vol():
    charles = RiskAgent(CONFIG)
    live = ["ORBIT", "FLUX"]
    candidates = [
        _candidate("ORBIT", "SPY", 1.0, vol=0.10),
        _candidate("FLUX", "SPY", 1.0, vol=0.30),
    ]
    slots = charles._build_slots(live)
    weights = charles._risk_parity(slots, candidates)
    assert weights["ORBIT"] > weights["FLUX"], "lower-vol trader should get more capital"


def test_risk_agent_nets_same_ticker_and_caps():
    charles = RiskAgent(CONFIG)
    live = ["ORBIT", "FLUX"]
    eligible_pairs = {("ORBIT", "QQQ"), ("FLUX", "QQQ")}
    weights = {"ORBIT": 0.20, "FLUX": 0.20}
    market = {"QQQ": object()}
    by_key = {("ORBIT", "QQQ"): _fake_result(1.0), ("FLUX", "QQQ"): _fake_result(1.0)}

    netted = charles._net_positions(live, eligible_pairs, weights, market, by_key)
    qqq = netted["QQQ"]
    assert qqq.capped
    assert abs(qqq.target_weight - charles.max_position_pct) < 1e-9
    assert set(qqq.contributors) == {"ORBIT", "FLUX"}


def test_risk_agent_only_eligible_pairs_contribute():
    charles = RiskAgent(CONFIG)
    live = ["ORBIT", "FLUX"]
    eligible_pairs = {("ORBIT", "QQQ")}  # FLUX failed validation on QQQ
    weights = {"ORBIT": 0.20, "FLUX": 0.20}
    market = {"QQQ": object()}
    by_key = {("ORBIT", "QQQ"): _fake_result(1.0), ("FLUX", "QQQ"): _fake_result(1.0)}

    netted = charles._net_positions(live, eligible_pairs, weights, market, by_key)
    assert netted["QQQ"].contributors == ["ORBIT"]
    assert abs(netted["QQQ"].target_weight - 0.20) < 1e-9


def test_risk_agent_end_to_end_shape():
    """Full run() over synthetic data must produce one candidate per
    non-blocked strategy/symbol pair and never crash on an all-fail day."""
    from agents.backtest_agent import BacktestAgent
    from agents.data_agent import DataAgent

    wong = DataAgent(CONFIG)
    market = {s: wong._synthetic(s) for s in CONFIG.universe}
    leo = BacktestAgent(CONFIG)
    results = leo.run_all(market)
    charles = RiskAgent(CONFIG)
    report = charles.run(leo.strategies, market, results)

    assert len(report.candidates) == len(CONFIG.universe) * len(leo.strategies)
    assert len(report.live_traders) <= charles.max_live_traders
    assert abs(sum(report.capital_weights.values()) - (1.0 if report.live_traders else 0.0)) < 1e-6


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
