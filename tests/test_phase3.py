"""Phase 3 smoke tests: regime classification, the regime-mismatch capital
cut + confirmation logic, chart helpers, and the reporting agent.

Run with:  python -m pytest tests -q      (or)  python tests/test_phase3.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.compliance_agent import ComplianceAgent  # noqa: E402
from agents.data_agent import DataAgent, MarketData  # noqa: E402
from agents.regime_agent import RegimeAgent, TraderRegimeAdjustment  # noqa: E402
from agents.reporting_agent import ReportingAgent  # noqa: E402
from agents.risk_agent import NettedPosition, RiskReport  # noqa: E402
from backtester.engine import BacktestResult  # noqa: E402
from strategies.base import build_strategies  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.metrics import PerformanceSummary  # noqa: E402
from utils.slack import post_message  # noqa: E402
from utils.svg_charts import bar_chart, line_chart  # noqa: E402

CONFIG = load_config()


def _trend(n: int = 400, start: float = 100.0, annual_drift: float = 0.6,
          annual_vol: float = 0.25, seed: int = 11) -> pd.DataFrame:
    """A geometric random walk with strong upward drift -- realistic enough
    that its *percentage* volatility stays roughly stationary (unlike a
    fixed-dollar-step ramp, whose % moves shrink as price compounds up and
    would wrongly starve the vol-percentile check). Should classify trending."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(end=pd.Timestamp("2026-01-01"), periods=n)
    daily_vol = annual_vol / np.sqrt(252)
    daily_drift = annual_drift / 252 - 0.5 * daily_vol ** 2
    close = start * np.exp(np.cumsum(rng.normal(daily_drift, daily_vol, n)))
    return pd.DataFrame({"open": close, "high": close * 1.003, "low": close * 0.997,
                         "close": close, "volume": 1e6}, index=index)


def _chop(n: int = 400, base: float = 100.0, amp: float = 4.0, period: int = 8,
         seed: int = 3) -> pd.DataFrame:
    """A tight, fast-oscillating bounce with a little noise -- should
    classify as choppy: low ADX, an MA slope that keeps flipping sign, price
    staying inside its own recent range."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(end=pd.Timestamp("2026-01-01"), periods=n)
    t = np.arange(n)
    close = base + amp * np.sin(2 * np.pi * t / period) + rng.normal(0, 0.2, n)
    return pd.DataFrame({"open": close, "high": close * 1.003, "low": close * 0.997,
                         "close": close, "volume": 1e6}, index=index)


def _fake_backtest_result(symbol: str, tail_positions: list[float]) -> BacktestResult:
    index = pd.bdate_range(end=pd.Timestamp("2026-01-01"), periods=len(tail_positions))
    return BacktestResult(
        strategy="X", symbol=symbol, asset_class="stocks", data_source="test",
        summary=PerformanceSummary(),
        position=pd.Series(tail_positions, index=index),
    )


# ------------------------------------------------------------------ classification

def test_regime_classifies_a_clean_trend_as_trending():
    greg = RegimeAgent(CONFIG)
    c = greg.classify(_trend(), symbol="TEST")
    assert c.regime == "trending", c.detail
    assert c.adx > greg.adx_trending
    assert c.slope_consistent


def test_regime_classifies_a_tight_bounce_as_choppy():
    greg = RegimeAgent(CONFIG)
    c = greg.classify(_chop(), symbol="TEST")
    assert c.regime == "choppy", c.detail
    assert c.adx < greg.adx_choppy
    assert not c.slope_consistent


def test_regime_short_history_is_undecided():
    greg = RegimeAgent(CONFIG)
    c = greg.classify(_trend(n=10), symbol="TEST")
    assert c.regime == "undecided"


def test_regime_is_per_ticker_not_portfolio_wide():
    greg = RegimeAgent(CONFIG)
    market = {"TREND": _market_data("TREND", _trend()), "CHOP": _market_data("CHOP", _chop())}
    regimes = greg.classify_universe(market)
    assert regimes["TREND"].regime == "trending"
    assert regimes["CHOP"].regime == "choppy"


def _market_data(symbol: str, bars: pd.DataFrame) -> MarketData:
    return MarketData(symbol, "stocks", bars, "test", datetime.now(timezone.utc))


# ------------------------------------------------------------------ mismatch adjustment

def test_matched_trader_gets_full_weight_no_confirmation_needed():
    greg = RegimeAgent(CONFIG)
    orbit = build_strategies(CONFIG)["ORBIT"]  # best_regimes = ("trending",)
    classification = greg.classify(_trend(), symbol="TEST")
    position = pd.Series([1.0])  # single day long, no history to confirm against
    adj = greg._adjust_one(orbit, "TEST", classification, base_weight=0.30, position=position)
    assert adj.matched
    assert adj.confirmed
    assert adj.active
    assert abs(adj.adjusted_weight - 0.30) < 1e-9


def test_mismatched_trader_weight_is_halved():
    greg = RegimeAgent(CONFIG)
    revert = build_strategies(CONFIG)["REVERT"]  # best_regimes = ("choppy",)
    classification = greg.classify(_trend(), symbol="TEST")  # trending -> mismatch
    assert classification.regime == "trending"
    position = pd.Series([1.0, 1.0, 1.0])
    adj = greg._adjust_one(revert, "TEST", classification, base_weight=0.30, position=position)
    assert not adj.matched
    assert abs(adj.adjusted_weight - 0.30 * greg.capital_cut) < 1e-9


def test_mismatched_trader_needs_confirmation_before_acting():
    greg = RegimeAgent(CONFIG)
    revert = build_strategies(CONFIG)["REVERT"]
    classification = greg.classify(_trend(), symbol="TEST")

    # Flipped long only yesterday: not yet confirmed (needs 2 consecutive days).
    flaky = pd.Series([0.0, 1.0])
    adj_unconfirmed = greg._adjust_one(revert, "TEST", classification, 0.30, flaky)
    assert not adj_unconfirmed.confirmed
    assert not adj_unconfirmed.active

    # Long for the last two days in a row: confirmed.
    steady = pd.Series([0.0, 1.0, 1.0])
    adj_confirmed = greg._adjust_one(revert, "TEST", classification, 0.30, steady)
    assert adj_confirmed.confirmed
    assert adj_confirmed.active


def test_undecided_regime_means_full_weight_for_everyone():
    greg = RegimeAgent(CONFIG)
    revert = build_strategies(CONFIG)["REVERT"]
    classification = greg.classify(_trend(n=5), symbol="TEST")  # too short -> undecided
    assert classification.regime == "undecided"
    position = pd.Series([1.0])
    adj = greg._adjust_one(revert, "TEST", classification, 0.30, position)
    assert adj.matched
    assert abs(adj.adjusted_weight - 0.30) < 1e-9


def test_no_trader_is_ever_fully_benched_by_regime():
    """Spec §8: mismatched traders are tilted, never zeroed out, as long as
    their position has been confirmed."""
    greg = RegimeAgent(CONFIG)
    revert = build_strategies(CONFIG)["REVERT"]
    classification = greg.classify(_trend(), symbol="TEST")
    position = pd.Series([1.0, 1.0, 1.0])
    adj = greg._adjust_one(revert, "TEST", classification, 0.30, position)
    assert adj.adjusted_weight > 0.0


# ------------------------------------------------------------------ netting

def test_regime_netting_sums_active_adjustments_and_caps():
    greg = RegimeAgent(CONFIG)
    adjustments = [
        TraderRegimeAdjustment("A", "QQQ", "trending", True, True, 0.20, 0.20, True),
        TraderRegimeAdjustment("B", "QQQ", "trending", True, True, 0.20, 0.20, True),
        TraderRegimeAdjustment("C", "QQQ", "choppy", False, False, 0.20, 0.10, False),
    ]
    market = {"QQQ": object()}
    netted = greg._net_positions(adjustments, market)
    assert set(netted["QQQ"].contributors) == {"A", "B"}
    assert netted["QQQ"].capped
    assert abs(netted["QQQ"].target_weight - greg.max_position_pct) < 1e-9


# ------------------------------------------------------------------ charts

def test_line_chart_produces_svg_with_all_series():
    svg = line_chart({"a": [1, 2, 3], "b": [3, 2, 1]}, title="test")
    assert svg.startswith("<svg")
    assert svg.count("<polyline") == 2


def test_line_chart_handles_empty_input():
    svg = line_chart({})
    assert svg.startswith("<svg")


def test_bar_chart_colors_by_sign():
    svg = bar_chart(["A", "B"], [5.0, -3.0])
    assert "#7FBF7F" in svg  # gain
    assert "#C97A7A" in svg  # loss


def test_slack_post_without_webhook_returns_false_and_does_not_raise():
    assert post_message(None, "hello") is False


# ------------------------------------------------------------------ reporting agent

def test_reporting_agent_end_to_end():
    """Full run() over synthetic data -- must produce an HTML file, a standup
    message, and never crash even on an all-fail, all-benched day."""
    from agents.backtest_agent import BacktestAgent
    from agents.regime_agent import RegimeAgent as _RegimeAgent
    from agents.risk_agent import RiskAgent

    wong = DataAgent(CONFIG)
    market = {s: wong._synthetic(s) for s in CONFIG.universe}
    david = ComplianceAgent(CONFIG)
    compliance = david.review(market)
    leo = BacktestAgent(CONFIG)
    results = leo.run_all(market)
    benchmarks = leo.benchmarks(market)
    charles = RiskAgent(CONFIG)
    risk_report = charles.run(leo.strategies, market, results)
    greg = _RegimeAgent(CONFIG)
    regime_report = greg.run(leo.strategies, market, results, risk_report)

    george = ReportingAgent(CONFIG)
    dashboard = george.run(market, compliance, results, benchmarks, risk_report,
                           regime_report, write_file=True, post_slack=False)

    assert dashboard.html.startswith("<!DOCTYPE") or "<html" in dashboard.html
    assert "MERIDIAN CAPITAL" in dashboard.html
    assert dashboard.html_path is not None and dashboard.html_path.exists()
    assert not dashboard.posted_to_slack
    assert "Wong (Data)" in dashboard.standup_text
    assert "Greg (Regime)" in dashboard.standup_text


def test_reporting_agent_handles_nothing_live():
    """The dashboard must render even when no trader is live and nothing is
    blocked -- the empty-state path spec §6 insists on."""
    george = ReportingAgent(CONFIG)
    market = {"SPY": _market_data("SPY", _trend())}
    compliance = {"SPY": ComplianceAgent(CONFIG).check(market["SPY"])}
    empty_risk = RiskReport(starting_capital=5000.0)
    from agents.regime_agent import RegimeReport
    empty_regime = RegimeReport(regimes={}, adjustments=[],
                                netted_positions={"SPY": NettedPosition("SPY", 0.0, False, [])})
    dashboard = george.run(market, compliance, [], {}, empty_risk, empty_regime,
                           write_file=False, post_slack=False)
    assert "MERIDIAN CAPITAL" in dashboard.html
    assert "No target positions" not in dashboard.standup_text  # standup doesn't echo HTML copy


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
