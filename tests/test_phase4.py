"""Phase 4 smoke tests: the paper broker's ledger math, the killswitch, and
the lifecycle (benching) agent's operator-approval gate.

Run with:  python -m pytest tests -q      (or)  python tests/test_phase4.py
"""

from __future__ import annotations

import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.data_agent import MarketData  # noqa: E402
from agents.lifecycle_agent import LifecycleAgent, TraderLifecycle  # noqa: E402
from agents.portfolio_agent import PaperBroker, Position  # noqa: E402
from agents.regime_agent import TraderRegimeAdjustment  # noqa: E402
from agents.risk_agent import NettedPosition  # noqa: E402
from utils.config import load_config  # noqa: E402

CONFIG = load_config()


def _tmp_broker() -> PaperBroker:
    broker = PaperBroker(CONFIG)
    broker.ledger_path = Path(tempfile.mkdtemp()) / "ledger.json"
    return broker


def _tmp_lifecycle() -> LifecycleAgent:
    lifecycle = LifecycleAgent(CONFIG)
    lifecycle.state_path = Path(tempfile.mkdtemp()) / "bench_state.json"
    return lifecycle


def _bars(price: float, n: int = 5) -> pd.DataFrame:
    index = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
    close = np.full(n, price, dtype="float64")
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                         "volume": 1e6}, index=index)


def _market(prices: dict[str, float], asset_classes: dict[str, str] | None = None
           ) -> dict[str, MarketData]:
    asset_classes = asset_classes or {}
    return {
        s: MarketData(s, asset_classes.get(s, "stocks"), _bars(p), "test",
                     datetime.now(timezone.utc))
        for s, p in prices.items()
    }


def _netted(symbol: str, weight: float, capped: bool = False) -> dict[str, NettedPosition]:
    return {symbol: NettedPosition(symbol, weight, capped, ["TEST"])}


# ------------------------------------------------------------------ paper broker

def test_fresh_ledger_starts_at_configured_capital():
    broker = _tmp_broker()
    ledger = broker.load_ledger()
    assert ledger.cash == broker.starting_capital
    assert ledger.positions == {}
    assert not ledger.halted


def test_execute_buys_toward_target_weight():
    broker = _tmp_broker()
    market = _market({"SPY": 100.0})
    ledger = broker.execute(market, _netted("SPY", 0.40))
    assert "SPY" in ledger.positions
    expected_value = 0.40 * broker.starting_capital
    actual_value = ledger.positions["SPY"].shares * 100.0
    # Within one cost-model bps of the target -- costs eat a hair of the fill.
    assert abs(actual_value - expected_value) / expected_value < 0.01
    assert ledger.cash < broker.starting_capital  # cash spent on the buy + cost


def test_execute_sells_toward_lower_target():
    broker = _tmp_broker()
    market = _market({"SPY": 100.0})
    ledger = broker.execute(market, _netted("SPY", 0.40))
    shares_before = ledger.positions["SPY"].shares
    ledger = broker.execute(market, _netted("SPY", 0.10))
    assert ledger.positions["SPY"].shares < shares_before
    assert ledger.cash > 0


def test_execute_flattens_to_zero_removes_position():
    broker = _tmp_broker()
    market = _market({"SPY": 100.0})
    broker.execute(market, _netted("SPY", 0.40))
    ledger = broker.execute(market, _netted("SPY", 0.0))
    assert "SPY" not in ledger.positions


def test_cash_constraint_scales_down_rather_than_overdrawing():
    broker = _tmp_broker()
    market = _market({"SPY": 100.0, "QQQ": 100.0, "BTC/USDT": 100.0},
                     {"BTC/USDT": "crypto"})
    # Three simultaneous 60% targets can't all be funded from one account.
    netted = {
        "SPY": NettedPosition("SPY", 0.60, False, ["A"]),
        "QQQ": NettedPosition("QQQ", 0.60, False, ["B"]),
        "BTC/USDT": NettedPosition("BTC/USDT", 0.60, False, ["C"]),
    }
    ledger = broker.execute(market, netted)
    assert ledger.cash >= -1e-6, "paper account went into overdraft"


def test_dust_trades_are_skipped():
    broker = _tmp_broker()
    market = _market({"SPY": 100.0})
    ledger = broker.execute(market, _netted("SPY", 0.40))
    trades_before = len(ledger.trades)
    # A 0.01% nudge is far below min_trade_dollars -- must not generate a trade.
    ledger = broker.execute(market, _netted("SPY", 0.4001))
    assert len(ledger.trades) == trades_before


def test_weighted_average_entry_price_on_increase():
    broker = _tmp_broker()
    market_low = _market({"SPY": 100.0})
    ledger = broker.execute(market_low, _netted("SPY", 0.20))
    entry_first = ledger.positions["SPY"].entry_price
    assert abs(entry_first - 100.0) < 1.0

    market_high = _market({"SPY": 200.0})
    ledger = broker.execute(market_high, _netted("SPY", 0.60))
    # Averaging in a much higher price must push entry price up from 100,
    # but a full position wasn't bought at 200 either.
    assert ledger.positions["SPY"].entry_price > entry_first
    assert ledger.positions["SPY"].entry_price < 200.0


def test_ledger_persists_across_broker_instances():
    broker = _tmp_broker()
    market = _market({"SPY": 100.0})
    broker.execute(market, _netted("SPY", 0.30))

    reloaded = PaperBroker(CONFIG)
    reloaded.ledger_path = broker.ledger_path
    ledger = reloaded.load_ledger()
    assert "SPY" in ledger.positions
    assert len(ledger.trades) == 1


def test_halted_ledger_refuses_to_trade():
    broker = _tmp_broker()
    ledger = broker.load_ledger()
    ledger.halted = True
    ledger.halt_reason = "test halt"
    broker.save_ledger(ledger)

    market = _market({"SPY": 100.0})
    result = broker.execute(market, _netted("SPY", 0.50))
    assert result.positions == {}
    assert result.cash == broker.starting_capital


def test_killswitch_flattens_every_position_and_halts():
    broker = _tmp_broker()
    market = _market({"SPY": 100.0})
    broker.execute(market, _netted("SPY", 0.40))
    assert broker.load_ledger().positions  # sanity: something is open

    ledger = broker.killswitch(market, reason="test kill")
    assert ledger.positions == {}
    assert ledger.halted
    assert ledger.halt_reason == "test kill"
    assert any(t.reason == "killswitch flatten" for t in ledger.trades)


def test_killswitch_without_market_data_uses_entry_price_fallback():
    broker = _tmp_broker()
    ledger = broker.load_ledger()
    ledger.positions["SPY"] = Position("SPY", "stocks", 2.0, 150.0, "2026-01-01")
    broker.save_ledger(ledger)

    result = broker.killswitch(market=None)
    assert result.positions == {}
    assert result.halted


def test_clear_halt_allows_trading_again():
    broker = _tmp_broker()
    market = _market({"SPY": 100.0})
    broker.execute(market, _netted("SPY", 0.40))
    broker.killswitch(market)
    assert broker.load_ledger().halted

    broker.clear_halt()
    ledger = broker.load_ledger()
    assert not ledger.halted

    ledger = broker.execute(market, _netted("SPY", 0.20))
    assert "SPY" in ledger.positions


def test_shadow_tracking_records_per_trader_returns():
    broker = _tmp_broker()
    market = _market({"SPY": 100.0})
    adjustments = [TraderRegimeAdjustment("ORBIT", "SPY", "trending", True, True, 0.30, 0.30, True)]
    ledger = broker.execute(market, _netted("SPY", 0.30), regime_adjustments=adjustments)
    assert "ORBIT" in ledger.trader_shadow
    assert len(ledger.trader_shadow["ORBIT"]) == 1


# ------------------------------------------------------------------ lifecycle agent

def test_consecutive_fails_increment_and_reset():
    lifecycle = _tmp_lifecycle()
    state = lifecycle.record_validation_results({"ORBIT"}, {"ORBIT": False})
    assert state["ORBIT"].consecutive_fails == 1
    state = lifecycle.record_validation_results({"ORBIT"}, {"ORBIT": False})
    assert state["ORBIT"].consecutive_fails == 2
    state = lifecycle.record_validation_results({"ORBIT"}, {"ORBIT": True})
    assert state["ORBIT"].consecutive_fails == 0


def test_bench_is_never_automatic():
    """Crossing the fail threshold must only produce a recommendation --
    validation_benched stays False until the operator explicitly applies it."""
    lifecycle = _tmp_lifecycle()
    state = {}
    for _ in range(lifecycle.fail_threshold):
        state = lifecycle.record_validation_results({"ORBIT"}, {"ORBIT": False})
    assert state["ORBIT"].consecutive_fails >= lifecycle.fail_threshold
    assert state["ORBIT"].validation_benched is False

    recs = lifecycle.recommend(state, {}, {})
    assert any(r.trader == "ORBIT" and r.trigger == "validation_fail" for r in recs)


def test_operator_apply_and_auto_reinstate():
    lifecycle = _tmp_lifecycle()
    for _ in range(lifecycle.fail_threshold):
        lifecycle.record_validation_results({"ORBIT"}, {"ORBIT": False})
    lifecycle.apply_validation_bench("ORBIT", "operator says bench")
    state = lifecycle.load_state()
    assert state["ORBIT"].validation_benched

    # Passing again must auto-reinstate -- no operator action needed for this side.
    state = lifecycle.record_validation_results({"ORBIT"}, {"ORBIT": True})
    assert not state["ORBIT"].validation_benched


def test_drift_bench_reinstatement_is_manual_only():
    lifecycle = _tmp_lifecycle()
    lifecycle.apply_drift_bench("REVERT", "structural break")
    state = lifecycle.load_state()
    assert state["REVERT"].drift_benched

    # A clean validation pass must NOT clear a drift bench.
    state = lifecycle.record_validation_results({"REVERT"}, {"REVERT": True})
    assert state["REVERT"].drift_benched

    lifecycle.clear_drift_bench("REVERT")
    assert not lifecycle.load_state()["REVERT"].drift_benched


def test_effective_benched_reports_reason():
    lifecycle = _tmp_lifecycle()
    lifecycle.apply_validation_bench("SPARK", "3 fails")
    benched, reason = lifecycle.effective_benched("SPARK")
    assert benched and "3 fails" in reason
    benched, _ = lifecycle.effective_benched("ORBIT")
    assert not benched


def test_drift_recommendation_needs_minimum_track_record():
    lifecycle = _tmp_lifecycle()
    short_history = [{"date": f"2026-01-{i:02d}", "return": -0.05} for i in range(1, 4)]
    state = {"ORBIT": TraderLifecycle()}
    recs = lifecycle.recommend(state, {"ORBIT": 1.5}, {"ORBIT": short_history})
    assert not any(r.trigger == "live_drift" for r in recs)


def test_drift_recommendation_fires_on_sustained_underperformance():
    lifecycle = _tmp_lifecycle()
    rng_returns = [-0.01] * lifecycle.drift_min_days  # steady small losses
    history = [{"date": f"day{i}", "return": r} for i, r in enumerate(rng_returns)]
    state = {"ORBIT": TraderLifecycle()}
    recs = lifecycle.recommend(state, {"ORBIT": 2.0}, {"ORBIT": history})
    assert any(r.trigger == "live_drift" and r.trader == "ORBIT" for r in recs)


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
