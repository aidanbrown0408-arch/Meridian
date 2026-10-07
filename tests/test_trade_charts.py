"""ACTIVE TRADES charts: the overlays must be the exact series each trader
decides on, fills must come from the ledgers, and P&L must reconcile to the
ledger to the cent. Every test uses tmp ledgers -- never reports/.

Run with:  python -m pytest tests/test_trade_charts.py -q
"""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.data_agent import MarketData  # noqa: E402
from agents.options_broker import OptionsBroker, OptionsLedger  # noqa: E402
from agents.options_risk_agent import OptionsProposal, RiskDecision  # noqa: E402
from agents.portfolio_agent import PaperBroker, PaperLedger, Trade  # noqa: E402
from agents.risk_agent import NettedPosition  # noqa: E402
from agents.trade_charts import (  # noqa: E402
    _bar_for_fill, build_options_charts, build_stock_charts, data_status,
    expected_last_bar, indicator_frame, put_indicator_frame, signal_from_indicators,
)
from strategies.base import build_strategies  # noqa: E402
from strategies.options_strategy import _donchian_breakdown  # noqa: E402
from utils.config import load_config  # noqa: E402

CONFIG = load_config()
REPO = Path(__file__).resolve().parent.parent


def _walk(seed: int, n: int = 400, drift: float = 0.0004, vol: float = 0.018) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    spread = np.abs(rng.normal(0, vol / 2, n)) * close
    index = pd.bdate_range(end="2026-10-02", periods=n)
    return pd.DataFrame({"open": close * (1 + rng.normal(0, vol / 4, n)),
                         "high": close + spread, "low": close - spread,
                         "close": close, "volume": 1e6}, index=index)


def _bar_sets():
    sets = [_walk(s) for s in (1, 2, 3)]
    sets.append(_walk(4, drift=0.003, vol=0.01))       # strong trend
    sets.append(_walk(5, drift=-0.002, vol=0.025))     # selloff, deep RSI lows
    cached = REPO / "data_cache" / "AAPL.csv"          # real bars when present
    if cached.exists():
        sets.append(pd.read_csv(cached, index_col=0, parse_dates=True))
    return sets


# ------------------------------------------------------------------ overlays == decisions

@pytest.mark.parametrize("callsign", sorted(build_strategies(CONFIG)))
def test_overlays_reproduce_each_strategys_signal_exactly(callsign):
    """Apply the strategy's rule to the charted series and get back
    generate_signals() bar for bar. If a strategy's formula changes and the
    chart doesn't, this fails."""
    strat = build_strategies(CONFIG)[callsign]
    for bars in _bar_sets():
        ind = indicator_frame(strat, bars)
        rebuilt = signal_from_indicators(strat, bars, ind)
        actual = strat.generate_signals(bars).astype("float64")
        pd.testing.assert_series_equal(rebuilt.fillna(0.0), actual.fillna(0.0),
                                       check_names=False)
        assert actual.sum() > 0 or callsign in ("REVERT", "ANCHOR"), "fixture never traded"


def test_put_overlay_matches_the_options_put_trigger():
    strat = build_strategies(CONFIG)["SPARK"]
    for bars in _bar_sets():
        ind = put_indicator_frame(strat, bars)
        from strategies.base import Strategy
        rebuilt = Strategy._hold(bars["close"] < ind["series"]["lower"],
                                 bars["close"] > ind["series"]["upper"])
        expected = _donchian_breakdown(bars, int(strat.params["entry_window"]),
                                       int(strat.params["exit_window"]))
        pd.testing.assert_series_equal(rebuilt, expected, check_names=False)


# ------------------------------------------------------------------ ledger attribution

def _tmp_broker() -> PaperBroker:
    b = PaperBroker(CONFIG)
    b.ledger_path = Path(tempfile.mkdtemp()) / "ledger.json"
    return b


def _market(bars: pd.DataFrame, symbol: str = "TEST") -> dict:
    return {symbol: MarketData(symbol, "stocks", bars, "test", datetime.now(timezone.utc))}


def test_paper_fills_record_traders_and_the_priced_bar():
    broker = _tmp_broker()
    bars = _walk(7)
    ledger = broker.execute(_market(bars), {"TEST": NettedPosition("TEST", 0.3, False, ["SPARK", "FLUX"])})
    t = ledger.trades[-1]
    assert t.side == "buy" and t.traders == ["SPARK", "FLUX"]
    assert t.bar_date == bars.index[-1].date().isoformat()
    assert ledger.positions["TEST"].traders == ["SPARK", "FLUX"]

    ledger = broker.execute(_market(bars), {"TEST": NettedPosition("TEST", 0.0, False, [])})
    sell = ledger.trades[-1]
    assert sell.side == "sell" and sell.traders == ["SPARK", "FLUX"], \
        "a sell records who the position was held for"
    assert "TEST" not in ledger.positions


def test_old_ledgers_without_attribution_still_load():
    old = {"starting_capital": 5000.0, "cash": 4000.0,
           "positions": {"X": {"symbol": "X", "asset_class": "stocks", "shares": 1.0,
                               "entry_price": 10.0, "entry_date": "2026-10-01"}},
           "trades": [{"date": "2026-09-30", "symbol": "X", "side": "buy", "shares": 1.0,
                       "price": 10.0, "cost": 0.01, "reason": "r"}]}
    ledger = PaperLedger.from_dict(old)
    assert ledger.positions["X"].traders == [] and ledger.trades[0].bar_date == ""


# ------------------------------------------------------------------ fills & P&L

def test_fill_without_bar_date_is_pinned_to_the_close_it_matched():
    bars = _walk(8)
    prev = bars.index[-2].date().isoformat()
    run = bars.index[-1].date().isoformat()
    price = float(bars["close"].iloc[-2])
    bar, note = _bar_for_fill(bars, run, price)
    assert bar == prev and "filled at the" in note


def test_stock_card_reconciles_to_the_ledger_to_the_cent():
    broker = _tmp_broker()
    bars = _walk(9)
    m1 = _market(bars.iloc[:-5])
    broker.execute(m1, {"TEST": NettedPosition("TEST", 0.4, False, ["SPARK"])})
    m2 = _market(bars.iloc[:-2])
    broker.execute(m2, {"TEST": NettedPosition("TEST", 0.2, False, ["SPARK"])})   # partial sell
    ledger = broker.load_ledger()
    m = _market(bars)
    out = build_stock_charts(CONFIG, m, ledger)
    assert len(out["open"]) == 1 and not out["closed"]
    card = out["open"][0]
    last = float(bars["close"].iloc[-1])
    pos = ledger.positions["TEST"]
    assert card["headline"]["raw"] == round(pos.unrealized_pnl(last), 2)
    # P&L incl. costs = equity change of the account (only one symbol traded).
    assert abs(card["pnl"][-1]["value"] - (ledger.mark_to_market({"TEST": last}) - 5000.0)) < 0.01
    fills = [mk for mk in card["markers"] if mk["shape"] in ("arrowUp", "arrowDown")]
    assert [f["time"] for f in fills] == [bars.index[-6].date().isoformat(),
                                          bars.index[-3].date().isoformat()]
    assert card["attribution"] == "recorded" and card["traders"][0]["callsign"] == "SPARK"
    assert card["overlays"] and card["overlays"][0]["label"].startswith("SPARK")


def test_card_ships_long_history_and_a_trade_focus_window():
    broker = _tmp_broker()
    bars = _walk(12, n=900)
    broker.execute(_market(bars.iloc[:-20]), {"TEST": NettedPosition("TEST", 0.4, False, ["SPARK"])})
    ledger = broker.load_ledger()
    card = build_stock_charts(CONFIG, _market(bars), ledger)["open"][0]
    dates = [c["time"] for c in card["candles"]]
    # Enough history for the 1Y / All range buttons and a weekly view...
    assert len(dates) == 756 and dates[-1] == bars.index[-1].date().isoformat()
    # ...while the chart still opens on the trade: 60 sessions before the buy.
    entry = bars.index[-21].date().isoformat()
    assert card["focus"]["to"] == dates[-1]
    assert dates.index(entry) - dates.index(card["focus"]["from"]) == 109  # 130-bar minimum wins
    # Overlays cover the whole shipped window once warmed up.
    assert card["overlays"][0]["points"][0]["time"] >= dates[0]


def test_pnl_percent_and_stats_track_the_replay():
    broker = _tmp_broker()
    bars = _walk(13)
    broker.execute(_market(bars.iloc[:-30]), {"TEST": NettedPosition("TEST", 0.4, False, ["SPARK"])})
    ledger = broker.load_ledger()
    card = build_stock_charts(CONFIG, _market(bars), ledger)["open"][0]
    pnl = card["pnl"]
    buy = ledger.trades[0]
    basis = buy.shares * buy.price + buy.cost
    for p in pnl:
        assert abs(p["pct"] - p["value"] / basis * 100) < 0.011
    st = card["pnl_stats"]
    vals = [p["value"] for p in pnl]
    assert st["current"] == vals[-1] and st["peak"] == max(vals) and st["trough"] == min(vals)
    assert st["change"] == round(vals[-1] - vals[-2], 2)
    assert st["giveback"] == (round(max(vals) - vals[-1], 2) if max(vals) > 0 else 0.0)


def test_closed_trade_card_shows_realized_pnl_after_costs():
    broker = _tmp_broker()
    bars = _walk(10)
    broker.execute(_market(bars.iloc[:-10]), {"TEST": NettedPosition("TEST", 0.4, False, ["ORBIT"])})
    ledger = broker.execute(_market(bars.iloc[:-4]), {"TEST": NettedPosition("TEST", 0.0, False, [])})
    out = build_stock_charts(CONFIG, _market(bars), ledger)
    assert not out["open"] and len(out["closed"]) == 1
    card = out["closed"][0]
    assert abs(card["headline"]["raw"] - (ledger.cash - 5000.0)) < 0.01
    assert any(o["label"] == "ORBIT slow MA" for o in card["overlays"])


def test_unattributed_position_is_inferred_and_labelled():
    bars = _walk(11)
    ledger = PaperLedger(starting_capital=5000.0, cash=4000.0)
    from agents.portfolio_agent import Position
    px = float(bars["close"].iloc[-3])
    ledger.positions["TEST"] = Position("TEST", "stocks", 1000 / px, px, "2026-09-28")
    ledger.trades.append(Trade(bars.index[-2].date().isoformat(), "TEST", "buy", 1000 / px, px, 1.0, "r"))
    out = build_stock_charts(CONFIG, _market(bars), ledger, contributors={"TEST": ["ANCHOR"]})
    card = out["open"][0]
    assert card["attribution"] == "inferred"
    assert any("attributed to today's contributor" in n for n in card["notes"])
    assert any("filled at the" in n for n in card["notes"])
    assert [o["label"] for o in card["overlays"]] == [
        "ANCHOR upper band", "ANCHOR middle (exit)", "ANCHOR lower (entry)"]


def test_revert_gets_an_rsi_pane_with_its_own_levels():
    broker = _tmp_broker()
    bars = _walk(12)
    ledger = broker.execute(_market(bars), {"TEST": NettedPosition("TEST", 0.3, False, ["REVERT"])})
    card = build_stock_charts(CONFIG, _market(bars), ledger)["open"][0]
    assert card["panes"] and [lv["value"] for lv in card["panes"][0]["levels"]] == [30.0, 55.0]
    assert all(0 <= p["value"] <= 100 for p in card["panes"][0]["points"])


# ------------------------------------------------------------------ staleness

def test_stale_stock_data_is_flagged():
    bars = _walk(13)                                  # ends Fri 2026-10-02
    md = MarketData("TEST", "stocks", bars, "yfinance", datetime.now(timezone.utc))
    monday_evening = datetime(2026, 10, 6, 0, 15, tzinfo=timezone.utc)   # 8:15 PM ET Mon
    st = data_status(md, CONFIG, monday_evening)
    assert st["stale"] and st["expected"] == "2026-10-05"
    friday_evening = datetime(2026, 10, 3, 0, 15, tzinfo=timezone.utc)
    assert not data_status(md, CONFIG, friday_evening)["stale"]
    saturday = datetime(2026, 10, 3, 15, 0, tzinfo=timezone.utc)
    assert expected_last_bar("stocks", CONFIG, saturday) == date(2026, 10, 2)


# ------------------------------------------------------------------ options

class _ApproveAll:
    name = "TheoStub"

    def evaluate(self, proposal, open_positions, bucket_equity):
        return RiskDecision(approved=True, reason="ok", proposal=proposal)


def _opt_broker() -> OptionsBroker:
    b = OptionsBroker(CONFIG, risk_agent=_ApproveAll())
    b.ledger_path = Path(tempfile.mkdtemp()) / "options_ledger.json"
    return b


def test_options_marks_build_a_pnl_line_and_a_card():
    broker = _opt_broker()
    exp = (date.today() + timedelta(days=20)).isoformat()
    prop = OptionsProposal("SPY", "long_call", 500.0, exp, 120.0, 1, "SPARK signals long on SPY")
    key = f"SPY|long_call|500|{exp}"
    ledger = broker.execute([prop], prices={key: 120.0})
    assert ledger.positions[key].marks[-1]["premium"] == 120.0
    ledger = broker.execute([], prices={key: 150.0})          # same day: replaced, not doubled
    assert len(ledger.positions[key].marks) == 1 and ledger.positions[key].marks[0]["premium"] == 150.0

    bars = _walk(14) * 5
    market = {"SPY": MarketData("SPY", "stocks", bars, "test", datetime.now(timezone.utc))}
    out = build_options_charts(CONFIG, market, ledger, prices={key: 150.0})
    card = out["open"][0]
    assert card["headline"]["raw"] == 30.0
    assert card["traders"][0]["callsign"] == "SPARK"
    assert card["price_lines"][0]["price"] == 500.0
    assert card["price_lines"][1]["price"] == 501.2      # strike + $1.20/share premium
    assert card["pnl"] == [{"time": card["pnl"][0]["time"], "value": 30.0, "pct": 25.0}]
    assert card["pnl_stats"]["current"] == 30.0 and card["pnl_stats"]["change"] is None

    ledger = broker.close(key, 90.0, reason="manual close")
    out = build_options_charts(CONFIG, market, ledger)
    assert not out["open"] and out["closed"][0]["headline"]["raw"] == round(90 - 120 - 2 * broker.commission_per_contract, 2)


def test_put_trigger_parsed_from_reason():
    broker = _opt_broker()
    exp = (date.today() + timedelta(days=20)).isoformat()
    prop = OptionsProposal("QQQ", "long_put", 400.0, exp, 90.0, 1, "SPARK-puts signals a breakdown on QQQ")
    ledger = broker.execute([prop], prices={})
    bars = _walk(15) * 4
    market = {"QQQ": MarketData("QQQ", "stocks", bars, "test", datetime.now(timezone.utc))}
    card = build_options_charts(CONFIG, market, ledger)["open"][0]
    assert card["traders"][0]["callsign"] == "SPARK-puts"
    assert any("no live quote" in n.lower() for n in card["notes"])
    assert card["headline"]["value"] == "no quote"


# ------------------------------------------------------------------ dashboard

def test_dashboard_renders_the_active_trades_tab_with_inline_library():
    from agents.reporting_agent import ReportingAgent
    broker = _tmp_broker()
    bars = _walk(16)
    ledger = broker.execute(_market(bars), {"TEST": NettedPosition("TEST", 0.3, False, ["SURGE"])})
    george = ReportingAgent(CONFIG)
    ctx = george._empty_stock_context()
    ctx["trade_charts"] = build_stock_charts(CONFIG, _market(bars), ledger)
    json.dumps(ctx["trade_charts"])                      # must be cache-safe
    html = george.env.get_template("report.html.j2").render(**ctx)
    assert 'data-tab="active"' in html and 'id="p-active"' in html
    assert "ACTIVE TRADES (1)" in html
    assert "TradingView Lightweight Charts" in html and "createChart" in html
    assert "SURGE entry line" in html
