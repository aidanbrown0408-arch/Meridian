"""Jonathan (trade reviews): P&L must reconcile to the ledger to the cent,
open trades are never reviewed, findings follow their stated rules, and
reviews land on disk and on the dashboard. Every test uses tmp dirs.

Run with:  python -m pytest tests/test_review_agent.py -q
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.data_agent import MarketData  # noqa: E402
from agents.options_broker import OptionsLedger, OptionsPosition, OptionsTrade  # noqa: E402
from agents.portfolio_agent import PaperBroker, PaperLedger, Position, Trade  # noqa: E402
from agents.reporting_agent import ReportingAgent  # noqa: E402
from agents.review_agent import TradeReviewAgent, render_markdown  # noqa: E402
from main import build_parser  # noqa: E402
from utils.config import load_config  # noqa: E402

CONFIG = load_config()


def _bars(close: np.ndarray, end: str = "2026-10-16") -> pd.DataFrame:
    idx = pd.bdate_range(end=pd.Timestamp(end), periods=len(close))
    return pd.DataFrame({"open": close, "high": close * 1.005, "low": close * 0.995,
                         "close": close, "volume": 1e6}, index=idx)


def _md(symbol: str, bars: pd.DataFrame, source: str = "yfinance") -> MarketData:
    return MarketData(symbol, "stocks", bars, source, datetime.now(timezone.utc))


def _day(bars, i) -> str:
    return bars.index[i].strftime("%Y-%m-%d")


def _round_trip(bars, i, j, traders=("ORBIT",)) -> PaperLedger:
    """Buy at bar i's close and sell everything at bar j's, through
    Cornelius's own rebalance code, so costs and cash are the real thing."""
    broker = PaperBroker(CONFIG)
    ledger = PaperLedger(starting_capital=5000.0, cash=5000.0)
    when = lambda k: datetime.combine(bars.index[k].date(), datetime.min.time(),  # noqa: E731
                                      tzinfo=timezone.utc).replace(hour=20)
    broker._rebalance_one(ledger, "NVDA", "stocks", float(bars["close"].iloc[i]), 0.25,
                          5000.0, when(i), list(traders), _day(bars, i))
    broker._rebalance_one(ledger, "NVDA", "stocks", float(bars["close"].iloc[j]), 0.0,
                          ledger.cash + 1250.0, when(j), [], _day(bars, j))
    return ledger


def _ramp_then_fade(n=300) -> np.ndarray:
    """Steady uptrend, a +12% pop over ten sessions, then a fade back."""
    base = 100 * np.exp(np.cumsum(np.full(n, 0.002)))
    pop = np.concatenate([np.linspace(0, 0.12, 10), np.linspace(0.12, 0.0, 10)])
    base[-25:-5] *= 1 + pop
    return base


def test_stock_review_reconciles_to_ledger_cash(tmp_path):
    bars = _bars(_ramp_then_fade())
    i, j = len(bars) - 26, len(bars) - 6
    ledger = _round_trip(bars, i, j)
    jon = TradeReviewAgent(CONFIG, tmp_path)
    run = jon.run(ledger, None, {"NVDA": _md("NVDA", bars), "SPY": _md("SPY", bars)})

    assert len(run.new) == 1
    r = run.new[0]
    assert abs(r.pnl - round(ledger.cash - 5000.0, 2)) < 0.01, "P&L must match the ledger's cash"
    assert r.traders == ["ORBIT"] and r.desk == "stock"
    assert r.sessions_held == j - i
    tags = {f.tag for f in r.findings}
    # Up ~12% at the best close, exited near flat -> gave most of it back.
    assert "gave_back" in tags
    assert r.metrics["mfe"] > 0.10
    # ORBIT was still long when the desk sold, so the exit was a portfolio call.
    assert "portfolio_exit" in tags
    assert not r.provisional and r.lesson


def test_review_files_index_and_scorecard_written(tmp_path):
    bars = _bars(_ramp_then_fade())
    ledger = _round_trip(bars, len(bars) - 26, len(bars) - 6)
    run = TradeReviewAgent(CONFIG, tmp_path).run(ledger, None, {"NVDA": _md("NVDA", bars)})
    r = run.new[0]
    d = tmp_path / "trade_reviews"
    md = (d / r.file).read_text()
    assert md.startswith("# NVDA · ORBIT ·")
    assert "## What worked" in md and "## What didn't" in md and "## Lesson" in md
    index = json.loads((d / "index.json").read_text())
    assert index["reviews"][0]["id"] == r.id
    score = (d / "SCORECARD.md").read_text()
    assert "| ORBIT | 1 |" in score and "**All trades**" in score


def test_reviews_are_written_once(tmp_path):
    bars = _bars(_ramp_then_fade())
    ledger = _round_trip(bars, len(bars) - 26, len(bars) - 6)
    market = {"NVDA": _md("NVDA", bars)}
    jon = TradeReviewAgent(CONFIG, tmp_path)
    assert len(jon.run(ledger, None, market).new) == 1
    again = jon.run(ledger, None, market)
    assert again.new == [] and len(again.reviews) == 1
    assert len(jon.run(ledger, None, market, rebuild=True).new) == 1


def test_open_position_is_not_reviewed(tmp_path):
    bars = _bars(_ramp_then_fade())
    i = len(bars) - 10
    ledger = PaperLedger(starting_capital=5000.0, cash=3750.0)
    price = float(bars["close"].iloc[i])
    ledger.trades.append(Trade(_day(bars, i), "NVDA", "buy", 1250 / price, price, 1.25,
                               "rebalance to target", ["ORBIT"], _day(bars, i)))
    ledger.positions["NVDA"] = Position("NVDA", "stocks", 1250 / price, price, _day(bars, i),
                                        ["ORBIT"])
    run = TradeReviewAgent(CONFIG, tmp_path).run(ledger, None, {"NVDA": _md("NVDA", bars)})
    assert run.new == [] and run.reviews == []


def test_synthetic_data_is_provisional_then_rebuilt(tmp_path):
    bars = _bars(_ramp_then_fade())
    ledger = _round_trip(bars, len(bars) - 26, len(bars) - 6)
    jon = TradeReviewAgent(CONFIG, tmp_path)
    first = jon.run(ledger, None, {"NVDA": _md("NVDA", bars, source="synthetic")}).new[0]
    assert first.provisional
    assert "ride" not in {f.area.lower() for f in first.findings}
    # Still synthetic: nothing rewritten.
    assert jon.run(ledger, None, {"NVDA": _md("NVDA", bars, source="synthetic")}).new == []
    # Real bars arrive: rebuilt in full.
    real = jon.run(ledger, None, {"NVDA": _md("NVDA", bars)}).new
    assert len(real) == 1 and not real[0].provisional
    assert "gave_back" in {f.tag for f in real[0].findings}


def test_whipsaw_and_chase_flags_on_a_quick_loser(tmp_path):
    close = 100 * np.exp(np.cumsum(np.full(300, 0.001)))
    close[-8:-3] *= np.array([1.02, 1.04, 1.06, 1.08, 1.10])   # +10% run into the buy
    close[-3:] = close[-4] * np.array([0.97, 0.95, 0.94])      # then it rolls over
    bars = _bars(close)
    ledger = _round_trip(bars, len(bars) - 4, len(bars) - 1, traders=("SPARK",))
    r = TradeReviewAgent(CONFIG, tmp_path).run(ledger, None, {"NVDA": _md("NVDA", bars)}).new[0]
    tags = {f.tag for f in r.didnt}
    assert r.outcome == "loss"
    assert {"chased", "whipsaw"} <= tags
    assert "run in five sessions" in r.lesson or "false signal" in r.lesson


def test_unattributed_fills_still_reviewed(tmp_path):
    bars = _bars(_ramp_then_fade())
    ledger = _round_trip(bars, len(bars) - 26, len(bars) - 6, traders=())
    r = TradeReviewAgent(CONFIG, tmp_path).run(ledger, None, {"NVDA": _md("NVDA", bars)}).new[0]
    assert r.traders == []
    assert any("No trader is recorded" in f.text for f in r.findings)
    card = TradeReviewAgent.scorecard([r])
    assert [row["trader"] for row in card["rows"]] == ["Unattributed", "All trades"]


def _options_ledger(open_day: str, close_day: str, settle: float, reason: str) -> OptionsLedger:
    ledger = OptionsLedger(starting_capital=1000.0, cash=1000.0)
    ledger.trades.append(OptionsTrade(open_day, "SPY", "long_call", 600.0, "2026-10-16", "open",
                                      1, 120.0, 0.65, -120.65, 0.0, "SPARK signals long on SPY"))
    pnl = settle - 120.65
    ledger.trades.append(OptionsTrade(close_day, "SPY", "long_call", 600.0, "2026-10-16",
                                      "close", 1, settle, 0.0, settle, pnl, reason))
    return ledger


def test_option_review_uses_snapshotted_marks(tmp_path):
    close = np.linspace(560, 610, 300)
    bars = _bars(close)
    jon = TradeReviewAgent(CONFIG, tmp_path)
    # While open, Joseph's position carries marks; Jonathan snapshots them.
    open_ledger = OptionsLedger(starting_capital=1000.0, cash=879.35)
    pos = OptionsPosition("SPY", "long_call", 600.0, "2026-10-16", 1, 120.65, 120.0,
                          "2026-10-01T20:00:00+00:00", "SPARK signals long on SPY",
                          marks=[{"date": "2026-10-05", "premium": 260.0},
                                 {"date": "2026-10-09", "premium": 90.0}])
    open_ledger.positions[pos.key] = pos
    jon.run(None, open_ledger, {})
    # Then it expires worthless; the closed position no longer has marks.
    closed = _options_ledger("2026-10-01", "2026-10-17", 0.0, "expired")
    r = jon.run(None, closed, {"SPY": _md("SPY", bars)}).new[0]
    assert r.desk == "options" and r.label == "SPY 2026-10-16 600C"
    assert r.traders == ["SPARK"]
    assert abs(r.pnl + 120.65) < 0.01
    tags = {f.tag for f in r.didnt}
    assert {"expired", "gave_back"} <= tags
    assert r.metrics["best_mark"] == 260.0


def test_dashboard_shows_reviews_and_jonathan(tmp_path):
    bars = _bars(_ramp_then_fade())
    ledger = _round_trip(bars, len(bars) - 26, len(bars) - 6)
    TradeReviewAgent(CONFIG, tmp_path).run(ledger, None, {"NVDA": _md("NVDA", bars)})
    george = ReportingAgent(CONFIG)
    george.reports_dir = tmp_path
    html = george.rerender().read_text()
    assert "Trade reviews" in html and "What worked" in html and "Lesson" in html
    assert "Jonathan" in html and "Trade Review Agent" in html
    assert "1 closed trade(s) reviewed" in html


def test_dashboard_without_reviews_still_renders(tmp_path):
    george = ReportingAgent(CONFIG)
    george.reports_dir = tmp_path
    html = george.rerender().read_text()
    assert "No closed trades yet" in html and "Jonathan" in html


def test_markdown_render_is_stable_for_a_flat_trade(tmp_path):
    bars = _bars(np.full(300, 100.0))
    ledger = _round_trip(bars, 250, 260, traders=("REVERT",))
    r = TradeReviewAgent(CONFIG, tmp_path).run(ledger, None, {"NVDA": _md("NVDA", bars)}).new[0]
    text = render_markdown(r)
    assert "# NVDA · REVERT ·" in text and r.lesson in text


def test_cli_has_review_mode():
    args = build_parser().parse_args(["review", "--rebuild"])
    assert args.mode == "review" and args.rebuild
