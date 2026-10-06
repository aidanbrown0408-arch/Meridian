"""Stale-feed guard (added 2026-10-05): every evening run from 9/30 to 10/05
decided and filled on the PREVIOUS session's close. These tests pin the fix:
Wong asks for today's bar explicitly and refetches a cache that's behind;
David blocks a ticker whose newest bar isn't the newest completed session;
Cornelius never trades a blocked ticker; David posts one Slack alert.
"""

from __future__ import annotations

import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.compliance_agent import ComplianceAgent  # noqa: E402
from agents.data_agent import DataAgent, MarketData  # noqa: E402
from agents.portfolio_agent import PaperBroker  # noqa: E402
from agents.risk_agent import NettedPosition  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.market_calendar import expected_last_bar, sessions_between  # noqa: E402

CONFIG = load_config()
UTC = timezone.utc
MON_8_15PM = datetime(2026, 10, 6, 0, 15, tzinfo=UTC)     # Mon 10/05 8:15 PM ET
MON_2PM = datetime(2026, 10, 5, 18, 0, tzinfo=UTC)        # Mon 10/05 2:00 PM ET


def _bars(last: str, n: int = 300, price: float = 100.0) -> pd.DataFrame:
    idx = pd.bdate_range(end=last, periods=n)
    close = price * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.01, n)))
    return pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                         "close": close, "volume": 1e6}, index=idx)


# ------------------------------------------------------------------ calendar

@pytest.mark.parametrize("now,asset,expected", [
    (MON_8_15PM, "stocks", date(2026, 10, 5)),            # after the close: today
    (MON_2PM, "stocks", date(2026, 10, 2)),               # intraday: Friday
    (datetime(2026, 10, 3, 16, 0, tzinfo=UTC), "stocks", date(2026, 10, 2)),   # Saturday
    (datetime(2026, 11, 27, 1, 0, tzinfo=UTC), "stocks", date(2026, 11, 25)),  # Thanksgiving night
    (datetime(2026, 9, 8, 1, 0, tzinfo=UTC), "stocks", date(2026, 9, 4)),      # Labor Day night
    (MON_8_15PM, "crypto", date(2026, 10, 5)),            # UTC candle closed at 00:00
])
def test_expected_last_bar(now, asset, expected):
    assert expected_last_bar(asset, CONFIG, now) == expected


def test_sessions_between_skips_weekends_and_holidays():
    assert sessions_between(date(2026, 10, 2), date(2026, 10, 5), "stocks") == 1
    assert sessions_between(date(2026, 9, 3), date(2026, 9, 8), "stocks") == 2   # Labor Day skipped


# ------------------------------------------------------------------ David

def _david(now):
    d = ComplianceAgent(CONFIG)
    d.now = now
    return d


def test_one_session_behind_is_blocked():
    md = MarketData("AAPL", "stocks", _bars("2026-10-02"), "yfinance", MON_8_15PM)
    rep = _david(MON_8_15PM).check(md)
    assert rep.blocked
    assert "last bar 2026-10-02 but the 2026-10-05 close is final" in rep.reason


def test_current_feed_passes_and_intraday_friday_bar_is_fine():
    assert not _david(MON_8_15PM).check(
        MarketData("AAPL", "stocks", _bars("2026-10-05"), "yfinance", MON_8_15PM)).blocked
    assert not _david(MON_2PM).check(
        MarketData("AAPL", "stocks", _bars("2026-10-02"), "yfinance", MON_2PM)).blocked


def test_synthetic_data_is_not_double_reported_as_stale():
    md = MarketData("AAPL", "stocks", _bars("2026-10-02"), "synthetic", MON_8_15PM)
    assert all(c.name != "current" for c in _david(MON_8_15PM).check(md).checks)


# ------------------------------------------------------------------ Wong

class _FakeTicker:
    calls: list = []

    def __init__(self, symbol):
        self.symbol = symbol

    def history(self, **kwargs):
        _FakeTicker.calls.append(kwargs)
        bars = _bars("2026-10-05")
        bars.columns = [c.capitalize() for c in bars.columns]
        return bars


def test_stock_fetch_asks_for_today_explicitly(monkeypatch):
    import types
    fake = types.SimpleNamespace(Ticker=_FakeTicker)
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    _FakeTicker.calls = []
    DataAgent(CONFIG)._fetch_stock("AAPL")
    kw = _FakeTicker.calls[-1]
    assert "end" in kw, "end must be explicit -- the default returned yesterday's bars"
    from utils.market_calendar import exchange_now
    assert kw["end"] > exchange_now().date().isoformat()


def test_intraday_bar_is_dropped_before_the_close():
    wong = DataAgent(CONFIG)
    bars = _bars("2026-10-05")
    kept = wong._drop_open_stock_bar("AAPL", bars, now=MON_2PM)
    assert kept.index[-1].date() == date(2026, 10, 2)
    assert len(wong._drop_open_stock_bar("AAPL", bars, now=MON_8_15PM)) == len(bars)


def test_cache_that_is_behind_gets_refetched(monkeypatch, tmp_path):
    wong = DataAgent(CONFIG)
    wong.cache_dir = tmp_path
    wong._write_cache("AAPL", _bars("2026-10-02"))           # fresh by TTL, stale by date
    monkeypatch.setattr("agents.data_agent.expected_last_bar",
                        lambda a, c=None, n=None: date(2026, 10, 5))
    monkeypatch.setattr(wong, "_fetch_stock", lambda s: _bars("2026-10-05"))
    monkeypatch.setattr(wong, "_drop_open_stock_bar", lambda s, b, now=None: b)
    md = wong.fetch("AAPL")
    assert md.data_source != "cache" and md.bars.index[-1].date() == date(2026, 10, 5)


def test_current_cache_is_still_used(monkeypatch, tmp_path):
    wong = DataAgent(CONFIG)
    wong.cache_dir = tmp_path
    wong._write_cache("AAPL", _bars("2026-10-05"))
    monkeypatch.setattr("agents.data_agent.expected_last_bar",
                        lambda a, c=None, n=None: date(2026, 10, 5))
    monkeypatch.setattr(wong, "_fetch_stock", lambda s: pytest.fail("should use cache"))
    assert wong.fetch("AAPL").data_source == "cache"


# ------------------------------------------------------------------ Cornelius

def _broker():
    b = PaperBroker(CONFIG)
    b.ledger_path = Path(tempfile.mkdtemp()) / "ledger.json"
    return b


def _mkt(bars):
    return {"AAPL": MarketData("AAPL", "stocks", bars, "yfinance", MON_8_15PM)}


def test_frozen_ticker_is_neither_sold_nor_bought():
    broker = _broker()
    bars = _bars("2026-10-02")
    ledger = broker.execute(_mkt(bars), {"AAPL": NettedPosition("AAPL", 0.3, False, ["SPARK"])})
    held = ledger.positions["AAPL"].shares
    n = len(ledger.trades)
    reason = {"AAPL": "stale feed"}
    ledger = broker.execute(_mkt(bars), {"AAPL": NettedPosition("AAPL", 0.0, False, [])}, frozen=reason)
    assert ledger.positions["AAPL"].shares == held and len(ledger.trades) == n
    ledger = broker.execute(_mkt(bars), {"AAPL": NettedPosition("AAPL", 0.6, False, ["SPARK"])}, frozen=reason)
    assert ledger.positions["AAPL"].shares == held and len(ledger.trades) == n
    assert ledger.equity_history[-1]["equity"] == pytest.approx(ledger.mark_to_market(
        {"AAPL": float(bars["close"].iloc[-1])}))


# ------------------------------------------------------------------ Slack

def test_data_alert_posts_once_as_urgent(monkeypatch):
    from utils import notifications
    sent = []
    monkeypatch.setattr(notifications, "post_message", lambda url, text, blocks: sent.append(text) or True)
    n = notifications.Notifier(CONFIG)
    assert n.data_alert({"AAPL": "stale feed: last bar 2026-10-02 ...",
                         "MSFT": "stale feed: last bar 2026-10-02 ..."})
    assert len(sent) == 1 and "AAPL, MSFT" in sent[0]
    assert "data_alerts" in n.URGENT
    assert n.data_alert({}) is False
