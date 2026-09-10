"""Phase 5 smoke tests: Edwin's sentiment agent (informational only, never
raises) and the live-trading three-gate refusal check.

Run with:  python -m pytest tests -q      (or)  python tests/test_phase5.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.data_agent import MarketData  # noqa: E402
from agents.portfolio_agent import LiveBroker  # noqa: E402
from agents.sentiment_agent import SentimentAgent, _label_for  # noqa: E402
from main import run_live  # noqa: E402
from utils.config import load_config  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

CONFIG = load_config()


def _tmp_sentiment_agent(env: dict | None = None, monkeypatch=None) -> SentimentAgent:
    agent = SentimentAgent(CONFIG)
    agent.cache_dir = Path(tempfile.mkdtemp())
    return agent


def _market(symbols: list[str]) -> dict[str, MarketData]:
    bars = pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0],
                         "volume": [1.0]},
                        index=pd.bdate_range(end=pd.Timestamp.today(), periods=1))
    return {s: MarketData(s, "stocks", bars, "test", datetime.now(timezone.utc))
           for s in symbols}


def _config_without_key(env_var: str = "MERIDIAN_TEST_MISSING_KEY_XYZ") -> object:
    cfg = load_config()
    cfg._data["sentiment"]["api_key_env"] = env_var
    return cfg


# ------------------------------------------------------------------ sentiment bucketing

def test_label_for_buckets_match_alpha_vantage_thresholds():
    assert _label_for(-0.6) == "bearish"
    assert _label_for(-0.20) == "somewhat-bearish"
    assert _label_for(0.0) == "neutral"
    assert _label_for(0.20) == "somewhat-bullish"
    assert _label_for(0.60) == "bullish"


# ------------------------------------------------------------------ never raises / degrades

def test_disabled_sentiment_returns_unavailable_for_every_symbol():
    cfg = load_config()
    cfg._data["sentiment"]["enabled"] = False
    agent = SentimentAgent(cfg)
    agent.cache_dir = Path(tempfile.mkdtemp())
    snapshots = agent.run(_market(["SPY", "QQQ"]))
    assert set(snapshots) == {"SPY", "QQQ"}
    assert all(not s.available for s in snapshots.values())
    assert all(s.note for s in snapshots.values())


def test_missing_api_key_returns_unavailable_not_an_exception():
    agent = SentimentAgent(_config_without_key())
    agent.cache_dir = Path(tempfile.mkdtemp())
    snapshots = agent.run(_market(["SPY"]))
    assert snapshots["SPY"].available is False
    assert "MERIDIAN_TEST_MISSING_KEY_XYZ" in snapshots["SPY"].note


def test_fetch_failure_never_raises_and_reports_unavailable(monkeypatch):
    import os

    monkeypatch.setenv("MERIDIAN_TEST_FAKE_KEY", "dummy")
    cfg = load_config()
    cfg._data["sentiment"]["api_key_env"] = "MERIDIAN_TEST_FAKE_KEY"
    agent = SentimentAgent(cfg)
    agent.cache_dir = Path(tempfile.mkdtemp())

    def _boom(self, ticker, api_key):
        raise TimeoutError("simulated network failure")

    monkeypatch.setattr(SentimentAgent, "_request", _boom)
    snapshots = agent.run(_market(["SPY"]))
    assert snapshots["SPY"].available is False
    assert "simulated network failure" in snapshots["SPY"].note


# ------------------------------------------------------------------ parsing

def test_parse_averages_ticker_specific_scores_and_picks_top_headline():
    agent = _tmp_sentiment_agent()
    payload = {
        "feed": [
            {"title": "Big rally", "overall_sentiment_score": 0.1,
             "ticker_sentiment": [{"ticker": "SPY", "ticker_sentiment_score": "0.4"}]},
            {"title": "Some pullback", "overall_sentiment_score": -0.1,
             "ticker_sentiment": [{"ticker": "SPY", "ticker_sentiment_score": "0.2"}]},
        ]
    }
    snap = agent._parse("SPY", "SPY", payload)
    assert snap.available is True
    assert snap.article_count == 2
    assert snap.top_headline == "Big rally"
    assert abs(snap.score - 0.3) < 1e-9
    assert snap.label == "somewhat-bullish"


def test_parse_handles_empty_feed_as_neutral_not_unavailable():
    agent = _tmp_sentiment_agent()
    snap = agent._parse("QQQ", "QQQ", {"feed": []})
    assert snap.available is True
    assert snap.label == "neutral"
    assert snap.article_count == 0


def test_parse_treats_rate_limit_note_as_unavailable():
    agent = _tmp_sentiment_agent()
    snap = agent._parse("SPY", "SPY", {"Note": "Thank you for using Alpha Vantage! ..."})
    assert snap.available is False
    assert "Alpha Vantage" in snap.note


def test_crypto_ticker_map_translates_before_matching():
    agent = _tmp_sentiment_agent()
    assert agent.crypto_ticker_map.get("BTC/USDT") == "CRYPTO:BTC"
    payload = {"feed": [{"title": "BTC pops", "overall_sentiment_score": 0.0,
                         "ticker_sentiment": [{"ticker": "CRYPTO:BTC",
                                               "ticker_sentiment_score": "0.5"}]}]}
    snap = agent._parse("BTC/USDT", "CRYPTO:BTC", payload)
    assert snap.score == 0.5


# ------------------------------------------------------------------ live-trading gate

def test_live_refuses_when_no_gates_pass():
    cfg = load_config()
    assert cfg.get("live.enable_live_trading") is False
    assert run_live(cfg, risk_ack=False) == 2


def test_live_refuses_with_only_operator_ack():
    cfg = load_config()
    assert run_live(cfg, risk_ack=True) == 2


def test_live_refuses_with_only_config_flag():
    cfg = load_config()
    cfg._data["live"]["enable_live_trading"] = True
    assert run_live(cfg, risk_ack=False) == 2


def test_live_still_refuses_with_config_and_operator_gate_both_satisfied():
    """The third gate -- a real LiveBroker implementation -- is not
    satisfiable by config or CLI flags at all. Even with the other two
    gates open, live mode must still refuse."""
    cfg = load_config()
    cfg._data["live"]["enable_live_trading"] = True
    assert LiveBroker.IMPLEMENTED is False
    assert run_live(cfg, risk_ack=True) == 2


def test_live_broker_stub_raises_not_implemented():
    broker = LiveBroker(CONFIG)
    try:
        broker.execute()
        assert False, "LiveBroker.execute() should not succeed while IMPLEMENTED is False"
    except NotImplementedError:
        pass


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                if "monkeypatch" in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
                    print(f"SKIP  {name} (requires pytest monkeypatch fixture)")
                    continue
                fn()
                print(f"PASS  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL  {name}: {exc}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
