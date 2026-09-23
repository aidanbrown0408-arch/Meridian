"""Slack notification tests: message builders, routing/toggles, and that
the paper run, options leg, killswitch and failures each post what they
should. `post_message` is captured -- nothing touches the network.

Run with:  python -m pytest tests/test_notifications.py -q
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main  # noqa: E402
import utils.notifications as notifications  # noqa: E402
from agents.data_agent import MarketData  # noqa: E402
from agents.lifecycle_agent import Recommendation  # noqa: E402
from agents.options_broker import OptionsTrade  # noqa: E402
from agents.options_data_agent import OptionsDataAgent  # noqa: E402
from agents.options_risk_agent import OptionsProposal, RiskDecision  # noqa: E402
from agents.portfolio_agent import Trade  # noqa: E402
from utils.config import Config, load_config  # noqa: E402
from utils.notifications import (  # noqa: E402
    BookSnapshot, DeskLine, Notifier, StandupPayload, build_agent_line, build_failure,
    build_halt, build_options_activity, build_standup, build_standup_header,
    build_stock_fills,
)

BASE = load_config()
WEDNESDAY = datetime(2026, 9, 16, 16, 15)
SATURDAY = datetime(2026, 9, 19, 16, 15)


def _cfg(tmp_path, slack: dict | None = None, **options) -> Config:
    data = BASE.as_dict()
    data["options"]["ledger_path"] = str(tmp_path / "options_ledger.json")
    data["options"].update(options)
    data["execution"] = dict(data.get("execution", {}),
                             paper_ledger_path=str(tmp_path / "paper_ledger.json"))
    data["slack"] = dict(data.get("slack", {}), **(slack or {}))
    return Config(data)


@pytest.fixture
def posts(monkeypatch):
    """Capture every Slack POST as (url, text, blocks)."""
    sent = []

    def fake_post(url, text, blocks=None, timeout=10.0):
        sent.append((url, text, blocks or []))
        return True

    monkeypatch.setattr(notifications, "post_message", fake_post)
    monkeypatch.setenv("MERIDIAN_SLACK_WEBHOOK_URL", "https://hooks.test/main")
    return sent


@pytest.fixture
def bot_posts(monkeypatch):
    """Capture every bot-mode chat.postMessage call as
    (bot_token, channel, text, blocks, thread_ts). Does not itself set a
    channel or any agent token -- individual tests opt in."""
    sent = []
    counter = {"n": 0}

    def fake_post_as(bot_token, channel, text, blocks=None, thread_ts=None, timeout=10.0):
        if not bot_token or not channel:
            return None
        counter["n"] += 1
        sent.append((bot_token, channel, text, blocks or [], thread_ts))
        return {"ok": True, "ts": thread_ts or f"1700000000.{counter['n']:06d}"}

    monkeypatch.setattr(notifications, "post_as", fake_post_as)
    return sent


def _check_blocks(blocks):
    """Slack Block Kit limits we rely on."""
    assert 0 < len(blocks) <= 50
    for b in blocks:
        if b["type"] == "header":
            assert len(b["text"]["text"]) <= 150
        if b["type"] == "section":
            if "text" in b:
                assert len(b["text"]["text"]) <= 3000
            for f in b.get("fields", []):
                assert len(f["text"]) <= 2000
            assert len(b.get("fields", [])) <= 10


def _all_text(blocks) -> str:
    out = []
    for b in blocks:
        if "text" in b:
            out.append(b["text"]["text"])
        out += [f["text"] for f in b.get("fields", [])]
        out += [e["text"] for e in b.get("elements", [])]
    return "\n".join(out)


def _otrade(side="open", pnl=0.0, reason="SPARK signals long on SPY"):
    return OptionsTrade(date="2026-09-16", underlying="SPY", option_type="long_call",
                        strike=500.0, expiration="2026-10-16", side=side, contracts=1,
                        premium_per_contract=120.0, cost=0.65,
                        cash_delta=-120.65 if side == "open" else 120.0,
                        realized_pnl=pnl, reason=reason)


# ------------------------------------------------------------------ builders


def test_standup_has_scoreboard_both_desks_and_inquiries():
    payload = StandupPayload(
        mode="Paper",
        stock_lines=[DeskLine("🔍", "Wong", "Data", "Pulled SPY — all live.")],
        options_lines=[DeskLine("🎯", "Joseph", "Options Execution", "No trades today.")],
        inquiries=[Recommendation("SPARK", "validation_fail", "bench", "Failed 3 days <running>.")],
        books=[BookSnapshot("Stock desk", 5100.0, 5000.0, 2, 1),
               BookSnapshot("Options desk", 950.0, 1000.0, 1, 0, extra="$120.00",
                            extra_label="Premium at risk")],
        dashboard="https://example.com/latest.html",
    )
    text, blocks = build_standup(payload, WEDNESDAY)
    _check_blocks(blocks)
    body = _all_text(blocks)
    assert blocks[0]["type"] == "header"
    assert "Stock desk" in body and "Options desk" in body
    assert "$5,100.00  (+2.00%)" in body and "$950.00  (-5.00%)" in body
    assert "+$100.00" in body and "-$50.00" in body
    assert "*Wong*" in body and "*Joseph*" in body
    assert "*Premium at risk*\n$120.00" in body
    assert body.count("Stock desk") == 1  # one heading per desk, not two
    assert "Needs your approval" in body and "&lt;running&gt;" in body  # escaped
    assert "<https://example.com/latest.html|Open the full dashboard>" in body
    assert "Wed, Sep 16, 2026" in body
    assert text.startswith("Meridian daily standup")


def test_stock_fills_lists_every_trade():
    trades = [Trade("2026-09-16", "BTC/USDT", "buy", 0.01, 60000.0, 0.6, "rebalance"),
              Trade("2026-09-16", "SPY", "sell", 5.0, 500.0, 0.5, "rebalance")]
    text, blocks = build_stock_fills(trades, 5000.0, 1000.0, WEDNESDAY)
    _check_blocks(blocks)
    body = _all_text(blocks)
    assert "2 fills" in blocks[0]["text"]["text"]
    assert "*BUY* BTC/USDT" in body and "*SELL* SPY" in body and "$2,500.00" in body
    assert "+1 more" in text


def test_options_activity_shows_opens_closes_expiries_and_rejections():
    proposal = OptionsProposal("QQQ", "long_call", 450.0, "2026-10-16", 180.0, 1)
    trades = [_otrade("open"), _otrade("close", 35.0, "manual"), _otrade("close", -120.0, "expired")]
    _, blocks = build_options_activity(trades, [RiskDecision(False, "premium over $150 cap", proposal)],
                                       980.0, 700.0, 120.0, WEDNESDAY)
    _check_blocks(blocks)
    body = _all_text(blocks)
    assert "*OPENED* SPY 2026-10-16 500C" in body
    assert "*CLOSED*" in body and "+$35.00" in body
    assert "*EXPIRED*" in body and "-$120.00" in body
    assert "REJECTED by Theo" in body and "premium over $150 cap" in body


def test_long_lists_are_clipped():
    trades = [Trade("2026-09-16", f"S{i}", "buy", 1.0, 10.0, 0.0, "r") for i in range(30)]
    _, blocks = build_stock_fills(trades, 1.0, 1.0)
    _check_blocks(blocks)
    assert "and 18 more" in _all_text(blocks)


def test_halt_and_failure_messages_tell_the_operator_what_to_do():
    _, blocks = build_halt("Options desk", "bucket loss cutoff breached")
    assert "clear-halt --options" in _all_text(blocks)
    _, blocks = build_halt("Stock desk", "Operator killswitch", manual=True)
    assert "`python main.py clear-halt`" in _all_text(blocks)
    try:
        raise ValueError("bad <bar>")
    except ValueError as exc:
        _, blocks = build_failure("Paper run", exc)
    _check_blocks(blocks)
    assert "ValueError" in _all_text(blocks) and "bad &lt;bar&gt;" in _all_text(blocks)


# ------------------------------------------------------------------ Notifier


def test_urgent_messages_use_the_alerts_webhook(tmp_path, posts, monkeypatch):
    monkeypatch.setenv("MERIDIAN_SLACK_ALERTS_WEBHOOK_URL", "https://hooks.test/alerts")
    n = Notifier(_cfg(tmp_path))
    n.halt("Stock desk", "x")
    n.stock_fills([Trade("d", "SPY", "buy", 1, 1, 0, "r")], 1, 1)
    assert [p[0] for p in posts] == ["https://hooks.test/alerts", "https://hooks.test/main"]


def test_alerts_fall_back_to_the_main_webhook(tmp_path, posts):
    Notifier(_cfg(tmp_path)).halt("Stock desk", "x")
    assert posts[0][0] == "https://hooks.test/main"


def test_each_message_type_can_be_switched_off(tmp_path, posts):
    n = Notifier(_cfg(tmp_path, slack={"notify": {"trades": False}}))
    assert n.stock_fills([Trade("d", "SPY", "buy", 1, 1, 0, "r")], 1, 1) is False
    assert n.halt("Stock desk", "x") is True
    assert len(posts) == 1


def test_disabled_notifier_and_empty_activity_send_nothing(tmp_path, posts):
    assert Notifier(_cfg(tmp_path), enabled=False).halt("Stock desk", "x") is False
    n = Notifier(_cfg(tmp_path))
    assert n.stock_fills([], 1, 1) is False
    assert n.options_activity([], [], 1, 1, 0) is False
    assert posts == []


def test_standup_skips_weekends(tmp_path, posts):
    n = Notifier(_cfg(tmp_path))
    assert n.standup(StandupPayload(mode="Paper"), now=SATURDAY) is False
    assert n.standup(StandupPayload(mode="Paper"), now=WEDNESDAY) is True
    assert len(posts) == 1


def test_a_formatting_bug_never_raises(tmp_path, posts, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("builder bug")
    monkeypatch.setattr(notifications, "build_halt", boom)
    assert Notifier(_cfg(tmp_path)).halt("Stock desk", "x") is False
    assert posts == []


# ------------------------------------------------------------------ bot mode


def _bot_cfg(tmp_path):
    return _cfg(tmp_path, slack={
        "channel_id_env": "MERIDIAN_SLACK_CHANNEL_ID",
        "agent_tokens": {name: f"MERIDIAN_SLACK_TOKEN_{name.upper()}"
                        for name in ("Wong", "Cornelius", "George", "Joseph")},
    })


def test_bot_mode_is_off_without_a_channel_or_any_agent_token(tmp_path):
    assert Notifier(_bot_cfg(tmp_path)).bot_mode is False


def test_standup_threads_george_header_then_each_agent_line(tmp_path, bot_posts, monkeypatch):
    monkeypatch.setenv("MERIDIAN_SLACK_CHANNEL_ID", "C0MAIN")
    monkeypatch.setenv("MERIDIAN_SLACK_TOKEN_GEORGE", "xoxb-george")
    monkeypatch.setenv("MERIDIAN_SLACK_TOKEN_WONG", "xoxb-wong")
    n = Notifier(_bot_cfg(tmp_path))
    assert n.bot_mode is True

    payload = StandupPayload(
        mode="Paper",
        stock_lines=[DeskLine("🔍", "Wong", "Data", "Pulled SPY — all live.")],
        options_lines=[DeskLine("🎯", "Joseph", "Options Execution", "No trades today.")],
        books=[BookSnapshot("Stock desk", 5100.0, 5000.0, 2, 1)],
    )
    assert n.standup(payload, now=WEDNESDAY) is True

    header = bot_posts[0]
    assert header[0] == "xoxb-george" and header[1] == "C0MAIN" and header[4] is None
    thread_ts = "1700000000.000001"
    wong_line = next(p for p in bot_posts[1:] if p[0] == "xoxb-wong")
    assert wong_line[4] == thread_ts and "Pulled SPY" in wong_line[2]


def test_agent_without_a_token_posts_under_george_in_the_thread(tmp_path, bot_posts, monkeypatch):
    monkeypatch.setenv("MERIDIAN_SLACK_CHANNEL_ID", "C0MAIN")
    monkeypatch.setenv("MERIDIAN_SLACK_TOKEN_GEORGE", "xoxb-george")
    n = Notifier(_bot_cfg(tmp_path))  # only George has a token this rollout

    payload = StandupPayload(
        mode="Paper",
        options_lines=[DeskLine("🎯", "Joseph", "Options Execution", "No trades today.")],
    )
    assert n.standup(payload, now=WEDNESDAY) is True

    joseph_line = bot_posts[-1]
    assert joseph_line[0] == "xoxb-george"  # posted as George, not dropped
    assert "No trades today" in joseph_line[2]


def test_standup_falls_back_to_the_webhook_when_the_header_post_fails(
        tmp_path, posts, bot_posts, monkeypatch):
    monkeypatch.setenv("MERIDIAN_SLACK_CHANNEL_ID", "C0MAIN")
    monkeypatch.setenv("MERIDIAN_SLACK_TOKEN_WONG", "xoxb-wong")
    # No George token at all -- post_as(None, ...) returns None for the header,
    # even though Wong's token alone is enough to turn bot mode on.
    n = Notifier(_bot_cfg(tmp_path))
    assert n.bot_mode is True

    payload = StandupPayload(mode="Paper",
                             stock_lines=[DeskLine("🔍", "Wong", "Data", "Pulled SPY.")])
    assert n.standup(payload, now=WEDNESDAY) is True
    assert bot_posts == []  # header post never succeeded, nothing else was tried
    assert len(posts) == 1 and posts[0][0] == "https://hooks.test/main"


def test_bot_mode_routes_fills_and_halts_to_the_owning_agent(tmp_path, bot_posts, monkeypatch):
    monkeypatch.setenv("MERIDIAN_SLACK_CHANNEL_ID", "C0MAIN")
    monkeypatch.setenv("MERIDIAN_SLACK_TOKEN_CORNELIUS", "xoxb-cornelius")
    monkeypatch.setenv("MERIDIAN_SLACK_TOKEN_JOSEPH", "xoxb-joseph")
    n = Notifier(_bot_cfg(tmp_path))

    assert n.stock_fills([Trade("d", "SPY", "buy", 1, 1, 0, "r")], 1, 1) is True
    assert n.options_activity([_otrade("open")], [], 900.0, 700.0, 120.0) is True
    assert n.halt("Stock desk", "x") is True
    assert n.halt("Options desk", "y") is True

    by_token = {p[0] for p in bot_posts}
    assert by_token == {"xoxb-cornelius", "xoxb-joseph"}
    assert bot_posts[0][0] == "xoxb-cornelius"   # stock fill
    assert bot_posts[1][0] == "xoxb-joseph"      # options activity
    assert bot_posts[2][0] == "xoxb-cornelius"   # stock desk halt
    assert bot_posts[3][0] == "xoxb-joseph"      # options desk halt


def test_bot_post_failure_falls_back_to_the_webhook_for_a_single_message(
        tmp_path, posts, monkeypatch):
    monkeypatch.setenv("MERIDIAN_SLACK_CHANNEL_ID", "C0MAIN")
    monkeypatch.setenv("MERIDIAN_SLACK_TOKEN_CORNELIUS", "xoxb-bad-token")

    def rejecting_post_as(bot_token, channel, text, blocks=None, thread_ts=None, timeout=10.0):
        return None  # e.g. Cornelius's bot was never invited to the channel

    monkeypatch.setattr(notifications, "post_as", rejecting_post_as)
    n = Notifier(_bot_cfg(tmp_path))
    assert n.bot_mode is True

    assert n.stock_fills([Trade("d", "SPY", "buy", 1, 1, 0, "r")], 1, 1) is True
    assert len(posts) == 1 and posts[0][0] == "https://hooks.test/main"


# ------------------------------------------------------------------ integration


def _ramp(n=60, start=440.0):
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    close = start + pd.Series(range(n), dtype=float).values
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                         "volume": 1e6}, index=idx)


def test_options_leg_posts_the_open_and_returns_standup_lines(tmp_path, posts, monkeypatch):
    from tests.test_main_cli import _live_chain
    cfg = _cfg(tmp_path, enabled=True, underlyings=["SPY"])
    bars = _ramp()
    spot = float(bars["close"].iloc[-1])
    exp = (date.today() + timedelta(days=30)).isoformat()
    monkeypatch.setattr(OptionsDataAgent, "fetch_universe",
                        lambda self, symbols=None: {"SPY": _live_chain("SPY", spot, exp)})
    market = {"SPY": MarketData("SPY", "stocks", bars, "yfinance", pd.Timestamp.now(tz="UTC"))}

    leg = main._options_leg(cfg, market, notifier=Notifier(cfg))
    assert len(leg.new_trades) == 1
    assert any("OPENED" in _all_text(p[2]) for p in posts)
    agents = [l.agent for l in leg.lines]
    assert agents == ["Augustus", "SPARK", "Theo", "Joseph"]
    assert leg.book.name == "Options desk" and leg.book.fills_today == 1

    posts.clear()
    main._options_leg(cfg, market, notifier=Notifier(cfg))  # holds, no new trade
    assert posts == []


def test_options_cutoff_breach_posts_a_halt(tmp_path, posts, monkeypatch):
    from agents.options_broker import OptionsBroker
    cfg = _cfg(tmp_path, enabled=True, underlyings=["SPY"])
    broker = OptionsBroker(cfg)
    ledger = broker.load_ledger()
    ledger.cash = 400.0  # below the $500 floor
    broker.save_ledger(ledger)
    monkeypatch.setattr(OptionsDataAgent, "fetch_universe", lambda self, symbols=None: {})
    market = {"SPY": MarketData("SPY", "stocks", _ramp(), "yfinance", pd.Timestamp.now(tz="UTC"))}

    leg = main._options_leg(cfg, market, notifier=Notifier(cfg))
    assert leg.ledger.halted
    halts = [p for p in posts if "HALTED" in _all_text(p[2])]
    assert len(halts) == 1 and "clear-halt --options" in _all_text(halts[0][2])
    assert leg.lines[-1].tone == "bad"


def test_killswitch_and_clear_halt_post(tmp_path, posts):
    cfg = _cfg(tmp_path, enabled=True)
    assert main.run_killswitch(cfg, symbols=["SPY"], options=True) == 0
    assert main.run_clear_halt(cfg, options=True) == 0
    texts = [p[1] for p in posts]
    assert any("Options desk halted" in t for t in texts)
    assert any("Options desk halt cleared" in t for t in texts)


def test_a_crashed_paper_run_posts_a_failure_and_exits_1(tmp_path, posts, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(main, "load_config", lambda *a, **k: cfg)
    monkeypatch.setattr(main, "setup_logging", lambda **k: None)

    def crash(*a, **k):
        raise RuntimeError("data feed exploded")
    monkeypatch.setattr(main, "run_paper", crash)

    assert main.main(["paper"]) == 1
    assert len(posts) == 1 and "Paper run failed" in posts[0][1]


def test_no_slack_flag_silences_failure_alerts(tmp_path, posts, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(main, "load_config", lambda *a, **k: cfg)
    monkeypatch.setattr(main, "setup_logging", lambda **k: None)
    monkeypatch.setattr(main, "run_paper", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    assert main.main(["paper", "--no-slack"]) == 1
    assert posts == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
