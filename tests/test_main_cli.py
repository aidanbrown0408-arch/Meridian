"""Options routing tests — Augustus -> SPARK-calls strategy -> Theo -> Joseph
end-to-end via `main._run_options_leg`, plus the `--options` flag on
killswitch/clear-halt.

`_run_options_leg` no longer fetches its own stock bars (that was
`run_paper_options`'s job, now retired) -- it's handed a `market` dict the
same way `run_paper` hands it one, reusing whatever Wong already fetched
for the live pipeline that run. Tests here build that `market` dict
themselves via a plain `DataAgent(cfg).fetch_universe(symbols)` call, same
as `run_paper` does internally, then pass it straight to
`main._run_options_leg`.

Every test redirects the options ledger to a temp path (same trick
`test_options_broker.py` uses) so nothing here touches a real ledger file.
Most tests don't mock network: `allow_synthetic_fallback` is on by default,
so a synthetic stock/options fetch never produces a proposal (see
`strategies/options_strategy.py`) and these pass identically whether or not
yfinance is reachable from this machine. The "forced live" tests near the
bottom monkeypatch both data agents to prove the strategy actually opens a
position when real (non-synthetic) data says to -- the scenario the
operator will hit for real once yfinance is reachable.

Run with:  python -m pytest tests/test_main_cli.py -q
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main  # noqa: E402
import utils.notifications as main_notifications  # noqa: E402
from agents.data_agent import DataAgent, MarketData  # noqa: E402
from agents.options_broker import OptionsBroker  # noqa: E402
from agents.options_data_agent import (  # noqa: E402
    CHAIN_COLUMNS, OptionsChain, OptionsDataAgent,
)
from utils.config import Config, load_config  # noqa: E402

BASE_CONFIG = load_config()


def _options_config(tmp_path, **overrides) -> Config:
    """A full config with the options ledger redirected to a temp file, plus
    any dotted-key overrides under the `options` section."""
    data = BASE_CONFIG.as_dict()
    data["options"]["ledger_path"] = str(tmp_path / "options_ledger.json")
    for key, value in overrides.items():
        data["options"][key] = value
    return Config(data)


def _run_options_leg(cfg: Config, symbols: list[str]):
    """What `run_paper` does now: fetch the stock bars once, hand that same
    `market` dict to the options leg. Standing in here for the live
    pipeline's own Wong fetch, since these tests exercise the options leg
    in isolation."""
    market = DataAgent(cfg).fetch_universe(symbols)
    return main._run_options_leg(cfg, market)


# ------------------------------------------------------------------ argparse


def test_parser_accepts_paper_options_mode():
    args = main.build_parser().parse_args(["paper-options"])
    assert args.mode == "paper-options"


def test_parser_accepts_options_flag_on_killswitch():
    args = main.build_parser().parse_args(["killswitch", "--options"])
    assert args.mode == "killswitch"
    assert args.options is True


def test_options_flag_defaults_false():
    args = main.build_parser().parse_args(["killswitch"])
    assert args.options is False


# ------------------------------------------------------------------ options leg


def test_options_leg_is_noop_when_disabled(tmp_path):
    cfg = _options_config(tmp_path, enabled=False)
    assert _run_options_leg(cfg, ["SPY"]) is None


def test_options_leg_opens_nothing_without_live_data(tmp_path):
    """Without network, both Wong and Augustus fall back to synthetic. The
    strategy may well read a signal off a synthetic random walk, but a
    synthetic option chain is never tradable, so it's skipped before a
    proposal is ever built -- ledger stays untouched. Proves the wiring
    (fetch -> signal check -> Theo -> Joseph) runs end-to-end and fails
    safe with no live data."""
    cfg = _options_config(tmp_path, enabled=True)
    ledger = _run_options_leg(cfg, ["SPY"])
    assert ledger is not None
    assert ledger.positions == {}
    assert ledger.cash == ledger.starting_capital
    assert not ledger.halted


def test_options_leg_persists_the_ledger_across_runs(tmp_path):
    cfg = _options_config(tmp_path, enabled=True)
    first = _run_options_leg(cfg, ["SPY"])
    second = _run_options_leg(cfg, ["SPY"])
    assert first.created_at == second.created_at  # same ledger file, not recreated


# ------------------------------------------------------------------ forced-live (Phase B)


def _ramp_bars(n: int = 60, start: float = 440.0) -> pd.DataFrame:
    index = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    close = start + np.arange(n, dtype=float)
    return pd.DataFrame({"open": close, "high": close * 1.001,
                         "low": close * 0.999, "close": close,
                         "volume": 1e6}, index=index)


def _live_chain(symbol: str, spot: float, expiration: str) -> OptionsChain:
    rows = []
    for strike in (spot - 10, spot - 5, spot, spot + 5, spot + 10):
        mid = max(spot - strike, 0.0) * 0.1 + 1.0  # keeps ATM well under Theo's $150 cap
        rows.append({"underlying": symbol, "option_type": "long_call", "strike": strike,
                    "expiration": expiration, "bid": round((mid - 0.1) * 100, 2),
                    "ask": round((mid + 0.1) * 100, 2), "last": round(mid * 100, 2),
                    "volume": 100, "open_interest": 500, "implied_volatility": 0.2})
    calls = pd.DataFrame(rows, columns=CHAIN_COLUMNS)
    puts = pd.DataFrame(columns=CHAIN_COLUMNS)
    return OptionsChain(symbol, [expiration], calls, puts, "yfinance", pd.Timestamp.now(tz="UTC"))


def test_options_leg_opens_a_position_with_live_data(tmp_path, monkeypatch):
    """Force both agents to return real (non-synthetic) data with SPARK
    reading long -- the scenario your own machine hits once yfinance is
    reachable. Proves Joseph actually opens a position through the full
    stack, not just that the wiring runs without crashing."""
    cfg = _options_config(tmp_path, enabled=True)
    bars = _ramp_bars()
    spot = float(bars["close"].iloc[-1])
    expiration = (date.today() + timedelta(days=30)).isoformat()

    def fake_stock_fetch(self, symbols=None):
        return {"SPY": MarketData("SPY", "stocks", bars, "yfinance", pd.Timestamp.now(tz="UTC"))}

    def fake_chain_fetch(self, symbols=None):
        return {"SPY": _live_chain("SPY", spot, expiration)}

    monkeypatch.setattr(DataAgent, "fetch_universe", fake_stock_fetch)
    monkeypatch.setattr(OptionsDataAgent, "fetch_universe", fake_chain_fetch)

    ledger = _run_options_leg(cfg, ["SPY"])
    assert len(ledger.positions) == 1
    position = next(iter(ledger.positions.values()))
    assert position.underlying == "SPY" and position.option_type == "long_call"
    assert position.strike == spot
    assert ledger.cash < ledger.starting_capital  # premium was actually paid


def test_options_leg_does_not_pyramid_on_a_second_run(tmp_path, monkeypatch):
    """Same forced-live setup, run twice: the signal stays long both times,
    but the second run must not open a second SPY call."""
    cfg = _options_config(tmp_path, enabled=True)
    bars = _ramp_bars()
    spot = float(bars["close"].iloc[-1])
    expiration = (date.today() + timedelta(days=30)).isoformat()

    monkeypatch.setattr(DataAgent, "fetch_universe",
                        lambda self, symbols=None: {
                            "SPY": MarketData("SPY", "stocks", bars, "yfinance",
                                             pd.Timestamp.now(tz="UTC"))})
    monkeypatch.setattr(OptionsDataAgent, "fetch_universe",
                        lambda self, symbols=None: {"SPY": _live_chain("SPY", spot, expiration)})

    _run_options_leg(cfg, ["SPY"])
    second = _run_options_leg(cfg, ["SPY"])
    assert len(second.positions) == 1  # still just the one, not stacked


# ------------------------------------------------------------------ killswitch / clear-halt


def test_killswitch_options_flag_routes_to_joseph_not_cornelius(tmp_path):
    """The stock ledger must not exist/move; only the options ledger halts."""
    cfg = _options_config(tmp_path, enabled=True)
    _run_options_leg(cfg, ["SPY"])  # create the options ledger
    assert main.run_killswitch(cfg, symbols=["SPY"], options=True) == 0

    from agents.options_broker import OptionsBroker
    ledger = OptionsBroker(cfg).load_ledger()
    assert ledger.halted
    assert ledger.halt_reason == "operator killswitch"


def test_clear_halt_options_flag_clears_only_the_options_halt(tmp_path):
    cfg = _options_config(tmp_path, enabled=True)
    _run_options_leg(cfg, ["SPY"])
    main.run_killswitch(cfg, symbols=["SPY"], options=True)

    assert main.run_clear_halt(cfg, options=True) == 0

    from agents.options_broker import OptionsBroker
    ledger = OptionsBroker(cfg).load_ledger()
    assert not ledger.halted
    assert ledger.halt_reason == ""


# ------------------------------------------------------------------ routing (no double-exposure)


def test_route_options_underlyings_zeroes_only_the_options_symbols():
    """The core guarantee: SPY/QQQ's target weight is zeroed before it ever
    reaches Cornelius, every other symbol's netted position is untouched,
    and nothing else about the decision (capped flag, contributors, for
    the dashboard/shadow reporting) is lost in the process."""
    from agents.regime_agent import RegimeReport
    from agents.risk_agent import NettedPosition

    report = RegimeReport(netted_positions={
        "SPY": NettedPosition("SPY", target_weight=0.4, capped=False, contributors=["SPARK"]),
        "QQQ": NettedPosition("QQQ", target_weight=0.1, capped=True, contributors=["SPARK"]),
        "AAPL": NettedPosition("AAPL", target_weight=0.2, capped=False, contributors=["Momentum"]),
    })

    routed = main._route_options_underlyings(report, {"SPY", "QQQ"})

    assert routed.netted_positions["SPY"].target_weight == 0.0
    assert routed.netted_positions["QQQ"].target_weight == 0.0
    assert routed.netted_positions["AAPL"].target_weight == 0.2  # untouched
    # Everything else about the position survives -- only the weight moves.
    assert routed.netted_positions["QQQ"].capped is True
    assert routed.netted_positions["QQQ"].contributors == ["SPARK"]
    # The input report is never mutated in place.
    assert report.netted_positions["SPY"].target_weight == 0.4


def test_route_options_underlyings_ignores_symbols_with_no_signal_today():
    from agents.regime_agent import RegimeReport
    from agents.risk_agent import NettedPosition

    report = RegimeReport(netted_positions={
        "AAPL": NettedPosition("AAPL", target_weight=0.2, capped=False, contributors=["Momentum"]),
    })
    routed = main._route_options_underlyings(report, {"SPY", "QQQ"})
    assert routed.netted_positions == report.netted_positions


def test_paper_options_cli_mode_is_deprecated_and_no_longer_runs(tmp_path, monkeypatch):
    """`python main.py paper-options` used to run a standalone leg; now it
    should refuse and point the operator at `paper` instead, rather than
    silently doing nothing useful."""
    called = {"paper": False}
    monkeypatch.setattr(main, "run_paper", lambda *a, **k: called.__setitem__("paper", True))
    monkeypatch.setattr(main, "load_config",
                        lambda *a, **k: _options_config(tmp_path, enabled=True))
    rc = main.main(["paper-options"])
    assert rc == 1
    assert called["paper"] is False  # deprecated mode never silently runs the merged pipeline


# ------------------------------------------------------------------ options leg: data + compliance


def test_options_leg_fetches_underlyings_missing_from_market(tmp_path, monkeypatch):
    """`paper --symbols AAPL` hands the options leg a market with no SPY in
    it. The leg must fetch SPY itself rather than silently reading flat."""
    cfg = _options_config(tmp_path, enabled=True, underlyings=["SPY"])
    requested = []

    def fake_stock_fetch(self, symbols=None):
        requested.append(list(symbols or []))
        return {s: MarketData(s, "stocks", _ramp_bars(), "yfinance",
                              pd.Timestamp.now(tz="UTC")) for s in symbols}

    monkeypatch.setattr(DataAgent, "fetch_universe", fake_stock_fetch)
    monkeypatch.setattr(OptionsDataAgent, "fetch_universe", lambda self, symbols=None: {})
    seen = {}

    def spy_build(config, market, chains, ledger, augustus):
        seen.update(market)
        return [], []

    monkeypatch.setattr(main, "build_proposals", spy_build)
    main._run_options_leg(cfg, {"AAPL": object()})
    assert requested == [["SPY"]]          # only the missing one is fetched
    assert "SPY" in seen and "AAPL" in seen  # original market kept, SPY added


def test_options_leg_skips_compliance_blocked_underlyings(tmp_path, monkeypatch):
    """If David blocked SPY this run, the options desk must not open a call
    on it -- even with live data and SPARK reading long."""
    cfg = _options_config(tmp_path, enabled=True, underlyings=["SPY"])
    bars = _ramp_bars()
    spot = float(bars["close"].iloc[-1])
    expiration = (date.today() + timedelta(days=30)).isoformat()
    market = {"SPY": MarketData("SPY", "stocks", bars, "yfinance", pd.Timestamp.now(tz="UTC"))}
    monkeypatch.setattr(OptionsDataAgent, "fetch_universe",
                        lambda self, symbols=None: {"SPY": _live_chain("SPY", spot, expiration)})

    ledger = main._run_options_leg(cfg, market, blocked={"SPY": "stale data"})
    assert ledger.positions == {}

    unblocked = main._run_options_leg(cfg, market)  # sanity: same inputs, unblocked, opens
    assert len(unblocked.positions) == 1


# ------------------------------------------------------------------ run_paper wiring


def _stub_run_paper(monkeypatch, tmp_path, market, regime_report, *, options_fails=False):
    """Stub every agent around run_paper so the test sees exactly what
    Cornelius, George and the options leg were each handed."""
    from agents.regime_agent import RegimeAgent
    from agents.risk_agent import RiskReport

    cfg = _options_config(tmp_path, enabled=True, underlyings=["SPY", "QQQ"])
    seen: dict = {"order": []}

    class _Lifecycle:
        def recommend(self, *a, **k):
            return []

    monkeypatch.setattr(main, "_run_pipeline", lambda config, symbols=None: (
        market, {}, {"QQQ": "blocked for test"}, type("Leo", (), {"strategies": {}})(),
        [], {}, RiskReport(), regime_report))
    monkeypatch.setattr(main, "_lifecycle_step", lambda *a, **k: (_Lifecycle(), {}))
    monkeypatch.setattr(RegimeAgent, "__init__", lambda self, config: None)
    monkeypatch.setattr(RegimeAgent, "run", lambda self, *a, **k: regime_report)

    class _Ledger:
        trader_shadow: dict = {}
        trades: list = []
        cash = 5000.0
        starting_capital = 5000.0
        positions: dict = {}
        halted = False
        halt_reason = ""

        def mark_to_market(self, prices):
            return 5000.0

    def fake_execute(self, market_, netted, adjustments=None):
        seen["order"].append("cornelius")
        seen["cornelius"] = netted
        return _Ledger()

    def fake_george(self, market_, compliance, results, benchmarks, risk, regime, **k):
        seen["order"].append("george")
        seen["george"] = regime.netted_positions
        return type("Dash", (), {"html_path": None, "posted_to_slack": False,
                                 "standup_text": ""})()

    def fake_print(*a, **k):
        seen["order"].append("print")

    def fake_options(config, market_, blocked=None, notifier=None):
        seen["order"].append("options")
        seen["options_market"] = market_
        seen["options_blocked"] = blocked
        if options_fails:
            raise RuntimeError("chain fetch exploded")

    monkeypatch.setattr(main.PaperBroker, "execute", fake_execute)
    monkeypatch.setattr(main, "_warn_on_legacy_share_positions", lambda *a, **k: [])
    monkeypatch.setattr(main.ReportingAgent, "run", fake_george)
    monkeypatch.setattr(main, "_print_report", fake_print)
    monkeypatch.setattr(main, "results_frame", lambda results: pd.DataFrame())
    monkeypatch.setattr(main, "_options_leg", fake_options)
    monkeypatch.setattr(main, "_trade_count", lambda broker: 0)
    monkeypatch.setattr(main.ReportingAgent, "_prices", staticmethod(lambda market_: {}))
    monkeypatch.setattr(main.ReportingAgent, "standup_payload",
                        lambda self, *a, **k: main_notifications.StandupPayload(mode="Paper"))
    return cfg, seen


def _report_with_spy_and_aapl():
    from agents.regime_agent import RegimeReport
    from agents.risk_agent import NettedPosition
    return RegimeReport(netted_positions={
        "SPY": NettedPosition("SPY", 0.4, False, ["SPARK"]),
        "AAPL": NettedPosition("AAPL", 0.2, False, ["Momentum"]),
    })


def test_run_paper_routes_spy_to_options_and_keeps_the_dashboard_honest(tmp_path, monkeypatch):
    market = {"SPY": object(), "AAPL": object()}
    cfg, seen = _stub_run_paper(monkeypatch, tmp_path, market, _report_with_spy_and_aapl())
    main.run_paper(cfg, post_slack=False)

    assert seen["cornelius"]["SPY"].target_weight == 0.0   # never opened as shares
    assert seen["cornelius"]["AAPL"].target_weight == 0.2
    assert seen["george"]["SPY"].target_weight == 0.4      # dashboard shows the real decision
    assert seen["options_market"] is market                # same bars, not a second read
    assert seen["options_blocked"] == {"QQQ": "blocked for test"}
    assert seen["order"] == ["cornelius", "george", "print", "options"]


def test_run_paper_survives_an_options_leg_failure(tmp_path, monkeypatch):
    market = {"SPY": object()}
    cfg, seen = _stub_run_paper(monkeypatch, tmp_path, market, _report_with_spy_and_aapl(),
                                options_fails=True)
    main.run_paper(cfg, post_slack=False)  # must not raise
    assert seen["order"] == ["cornelius", "george", "print", "options"]


def test_run_paper_leaves_spy_as_shares_when_options_disabled(tmp_path, monkeypatch):
    market = {"SPY": object()}
    cfg, seen = _stub_run_paper(monkeypatch, tmp_path, market, _report_with_spy_and_aapl())
    cfg = _options_config(tmp_path, enabled=False)
    main.run_paper(cfg, post_slack=False)
    assert seen["cornelius"]["SPY"].target_weight == 0.4
    assert "options" not in seen["order"]


# ------------------------------------------------------------------ leftover shares


def test_routing_closes_leftover_spy_shares_and_says_so(tmp_path, caplog):
    """A zero target is a sell, not a skip: shares bought before routing
    existed are closed out, and the run logs that it is doing so."""
    import logging
    from datetime import datetime, timezone

    from agents.portfolio_agent import PaperBroker, Position
    from agents.regime_agent import RegimeReport
    from agents.risk_agent import NettedPosition

    cfg = _options_config(tmp_path, enabled=True, underlyings=["SPY"])
    broker = PaperBroker(cfg)
    broker.ledger_path = tmp_path / "paper_ledger.json"
    ledger = broker.load_ledger()
    ledger.positions["SPY"] = Position("SPY", "stocks", 5.0, 400.0,
                                       datetime.now(timezone.utc).date().isoformat())
    ledger.cash -= 2000.0
    broker.save_ledger(ledger)

    bars = _ramp_bars(n=5, start=400.0)
    market = {"SPY": MarketData("SPY", "stocks", bars, "test", pd.Timestamp.now(tz="UTC"))}
    report = RegimeReport(netted_positions={"SPY": NettedPosition("SPY", 0.4, False, ["SPARK"])})

    with caplog.at_level(logging.WARNING):
        legacy = main._warn_on_legacy_share_positions(broker, market, {"SPY"})
    assert legacy == ["SPY"]
    assert any("leftover share" in r.getMessage() for r in caplog.records)

    routed = main._route_options_underlyings(report, {"SPY"})
    after = broker.execute(market, routed.netted_positions, [])
    assert "SPY" not in after.positions


def test_allocation_report_marks_routed_symbols(capsys):
    """Regression: the console allocation table must render (not NameError)
    when there are target positions, and flag the options-routed ones."""
    from agents.regime_agent import RegimeReport
    from agents.risk_agent import NettedPosition, RiskReport

    report = RegimeReport(netted_positions={
        "SPY": NettedPosition("SPY", 0.4, False, ["SPARK"]),
        "BTC/USDT": NettedPosition("BTC/USDT", 0.2, False, ["Momentum"]),
    })
    main._print_allocation(RiskReport(), report, {"SPY"})
    out = capsys.readouterr().out
    assert "SPY" in out and "-> options desk, not shares" in out
    btc_line = next(l for l in out.splitlines() if "BTC/USDT" in l)
    assert "options desk" not in btc_line


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
