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


# ------------------------------------------------------------------ run_paper_options


def test_run_paper_options_is_noop_when_disabled(tmp_path):
    cfg = _options_config(tmp_path, enabled=False)
    assert _run_options_leg(cfg, ["SPY"]) is None


def test_run_paper_options_opens_nothing_without_live_data(tmp_path):
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


def test_run_paper_options_persists_the_ledger_across_runs(tmp_path):
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


def test_run_paper_options_opens_a_position_with_live_data(tmp_path, monkeypatch):
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


def test_run_paper_options_does_not_pyramid_on_a_second_run(tmp_path, monkeypatch):
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


def test_paper_options_cli_mode_is_deprecated_and_no_longer_runs(tmp_path, monkeypatch, capsys):
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


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
