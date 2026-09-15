"""`paper-options` CLI wiring tests — Augustus -> SPARK-calls strategy ->
Theo -> Joseph end-to-end, plus the `--options` flag on killswitch/clear-halt.

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
    assert main.run_paper_options(cfg, symbols=["SPY"]) is None


def test_run_paper_options_opens_nothing_without_live_data(tmp_path):
    """Without network, both Wong and Augustus fall back to synthetic. The
    strategy may well read a signal off a synthetic random walk, but a
    synthetic option chain is never tradable, so it's skipped before a
    proposal is ever built -- ledger stays untouched. Proves the wiring
    (fetch -> signal check -> Theo -> Joseph) runs end-to-end and fails
    safe with no live data."""
    cfg = _options_config(tmp_path, enabled=True)
    ledger = main.run_paper_options(cfg, symbols=["SPY"])
    assert ledger is not None
    assert ledger.positions == {}
    assert ledger.cash == ledger.starting_capital
    assert not ledger.halted


def test_run_paper_options_persists_the_ledger_across_runs(tmp_path):
    cfg = _options_config(tmp_path, enabled=True)
    first = main.run_paper_options(cfg, symbols=["SPY"])
    second = main.run_paper_options(cfg, symbols=["SPY"])
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

    ledger = main.run_paper_options(cfg, symbols=["SPY"])
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

    main.run_paper_options(cfg, symbols=["SPY"])
    second = main.run_paper_options(cfg, symbols=["SPY"])
    assert len(second.positions) == 1  # still just the one, not stacked


# ------------------------------------------------------------------ killswitch / clear-halt


def test_killswitch_options_flag_routes_to_joseph_not_cornelius(tmp_path):
    """The stock ledger must not exist/move; only the options ledger halts."""
    cfg = _options_config(tmp_path, enabled=True)
    main.run_paper_options(cfg, symbols=["SPY"])  # create the options ledger
    assert main.run_killswitch(cfg, symbols=["SPY"], options=True) == 0

    from agents.options_broker import OptionsBroker
    ledger = OptionsBroker(cfg).load_ledger()
    assert ledger.halted
    assert ledger.halt_reason == "operator killswitch"


def test_clear_halt_options_flag_clears_only_the_options_halt(tmp_path):
    cfg = _options_config(tmp_path, enabled=True)
    main.run_paper_options(cfg, symbols=["SPY"])
    main.run_killswitch(cfg, symbols=["SPY"], options=True)

    assert main.run_clear_halt(cfg, options=True) == 0

    from agents.options_broker import OptionsBroker
    ledger = OptionsBroker(cfg).load_ledger()
    assert not ledger.halted
    assert ledger.halt_reason == ""


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
