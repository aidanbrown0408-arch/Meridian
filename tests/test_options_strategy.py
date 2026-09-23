"""Tests for the Phase B options strategy — "buy a call when SPARK signals
long" (strategies/options_strategy.py).

Run with:  python -m pytest tests/test_options_strategy.py -q
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.data_agent import MarketData  # noqa: E402
from agents.options_broker import OptionsLedger, OptionsPosition  # noqa: E402
from agents.options_data_agent import (  # noqa: E402
    CHAIN_COLUMNS, OptionsChain, OptionsDataAgent,
)
from strategies.options_strategy import (  # noqa: E402
    build_proposals, check_put_signals, check_signals,
)
from utils.config import Config, load_config  # noqa: E402

CONFIG = load_config()
AUGUSTUS = OptionsDataAgent(CONFIG)


def _config_with(**overrides) -> Config:
    """A Config copy with one or more `options.strategy.*` keys overridden --
    for toggling `puts_enabled` without touching the real config file."""
    data = CONFIG.as_dict()
    data["options"]["strategy"].update(overrides)
    return Config(data)


# ------------------------------------------------------------------ fixtures


def _ramp(n: int = 60, start: float = 440.0, step: float = 1.0) -> pd.DataFrame:
    """Monotonically rising close series -- SPARK (a breakout trader) reads
    this as long once warmup clears, same helper style as test_phase1.py."""
    index = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    close = start + step * np.arange(n)
    return pd.DataFrame({"open": close, "high": close * 1.001,
                         "low": close * 0.999, "close": close,
                         "volume": 1e6}, index=index)


def _flat(n: int = 60, level: float = 440.0) -> pd.DataFrame:
    """Constant price -- never breaks its own rolling high, so SPARK stays
    flat the whole series."""
    index = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    close = np.full(n, level)
    return pd.DataFrame({"open": close, "high": close, "low": close,
                         "close": close, "volume": 1e6}, index=index)


def _falling(n: int = 60, start: float = 440.0, step: float = 1.0) -> pd.DataFrame:
    """Monotonically falling close series -- the mirror-image of `_ramp`.
    SPARK itself reads this as flat throughout (it only ever goes long, it
    never shorts), but the put-side Donchian breakdown reads it as active
    once warmup clears."""
    index = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    close = start - step * np.arange(n)
    return pd.DataFrame({"open": close, "high": close * 1.001,
                         "low": close * 0.999, "close": close,
                         "volume": 1e6}, index=index)


def _market(bars: pd.DataFrame, symbol: str = "SPY",
           source: str = "yfinance") -> dict[str, MarketData]:
    return {symbol: MarketData(symbol, "stocks", bars, source,
                               pd.Timestamp.now(tz="UTC"))}


def _chain(symbol: str, spot: float, expiration: str,
          synthetic: bool = False) -> OptionsChain:
    """A small, deterministic real-shaped chain: 5 strikes around spot, both
    sides populated (like a real yfinance chain always has calls and puts)."""
    strikes = [spot - 10, spot - 5, spot, spot + 5, spot + 10]

    def _side(option_type: str) -> pd.DataFrame:
        rows = []
        for strike in strikes:
            intrinsic = (max(spot - strike, 0.0) if option_type == "long_call"
                        else max(strike - spot, 0.0))
            mid = intrinsic + 3.0
            rows.append({"underlying": symbol, "option_type": option_type,
                        "strike": strike, "expiration": expiration,
                        "bid": round((mid - 0.1) * 100, 2), "ask": round((mid + 0.1) * 100, 2),
                        "last": round(mid * 100, 2), "volume": 100, "open_interest": 500,
                        "implied_volatility": 0.2})
        return pd.DataFrame(rows, columns=CHAIN_COLUMNS)

    source = "synthetic" if synthetic else "yfinance"
    return OptionsChain(symbol, [expiration], _side("long_call"), _side("long_put"),
                        source, pd.Timestamp.now(tz="UTC"))


def _empty_ledger() -> OptionsLedger:
    return OptionsLedger(starting_capital=1000.0, cash=1000.0)


def _exp(days_out: int = 30) -> str:
    return (date.today() + timedelta(days=days_out)).isoformat()


# ------------------------------------------------------------------ check_signals


def test_signal_is_long_on_a_rising_series():
    market = _market(_ramp())
    checks = check_signals(CONFIG, market)
    spy = next(c for c in checks if c.underlying == "SPY")
    assert spy.signal_long
    assert "long" in spy.detail


def test_signal_is_flat_on_a_flat_series():
    market = _market(_flat())
    checks = check_signals(CONFIG, market)
    spy = next(c for c in checks if c.underlying == "SPY")
    assert not spy.signal_long


def test_missing_symbol_reads_flat_with_reason():
    checks = check_signals(CONFIG, {})
    assert all(not c.signal_long and c.detail for c in checks)


def test_short_history_reads_flat_not_crashes():
    market = _market(_ramp(n=5))  # well under SPARK's warmup
    checks = check_signals(CONFIG, market)
    spy = next(c for c in checks if c.underlying == "SPY")
    assert not spy.signal_long
    assert "bars" in spy.detail


# ------------------------------------------------------------------ build_proposals


def test_long_signal_with_live_chain_produces_one_proposal():
    market = _market(_ramp())
    spot = float(market["SPY"].bars["close"].iloc[-1])
    exp = _exp(30)
    chains = {"SPY": _chain("SPY", spot, exp)}
    proposals, checks = build_proposals(CONFIG, market, chains, _empty_ledger(), AUGUSTUS)
    assert len(proposals) == 1
    p = proposals[0]
    assert p.underlying == "SPY" and p.option_type == "long_call"
    assert p.expiration == exp
    assert p.strike == spot  # exact ATM strike is in the fixture
    assert p.premium_per_contract > 0
    assert p.contracts == int(CONFIG.get("options.strategy.contracts_per_signal", 1))
    assert "SPARK" in p.reason


def test_flat_signal_produces_no_proposal():
    market = _market(_flat())
    spot = float(market["SPY"].bars["close"].iloc[-1])
    chains = {"SPY": _chain("SPY", spot, _exp(30))}
    proposals, _ = build_proposals(CONFIG, market, chains, _empty_ledger(), AUGUSTUS)
    assert proposals == []


def test_synthetic_chain_blocks_a_proposal_even_if_long():
    market = _market(_ramp())
    spot = float(market["SPY"].bars["close"].iloc[-1])
    chains = {"SPY": _chain("SPY", spot, _exp(30), synthetic=True)}
    proposals, _ = build_proposals(CONFIG, market, chains, _empty_ledger(), AUGUSTUS)
    assert proposals == []


def test_already_open_call_blocks_pyramiding():
    market = _market(_ramp())
    spot = float(market["SPY"].bars["close"].iloc[-1])
    chains = {"SPY": _chain("SPY", spot, _exp(30))}
    ledger = _empty_ledger()
    ledger.positions["SPY|long_call|450|2026-12-18"] = OptionsPosition(
        underlying="SPY", option_type="long_call", strike=450.0,
        expiration="2026-12-18", contracts=1, premium_paid=100.0,
        entry_premium_per_contract=100.0, entry_date=date.today().isoformat(),
    )
    proposals, _ = build_proposals(CONFIG, market, chains, ledger, AUGUSTUS)
    assert proposals == []


def test_expiration_too_close_is_skipped_for_a_later_one():
    """min_days_to_expiration (config, default 3) should push selection past
    an expiration that's inside the window."""
    market = _market(_ramp())
    spot = float(market["SPY"].bars["close"].iloc[-1])
    near, far = _exp(1), _exp(30)
    chain = _chain("SPY", spot, far)
    # Splice in a too-near expiration as an earlier candidate.
    near_rows = chain.calls.copy()
    near_rows["expiration"] = near
    chain.calls = pd.concat([near_rows, chain.calls], ignore_index=True)
    chain.expirations = [near, far]

    proposals, _ = build_proposals(CONFIG, market, chain and {"SPY": chain}, _empty_ledger(),
                                   AUGUSTUS)
    assert len(proposals) == 1
    assert proposals[0].expiration == far


def test_multiple_underlyings_each_evaluated_independently():
    market = {
        **_market(_ramp(), symbol="SPY"),
        **_market(_flat(), symbol="QQQ"),
    }
    spy_spot = float(market["SPY"].bars["close"].iloc[-1])
    qqq_spot = float(market["QQQ"].bars["close"].iloc[-1])
    chains = {
        "SPY": _chain("SPY", spy_spot, _exp(30)),
        "QQQ": _chain("QQQ", qqq_spot, _exp(30)),
    }
    proposals, checks = build_proposals(CONFIG, market, chains, _empty_ledger(), AUGUSTUS)
    assert {p.underlying for p in proposals} == {"SPY"}
    call_reads = {c.underlying: c.signal_long for c in checks if c.direction == "call"}
    assert call_reads == {"SPY": True, "QQQ": False}


# ------------------------------------------------------------------ check_put_signals


def test_put_signal_is_active_on_a_falling_series():
    market = _market(_falling())
    checks = check_put_signals(CONFIG, market)
    spy = next(c for c in checks if c.underlying == "SPY")
    assert spy.signal_long
    assert spy.direction == "put"
    assert "breakdown" in spy.detail


def test_put_signal_is_flat_on_a_rising_series():
    market = _market(_ramp())
    checks = check_put_signals(CONFIG, market)
    spy = next(c for c in checks if c.underlying == "SPY")
    assert not spy.signal_long


def test_put_signals_empty_when_disabled():
    market = _market(_falling())
    checks = check_put_signals(_config_with(puts_enabled=False), market)
    assert checks == []


# ------------------------------------------------------------------ build_proposals (puts)


def test_falling_signal_with_live_chain_produces_one_put_proposal():
    market = _market(_falling())
    spot = float(market["SPY"].bars["close"].iloc[-1])
    exp = _exp(30)
    chains = {"SPY": _chain("SPY", spot, exp)}
    proposals, checks = build_proposals(CONFIG, market, chains, _empty_ledger(), AUGUSTUS)
    assert len(proposals) == 1
    p = proposals[0]
    assert p.underlying == "SPY" and p.option_type == "long_put"
    assert p.expiration == exp
    assert p.strike == spot
    assert p.premium_per_contract > 0
    assert "SPARK" in p.reason


def test_open_call_blocks_a_put_proposal_on_the_same_underlying():
    """The no-straddling rule: an open call on SPY must block a new put
    proposal on SPY too, not just another call."""
    market = _market(_falling())
    spot = float(market["SPY"].bars["close"].iloc[-1])
    chains = {"SPY": _chain("SPY", spot, _exp(30))}
    ledger = _empty_ledger()
    ledger.positions["SPY|long_call|450|2026-12-18"] = OptionsPosition(
        underlying="SPY", option_type="long_call", strike=450.0,
        expiration="2026-12-18", contracts=1, premium_paid=100.0,
        entry_premium_per_contract=100.0, entry_date=date.today().isoformat(),
    )
    proposals, _ = build_proposals(CONFIG, market, chains, ledger, AUGUSTUS)
    assert proposals == []


def test_open_put_blocks_a_call_proposal_on_the_same_underlying():
    market = _market(_ramp())
    spot = float(market["SPY"].bars["close"].iloc[-1])
    chains = {"SPY": _chain("SPY", spot, _exp(30))}
    ledger = _empty_ledger()
    ledger.positions["SPY|long_put|430|2026-12-18"] = OptionsPosition(
        underlying="SPY", option_type="long_put", strike=430.0,
        expiration="2026-12-18", contracts=1, premium_paid=100.0,
        entry_premium_per_contract=100.0, entry_date=date.today().isoformat(),
    )
    proposals, _ = build_proposals(CONFIG, market, chains, ledger, AUGUSTUS)
    assert proposals == []


def test_puts_disabled_never_produces_a_put_proposal():
    market = _market(_falling())
    spot = float(market["SPY"].bars["close"].iloc[-1])
    chains = {"SPY": _chain("SPY", spot, _exp(30))}
    proposals, checks = build_proposals(_config_with(puts_enabled=False), market, chains,
                                        _empty_ledger(), AUGUSTUS)
    assert proposals == []
    assert all(c.direction == "call" for c in checks)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
