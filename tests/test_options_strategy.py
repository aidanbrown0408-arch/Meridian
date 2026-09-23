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
    _market_gates, build_proposals, check_put_signals, check_signals,
)
from utils.config import Config, load_config  # noqa: E402

CONFIG = load_config()
AUGUSTUS = OptionsDataAgent(CONFIG)


def _config_with(**overrides) -> Config:
    """A Config copy with one or more `options.strategy.*` keys overridden --
    for toggling `puts_enabled`/`market_gate_enabled` without touching the
    real config file."""
    data = CONFIG.as_dict()
    data["options"]["strategy"].update(overrides)
    return Config(data)


# The synthetic fixtures below (_ramp/_falling/_flat) are short, clean, and
# built purely to pin down signal/proposal mechanics -- not to land in any
# particular regime. Empirically (verified against the real RegimeAgent/
# RiskAgent) a 60-bar _ramp reads "undecided" (its % volatility shrinks as
# price compounds up, starving the vol-percentile check) and a 60-bar
# _falling reads "trending" but 0/5 walk-forward folds (SPARK never trades
# on a series that only ever falls). Either way the market gate (added
# after Phase B2, see "Market gate" in the module docstring) would block
# them, which is correct behavior but not what these particular tests are
# about -- they use GATE_OFF so gate behavior stays confined to its own
# section below.
GATE_OFF = _config_with(market_gate_enabled=False)


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


def _trend(n: int = 400, start: float = 100.0, annual_drift: float = 0.6,
          annual_vol: float = 0.25, seed: int = 11) -> pd.DataFrame:
    """Same fixture as `tests/test_phase3.py` -- a geometric random walk with
    upward drift, realistic enough that % vol stays roughly stationary.
    Classifies trending per Greg (see `test_regime_classifies_a_clean_trend_
    as_trending` in test_phase3.py); whether SPARK also clears Charles's
    walk-forward grading on a given seed/drift varies -- the default
    (seed=11, drift=0.6) does NOT clear it, seed=15/drift=0.9 does (verified
    empirically), which is exactly the trending-but-unvalidated vs.
    trending-and-validated split the gate tests below need."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(end=pd.Timestamp("2026-01-01"), periods=n)
    daily_vol = annual_vol / np.sqrt(252)
    daily_drift = annual_drift / 252 - 0.5 * daily_vol ** 2
    close = start * np.exp(np.cumsum(rng.normal(daily_drift, daily_vol, n)))
    return pd.DataFrame({"open": close, "high": close * 1.003, "low": close * 0.997,
                         "close": close, "volume": 1e6}, index=index)


def _chop(n: int = 400, base: float = 100.0, amp: float = 4.0, period: int = 8,
         seed: int = 3) -> pd.DataFrame:
    """Same fixture as `tests/test_phase3.py` -- a tight, fast-oscillating
    bounce. Classifies choppy per Greg."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(end=pd.Timestamp("2026-01-01"), periods=n)
    t = np.arange(n)
    close = base + amp * np.sin(2 * np.pi * t / period) + rng.normal(0, 0.2, n)
    return pd.DataFrame({"open": close, "high": close * 1.003, "low": close * 0.997,
                         "close": close, "volume": 1e6}, index=index)


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
    proposals, checks = build_proposals(GATE_OFF, market, chains, _empty_ledger(), AUGUSTUS)
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
    proposals, _ = build_proposals(GATE_OFF, market, chains, _empty_ledger(), AUGUSTUS)
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
    proposals, _ = build_proposals(GATE_OFF, market, chains, ledger, AUGUSTUS)
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

    proposals, _ = build_proposals(GATE_OFF, market, chain and {"SPY": chain}, _empty_ledger(),
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
    proposals, checks = build_proposals(GATE_OFF, market, chains, _empty_ledger(), AUGUSTUS)
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
    proposals, checks = build_proposals(GATE_OFF, market, chains, _empty_ledger(), AUGUSTUS)
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
    proposals, _ = build_proposals(GATE_OFF, market, chains, ledger, AUGUSTUS)
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
    proposals, _ = build_proposals(GATE_OFF, market, chains, ledger, AUGUSTUS)
    assert proposals == []


def test_puts_disabled_never_produces_a_put_proposal():
    market = _market(_falling())
    spot = float(market["SPY"].bars["close"].iloc[-1])
    chains = {"SPY": _chain("SPY", spot, _exp(30))}
    proposals, checks = build_proposals(_config_with(puts_enabled=False), market, chains,
                                        _empty_ledger(), AUGUSTUS)
    assert proposals == []
    assert all(c.direction == "call" for c in checks)


# ------------------------------------------------------------------ market gate


def test_gate_passes_when_trending_and_validated():
    """Empirically verified fixture (seed=15, drift=0.9): Greg reads this as
    trending AND SPARK clears Charles's walk-forward + drawdown grading on
    it (3/5 folds)."""
    market = _market(_trend(seed=15, annual_drift=0.9))
    gates = _market_gates(CONFIG, market, ["SPY"])
    assert gates["SPY"].regime == "trending"
    assert gates["SPY"].passed, gates["SPY"].reason


def test_gate_blocks_on_choppy_regime():
    market = _market(_chop())
    gates = _market_gates(CONFIG, market, ["SPY"])
    assert gates["SPY"].regime == "choppy"
    assert not gates["SPY"].passed
    assert "not trending" in gates["SPY"].reason


def test_gate_blocks_when_trending_but_unvalidated():
    """Default _trend() (seed=11, drift=0.6): trending per Greg, but SPARK
    only clears 2/5 walk-forward folds on this particular series -- below
    the 3/5 bar, so the gate should still block it."""
    market = _market(_trend())
    gates = _market_gates(CONFIG, market, ["SPY"])
    assert gates["SPY"].regime == "trending"
    assert not gates["SPY"].passed
    assert "fails validation" in gates["SPY"].reason


def test_gate_disabled_returns_empty():
    market = _market(_trend(seed=15, annual_drift=0.9))
    gates = _market_gates(_config_with(market_gate_enabled=False), market, ["SPY"])
    assert gates == {}


def test_build_proposals_blocked_by_gate_even_with_signal_and_chain():
    """The interaction test: SPARK's own signal is long AND there's a live,
    tradable chain AND no existing position -- everything that used to be
    sufficient for a proposal -- but the gate still blocks it because this
    60-bar ramp reads as an undecided regime (too little history for a
    confident trending read at this length)."""
    market = _market(_ramp())
    spot = float(market["SPY"].bars["close"].iloc[-1])
    chains = {"SPY": _chain("SPY", spot, _exp(30))}
    proposals, checks = build_proposals(CONFIG, market, chains, _empty_ledger(), AUGUSTUS)
    assert proposals == []
    call_check = next(c for c in checks if c.direction == "call")
    assert call_check.signal_long          # the trigger itself did fire
    assert not call_check.gate_passed      # but the gate held it back
    assert call_check.gate_detail


def test_build_proposals_opens_once_gate_clears():
    """Same shape as the blocked case above, but on a fixture that clears
    the gate -- confirms the gate is a real block, not a no-op that happens
    to always read False in tests."""
    market = _market(_trend(seed=15, annual_drift=0.9), symbol="SPY")
    spot = float(market["SPY"].bars["close"].iloc[-1])
    chains = {"SPY": _chain("SPY", spot, _exp(30))}
    proposals, checks = build_proposals(CONFIG, market, chains, _empty_ledger(), AUGUSTUS)
    call_check = next(c for c in checks if c.direction == "call")
    assert call_check.gate_passed
    # Whether SPARK's OWN signal happens to be long on this particular
    # random-walk fixture's last bar is a separate question from the gate;
    # either way the gate itself must not be what's blocking it.
    if call_check.signal_long:
        assert len(proposals) == 1
        assert proposals[0].option_type == "long_call"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
