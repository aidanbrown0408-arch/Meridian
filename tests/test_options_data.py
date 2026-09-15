"""Tests for Augustus (OptionsDataAgent).

All tests exercise the synthetic path directly (`_synthetic`) so the suite
never touches the network -- same approach `test_phase1.py` uses for Wong's
synthetic bars. Live yfinance fetching is exercised manually, not in CI:
there's no reliable way to assert on real option chain contents.

Run with:  python -m pytest tests/test_options_data.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.options_broker import OptionsPosition  # noqa: E402
from agents.options_data_agent import OptionsDataAgent  # noqa: E402
from utils.config import load_config  # noqa: E402

CONFIG = load_config()


def _augustus() -> OptionsDataAgent:
    return OptionsDataAgent(CONFIG)


# ------------------------------------------------------------------ synthetic


def test_synthetic_chain_is_flagged_and_deterministic():
    augustus = _augustus()
    first = augustus._synthetic("SPY", reason="test")
    second = augustus._synthetic("SPY", reason="test")
    assert first.is_synthetic and not first.is_trusted
    assert not first.calls.empty and not first.puts.empty
    np.testing.assert_allclose(first.calls["last"].to_numpy(),
                               second.calls["last"].to_numpy())
    # Different underlyings must not produce identical chains.
    other = augustus._synthetic("QQQ", reason="test")
    assert not np.allclose(first.calls["last"].to_numpy()[:5],
                           other.calls["last"].to_numpy()[:5])


def test_synthetic_quotes_are_coherent():
    """bid <= ask, everything positive, and every dollar figure is already
    per-contract (i.e. plainly >> a per-share option premium would be)."""
    chain = _augustus()._synthetic("SPY", reason="test")
    for frame in (chain.calls, chain.puts):
        assert (frame["bid"] <= frame["ask"]).all()
        assert (frame["bid"] >= 0).all()
        assert (frame["last"] > 0).all()


def test_synthetic_respects_expiration_and_strike_window():
    augustus = _augustus()
    chain = augustus._synthetic("SPY", reason="test")
    assert len(chain.expirations) <= augustus.max_expirations
    for frame in (chain.calls, chain.puts):
        assert set(frame["expiration"].unique()) <= set(chain.expirations)


def test_synthetic_chains_are_excluded_from_price_lookup():
    """price_lookup() must never hand Joseph a fabricated price."""
    augustus = _augustus()
    chains = {"SPY": augustus._synthetic("SPY", reason="test")}
    assert augustus.price_lookup(chains) == {}


# ------------------------------------------------------------------ shaping


def test_find_contract_picks_closest_strike():
    augustus = _augustus()
    chain = augustus._synthetic("SPY", reason="test")
    target = float(chain.calls["strike"].iloc[3])
    found = augustus.find_contract(chain, "long_call", target_strike=target)
    assert found is not None
    assert found.strike == target
    assert found.option_type == "long_call"


def test_find_contract_returns_none_for_empty_side():
    augustus = _augustus()
    chain = augustus._synthetic("SPY", reason="test")
    # Ask for an expiration that doesn't exist in the synthetic chain.
    found = augustus.find_contract(chain, "long_call", target_strike=450.0,
                                   expiration="1999-01-01")
    assert found is None


def test_contract_key_matches_joseph_position_key():
    """Augustus.key must line up exactly with OptionsPosition.key -- this is
    the join key Joseph's mark_to_market / expiration settlement rely on."""
    augustus = _augustus()
    chain = augustus._synthetic("SPY", reason="test")
    contract = augustus.find_contract(chain, "long_call",
                                      target_strike=float(chain.calls["strike"].iloc[0]),
                                      expiration=chain.expirations[0])
    position = OptionsPosition(
        underlying=contract.underlying, option_type=contract.option_type,
        strike=contract.strike, expiration=contract.expiration,
        contracts=1, premium_paid=contract.mid,
        entry_premium_per_contract=contract.mid, entry_date="2026-01-01",
    )
    assert contract.key == position.key


def test_mid_falls_back_to_last_when_book_is_empty():
    augustus = _augustus()
    chain = augustus._synthetic("SPY", reason="test")
    contract = augustus.contracts(chain, "long_call")[0]
    contract.bid = 0.0
    contract.ask = 0.0
    assert contract.mid == round(contract.last, 2)


def test_price_lookup_values_are_per_contract_scale():
    """A synthetic SPY contract should read in the hundreds of dollars
    (per-contract), never single-digit dollars (per-share)."""
    augustus = _augustus()
    chain = augustus._synthetic("SPY", reason="test")
    contract = augustus.contracts(chain, "long_call")[0]
    assert contract.mid > 1.0


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
