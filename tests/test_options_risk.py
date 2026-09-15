"""Smoke tests for Theo (OptionsRiskAgent).

Run with:  python -m pytest tests/test_options_risk.py -q

Verifies each of the four caps rejects when tripped, and approves in the
clean case. Uses a fake stand-in for open positions so we don't need Joseph
to exist yet.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.options_risk_agent import OptionsProposal, OptionsRiskAgent  # noqa: E402
from utils.config import load_config  # noqa: E402

CONFIG = load_config()


@dataclass
class _FakePosition:
    """Stands in for a real ledger position until Joseph exists."""
    premium_paid: float
    contracts: int


def _clean_proposal(**overrides) -> OptionsProposal:
    """A proposal that passes all caps in a fresh bucket."""
    defaults = dict(
        underlying="SPY",
        option_type="long_call",
        strike=450.0,
        expiration="2026-10-17",
        premium_per_contract=120.0,
        contracts=1,
        reason="test",
    )
    defaults.update(overrides)
    return OptionsProposal(**defaults)


def _fresh_bucket_equity() -> float:
    return float(CONFIG.get("options.bucket.starting_capital"))


# ------------------------------------------------------------------ happy path

def test_clean_proposal_is_approved():
    theo = OptionsRiskAgent(CONFIG)
    result = theo.evaluate(_clean_proposal(), [], _fresh_bucket_equity())
    assert result.approved, result.reason


# ------------------------------------------------------------------ per-contract cap

def test_premium_over_cap_is_rejected():
    theo = OptionsRiskAgent(CONFIG)
    over_cap = theo.max_premium_per_contract + 1.0
    result = theo.evaluate(
        _clean_proposal(premium_per_contract=over_cap), [], _fresh_bucket_equity()
    )
    assert not result.approved
    assert "per-contract cap" in result.reason


def test_premium_exactly_at_cap_is_approved():
    theo = OptionsRiskAgent(CONFIG)
    at_cap = theo.max_premium_per_contract
    result = theo.evaluate(
        _clean_proposal(premium_per_contract=at_cap), [], _fresh_bucket_equity()
    )
    assert result.approved


# ------------------------------------------------------------------ concurrent count cap

def test_would_exceed_concurrent_cap_is_rejected():
    theo = OptionsRiskAgent(CONFIG)
    # Fill the bucket to the concurrent cap already
    open_positions = [
        _FakePosition(premium_paid=100.0, contracts=1)
        for _ in range(theo.max_concurrent_contracts)
    ]
    result = theo.evaluate(_clean_proposal(), open_positions, _fresh_bucket_equity())
    assert not result.approved
    assert "contracts total" in result.reason


# ------------------------------------------------------------------ total premium cap

def test_would_exceed_total_open_premium_is_rejected():
    theo = OptionsRiskAgent(CONFIG)
    # One giant fake position that already sits at the total-premium cap
    open_positions = [_FakePosition(premium_paid=theo.max_total_open_premium, contracts=1)]
    result = theo.evaluate(_clean_proposal(), open_positions, _fresh_bucket_equity())
    assert not result.approved
    assert "at risk" in result.reason


# ------------------------------------------------------------------ bucket loss cutoff

def test_bucket_at_loss_cutoff_is_rejected():
    theo = OptionsRiskAgent(CONFIG)
    halted_equity = theo.starting_capital * (1.0 - theo.bucket_loss_cutoff_pct)
    result = theo.evaluate(_clean_proposal(), [], halted_equity)
    assert not result.approved
    assert "bucket halted" in result.reason


def test_bucket_just_above_cutoff_is_approved():
    theo = OptionsRiskAgent(CONFIG)
    just_above = theo.starting_capital * (1.0 - theo.bucket_loss_cutoff_pct) + 1.0
    result = theo.evaluate(_clean_proposal(), [], just_above)
    assert result.approved


# ------------------------------------------------------------------ whitelist checks

def test_disallowed_underlying_is_rejected():
    theo = OptionsRiskAgent(CONFIG)
    result = theo.evaluate(
        _clean_proposal(underlying="TSLA"), [], _fresh_bucket_equity()
    )
    assert not result.approved
    assert "not in allowed_underlyings" in result.reason


def test_disallowed_option_type_is_rejected():
    theo = OptionsRiskAgent(CONFIG)
    # Even if the strategy tried to short a call, allowed_types blocks it.
    result = theo.evaluate(
        _clean_proposal(option_type="short_call"), [], _fresh_bucket_equity()
    )
    assert not result.approved
    assert "not in allowed_types" in result.reason


# ------------------------------------------------------------------ basic sanity

def test_zero_contracts_is_rejected():
    theo = OptionsRiskAgent(CONFIG)
    result = theo.evaluate(_clean_proposal(contracts=0), [], _fresh_bucket_equity())
    assert not result.approved


def test_negative_premium_is_rejected():
    theo = OptionsRiskAgent(CONFIG)
    result = theo.evaluate(
        _clean_proposal(premium_per_contract=-1.0), [], _fresh_bucket_equity()
    )
    assert not result.approved


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
