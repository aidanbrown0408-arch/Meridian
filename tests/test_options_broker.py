"""Joseph (OptionsBroker) tests — ledger math, expiration settlement, the
bucket loss cutoff, and the killswitch/clear-halt gate.

Most tests inject a stub risk agent so they exercise Joseph's mechanics
without depending on Theo's internals (Theo has its own suite in
tests/test_options_risk.py). Two tests at the bottom run the real Theo to
prove the two agents actually talk to each other.

Run with:  python -m pytest tests -q      (or)  python tests/test_options_broker.py
"""

from __future__ import annotations

import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.options_broker import (  # noqa: E402
    OptionsBroker, OptionsLedger, OptionsPosition,
)
from agents.options_risk_agent import (  # noqa: E402
    OptionsProposal, OptionsRiskAgent, RiskDecision,
)
from utils.config import load_config  # noqa: E402

CONFIG = load_config()


# ------------------------------------------------------------------ scaffolding

class _ApproveAll:
    """Stand-in for Theo that waves everything through."""
    name = "TheoStub"

    def evaluate(self, proposal, open_positions, bucket_equity) -> RiskDecision:
        return RiskDecision(approved=True, reason="stub approve", proposal=proposal)


class _RejectAll:
    name = "TheoStub"

    def evaluate(self, proposal, open_positions, bucket_equity) -> RiskDecision:
        return RiskDecision(approved=False, reason="stub reject", proposal=proposal)


def _tmp_broker(risk_agent=None) -> OptionsBroker:
    broker = OptionsBroker(CONFIG, risk_agent=risk_agent or _ApproveAll())
    broker.ledger_path = Path(tempfile.mkdtemp()) / "options_ledger.json"
    return broker


def _iso(days_out: int) -> str:
    return (date.today() + timedelta(days=days_out)).isoformat()


def _proposal(premium: float = 50.0, contracts: int = 1, underlying: str = "SPY",
              option_type: str = "long_call", strike: float = 500.0,
              days_out: int = 30) -> OptionsProposal:
    return OptionsProposal(
        underlying=underlying, option_type=option_type, strike=strike,
        expiration=_iso(days_out), premium_per_contract=premium,
        contracts=contracts, reason="test",
    )


def _position(broker: OptionsBroker, premium: float = 100.0, contracts: int = 1,
              days_out: int = 30, strike: float = 500.0) -> OptionsPosition:
    return OptionsPosition(
        underlying="SPY", option_type="long_call", strike=strike,
        expiration=_iso(days_out), contracts=contracts,
        premium_paid=premium * contracts, entry_premium_per_contract=premium,
        entry_date=datetime.now(timezone.utc).isoformat(),
    )


# ------------------------------------------------------------------ ledger basics

def test_fresh_ledger_starts_at_bucket_capital():
    broker = _tmp_broker()
    ledger = broker.load_ledger()
    assert ledger.cash == broker.starting_capital
    assert ledger.positions == {}
    assert not ledger.halted
    assert ledger.realized_pnl == 0.0


def test_options_bucket_is_separate_from_the_stock_account():
    """The whole point of the parallel track: this bucket must never be the
    $5,000 stock ledger."""
    broker = _tmp_broker()
    stock_capital = float(CONFIG.get("capital.starting_paper_capital"))
    assert broker.starting_capital != stock_capital
    assert "options" in broker.ledger_path.name


def test_execute_opens_an_approved_position():
    broker = _tmp_broker()
    ledger = broker.execute([_proposal(premium=50.0, contracts=2)])
    assert len(ledger.positions) == 1
    pos = next(iter(ledger.positions.values()))
    assert pos.contracts == 2
    assert pos.premium_paid == 100.0
    expected_cash = broker.starting_capital - 100.0 - 2 * broker.commission_per_contract
    assert abs(ledger.cash - expected_cash) < 1e-9
    assert len(ledger.trades) == 1 and ledger.trades[0].side == "open"


def test_rejected_proposal_changes_nothing():
    broker = _tmp_broker(risk_agent=_RejectAll())
    ledger = broker.execute([_proposal()])
    assert ledger.positions == {}
    assert ledger.cash == broker.starting_capital
    assert ledger.trades == []


def test_second_fill_on_same_contract_merges_into_one_position():
    broker = _tmp_broker()
    broker.execute([_proposal(premium=40.0)])
    ledger = broker.execute([_proposal(premium=60.0)])
    assert len(ledger.positions) == 1
    pos = next(iter(ledger.positions.values()))
    assert pos.contracts == 2
    assert pos.premium_paid == 100.0
    assert pos.entry_premium_per_contract == 50.0   # weighted average


def test_insufficient_cash_skips_the_trade_rather_than_sizing_down():
    broker = _tmp_broker()
    ledger = broker.load_ledger()
    ledger.cash = 40.0
    broker.save_ledger(ledger)

    result = broker.execute([_proposal(premium=100.0)])
    assert result.positions == {}
    assert result.cash == 40.0


def test_ledger_persists_across_broker_instances():
    broker = _tmp_broker()
    broker.execute([_proposal()])

    reloaded = OptionsBroker(CONFIG, risk_agent=_ApproveAll())
    reloaded.ledger_path = broker.ledger_path
    ledger = reloaded.load_ledger()
    assert len(ledger.positions) == 1
    assert len(ledger.trades) == 1


# ------------------------------------------------------------------ marking

def test_mark_to_market_uses_quotes_when_given():
    broker = _tmp_broker()
    ledger = broker.execute([_proposal(premium=100.0)])
    key = next(iter(ledger.positions))
    # Contract doubled: equity should rise by the same $100.
    assert abs(ledger.mark_to_market({key: 200.0})
               - (ledger.cash + 200.0)) < 1e-9


def test_mark_to_market_without_quotes_marks_at_cost():
    broker = _tmp_broker()
    ledger = broker.execute([_proposal(premium=100.0)])
    # Cash + premium at risk, minus the commission already paid.
    expected = broker.starting_capital - broker.commission_per_contract
    assert abs(ledger.mark_to_market() - expected) < 1e-9


# ------------------------------------------------------------------ expirations

def test_expired_position_without_a_quote_books_worthless():
    broker = _tmp_broker()
    ledger = broker.load_ledger()
    pos = _position(broker, premium=120.0, days_out=-1)
    ledger.positions[pos.key] = pos
    ledger.cash -= pos.premium_paid
    broker.save_ledger(ledger)

    result = broker.execute([])
    assert result.positions == {}
    assert result.realized_pnl == -120.0
    assert any(t.reason == "expired" for t in result.trades)


def test_expired_position_settles_at_a_supplied_price():
    broker = _tmp_broker()
    ledger = broker.load_ledger()
    pos = _position(broker, premium=120.0, days_out=-1)
    ledger.positions[pos.key] = pos
    ledger.cash -= pos.premium_paid
    broker.save_ledger(ledger)

    result = broker.execute([], prices={pos.key: 300.0})
    assert result.positions == {}
    assert result.realized_pnl > 0


def test_position_expiring_today_is_not_settled_yet():
    broker = _tmp_broker()
    ledger = broker.load_ledger()
    pos = _position(broker, days_out=1)
    ledger.positions[pos.key] = pos
    broker.save_ledger(ledger)

    result = broker.execute([])
    assert pos.key in result.positions


def test_manual_close_books_realized_pnl():
    broker = _tmp_broker()
    ledger = broker.execute([_proposal(premium=100.0)])
    key = next(iter(ledger.positions))

    result = broker.close(key, premium_per_contract=175.0, reason="take profit")
    assert result.positions == {}
    assert result.realized_pnl > 0
    assert result.trades[-1].side == "close"


# ------------------------------------------------------------------ safety gates

def test_halted_bucket_refuses_to_trade():
    broker = _tmp_broker()
    ledger = broker.load_ledger()
    ledger.halted = True
    ledger.halt_reason = "test halt"
    broker.save_ledger(ledger)

    result = broker.execute([_proposal()])
    assert result.positions == {}
    assert result.cash == broker.starting_capital


def test_bucket_loss_cutoff_flattens_and_halts():
    broker = _tmp_broker()
    ledger = broker.load_ledger()
    ledger.cash = broker.halt_floor - 50.0          # already through the floor
    broker.save_ledger(ledger)

    result = broker.execute([_proposal()])
    assert result.halted
    assert "cutoff" in result.halt_reason
    assert result.positions == {}


def test_bucket_just_above_the_cutoff_still_trades():
    broker = _tmp_broker()
    ledger = broker.load_ledger()
    ledger.cash = broker.halt_floor + 200.0
    broker.save_ledger(ledger)

    result = broker.execute([_proposal(premium=50.0)])
    assert not result.halted
    assert len(result.positions) == 1


def test_killswitch_flattens_every_contract_and_halts():
    broker = _tmp_broker()
    ledger = broker.execute([_proposal(premium=50.0)])
    key = next(iter(ledger.positions))

    result = broker.killswitch({key: 60.0}, reason="test kill")
    assert result.positions == {}
    assert result.halted
    assert result.halt_reason == "test kill"
    assert any(t.reason == "killswitch flatten" for t in result.trades)


def test_killswitch_without_quotes_falls_back_to_entry_premium():
    broker = _tmp_broker()
    broker.execute([_proposal(premium=50.0)])
    result = broker.killswitch()
    assert result.positions == {}
    assert result.halted


def test_clear_halt_allows_trading_again():
    broker = _tmp_broker()
    broker.execute([_proposal(premium=50.0)])
    broker.killswitch()
    assert broker.load_ledger().halted

    broker.clear_halt()
    assert not broker.load_ledger().halted
    ledger = broker.execute([_proposal(premium=50.0)])
    assert len(ledger.positions) == 1


def test_halt_floor_matches_the_configured_cutoff():
    broker = _tmp_broker()
    assert broker.halt_floor == broker.starting_capital * (1 - broker.loss_cutoff_pct)


# ------------------------------------------------------- integration with Theo

def test_real_theo_approves_a_within_caps_proposal():
    broker = _tmp_broker(risk_agent=OptionsRiskAgent(CONFIG))
    ledger = broker.execute([_proposal(premium=50.0, contracts=1)])
    assert len(ledger.positions) == 1


def test_real_theo_blocks_an_oversized_premium():
    broker = _tmp_broker(risk_agent=OptionsRiskAgent(CONFIG))
    ledger = broker.execute([_proposal(premium=10_000.0, contracts=1)])
    assert ledger.positions == {}
    assert ledger.cash == broker.starting_capital


def test_real_theo_blocks_an_off_whitelist_underlying():
    broker = _tmp_broker(risk_agent=OptionsRiskAgent(CONFIG))
    ledger = broker.execute([_proposal(underlying="TSLA", premium=50.0)])
    assert ledger.positions == {}


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
