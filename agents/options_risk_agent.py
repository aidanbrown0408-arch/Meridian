"""Theo — Options Risk Agent.

Enforces the four hard caps on every proposed options trade:

  1. Per-contract premium ceiling      (options.caps.max_premium_per_contract)
  2. Total open premium ceiling         (options.caps.max_total_open_premium)
  3. Concurrent contract count ceiling (options.caps.max_concurrent_contracts)
  4. Bucket loss cutoff halt            (options.caps.bucket_loss_cutoff_pct)

Also refuses any option type not on `options.allowed_types` -- currently
long_call and long_put only, so shorts/naked writes are impossible by
construction.

Deliberately much simpler than Charles (stock RiskAgent). No risk parity, no
walk-forward validation, no correlated-slot logic -- for long-only,
capped-loss options at this account size, the risk model IS the caps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from utils.config import Config
from utils.logging_setup import get_logger

log = get_logger("options_risk", agent="Theo")


OptionType = Literal["long_call", "long_put"]


@dataclass
class OptionsProposal:
    """A strategy's request to open one options position.

    `premium_per_contract` is the total dollar cost of a single contract
    (i.e. mid or ask price * 100), not the per-share premium.
    """
    underlying: str            # e.g. "SPY"
    option_type: OptionType    # "long_call" | "long_put"
    strike: float
    expiration: str            # ISO date, e.g. "2026-10-17"
    premium_per_contract: float
    contracts: int             # count of contracts to buy
    reason: str = ""           # free-text, for logging (e.g. "SPARK breakout")


@dataclass
class RiskDecision:
    """Theo's verdict on one proposal."""
    approved: bool
    reason: str
    proposal: OptionsProposal


class OptionsRiskAgent:
    name = "Theo"
    role = "Risk (Options)"

    def __init__(self, config: Config):
        self.config = config
        self.enabled = bool(config.get("options.enabled", False))
        self.starting_capital = float(config.get("options.bucket.starting_capital"))
        caps = config.section("options.caps")
        self.max_premium_per_contract = float(caps["max_premium_per_contract"])
        self.max_total_open_premium = float(caps["max_total_open_premium"])
        self.max_concurrent_contracts = int(caps["max_concurrent_contracts"])
        self.bucket_loss_cutoff_pct = float(caps["bucket_loss_cutoff_pct"])
        self.allowed_types = set(config.get("options.allowed_types"))
        self.allowed_underlyings = set(config.get("options.underlyings"))

    def evaluate(self, proposal: OptionsProposal,
                 open_positions: list,
                 bucket_equity: float) -> RiskDecision:
        """Approve or reject one proposal.

        `open_positions` is the list of currently-open positions from Joseph's
        ledger; each must expose `.premium_paid` (dollars) and `.contracts` (int).
        `bucket_equity` is Joseph's mark-to-market bucket value right now.
        """
        # -- Master switch ----------------------------------------------------
        if not self.enabled:
            return self._reject(proposal, "options.enabled is false")

        # -- Bucket loss cutoff (halt) ---------------------------------------
        min_equity = self.starting_capital * (1.0 - self.bucket_loss_cutoff_pct)
        if bucket_equity <= min_equity:
            return self._reject(
                proposal,
                f"bucket halted: equity ${bucket_equity:,.2f} at/below "
                f"loss-cutoff ${min_equity:,.2f} "
                f"({self.bucket_loss_cutoff_pct:.0%} of ${self.starting_capital:,.2f})"
            )

        # -- Whitelist checks -------------------------------------------------
        if proposal.underlying not in self.allowed_underlyings:
            return self._reject(
                proposal,
                f"underlying {proposal.underlying} not in allowed_underlyings "
                f"({sorted(self.allowed_underlyings)})"
            )
        if proposal.option_type not in self.allowed_types:
            return self._reject(
                proposal,
                f"option_type {proposal.option_type!r} not in allowed_types "
                f"({sorted(self.allowed_types)})"
            )

        # -- Basic sanity -----------------------------------------------------
        if proposal.contracts < 1:
            return self._reject(proposal, "contracts must be >= 1")
        if proposal.premium_per_contract <= 0:
            return self._reject(proposal, "premium_per_contract must be > 0")

        # -- Per-contract premium cap ----------------------------------------
        if proposal.premium_per_contract > self.max_premium_per_contract:
            return self._reject(
                proposal,
                f"premium ${proposal.premium_per_contract:,.2f} exceeds "
                f"per-contract cap ${self.max_premium_per_contract:,.2f}"
            )

        # -- Concurrent contract count cap -----------------------------------
        currently_open = sum(int(p.contracts) for p in open_positions)
        after_open = currently_open + proposal.contracts
        if after_open > self.max_concurrent_contracts:
            return self._reject(
                proposal,
                f"would open {after_open} contracts total; cap is "
                f"{self.max_concurrent_contracts} "
                f"(currently {currently_open} open)"
            )

        # -- Total open premium cap ------------------------------------------
        current_open_premium = sum(float(p.premium_paid) for p in open_positions)
        proposal_premium = proposal.premium_per_contract * proposal.contracts
        after_premium = current_open_premium + proposal_premium
        if after_premium > self.max_total_open_premium:
            return self._reject(
                proposal,
                f"would put ${after_premium:,.2f} at risk; cap is "
                f"${self.max_total_open_premium:,.2f} "
                f"(currently ${current_open_premium:,.2f} open)"
            )

        # -- All checks passed ------------------------------------------------
        log.info(
            "APPROVED %s %s $%s exp %s x%d @ $%.2f/contract (%s)",
            proposal.option_type, proposal.underlying, proposal.strike,
            proposal.expiration, proposal.contracts,
            proposal.premium_per_contract, proposal.reason or "no reason given",
        )
        return RiskDecision(approved=True, reason="all caps satisfied",
                            proposal=proposal)

    def _reject(self, proposal: OptionsProposal, reason: str) -> RiskDecision:
        log.warning("REJECTED %s %s: %s",
                    proposal.option_type, proposal.underlying, reason)
        return RiskDecision(approved=False, reason=reason, proposal=proposal)
