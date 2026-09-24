"""Joseph — Options Execution Agent.

The options-side counterpart to Cornelius's `PaperBroker`. Joseph owns a
completely separate paper ledger (`reports/options_ledger.json`) funded with
its own bucket capital -- the $5,000 stock account and this bucket are never
blended, never net against each other, and never share a halt.

Structure deliberately mirrors `agents/portfolio_agent.py`:
  * JSON ledger on disk so the bucket survives between runs
  * killswitch -> halt, cleared only by an explicit operator command
  * an unreadable ledger is an error, never something to guess around

What is different, because options are different:
  * positions are contracts, not fractional shares
  * there is no "rebalance toward a target weight" -- an option position is a
    discrete open/close event
  * every position has an expiration, and an expired long option that nobody
    closed is worth zero. Joseph books that loss on the next run rather than
    letting a dead contract sit in the ledger inflating equity.
  * pre-trade risk is Theo's call (`agents/options_risk_agent.py`), not
    risk-parity math

Premium convention: every premium figure in this module is DOLLARS PER
CONTRACT (i.e. the quoted per-share premium already multiplied by 100).
Augustus is responsible for that conversion when it builds proposals, so
that Theo's "$150 per contract" cap reads in the same units everywhere.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

from agents.options_risk_agent import OptionsProposal, OptionsRiskAgent
from utils.dates import local_date
from utils.config import Config
from utils.logging_setup import get_logger

log = get_logger("options_execution", agent="Joseph")


def _cfg(config: Config, paths: list[str], default):
    """First key that exists wins. The `options:` block was hand-written in
    Phase A1; this keeps Joseph from hard-failing on a near-miss key name."""
    for path in paths:
        value = config.get(path, None)
        if value is not None:
            return value
    return default


# ------------------------------------------------------------------ data model


@dataclass
class OptionsPosition:
    """One open long option position. `premium_paid` and `contracts` are the
    two fields Theo introspects -- that contract is load-bearing, don't
    rename them without updating `OptionsRiskAgent.evaluate`."""

    underlying: str                    # "SPY" | "QQQ"
    option_type: str                   # "long_call" | "long_put"
    strike: float
    expiration: str                    # ISO date, "2026-10-17"
    contracts: int
    premium_paid: float                # total dollars of premium at risk
    entry_premium_per_contract: float
    entry_date: str                    # ISO datetime of the opening fill
    reason: str = ""

    @property
    def key(self) -> str:
        """Ledger key. Two fills on the same contract merge into one position."""
        return f"{self.underlying}|{self.option_type}|{self.strike:g}|{self.expiration}"

    @property
    def label(self) -> str:
        kind = "C" if self.option_type == "long_call" else "P"
        return f"{self.underlying} {self.expiration} {self.strike:g}{kind}"

    def days_held(self, as_of: datetime | None = None) -> int:
        as_of = as_of or datetime.now(timezone.utc)
        entered = datetime.fromisoformat(self.entry_date)
        return max(0, (as_of.date() - entered.date()).days)

    def expiration_date(self) -> date:
        return date.fromisoformat(self.expiration)

    def days_to_expiration(self, as_of: datetime | None = None) -> int:
        as_of = as_of or datetime.now(timezone.utc)
        return (self.expiration_date() - as_of.date()).days

    def is_expired(self, as_of: datetime | None = None) -> bool:
        """Expired only once the expiration date has passed. A contract on its
        expiration date is still tradable."""
        as_of = as_of or datetime.now(timezone.utc)
        return as_of.date() > self.expiration_date()

    def market_value(self, premium_per_contract: float | None = None) -> float:
        """Value at a current premium. With no quote, falls back to the entry
        premium -- scaffolding behaviour until Augustus supplies live chains.
        That fallback is a placeholder, not a mark: it makes equity look flat,
        never favourable."""
        if premium_per_contract is None:
            return self.premium_paid
        return self.contracts * premium_per_contract

    def unrealized_pnl(self, premium_per_contract: float | None = None) -> float:
        return self.market_value(premium_per_contract) - self.premium_paid


@dataclass
class OptionsTrade:
    date: str
    underlying: str
    option_type: str
    strike: float
    expiration: str
    side: str                    # "open" | "close"
    contracts: int
    premium_per_contract: float
    cost: float                  # dollars, commission
    cash_delta: float            # signed: negative on open, positive on close
    realized_pnl: float          # 0.0 on open
    reason: str


@dataclass
class OptionsLedger:
    """Mirrors `PaperLedger`, minus the trader-shadow machinery (there are no
    competing options traders to drift-check yet) and plus realized P&L, which
    matters more here because positions terminate rather than rebalance."""

    starting_capital: float
    cash: float
    positions: dict = field(default_factory=dict)        # key -> OptionsPosition
    trades: list = field(default_factory=list)           # OptionsTrade, newest last
    equity_history: list = field(default_factory=list)   # [{"date":..., "equity":...}]
    realized_pnl: float = 0.0
    halted: bool = False
    halt_reason: str = ""
    created_at: str = ""
    updated_at: str = ""

    @property
    def equity(self) -> float:
        """Cash-only equity; use `mark_to_market` for the version including
        open premium."""
        return self.cash

    @property
    def open_premium(self) -> float:
        return sum(p.premium_paid for p in self.positions.values())

    @property
    def open_contracts(self) -> int:
        return sum(p.contracts for p in self.positions.values())

    def mark_to_market(self, prices: dict | None = None) -> float:
        prices = prices or {}
        total = self.cash
        for key, pos in self.positions.items():
            total += pos.market_value(prices.get(key))
        return total

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "OptionsLedger":
        positions = {k: OptionsPosition(**p) for k, p in d.get("positions", {}).items()}
        trades = [OptionsTrade(**t) for t in d.get("trades", [])]
        return cls(
            starting_capital=d["starting_capital"], cash=d["cash"],
            positions=positions, trades=trades,
            equity_history=d.get("equity_history", []),
            realized_pnl=d.get("realized_pnl", 0.0),
            halted=d.get("halted", False), halt_reason=d.get("halt_reason", ""),
            created_at=d.get("created_at", ""), updated_at=d.get("updated_at", ""),
        )


# ------------------------------------------------------------------ the agent


class OptionsBroker:
    """Joseph."""

    name = "Joseph"
    role = "Execution (Options)"

    def __init__(self, config: Config, risk_agent: OptionsRiskAgent | None = None):
        self.config = config
        self.ledger_path: Path = config.repo_path(
            _cfg(config, ["options.ledger_path", "options.paper_ledger_path"],
                 "reports/options_ledger.json"))
        self.starting_capital = float(_cfg(
            config,
            ["options.bucket.starting_capital", "options.starting_capital",
             "options.capital.starting_capital"],
            1000.0))
        self.loss_cutoff_pct = float(_cfg(
            config,
            ["options.caps.bucket_loss_cutoff_pct", "options.caps.bucket_loss_cutoff",
             "options.bucket.loss_cutoff_pct"],
            0.50))
        self.commission_per_contract = float(_cfg(
            config, ["options.costs.commission_per_contract"], 0.65))
        # Theo gets the final say on every open. Injectable for tests.
        self.risk_agent = risk_agent or OptionsRiskAgent(config)
        # Theo's rejections from the most recent execute(), for notifications.
        self.rejections: list = []

    @property
    def halt_floor(self) -> float:
        """Equity at or below this halts the bucket (50% of $1,000 = $500)."""
        return self.starting_capital * (1.0 - self.loss_cutoff_pct)

    # ------------------------------------------------------------- persistence

    def load_ledger(self) -> OptionsLedger:
        if not self.ledger_path.exists():
            now = datetime.now(timezone.utc).isoformat()
            return OptionsLedger(starting_capital=self.starting_capital,
                                 cash=self.starting_capital,
                                 created_at=now, updated_at=now)
        try:
            return OptionsLedger.from_dict(json.loads(self.ledger_path.read_text()))
        except Exception as exc:
            log.error("Options ledger at %s is unreadable (%s) -- refusing to guess at "
                      "state. Move or delete it to start a fresh options bucket.",
                      self.ledger_path, exc)
            raise

    def save_ledger(self, ledger: OptionsLedger) -> None:
        ledger.updated_at = datetime.now(timezone.utc).isoformat()
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.ledger_path.write_text(json.dumps(ledger.to_dict(), indent=2, default=str))

    # ------------------------------------------------------------------ trading

    def execute(self, proposals: list[OptionsProposal] | None = None,
                prices: dict | None = None) -> OptionsLedger:
        """One options run: settle expirations, re-check the bucket cutoff,
        then offer each proposal to Theo and open whatever survives.

        `prices` maps position key -> current dollars per contract. Empty is
        fine during Phase A scaffolding; positions then mark at cost.
        """
        proposals = proposals or []
        prices = prices or {}
        self.rejections = []
        ledger = self.load_ledger()
        today = datetime.now(timezone.utc)

        if ledger.halted:
            log.warning("HALTED (%s) -- refusing to trade until the operator clears it "
                        "(python main.py clear-halt --options).", ledger.halt_reason)
            return ledger

        # 1. Dead contracts come off the books before anything else is decided.
        self._settle_expirations(ledger, prices, today)

        # 2. Bucket cutoff. Checked before opening anything new, so a bucket
        #    that has already breached never adds risk on the way down.
        equity = ledger.mark_to_market(prices)
        if equity <= self.halt_floor:
            log.error("Options bucket equity $%.2f is at or below the %.0f%% cutoff "
                      "($%.2f) -- flattening and halting.",
                      equity, self.loss_cutoff_pct * 100, self.halt_floor)
            self.save_ledger(ledger)
            return self.killswitch(prices, reason=(
                f"bucket loss cutoff breached: equity ${equity:,.2f} "
                f"<= ${self.halt_floor:,.2f}"))

        # 3. Proposals, one at a time -- each one re-reads the book, so two
        #    proposals in the same batch can't both slip under the same cap.
        for proposal in proposals:
            self._consider(ledger, proposal, prices, today)

        equity_after = ledger.mark_to_market(prices)
        ledger.equity_history.append({"date": local_date(self.config, today),
                                      "equity": equity_after})
        ledger.equity_history = ledger.equity_history[-800:]

        self.save_ledger(ledger)
        log.info("Options bucket: $%.2f equity (cash $%.2f, %d position(s), "
                 "%d contract(s), $%.2f premium at risk)",
                 equity_after, ledger.cash, len(ledger.positions),
                 ledger.open_contracts, ledger.open_premium)
        return ledger

    def _consider(self, ledger: OptionsLedger, proposal: OptionsProposal,
                  prices: dict, today: datetime) -> None:
        equity = ledger.mark_to_market(prices)
        decision = self.risk_agent.evaluate(proposal, list(ledger.positions.values()),
                                            equity)
        if not decision.approved:
            self.rejections.append(decision)
            log.info("Theo rejected %s %s %g %s: %s", proposal.underlying,
                     proposal.option_type, proposal.strike, proposal.expiration,
                     decision.reason)
            return

        premium = proposal.premium_per_contract * proposal.contracts
        cost = proposal.contracts * self.commission_per_contract
        if premium + cost > ledger.cash:
            log.warning("Skipping %s %g %s: needs $%.2f (premium + commission) but the "
                        "bucket only has $%.2f in cash. Options fill whole or not at "
                        "all -- no sizing down.",
                        proposal.underlying, proposal.strike, proposal.expiration,
                        premium + cost, ledger.cash)
            return

        self._open(ledger, proposal, cost, today)

    def _open(self, ledger: OptionsLedger, proposal: OptionsProposal,
              cost: float, today: datetime) -> None:
        premium = proposal.premium_per_contract * proposal.contracts
        ledger.cash -= premium + cost

        position = OptionsPosition(
            underlying=proposal.underlying, option_type=proposal.option_type,
            strike=float(proposal.strike), expiration=proposal.expiration,
            contracts=int(proposal.contracts), premium_paid=premium,
            entry_premium_per_contract=float(proposal.premium_per_contract),
            entry_date=today.isoformat(), reason=proposal.reason,
        )
        existing = ledger.positions.get(position.key)
        if existing is not None:
            # Same contract, second fill: merge and carry a weighted-average entry.
            total_contracts = existing.contracts + position.contracts
            total_premium = existing.premium_paid + position.premium_paid
            existing.contracts = total_contracts
            existing.premium_paid = total_premium
            existing.entry_premium_per_contract = total_premium / total_contracts
        else:
            ledger.positions[position.key] = position

        ledger.trades.append(OptionsTrade(
            date=local_date(self.config, today), underlying=position.underlying,
            option_type=position.option_type, strike=position.strike,
            expiration=position.expiration, side="open", contracts=position.contracts,
            premium_per_contract=float(proposal.premium_per_contract), cost=cost,
            cash_delta=-(premium + cost), realized_pnl=0.0,
            reason=proposal.reason or "opened on approved proposal",
        ))
        log.info("OPEN  %s x%d @ $%.2f/contract  ($%.2f premium, $%.2f commission)",
                 position.label, position.contracts, proposal.premium_per_contract,
                 premium, cost)

    # ------------------------------------------------------------------ closing

    def close(self, key: str, premium_per_contract: float = 0.0,
              reason: str = "manual close", ledger: OptionsLedger | None = None,
              charge_commission: bool = True) -> OptionsLedger:
        """Close one position in full. Pass the ledger in to close several
        within one run; omit it and Joseph loads, closes, and saves."""
        standalone = ledger is None
        ledger = ledger or self.load_ledger()
        position = ledger.positions.get(key)
        if position is None:
            log.warning("No open options position keyed %s -- nothing to close.", key)
            return ledger

        today = datetime.now(timezone.utc)
        # Worthless contracts aren't sold, they just lapse -- no commission.
        cost = (position.contracts * self.commission_per_contract
                if charge_commission and premium_per_contract > 0 else 0.0)
        proceeds = position.contracts * premium_per_contract - cost
        pnl = proceeds - position.premium_paid

        ledger.cash += proceeds
        ledger.realized_pnl += pnl
        ledger.trades.append(OptionsTrade(
            date=local_date(self.config, today), underlying=position.underlying,
            option_type=position.option_type, strike=position.strike,
            expiration=position.expiration, side="close", contracts=position.contracts,
            premium_per_contract=premium_per_contract, cost=cost, cash_delta=proceeds,
            realized_pnl=pnl, reason=reason,
        ))
        del ledger.positions[key]
        log.info("CLOSE %s x%d @ $%.2f/contract  (%s)  P&L $%+.2f",
                 position.label, position.contracts, premium_per_contract, reason, pnl)

        if standalone:
            self.save_ledger(ledger)
        return ledger

    def _settle_expirations(self, ledger: OptionsLedger, prices: dict,
                            today: datetime) -> None:
        """A long option past its expiration is worth whatever it settled for,
        and zero if nobody told us otherwise. Booking it worthless is the
        conservative read: the loss is capped at premium paid either way, so
        the worst this does is understate equity until a real settlement price
        arrives."""
        for key, position in list(ledger.positions.items()):
            if not position.is_expired(today):
                continue
            settle = float(prices.get(key, 0.0))
            if key not in prices:
                log.warning("%s expired %s with no settlement price -- booking it "
                            "worthless (loss $%.2f).", position.label,
                            position.expiration, position.premium_paid)
            self.close(key, settle, reason="expired", ledger=ledger,
                       charge_commission=False)

    # ------------------------------------------------------------------- safety

    def killswitch(self, prices: dict | None = None,
                   reason: str = "operator killswitch") -> OptionsLedger:
        """Close every open contract and halt the options bucket. Nothing
        trades again until `clear_halt()` -- never automatic, same rule as the
        stock side."""
        prices = prices or {}
        ledger = self.load_ledger()

        for key, position in list(ledger.positions.items()):
            premium = prices.get(key)
            if premium is None:
                premium = position.entry_premium_per_contract
                log.warning("No live quote for %s at killswitch -- flattening at the "
                            "entry premium $%.2f/contract instead. Reconcile this "
                            "position by hand.", position.label, premium)
            self.close(key, float(premium), reason="killswitch flatten", ledger=ledger)

        ledger.halted = True
        ledger.halt_reason = reason
        self.save_ledger(ledger)
        log.warning("OPTIONS KILLSWITCH: %s. Bucket halted -- clear with "
                    "'python main.py clear-halt --options' when ready.", reason)
        return ledger

    def clear_halt(self) -> OptionsLedger:
        ledger = self.load_ledger()
        ledger.halted = False
        ledger.halt_reason = ""
        self.save_ledger(ledger)
        log.info("Options halt cleared -- the bucket trades again on the next run.")
        return ledger
