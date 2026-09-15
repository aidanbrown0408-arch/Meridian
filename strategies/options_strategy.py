"""First options strategy (Phase B) — "buy a call when SPARK signals long."

Reuses Leo's exact `SparkStrategy` (same params as `strategies.SPARK` in
config, so the options entry never drifts from what the stock side sees)
against Wong's stock bars for each configured options underlying. Reading
those bars here is read-only: this module never proposes, opens, or
touches anything on the $5,000 stock ledger. It's the same market data
Leo/Charles/Greg already use for the equity signal, borrowed once to
answer one question — "is the trigger long today."

No walk-forward gate, no eligibility check, on purpose (see Theo's
docstring in `agents/options_risk_agent.py`): for long-only, capped-loss
options at this account size, the risk model IS the caps, and Theo is the
only thing standing between a signal and an open position.

Entry rule, deliberately simple:
  * SPARK's live signal (today's decision — `generate_signals(bars)
    .iloc[-1]`, not yet shifted the way `positions()` shifts it for the
    stock backtester) is long on `symbol`
  * AND there is no already-open long_call position on that underlying —
    one position per name at a time, no pyramiding while a signal stays
    long. Closing (expiration or manual) is entirely Joseph's job.

Contract selection: the nearest expiration Augustus kept that's still at
least `options.strategy.min_days_to_expiration` out, and the strike
closest to the current spot price ("ATM"). Proposed at the quoted ask
(falling back to mid if the book is empty) — paying the ask is the
realistic side of the spread for a strategy that's buying, even though
Augustus/Joseph mark existing positions at the mid.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from agents.data_agent import MarketData
from agents.options_broker import OptionsLedger
from agents.options_data_agent import OptionsChain, OptionsDataAgent
from agents.options_risk_agent import OptionsProposal
from strategies.base import build_strategies
from utils.config import Config
from utils.logging_setup import get_logger

log = get_logger("options_strategy", agent="SPARK-Calls")


@dataclass
class SignalCheck:
    """One underlying's trigger read for today — surfaced in the console
    report even when it produces no proposal, so 'nothing happened today'
    is visible and explained rather than silent."""

    underlying: str
    trigger: str
    signal_long: bool
    detail: str


def check_signals(config: Config, market: dict[str, MarketData]) -> list[SignalCheck]:
    """Today's read of the trigger strategy for every configured options
    underlying Wong actually fetched. A symbol Wong didn't return, or that
    doesn't have enough history yet, reads as flat with a reason — never
    guessed at."""
    trigger_callsign = str(config.get("options.strategy.trigger_strategy", "SPARK"))
    underlyings = list(config.get("options.underlyings"))
    trigger = build_strategies(config).get(trigger_callsign)
    if trigger is None:
        log.error("options.strategy.trigger_strategy=%r is not a registered/enabled "
                 "equity strategy — no options signals can be checked.", trigger_callsign)
        return [SignalCheck(s, trigger_callsign, False, "trigger strategy unavailable")
               for s in underlyings]

    checks: list[SignalCheck] = []
    for symbol in underlyings:
        data = market.get(symbol)
        if data is None or data.bars.empty:
            checks.append(SignalCheck(symbol, trigger_callsign, False,
                                      "no stock-side data available"))
            continue
        if len(data.bars) < trigger.warmup:
            checks.append(SignalCheck(symbol, trigger_callsign, False,
                                      f"only {len(data.bars)} bars, needs {trigger.warmup}"))
            continue

        signal = trigger.generate_signals(data.bars).iloc[-1]
        is_long = bool(signal and signal > 0)
        note = "synthetic stock data, not trusted" if data.is_synthetic else "live"
        checks.append(SignalCheck(symbol, trigger_callsign, is_long,
                                  f"{'long' if is_long else 'flat'} ({note})"))
    return checks


def build_proposals(config: Config, market: dict[str, MarketData],
                    chains: dict[str, OptionsChain], ledger: OptionsLedger,
                    augustus: OptionsDataAgent
                    ) -> tuple[list[OptionsProposal], list[SignalCheck]]:
    """The one thing this module hands to Joseph: a (usually empty) list of
    proposals, plus the signal reads for the console report. Building a
    proposal never opens anything — Theo still has to approve it and
    Joseph still has to execute it."""
    checks = check_signals(config, market)
    contracts_per_signal = int(config.get("options.strategy.contracts_per_signal", 1))
    min_dte = int(config.get("options.strategy.min_days_to_expiration", 3))
    today = date.today()

    open_calls = {p.underlying for p in ledger.positions.values()
                 if p.option_type == "long_call"}

    proposals: list[OptionsProposal] = []
    for check in checks:
        if not check.signal_long:
            continue
        if check.underlying in open_calls:
            log.info("%s: %s long but a call is already open — holding, not "
                     "pyramiding.", check.underlying, check.trigger)
            continue

        chain = chains.get(check.underlying)
        if chain is None or chain.is_synthetic:
            log.warning("%s: %s signals long but no live option chain — skipping "
                       "(a synthetic chain is never tradable).",
                       check.underlying, check.trigger)
            continue

        data = market.get(check.underlying)
        if data is None or data.bars.empty:
            continue
        spot = float(data.bars["close"].iloc[-1])

        candidates = [e for e in chain.expirations
                     if (date.fromisoformat(e) - today).days >= min_dte]
        expiration = (candidates or chain.expirations or [None])[0]
        if expiration is None:
            log.warning("%s: no usable expiration in the chain — skipping.",
                       check.underlying)
            continue

        contract = augustus.find_contract(chain, "long_call", target_strike=spot,
                                          expiration=expiration)
        if contract is None:
            log.warning("%s: no call contracts for expiration %s — skipping.",
                       check.underlying, expiration)
            continue

        premium = contract.ask if contract.ask > 0 else contract.mid
        if premium <= 0:
            log.warning("%s: no usable quote (ask and mid both 0) — skipping.",
                       contract.label)
            continue

        proposals.append(OptionsProposal(
            underlying=check.underlying, option_type="long_call",
            strike=contract.strike, expiration=expiration,
            premium_per_contract=round(premium, 2), contracts=contracts_per_signal,
            reason=f"{check.trigger} signals long on {check.underlying}",
        ))
    return proposals, checks
