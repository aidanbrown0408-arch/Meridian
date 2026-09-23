"""Options strategy (Phase B/B2) — "buy a call when SPARK breaks out long,
buy a put when the same Donchian mechanic breaks down."

Calls reuse Leo's exact `SparkStrategy` (same params as `strategies.SPARK` in
config, so the options entry never drifts from what the stock side sees)
against Wong's stock bars for each configured options underlying. Puts mirror
that same breakout mechanic on the downside — SPARK itself is long-or-flat
only (0..1, no short concept), so the put trigger is a standalone Donchian
breakdown computed here with SPARK's own `entry_window`/`exit_window`
params, never a change to `strategies/spark.py` or the stock backtester.
Reading bars here is read-only either way: this module never proposes,
opens, or touches anything on the $5,000 stock ledger.

Market gate (added after Phase B2): before EITHER direction's signal is
even allowed to become a proposal, `_market_gates()` requires two things
on that underlying, run fresh every time this module is called:
  1. Greg's regime read is "trending" — a Donchian breakout/breakdown is a
     trend-following mechanic (this is SPARK's own `best_regimes`) and is
     expected to whipsaw in chop, so "choppy" and "undecided" both block.
  2. SPARK clears Charles's own walk-forward + drawdown grading on that
     specific underlying — reusing `RiskAgent._grade()` verbatim so this
     never silently drifts from what the stock side considers "validated."
     Run fresh here rather than read off Charles's live stock roster, so
     the options desk's answer never depends on whether SPARK happens to
     be one of today's 3 live STOCK traders — an unrelated capital-roster
     decision on a completely separate ledger.
Both checks apply identically to calls and puts: it's the same Donchian
mechanism on the same underlying, just mirrored, so it's trusted (or not)
the same way in both directions. Toggle with
`options.strategy.market_gate_enabled: false` to fall back to Theo's caps
alone (the original Phase B/B2 behavior).

Once a signal clears the gate, Theo is still the only thing standing
between it and an open position (see Theo's docstring in
`agents/options_risk_agent.py`) — the gate answers "should this even be
considered," not "is it sized/capped correctly," which stays entirely
Theo's job.

Entry rules, deliberately simple and symmetric:
  * Call: SPARK's live signal (today's decision — `generate_signals(bars)
    .iloc[-1]`, not yet shifted the way `positions()` shifts it for the
    stock backtester) is long on `symbol`.
  * Put: price closes below the prior `entry_window`-day low (mirrors
    SPARK's "close above the prior entry_window-day high" call trigger),
    and stays "active" until it recovers above the prior `exit_window`-day
    high — same hold-state mechanic as `Strategy._hold`, just flipped.
  * Either direction requires there be no already-open position (call OR
    put) on that underlying — one position per name at a time, no
    pyramiding and no straddling. Closing (expiration or manual) is
    entirely Joseph's job. Puts can be turned off without touching this
    module via `options.strategy.puts_enabled: false`.

Contract selection: the nearest expiration Augustus kept that's still at
least `options.strategy.min_days_to_expiration` out, and the strike
closest to the current spot price ("ATM"), on whichever side (calls/puts)
the signal calls for. Proposed at the quoted ask (falling back to mid if
the book is empty) — paying the ask is the realistic side of the spread
for a strategy that's buying, even though Augustus/Joseph mark existing
positions at the mid.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from agents.data_agent import MarketData
from agents.options_broker import OptionsLedger
from agents.options_data_agent import OptionsChain, OptionsDataAgent
from agents.options_risk_agent import OptionsProposal
from agents.regime_agent import RegimeAgent
from agents.risk_agent import RiskAgent, TraderCandidate
from backtester.engine import BacktestEngine
from backtester.walkforward import WalkForwardValidator
from strategies.base import Strategy, build_strategies
from utils.config import Config
from utils.logging_setup import get_logger

log = get_logger("options_strategy", agent="SPARK-Calls")


@dataclass
class SignalCheck:
    """One underlying's trigger read for today, for one direction — surfaced
    in the console report even when it produces no proposal, so 'nothing
    happened today' is visible and explained rather than silent."""

    underlying: str
    trigger: str
    signal_long: bool
    detail: str
    direction: str = "call"      # "call" | "put"
    gate_passed: bool = True     # today's regime/validation read, see MarketGate
    gate_detail: str = ""


@dataclass
class MarketGate:
    """One underlying's pre-signal verdict: has today's market actually
    earned the right to be traded, independent of which direction (if any)
    is signaling. See the "Market gate" section of the module docstring."""

    underlying: str
    regime: str
    passed: bool
    reason: str


def _donchian_breakdown(bars: pd.DataFrame, entry_window: int, exit_window: int) -> pd.Series:
    """Mirror of `SurgeStrategy`/`SparkStrategy.generate_signals` (see
    `strategies/surge.py`), flipped for the downside: "active" (bearish)
    once price closes below the prior `entry_window`-day low, released once
    it closes back above the prior `exit_window`-day high. Deliberately not
    a `Strategy` subclass — those are long-or-flat only and feed the stock
    backtester; this exists solely to trigger a put proposal here and is
    never seen by Leo/Charles/Greg."""
    close = bars["close"]
    high = bars["high"].fillna(close)
    low = bars["low"].fillna(close)

    # Prior-window extremes, same "exclude today" shift SURGE/SPARK use so a
    # new low is a genuine break, not today's own bar leaking into its
    # threshold.
    lower = low.rolling(entry_window, min_periods=entry_window).min().shift(1)
    upper = high.rolling(exit_window, min_periods=exit_window).max().shift(1)

    return Strategy._hold(close < lower, close > upper)


def check_signals(config: Config, market: dict[str, MarketData]) -> list[SignalCheck]:
    """Today's read of the call trigger (SPARK breakout) for every configured
    options underlying Wong actually fetched. A symbol Wong didn't return, or
    that doesn't have enough history yet, reads as flat with a reason —
    never guessed at."""
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
                                  f"{'long' if is_long else 'flat'} ({note})",
                                  direction="call"))
    return checks


def check_put_signals(config: Config, market: dict[str, MarketData]) -> list[SignalCheck]:
    """Today's read of the put trigger (Donchian breakdown, mirroring the
    call trigger's own params) for every configured options underlying.
    Same never-guess-at-it shape as `check_signals`. Returns an empty list
    outright if `options.strategy.puts_enabled` is false, or if the
    configured call trigger isn't a registered strategy (there's nothing to
    mirror the params of)."""
    if not bool(config.get("options.strategy.puts_enabled", True)):
        return []

    trigger_callsign = str(config.get("options.strategy.trigger_strategy", "SPARK"))
    underlyings = list(config.get("options.underlyings"))
    trigger = build_strategies(config).get(trigger_callsign)
    put_trigger_name = f"{trigger_callsign}-puts"
    if trigger is None:
        log.error("options.strategy.trigger_strategy=%r is not a registered/enabled "
                 "equity strategy — no put signals can be checked.", trigger_callsign)
        return [SignalCheck(s, put_trigger_name, False, "trigger strategy unavailable",
                            direction="put")
               for s in underlyings]

    entry_window = int(trigger.params.get("entry_window", 20))
    exit_window = int(trigger.params.get("exit_window", 10))

    checks: list[SignalCheck] = []
    for symbol in underlyings:
        data = market.get(symbol)
        if data is None or data.bars.empty:
            checks.append(SignalCheck(symbol, put_trigger_name, False,
                                      "no stock-side data available", direction="put"))
            continue
        if len(data.bars) < trigger.warmup:
            checks.append(SignalCheck(symbol, put_trigger_name, False,
                                      f"only {len(data.bars)} bars, needs {trigger.warmup}",
                                      direction="put"))
            continue

        state = _donchian_breakdown(data.bars, entry_window, exit_window)
        is_active = bool(state.iloc[-1] and state.iloc[-1] > 0)
        note = "synthetic stock data, not trusted" if data.is_synthetic else "live"
        checks.append(SignalCheck(symbol, put_trigger_name, is_active,
                                  f"{'breakdown' if is_active else 'flat'} ({note})",
                                  direction="put"))
    return checks


def _market_gates(config: Config, market: dict[str, MarketData],
                  underlyings: list[str]) -> dict[str, MarketGate]:
    """Today's trending-regime + walk-forward-validation verdict for every
    configured underlying. Returns a gate for exactly the underlyings Wong
    actually has data for; a symbol missing from `market` simply has no
    entry (callers treat that as blocked — see `_propose_one`).

    Deliberately re-runs Greg's classifier and Charles's grading fresh on
    each call rather than reading Charles's/Greg's own daily RiskReport/
    RegimeReport (which only cover the stock side's already-chosen live
    roster) — this way the options desk's answer depends only on today's
    market and SPARK's own track record on that ticker, never on whether
    SPARK happens to be one of the 3 stock traders Charles picked today.

    Returns an empty dict outright if `options.strategy.market_gate_enabled`
    is false, or if the configured trigger isn't a registered strategy."""
    if not bool(config.get("options.strategy.market_gate_enabled", True)):
        return {}

    trigger_callsign = str(config.get("options.strategy.trigger_strategy", "SPARK"))
    trigger = build_strategies(config).get(trigger_callsign)
    if trigger is None:
        return {}

    regime_agent = RegimeAgent(config)
    charles = RiskAgent(config)
    validator = WalkForwardValidator(config)
    engine = BacktestEngine(config)

    gates: dict[str, MarketGate] = {}
    for symbol in underlyings:
        data = market.get(symbol)
        if data is None or data.bars.empty:
            continue

        classification = regime_agent.classify(data.bars, symbol)
        if classification.regime != "trending":
            gates[symbol] = MarketGate(
                symbol, classification.regime, False,
                f"regime is {classification.regime}, not trending ({classification.detail})")
            continue

        bt = engine.run(trigger, data.bars, symbol, asset_class=data.asset_class,
                        data_source=data.data_source)
        if bt.blocked:
            gates[symbol] = MarketGate(
                symbol, classification.regime, False,
                f"trending, but {trigger_callsign} has insufficient history to grade "
                f"({bt.block_reason})")
            continue

        wf = validator.validate(trigger, data.bars, symbol, data.asset_class)
        candidate = TraderCandidate(
            strategy=trigger_callsign, symbol=symbol, asset_class=data.asset_class,
            walkforward=wf, backtest_max_drawdown=bt.summary.max_drawdown,
            annual_vol=bt.summary.annual_vol, sharpe=bt.summary.sharpe,
        )
        # Reuse Charles's own grading verbatim (walk-forward pass/fail, then
        # the hard full-period drawdown cap) so this can never silently
        # drift from what "validated" means on the stock side.
        eligible, reject_reason = charles._grade(candidate)
        if not eligible:
            gates[symbol] = MarketGate(
                symbol, classification.regime, False,
                f"trending, but {trigger_callsign} fails validation: {reject_reason}")
        else:
            gates[symbol] = MarketGate(symbol, classification.regime, True,
                                       "trending & validated")
    return gates


def _propose_one(config: Config, check: SignalCheck, market: dict[str, MarketData],
                 chains: dict[str, OptionsChain], augustus: OptionsDataAgent,
                 open_underlyings: set[str], min_dte: int, today: date,
                 contracts_per_signal: int, option_type: str
                 ) -> OptionsProposal | None:
    """Shared plumbing for turning one active SignalCheck into an
    OptionsProposal, whichever side of the chain it reads from."""
    if not check.signal_long:
        return None
    if not check.gate_passed:
        log.info("%s: %s active but blocked by today's market gate — %s",
                 check.underlying, check.trigger, check.gate_detail)
        return None
    if check.underlying in open_underlyings:
        log.info("%s: %s active but a position is already open — holding, not "
                 "pyramiding or straddling.", check.underlying, check.trigger)
        return None

    chain = chains.get(check.underlying)
    if chain is None or chain.is_synthetic:
        log.warning("%s: %s signals active but no live option chain — skipping "
                   "(a synthetic chain is never tradable).",
                   check.underlying, check.trigger)
        return None

    data = market.get(check.underlying)
    if data is None or data.bars.empty:
        return None
    spot = float(data.bars["close"].iloc[-1])

    candidates = [e for e in chain.expirations
                 if (date.fromisoformat(e) - today).days >= min_dte]
    expiration = (candidates or chain.expirations or [None])[0]
    if expiration is None:
        log.warning("%s: no usable expiration in the chain — skipping.",
                   check.underlying)
        return None

    contract = augustus.find_contract(chain, option_type, target_strike=spot,
                                      expiration=expiration)
    if contract is None:
        side = "call" if option_type == "long_call" else "put"
        log.warning("%s: no %s contracts for expiration %s — skipping.",
                   check.underlying, side, expiration)
        return None

    premium = contract.ask if contract.ask > 0 else contract.mid
    if premium <= 0:
        log.warning("%s: no usable quote (ask and mid both 0) — skipping.",
                   contract.label)
        return None

    verb = "signals long on" if option_type == "long_call" else "signals a breakdown on"
    return OptionsProposal(
        underlying=check.underlying, option_type=option_type,
        strike=contract.strike, expiration=expiration,
        premium_per_contract=round(premium, 2), contracts=contracts_per_signal,
        reason=f"{check.trigger} {verb} {check.underlying}",
    )


def build_proposals(config: Config, market: dict[str, MarketData],
                    chains: dict[str, OptionsChain], ledger: OptionsLedger,
                    augustus: OptionsDataAgent
                    ) -> tuple[list[OptionsProposal], list[SignalCheck]]:
    """The one thing this module hands to Joseph: a (usually empty) list of
    proposals, plus the signal reads for the console report. Building a
    proposal never opens anything — Theo still has to approve it and
    Joseph still has to execute it.

    Calls and puts are evaluated against the SAME open-position set, so a
    symbol already holding either type never gets a second proposal of
    either type in the same run — one directional bet per underlying at a
    time, by design (see module docstring)."""
    call_checks = check_signals(config, market)
    put_checks = check_put_signals(config, market)

    underlyings = list(config.get("options.underlyings"))
    gates = _market_gates(config, market, underlyings)
    for c in (*call_checks, *put_checks):
        gate = gates.get(c.underlying)
        if gate is not None:
            c.gate_passed = gate.passed
            c.gate_detail = gate.reason
        # A symbol missing from `gates` (e.g. Wong has no data for it, or
        # the gate is disabled) reads as gate_passed=True, its default —
        # "no opinion" never blocks on its own; the earlier per-symbol
        # data checks in check_signals/check_put_signals already handle a
        # genuinely missing underlying.

    # Interleave per underlying (call read, then put read) so the console/
    # dashboard reports both directions together for each name.
    put_by_symbol = {c.underlying: c for c in put_checks}
    checks: list[SignalCheck] = []
    for c in call_checks:
        checks.append(c)
        if c.underlying in put_by_symbol:
            checks.append(put_by_symbol[c.underlying])

    contracts_per_signal = int(config.get("options.strategy.contracts_per_signal", 1))
    min_dte = int(config.get("options.strategy.min_days_to_expiration", 3))
    today = date.today()

    # ANY open position (call or put) on a name blocks a new proposal on
    # that same name — enforced once, here, ahead of both directions.
    open_underlyings = {p.underlying for p in ledger.positions.values()}

    proposals: list[OptionsProposal] = []
    for check in call_checks:
        proposal = _propose_one(config, check, market, chains, augustus,
                                open_underlyings, min_dte, today,
                                contracts_per_signal, "long_call")
        if proposal is not None:
            proposals.append(proposal)
            open_underlyings.add(check.underlying)  # don't also open a put this run

    for check in put_checks:
        proposal = _propose_one(config, check, market, chains, augustus,
                                open_underlyings, min_dte, today,
                                contracts_per_signal, "long_put")
        if proposal is not None:
            proposals.append(proposal)
            open_underlyings.add(check.underlying)

    return proposals, checks
