"""Meridian Capital — orchestrator CLI.

    python main.py research           # backtest + validate + console report
    python main.py paper              # + simulated fills, incl. the options leg
    python main.py live               # real money             (Phase 5, triple-gated)
    python main.py killswitch [--options]     # flatten open positions + halt
    python main.py clear-halt [--options]     # manually clear a killswitch halt
    python main.py bench   --trader X --trigger validation|drift --reason "..."
    python main.py unbench --trader X --trigger validation|drift

Research and paper mode share one pipeline: Wong fetches, David blocks broken
data, Leo backtests with cost modeling, Charles walk-forward validates and
risk-parity allocates, Greg classifies each ticker's regime and tilts
capital, and George renders the HTML dashboard + Slack standup. Paper mode
additionally runs Cornelius's PaperBroker against a persisted $5,000 virtual
ledger and the lifecycle agent's benching recommendations. `bench`/`unbench`/
`clear-halt` are the operator's explicit approval mechanism for section 10's
"no automatic benching without operator approval" rule -- not one of the
spec's four named entry points, but a necessary way to exercise it from a
CLI rather than an interactive Slack app.

Options is a separate ledger and a separate set of agents, but no longer a
separate run. Whatever Wong/David/Leo/Charles/Greg agree on for SPY or QQQ
inside `paper` is the same decision handed to the options desk: if
`options.enabled` is on, SPARK's slice of SPY/QQQ's target weight is removed
before Cornelius acts (SPARK's view trades as options, never also as shares),
while every other live strategy's slice of SPY/QQQ is bought as real shares
like any other ticker. That same agreement --
reusing the very same `market` bars, not a second independent read -- is
what the SPARK-calls strategy (`strategies/options_strategy.py`) checks.
Augustus fetches SPY/QQQ option chains, Theo enforces the four hard caps,
and Joseph executes against its own $1,000 ledger
(`reports/options_ledger.json`) -- never blended with the $5,000 stock
ledger, never sharing a halt. `killswitch` and `clear-halt` still take an
explicit `--options` flag to pick which bucket they target -- omitting it
always means the stock side, so a stock-side habit
(`python main.py killswitch`) can never accidentally reach into the options
bucket, or vice versa.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from agents.backtest_agent import BacktestAgent
from agents.compliance_agent import ComplianceAgent
from agents.data_agent import DataAgent
from agents.lifecycle_agent import LifecycleAgent
from agents.options_broker import OptionsBroker
from agents.options_data_agent import OptionsDataAgent
from agents.options_risk_agent import OptionsRiskAgent
from agents.portfolio_agent import PaperBroker
from agents.regime_agent import RegimeAgent, RegimeReport
from agents.reporting_agent import ReportingAgent
from agents.risk_agent import RiskAgent, RiskReport
from backtester.engine import results_frame
from strategies.options_strategy import SignalCheck, build_proposals
from utils.config import load_config
from utils.notifications import DeskLine, Notifier
from utils.logging_setup import get_logger, setup_logging

log = get_logger("orchestrator")

BANNER = r"""
   MERIDIAN CAPITAL
   multi-agent trading research desk · v1.0 · Phase 4
"""

PHASE_PENDING = {
    "live": ("Phase 5", "LiveBroker stub plus the three-gate refusal logic"),
}


# ------------------------------------------------------------------ shared pipeline

def _run_pipeline(config, symbols: list[str] | None = None):
    """Wong -> David -> Leo -> Charles -> Greg. Shared by research and paper
    mode; execution (Cornelius) is the only thing that differs between them."""
    wong = DataAgent(config)
    market = wong.fetch_universe(symbols)

    david = ComplianceAgent(config)
    compliance = david.review(market)
    blocked = {s: r.reason for s, r in compliance.items() if r.blocked}
    for symbol, reason in blocked.items():
        log.warning("%s blocked: %s", symbol, reason)
    if not blocked:
        log.info("No blocks.")

    leo = BacktestAgent(config)
    results = leo.run_all(market, blocked=blocked)
    benchmarks = leo.benchmarks(market)

    charles = RiskAgent(config)
    risk_report = charles.run(leo.strategies, market, results)

    greg = RegimeAgent(config)
    regime_report = greg.run(leo.strategies, market, results, risk_report)

    return market, compliance, blocked, leo, results, benchmarks, risk_report, regime_report


def _apply_lifecycle_bench(risk_report: RiskReport, lifecycle: LifecycleAgent,
                           state: dict) -> None:
    """Move any operator-approved-benched trader out of the live roster
    before capital gets sized or executed. This is the only place a bench
    actually takes effect -- Charles's own roster pick never sees it."""
    still_live = []
    for trader in risk_report.live_traders:
        benched, reason = lifecycle.effective_benched(trader, state)
        if benched:
            risk_report.benched_traders[trader] = f"operator bench: {reason}"
            risk_report.capital_weights.pop(trader, None)
            log.warning("%s excluded from today's live roster (%s)", trader, reason)
        else:
            still_live.append(trader)
    risk_report.live_traders = still_live


def _lifecycle_step(config, leo, risk_report: RiskReport):
    """Update consecutive-fail counters, apply any standing operator bench,
    and produce today's recommendations (never applied automatically)."""
    lifecycle = LifecycleAgent(config)
    all_traders = set(leo.strategies)
    passed_by_trader = {t: False for t in all_traders}
    for c in risk_report.candidates:
        if c.eligible:
            passed_by_trader[c.strategy] = True
    state = lifecycle.record_validation_results(all_traders, passed_by_trader)
    _apply_lifecycle_bench(risk_report, lifecycle, state)
    return lifecycle, state


# ------------------------------------------------------------------ research

def run_research(config, symbols: list[str] | None = None,
                 post_slack: bool = True) -> pd.DataFrame:
    """Fetch, check, backtest, validate, classify, allocate, report. No
    execution, safe to run any time."""
    market, compliance, blocked, leo, results, benchmarks, risk_report, regime_report = (
        _run_pipeline(config, symbols))

    lifecycle, state = _lifecycle_step(config, leo, risk_report)
    recommendations = lifecycle.recommend(state, {}, {})

    george = ReportingAgent(config)
    dashboard = george.run(market, compliance, results, benchmarks, risk_report,
                           regime_report, recommendations=recommendations,
                           post_slack=post_slack)

    frame = results_frame(results)
    _print_report(config, market, frame, benchmarks, blocked, risk_report,
                  regime_report, dashboard, recommendations, ledger=None)
    return frame


# ------------------------------------------------------------------ paper

def run_paper(config, symbols: list[str] | None = None, post_slack: bool = True):
    """Same pipeline as research, plus Cornelius's simulated fills against
    the persisted paper ledger."""
    market, compliance, blocked, leo, results, benchmarks, risk_report, regime_report = (
        _run_pipeline(config, symbols))

    lifecycle, state = _lifecycle_step(config, leo, risk_report)
    # Re-run Greg's netting now that any operator bench has pulled traders
    # out of the roster -- the earlier netted_positions may include a
    # now-benched trader's contribution.
    greg = RegimeAgent(config)
    regime_report = greg.run(leo.strategies, market, results, risk_report)

    # If the options bucket is on, SPARK's slice of SPY/QQQ is removed
    # before Cornelius sees it -- SPARK's view on those trades on the options
    # desk (Augustus/Theo/Joseph, same market dict) instead. Every other
    # live strategy's slice of SPY/QQQ still goes to Cornelius as shares.
    options_underlyings = set(config.get("options.underlyings", [])) \
        if config.get("options.enabled", False) else set()

    notifier = Notifier(config, enabled=post_slack)
    cornelius = PaperBroker(config)
    fills_before = _trade_count(cornelius)
    stock_positions = regime_report.netted_positions
    if options_underlyings:
        stock_positions = _route_options_underlyings(
            regime_report, options_underlyings, _options_routed_strategies(config)
        ).netted_positions
        _warn_on_legacy_share_positions(cornelius, market, options_underlyings,
                                        stock_positions)
    ledger = cornelius.execute(market, stock_positions, regime_report.adjustments)
    new_fills = ledger.trades[fills_before:]
    stock_prices = ReportingAgent._prices(market)
    notifier.stock_fills(new_fills, ledger.mark_to_market(stock_prices), ledger.cash)

    predicted_sharpe = {}
    for c in risk_report.candidates:
        if c.eligible and c.strategy in risk_report.live_traders:
            predicted_sharpe[c.strategy] = max(predicted_sharpe.get(c.strategy, 0.0), c.sharpe)
    recommendations = lifecycle.recommend(state, predicted_sharpe, ledger.trader_shadow)

    # George and the console report get the UNROUTED regime report, so the
    # dashboard still shows what the desk actually agreed on for SPY/QQQ.
    # Only Cornelius sees the routed weights (SPARK's SPY/QQQ slice removed).
    george = ReportingAgent(config)
    dashboard = george.run(market, compliance, results, benchmarks, risk_report,
                           regime_report, ledger=ledger, recommendations=recommendations,
                           post_slack=False)  # standup is posted below, once options has run
    dashboard.posted_to_slack = None

    frame = results_frame(results)
    _print_report(config, market, frame, benchmarks, blocked, risk_report,
                  regime_report, dashboard, recommendations, ledger=ledger)

    # The options leg runs last: stock fills are already saved and the stock
    # report already printed, so an options failure can never hide them.
    # It prints its own report and refreshes the options section of
    # latest.html itself.
    payload = george.standup_payload(market, compliance, risk_report, regime_report,
                                     ledger, recommendations, dashboard.html_path,
                                     new_fills=new_fills)
    if options_underlyings:
        try:
            leg = _options_leg(config, market, blocked=blocked, notifier=notifier)
            if leg is not None:
                payload.options_lines = leg.lines
                payload.books.append(leg.book)
        except Exception as exc:
            log.exception("Options leg failed -- stock fills and the stock report "
                          "above are unaffected; the options ledger was not updated "
                          "past the point of failure.")
            notifier.failure("Options leg", exc, impact=(
                "Stock fills and the stock report are unaffected. The options ledger "
                "was not updated past the point of failure."))
            payload.options_lines = [DeskLine(
                "\U0001F3AF", "Joseph", "Options Execution",
                f"Options leg failed this run ({type(exc).__name__}) — see the alert.",
                "bad")]

    if post_slack:
        posted = notifier.standup(payload)
        print(f"  Slack: standup {'posted' if posted else 'not posted (see log)'}, "
              f"{len(notifier.sent)} message(s) sent this run.\n")
    return frame


def _trade_count(broker) -> int:
    """How many trades a ledger already holds, so this run's new ones can be
    sliced off afterwards. A broken ledger is execute()'s problem to report."""
    try:
        return len(broker.load_ledger().trades)
    except Exception:
        return 0


# ------------------------------------------------------------------ options routing

DEFAULT_OPTIONS_ROUTED_STRATEGIES = frozenset({"SPARK"})


def _options_routed_strategies(config) -> frozenset:
    """Which strategies' SPY/QQQ views trade on the options desk instead of
    as shares. Only SPARK drives the options desk today (see
    strategies/options_strategy.py), so only SPARK's slice is routed."""
    return frozenset(config.get("options.routed_strategies",
                                sorted(DEFAULT_OPTIONS_ROUTED_STRATEGIES)))


def _route_options_underlyings(regime_report: RegimeReport, underlyings: set[str],
                               routed_strategies=DEFAULT_OPTIONS_ROUTED_STRATEGIES
                               ) -> "RegimeReport":
    """Strip the options-routed strategies' slice (SPARK) out of SPY/QQQ's
    netted target before Cornelius sees it. SPARK's view on those symbols
    trades on the options desk -- never as shares too, so one strategy's one
    decision never becomes two positions. Every OTHER live strategy's slice
    of SPY/QQQ passes through and is bought as real shares, same as any
    other ticker. Symbols outside `underlyings` pass through untouched.

    Uses NettedPosition.per_trader (each contributor's pre-cap slice). The
    recomputed weight is re-capped at the original target -- removing a
    positive slice can only lower the blend, so `min(new_raw, old_target)`
    is exactly `min(new_raw, max_position_pct)` without needing the config.
    A position built without a per-strategy breakdown can't be split
    safely, so if any routed strategy contributed it's zeroed (the old,
    conservative behavior)."""
    import dataclasses
    routed_strategies = frozenset(routed_strategies)
    netted = dict(regime_report.netted_positions)
    for symbol in underlyings:
        pos = netted.get(symbol)
        if pos is None or pos.target_weight == 0.0:
            continue
        if not any(c in routed_strategies for c in pos.contributors):
            continue  # nobody routed to options contributed -- all shares
        per_trader = dict(getattr(pos, "per_trader", None) or {})
        if not per_trader:
            netted[symbol] = dataclasses.replace(pos, target_weight=0.0)
            continue
        kept = {k: v for k, v in per_trader.items() if k not in routed_strategies}
        raw = sum(kept.values())
        netted[symbol] = dataclasses.replace(
            pos,
            target_weight=max(0.0, min(raw, pos.target_weight)),
            capped=pos.capped and raw > pos.target_weight,
            contributors=[c for c in pos.contributors if c not in routed_strategies],
            per_trader=kept,
        )
    return dataclasses.replace(regime_report, netted_positions=netted)


def _warn_on_legacy_share_positions(cornelius: PaperBroker, market: dict,
                                    underlyings: set[str],
                                    routed_positions: dict | None = None) -> list[str]:
    """SPY/QQQ can be held as shares now, but only on behalf of strategies
    other than SPARK. When the routed target for one of them is zero (e.g.
    SPARK was the only strategy long it today), Cornelius SELLS any shares
    still held there. That's correct, but on an options underlying it should
    never happen silently -- log which shares are being closed and why.

    `routed_positions` is the post-routing netted positions; if omitted,
    every held options underlying is reported (the pre-2026-09-23 behavior)."""
    try:
        held = cornelius.load_ledger().positions
    except Exception:
        return []  # execute() will raise its own, clearer error

    def target(symbol: str) -> float:
        if routed_positions is None:
            return 0.0
        pos = routed_positions.get(symbol)
        return 0.0 if pos is None else float(pos.target_weight)

    legacy = sorted(s for s in underlyings
                    if s in held and s in market and held[s].shares != 0
                    and target(s) == 0.0)
    for symbol in legacy:
        log.warning("%s: closing %.4f leftover share(s) on the stock ledger -- no "
                    "strategy other than SPARK is long %s today, and SPARK's view on "
                    "%s trades on the options desk, not as shares.",
                    symbol, held[symbol].shares, symbol, symbol)
    return legacy


@dataclass
class OptionsLegResult:
    ledger: object
    lines: list = field(default_factory=list)   # DeskLine, for the standup
    book: object = None                         # BookSnapshot, for the standup
    new_trades: list = field(default_factory=list)


def _run_options_leg(config, market: dict, blocked: dict | None = None,
                     notifier: Notifier | None = None):
    """The options leg, returning just Joseph's ledger (None if disabled)."""
    leg = _options_leg(config, market, blocked=blocked, notifier=notifier)
    return leg.ledger if leg is not None else None


def _options_leg(config, market: dict, blocked: dict | None = None,
                 notifier: Notifier | None = None) -> OptionsLegResult | None:
    """Augustus -> the SPARK-calls strategy -> Theo -> Joseph, reusing the
    SAME `market` dict Wong already fetched for the live stock pipeline
    this run -- not a second, independent read of the bars. This is what
    makes it "the agents agreed on a trade, so open a call instead of
    shares" rather than a parallel check that happens to agree.

    Strategy itself is untouched (`strategies/options_strategy.py`, same
    SPARK read as before). Only the trigger for calling this moved -- it
    now fires inside `run_paper`, right after Cornelius, instead of on its
    own daily schedule.

    Any configured underlying missing from `market` (e.g. `paper --symbols
    AAPL`, or SPY/QQQ dropped from `universe`) is fetched here so the signal
    is never silently read as flat. Any underlying David blocked this run is
    dropped from the market handed to the strategy, so it reads flat and
    nothing is opened on it -- same rule the stock desk follows.
    """
    if not config.get("options.enabled", False):
        log.warning("options.enabled is false in config -- the options leg is a "
                   "no-op. Set options.enabled: true to turn the bucket on.")
        return None

    underlyings = list(config.get("options.underlyings"))
    blocked = blocked or {}

    missing = [u for u in underlyings if u not in market]
    if missing:
        log.info("Options underlyings not in this run's market data, fetching: %s",
                 ", ".join(missing))
        fetched = DataAgent(config).fetch_universe(missing)
        market = {**market, **{s: d for s, d in fetched.items() if s in missing}}

    for symbol in underlyings:
        if symbol in blocked:
            log.warning("%s is blocked by compliance (%s) -- no options entry today.",
                        symbol, blocked[symbol])
    market = {s: d for s, d in market.items() if s not in blocked}

    augustus = OptionsDataAgent(config)
    chains = augustus.fetch_universe(underlyings)
    prices = augustus.price_lookup(chains)

    theo = OptionsRiskAgent(config)
    joseph = OptionsBroker(config, risk_agent=theo)

    # Peek at the current ledger to avoid pyramiding (one call per
    # underlying at a time) -- read-only, Joseph's execute() below is
    # still the only thing that writes it.
    current = joseph.load_ledger()
    trades_before, was_halted = len(current.trades), current.halted
    proposals, checks = build_proposals(config, market, chains, current, augustus)

    ledger = joseph.execute(proposals=proposals, prices=prices)
    new_trades = ledger.trades[trades_before:]
    rejections = list(joseph.rejections)

    if notifier is not None:
        notifier.options_activity(new_trades, rejections, ledger.mark_to_market(prices),
                                  ledger.cash, ledger.open_premium)
        if ledger.halted and not was_halted:
            notifier.halt("Options desk", ledger.halt_reason, detail=(
                f"Loss cutoff is {joseph.loss_cutoff_pct:.0%} of the "
                f"${joseph.starting_capital:,.0f} bucket (floor ${joseph.halt_floor:,.2f})."))

    _print_options_report(chains, ledger, prices, checks)

    # Refresh the options section of reports/latest.html. A dashboard
    # problem must never fail the trading run, so it only logs.
    try:
        summary = ReportingAgent.build_options_summary(
            ledger, prices, chains=chains, checks=checks, halt_floor=joseph.halt_floor)
        path = ReportingAgent(config).refresh_options(summary)
        print(f"    Dashboard updated: {path}")
    except Exception as exc:
        log.warning("Options dashboard refresh failed (%s) -- ledger is unaffected.", exc)

    lines = ReportingAgent.options_desk_lines(chains, checks, proposals, rejections,
                                              new_trades, ledger, prices)
    book = ReportingAgent.options_book(ledger, prices, fills_today=len(new_trades))
    return OptionsLegResult(ledger, lines, book, new_trades)


def _print_options_report(chains: dict, ledger, prices: dict,
                          checks: list[SignalCheck]) -> None:
    print("\n" + "=" * 78)
    print(f"  OPTIONS PAPER REPORT — {datetime.now():%Y-%m-%d %H:%M}")
    print("=" * 78)

    print("\n  DATA (Augustus)")
    for symbol, chain in chains.items():
        flag = "  ⚠ SYNTHETIC — not tradable" if chain.is_synthetic else ""
        print(f"    {symbol:<10} {len(chain.expirations)} exp  "
              f"{len(chain.calls):>3} calls  {len(chain.puts):>3} puts  "
              f"source={chain.data_source:<10}{flag}")

    if checks:
        print("\n  STRATEGY SIGNAL (SPARK breakout — calls + puts, Phase B/B2)")
        for c in checks:
            if c.signal_long:
                mark = "📈" if c.direction == "call" else "📉"
            else:
                mark = "  "
            gate = ""
            if c.signal_long and not c.gate_passed:
                gate = f"  ⛔ gate: {c.gate_detail}"
            print(f"    {mark} {c.underlying:<10} [{c.direction:<4}] {c.detail}{gate}")

    equity = ledger.mark_to_market(prices)
    pnl = equity - ledger.starting_capital
    pnl_pct = (pnl / ledger.starting_capital) if ledger.starting_capital else 0.0
    print("\n  OPTIONS LEDGER (Joseph) — separate $%.0f bucket, never blended "
          "with the stock ledger" % ledger.starting_capital)
    print(f"    Equity ${equity:,.2f}  (cash ${ledger.cash:,.2f})   "
          f"P&L {pnl_pct:+.2%} (${pnl:+,.2f}) since inception   "
          f"realized ${ledger.realized_pnl:+,.2f}   "
          f"{len(ledger.trades)} trade(s) total")
    if ledger.halted:
        print(f"    ⛔ HALTED: {ledger.halt_reason} "
              "-- clear with 'python main.py clear-halt --options'")
    if ledger.positions:
        for key, pos in ledger.positions.items():
            price = prices.get(key)
            mv = pos.market_value(price)
            quote = "" if price is not None else "  (no live quote, marked at cost)"
            print(f"      {pos.label:<24} x{pos.contracts}  paid ${pos.premium_paid:>8.2f}  "
                  f"now ${mv:>8.2f}  {pos.days_held()}d held  "
                  f"{pos.days_to_expiration()}d to exp  "
                  f"P&L ${pos.unrealized_pnl(price):+,.2f}{quote}")
    else:
        print("      No open positions.")

    synthetic = [s for s, c in chains.items() if c.is_synthetic]
    if synthetic:
        print(f"\n  ⚠ {', '.join(synthetic)} used a synthetic option chain. Not tradable "
              "data -- any signal there was skipped, never proposed.")

    print("=" * 78 + "\n")




def _print_report(config, market, frame: pd.DataFrame, benchmarks: dict,
                  blocked: dict, risk_report: RiskReport,
                  regime_report: RegimeReport, dashboard,
                  recommendations: list, ledger=None) -> None:
    print("\n" + "=" * 78)
    label = "PAPER" if ledger is not None else "RESEARCH"
    print(f"  {label} REPORT — {datetime.now():%Y-%m-%d %H:%M}")
    print("=" * 78)

    print("\n  DATA")
    for symbol, data in market.items():
        flag = "  ⚠ SYNTHETIC — not trusted" if data.is_synthetic else ""
        last = data.last_date.date() if data.last_date is not None else "n/a"
        print(f"    {symbol:<10} {len(data.bars):>5} bars  "
              f"source={data.data_source:<10} last={last}{flag}")

    print("\n  COMPLIANCE (David)")
    if blocked:
        for symbol, reason in blocked.items():
            print(f"    ⛔ {symbol:<10} {reason}")
    else:
        print("    No blocks.")

    if frame.empty:
        print("\n  No results produced.")
        print("=" * 78 + "\n")
        return

    print("\n  RESULTS (ranked by Sharpe, in-sample)")
    cols = ["strategy", "symbol", "sharpe", "cagr", "max_drawdown",
            "win_rate", "trades", "total_cost", "trusted"]
    view = frame[cols].copy()
    for col in ("cagr", "max_drawdown", "win_rate", "total_cost"):
        view[col] = (view[col] * 100).map(lambda v: f"{v:6.2f}%")
    view["sharpe"] = view["sharpe"].map(lambda v: f"{v:6.2f}")
    print(view.to_string(index=False, max_rows=60))

    print("\n  BUY-AND-HOLD BENCHMARK")
    for symbol, summary in benchmarks.items():
        print(f"    {symbol:<10} Sharpe {summary.sharpe:6.2f}   "
              f"CAGR {summary.cagr * 100:7.2f}%   "
              f"MaxDD {summary.max_drawdown * 100:6.2f}%")

    _print_validation(config, risk_report)
    options_routed = (set(config.get("options.underlyings", []))
                      if ledger is not None and config.get("options.enabled", False)
                      else set())
    _print_regime(regime_report)
    _print_allocation(risk_report, regime_report, options_routed,
                      _options_routed_strategies(config))
    _print_lifecycle(recommendations)
    if ledger is not None:
        _print_ledger(market, ledger)

    synthetic = [s for s, d in market.items() if d.is_synthetic]
    if synthetic:
        print(f"\n  ⚠ {', '.join(synthetic)} used synthetic data. These numbers describe"
              "\n    a random walk, not a market. Do not act on them.")

    print("\n  DASHBOARD (George)")
    if dashboard.html_path is not None:
        print(f"    HTML report: {dashboard.html_path}")
    if dashboard.posted_to_slack is not None:
        print(f"    Slack: {'posted' if dashboard.posted_to_slack else 'not posted (see log above)'}")
    print("    Standup message:")
    for line in dashboard.standup_text.splitlines():
        print(f"      {line}")

    if ledger is None:
        print("\n  Run 'python main.py paper' to trade this allocation against the"
              "\n  persisted $5,000 paper ledger.")
    print("=" * 78 + "\n")


def _print_regime(regime_report: RegimeReport) -> None:
    print("\n  REGIME (Greg)")
    for symbol, c in regime_report.regimes.items():
        print(f"    {symbol:<10} {c.regime:<10} {c.detail}")


def _print_validation(config, risk_report: RiskReport) -> None:
    folds = int(config.get("validation.folds"))
    passing = int(config.get("validation.min_folds_passing"))
    print(f"\n  WALK-FORWARD VALIDATION (Charles) — {passing}-of-{folds} folds must pass")
    if not risk_report.candidates:
        print("    Nothing to validate (every symbol blocked or no strategies ran).")
        return
    rows = sorted(risk_report.candidates, key=lambda c: (-c.eligible, -c.sharpe))
    for c in rows:
        mark = "✔" if c.eligible else "✘"
        detail = "" if c.eligible else f"  — {c.reject_reason}"
        print(f"    {mark} {c.strategy:<8} {c.symbol:<10} "
              f"{c.walkforward.folds_passed}/{c.walkforward.folds_total} folds  "
              f"Sharpe {c.sharpe:6.2f}  MaxDD {c.backtest_max_drawdown * 100:6.2f}%{detail}")
    if not any(c.eligible for c in risk_report.candidates):
        print("\n    No strategy passed validation today — closest contenders above.")
        print("    This is deliberate: no delivery is worse than 'nothing today.'")


def _print_allocation(risk_report: RiskReport, regime_report: RegimeReport,
                      options_routed: set[str] | None = None,
                      routed_strategies=DEFAULT_OPTIONS_ROUTED_STRATEGIES) -> None:
    options_routed = options_routed or set()
    routed_report = (_route_options_underlyings(regime_report, options_routed,
                                                routed_strategies)
                     if options_routed else regime_report)
    print(f"\n  RISK & ALLOCATION (Charles + Greg) — {len(risk_report.live_traders) or 0} of "
          f"{len(risk_report.live_traders) + len(risk_report.benched_traders)} traders live")
    if risk_report.live_traders:
        for slot in risk_report.slots:
            label = "+".join(slot.members)
            print(f"    LIVE  {label:<16} slot weight {slot.weight * 100:5.1f}%")
            for member in slot.members:
                weight = slot.member_weights.get(member, 0.0)
                print(f"          {member:<8} base weight {weight * 100:5.1f}%  "
                      f"${risk_report.dollars(member):,.0f}")
    else:
        print("    No traders live today.")

    if risk_report.benched_traders:
        print("\n    BENCHED")
        for trader, reason in sorted(risk_report.benched_traders.items()):
            print(f"    🪑 {trader:<8} {reason}")

    if regime_report.adjustments:
        print("\n    REGIME ADJUSTMENTS")
        for adj in regime_report.adjustments:
            status = "matched" if adj.matched else ("confirmed" if adj.confirmed else "unconfirmed")
            print(f"    {adj.strategy:<8} {adj.symbol:<10} {adj.regime:<10} {status:<11} "
                  f"{adj.base_weight * 100:5.1f}% -> {adj.adjusted_weight * 100:5.1f}%  "
                  f"{adj.note}")

    if any(p.contributors for p in regime_report.netted_positions.values()):
        print("\n    TARGET ALLOCATION (post-regime)")
        for symbol, pos in regime_report.netted_positions.items():
            if not pos.contributors:
                continue
            cap = "  (capped)" if pos.capped else ""
            routed = ""
            if symbol in options_routed:
                opt = sorted(c for c in pos.contributors if c in routed_strategies)
                shares = routed_report.netted_positions[symbol].target_weight
                if opt and shares > 0:
                    routed = (f"  -> {', '.join(opt)} to options desk; "
                              f"{shares * 100:.1f}% bought as shares")
                elif opt:
                    routed = f"  -> {', '.join(opt)} to options desk, no shares"
                else:
                    routed = "  -> bought as shares"
            print(f"    {symbol:<10} {pos.target_weight * 100:5.1f}% of capital"
                  f"{cap}  <- {', '.join(pos.contributors)}{routed}")
    else:
        print("\n    No target positions today.")


def _print_lifecycle(recommendations: list) -> None:
    if not recommendations:
        return
    print("\n  LIFECYCLE — IMPORTANT INQUIRY (Charles + George)")
    for rec in recommendations:
        print(f"    🪑 {rec.trader:<8} [{rec.trigger}] {rec.action.upper()}  {rec.detail}")
    print("    Recommendation only -- nothing changes without:")
    print("      python main.py bench --trader <CALLSIGN> --trigger validation|drift --reason \"...\"")


def _print_ledger(market, ledger) -> None:
    prices = {s: float(d.bars["close"].iloc[-1]) for s, d in market.items() if not d.bars.empty}
    equity = ledger.mark_to_market(prices)
    pnl = equity - ledger.starting_capital
    pnl_pct = (pnl / ledger.starting_capital) if ledger.starting_capital else 0.0
    print("\n  PAPER LEDGER (Cornelius)")
    print(f"    Equity ${equity:,.2f}  (cash ${ledger.cash:,.2f})   "
          f"P&L {pnl_pct:+.2%} (${pnl:+,.2f}) since inception   "
          f"{len(ledger.trades)} trade(s) total")
    if ledger.halted:
        print(f"    ⛔ HALTED: {ledger.halt_reason} -- clear with 'python main.py clear-halt'")
    if ledger.positions:
        for symbol, pos in ledger.positions.items():
            price = prices.get(symbol, pos.entry_price)
            print(f"      {symbol:<10} {pos.shares:>10.4f} sh  entry ${pos.entry_price:>9.2f}  "
                  f"now ${price:>9.2f}  {pos.days_held()}d held  "
                  f"P&L ${pos.unrealized_pnl(price):+,.2f}")
    else:
        print("      No open positions.")


# ------------------------------------------------------------------ killswitch / lifecycle CLI

def run_killswitch(config, symbols: list[str] | None = None, options: bool = False) -> int:
    if options:
        augustus = OptionsDataAgent(config)
        try:
            prices = augustus.price_lookup(augustus.fetch_universe(symbols))
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("Could not fetch live option prices for killswitch (%s) -- "
                       "flattening open positions at their entry premium instead.", exc)
            prices = {}
        before = OptionsBroker(config).load_ledger()
        OptionsBroker(config).killswitch(prices)
        Notifier(config).halt("Options desk", "Operator killswitch", manual=True, detail=(
            f"Flattened {len(before.positions)} open position(s)."))
        return 0

    wong = DataAgent(config)
    try:
        market = wong.fetch_universe(symbols)
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("Could not fetch live prices for killswitch (%s) -- flattening at "
                   "last known entry prices instead.", exc)
        market = {}
    cornelius = PaperBroker(config)
    open_before = len(cornelius.load_ledger().positions)
    cornelius.killswitch(market)
    Notifier(config).halt("Stock desk", "Operator killswitch", manual=True, detail=(
        f"Flattened {open_before} open position(s)."))
    return 0


def run_clear_halt(config, options: bool = False) -> int:
    if options:
        OptionsBroker(config).clear_halt()
        Notifier(config).halt_cleared("Options desk")
        return 0
    PaperBroker(config).clear_halt()
    Notifier(config).halt_cleared("Stock desk")
    return 0


def run_bench(config, trader: str | None, trigger: str, reason: str, unbench: bool) -> int:
    if not trader:
        log.error("--trader is required for %s.", "unbench" if unbench else "bench")
        return 2
    lifecycle = LifecycleAgent(config)
    Notifier(config).operator_action(
        "Unbenched" if unbench else "Benched",
        f"{trader} ({trigger})" + (f" — {reason}" if reason and not unbench else ""))
    if trigger == "validation":
        if unbench:
            lifecycle.clear_validation_bench(trader)
        else:
            lifecycle.apply_validation_bench(trader, reason)
    else:
        if unbench:
            lifecycle.clear_drift_bench(trader)
        else:
            lifecycle.apply_drift_bench(trader, reason)
    return 0


# ------------------------------------------------------------------ CLI

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py", description="Meridian Capital orchestrator")
    parser.add_argument("mode", choices=["research", "paper", "paper-options", "live",
                                         "killswitch", "clear-halt", "bench", "unbench"])
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="override the configured universe")
    parser.add_argument("--options", action="store_true",
                        help="target the options bucket (Joseph) for killswitch/"
                             "clear-halt instead of the stock bucket (Cornelius)")
    parser.add_argument("--i-understand-the-risk", action="store_true",
                        dest="risk_ack", help="required for live mode (Phase 5)")
    parser.add_argument("--trader", default=None, help="callsign for bench/unbench")
    parser.add_argument("--trigger", choices=["validation", "drift"], default="drift",
                        help="which bench type applies to bench/unbench")
    parser.add_argument("--reason", default="", help="reason text for 'bench'")
    parser.add_argument("--no-slack", action="store_true",
                        help="skip posting the standup message even if a webhook is configured")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    setup_logging(
        level="DEBUG" if args.verbose else config.get("logging.level", "INFO"),
        log_file=config.repo_path(config.get("logging.file", "reports/meridian.log")),
        console=config.get("logging.console", True),
    )
    print(BANNER)

    if args.mode in ("research", "paper"):
        runner = run_research if args.mode == "research" else run_paper
        try:
            runner(config, args.symbols, post_slack=not args.no_slack)
        except Exception as exc:
            log.exception("%s run failed.", args.mode)
            Notifier(config, enabled=not args.no_slack).failure(
                f"{args.mode.capitalize()} run", exc,
                impact="Anything saved before the failure (e.g. stock fills) is kept; "
                       "nothing after it ran. Check the log, fix, and re-run.")
            return 1
        return 0
    if args.mode == "paper-options":
        log.error("`paper-options` no longer runs standalone -- SPY/QQQ options "
                  "entries now run inside `python main.py paper`, right after the "
                  "stock fills. Run `python main.py paper` instead (the launchd "
                  "job in scripts/ already does).")
        return 1
    if args.mode == "killswitch":
        return run_killswitch(config, args.symbols, options=args.options)
    if args.mode == "clear-halt":
        return run_clear_halt(config, options=args.options)
    if args.mode in ("bench", "unbench"):
        return run_bench(config, args.trader, args.trigger, args.reason,
                         unbench=(args.mode == "unbench"))

    phase, what = PHASE_PENDING[args.mode]
    log.error("Mode '%s' is not implemented yet — it arrives in %s (%s).",
              args.mode, phase, what)
    if args.mode == "live":
        log.error("Live mode will still require all three gates: a real "
                  "LiveBroker implementation, enable_live_trading: true in "
                  "config, and --i-understand-the-risk.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
