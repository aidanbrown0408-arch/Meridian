"""Meridian Capital — orchestrator CLI.

    python main.py research           # backtest + validate + console report
    python main.py paper              # + simulated fills against a persisted ledger
    python main.py live               # real money             (Phase 5, triple-gated)
    python main.py killswitch         # flatten every open paper position + halt
    python main.py clear-halt         # manually clear a killswitch halt
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
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime

import pandas as pd

from agents.backtest_agent import BacktestAgent
from agents.compliance_agent import ComplianceAgent
from agents.data_agent import DataAgent
from agents.lifecycle_agent import LifecycleAgent
from agents.portfolio_agent import PaperBroker
from agents.regime_agent import RegimeAgent, RegimeReport
from agents.reporting_agent import ReportingAgent
from agents.risk_agent import RiskAgent, RiskReport
from backtester.engine import results_frame
from utils.config import load_config
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
                           strategies=leo.strategies, post_slack=post_slack)

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

    cornelius = PaperBroker(config)
    ledger = cornelius.execute(market, regime_report.netted_positions, regime_report.adjustments)

    predicted_sharpe = {}
    for c in risk_report.candidates:
        if c.eligible and c.strategy in risk_report.live_traders:
            predicted_sharpe[c.strategy] = max(predicted_sharpe.get(c.strategy, 0.0), c.sharpe)
    recommendations = lifecycle.recommend(state, predicted_sharpe, ledger.trader_shadow)

    george = ReportingAgent(config)
    dashboard = george.run(market, compliance, results, benchmarks, risk_report,
                           regime_report, ledger=ledger, recommendations=recommendations,
                           strategies=leo.strategies, post_slack=post_slack)

    frame = results_frame(results)
    _print_report(config, market, frame, benchmarks, blocked, risk_report,
                  regime_report, dashboard, recommendations, ledger=ledger)
    return frame


# ------------------------------------------------------------------ console report

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
    _print_regime(regime_report)
    _print_allocation(risk_report, regime_report)
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


def _print_allocation(risk_report: RiskReport, regime_report: RegimeReport) -> None:
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
            print(f"    {symbol:<10} {pos.target_weight * 100:5.1f}% of capital"
                  f"{cap}  <- {', '.join(pos.contributors)}")
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

def run_killswitch(config, symbols: list[str] | None = None) -> int:
    wong = DataAgent(config)
    try:
        market = wong.fetch_universe(symbols)
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("Could not fetch live prices for killswitch (%s) -- flattening at "
                   "last known entry prices instead.", exc)
        market = {}
    cornelius = PaperBroker(config)
    cornelius.killswitch(market)
    return 0


def run_clear_halt(config) -> int:
    PaperBroker(config).clear_halt()
    return 0


def run_bench(config, trader: str | None, trigger: str, reason: str, unbench: bool) -> int:
    if not trader:
        log.error("--trader is required for %s.", "unbench" if unbench else "bench")
        return 2
    lifecycle = LifecycleAgent(config)
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
    parser.add_argument("mode", choices=["research", "paper", "live", "killswitch",
                                         "clear-halt", "bench", "unbench"])
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="override the configured universe")
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

    if args.mode == "research":
        run_research(config, args.symbols, post_slack=not args.no_slack)
        return 0
    if args.mode == "paper":
        run_paper(config, args.symbols, post_slack=not args.no_slack)
        return 0
    if args.mode == "killswitch":
        return run_killswitch(config, args.symbols)
    if args.mode == "clear-halt":
        return run_clear_halt(config)
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
