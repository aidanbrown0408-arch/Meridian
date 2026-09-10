"""Meridian Capital — orchestrator CLI.

    python main.py research      # backtest + validate + console report (working)
    python main.py paper         # simulated fills        (Phase 4)
    python main.py live          # real money             (Phase 5, triple-gated)
    python main.py killswitch    # flatten + halt         (Phase 4)

Research mode now runs the full pre-execution pipeline: Wong fetches, David
blocks anything with broken data, Leo backtests with cost modeling, Charles
walk-forward validates the survivors and turns passing traders into a
risk-parity allocation, and the results print as a ranked table plus a
validation and allocation report. The other modes exist as explicit refusals
rather than missing commands, so running them tells you which phase they
arrive in instead of a stack trace.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime

import pandas as pd

from agents.backtest_agent import BacktestAgent
from agents.compliance_agent import ComplianceAgent
from agents.data_agent import DataAgent
from agents.risk_agent import RiskAgent, RiskReport
from backtester.engine import results_frame
from utils.config import load_config
from utils.logging_setup import get_logger, setup_logging

log = get_logger("orchestrator")

BANNER = r"""
   MERIDIAN CAPITAL
   multi-agent trading research desk · v1.0 · Phase 2
"""

PHASE_PENDING = {
    "paper": ("Phase 4", "PaperBroker with a persisted $5,000 virtual ledger"),
    "live": ("Phase 5", "LiveBroker stub plus the three-gate refusal logic"),
    "killswitch": ("Phase 4", "position flattening and the halt flag"),
}


def run_research(config, symbols: list[str] | None = None) -> pd.DataFrame:
    """Fetch, check, backtest, validate, allocate, report. No execution, safe
    to run any time."""
    wong = DataAgent(config)
    market = wong.fetch_universe(symbols)

    david = ComplianceAgent(config)
    blocked = david.blocked_symbols(market)

    leo = BacktestAgent(config)
    results = leo.run_all(market, blocked=blocked)
    benchmarks = leo.benchmarks(market)

    charles = RiskAgent(config)
    risk_report = charles.run(leo.strategies, market, results)

    frame = results_frame(results)
    _print_report(config, market, frame, benchmarks, blocked, risk_report)
    return frame


def _print_report(config, market, frame: pd.DataFrame, benchmarks: dict,
                  blocked: dict, risk_report: RiskReport) -> None:
    """Console stand-in for George's dashboard, which lands in Phase 3."""
    print("\n" + "=" * 78)
    print(f"  RESEARCH REPORT — {datetime.now():%Y-%m-%d %H:%M}")
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
    _print_allocation(risk_report)

    synthetic = [s for s, d in market.items() if d.is_synthetic]
    if synthetic:
        print(f"\n  ⚠ {', '.join(synthetic)} used synthetic data. These numbers describe"
              "\n    a random walk, not a market. Do not act on them.")

    print("\n  Phase 3 adds regime detection and the Slack dashboard. Until then"
          "\n  this console report and the walk-forward gate above are it.")
    print("=" * 78 + "\n")


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


def _print_allocation(risk_report: RiskReport) -> None:
    print(f"\n  RISK & ALLOCATION (Charles) — max {len(risk_report.live_traders) or 0} of "
          f"{len(risk_report.live_traders) + len(risk_report.benched_traders)} traders live")
    if risk_report.live_traders:
        for slot in risk_report.slots:
            label = "+".join(slot.members)
            print(f"    LIVE  {label:<16} slot weight {slot.weight * 100:5.1f}%")
            for member in slot.members:
                weight = slot.member_weights.get(member, 0.0)
                print(f"          {member:<8} weight {weight * 100:5.1f}%  "
                      f"${risk_report.dollars(member):,.0f}")
    else:
        print("    No traders live today.")

    if risk_report.benched_traders:
        print("\n    BENCHED")
        for trader, reason in sorted(risk_report.benched_traders.items()):
            print(f"    🪑 {trader:<8} {reason}")

    if any(p.contributors for p in risk_report.netted_positions.values()):
        print("\n    NETTED POSITIONS")
        for symbol, pos in risk_report.netted_positions.items():
            if not pos.contributors:
                continue
            cap = "  (capped)" if pos.capped else ""
            print(f"    {symbol:<10} {pos.target_weight * 100:5.1f}% of capital"
                  f"{cap}  <- {', '.join(pos.contributors)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py", description="Meridian Capital orchestrator")
    parser.add_argument("mode", choices=["research", "paper", "live", "killswitch"])
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="override the configured universe")
    parser.add_argument("--i-understand-the-risk", action="store_true",
                        dest="risk_ack", help="required for live mode (Phase 5)")
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
        run_research(config, args.symbols)
        return 0

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
