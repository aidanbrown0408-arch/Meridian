"""Meridian Capital — orchestrator CLI.

    python main.py research      # backtest + console report (Phase 1: working)
    python main.py paper         # simulated fills        (Phase 4)
    python main.py live          # real money             (Phase 5, triple-gated)
    python main.py killswitch    # flatten + halt         (Phase 4)

Phase 1 implements the research pipeline end to end: Wong fetches, Leo
backtests with cost modeling, and the results print as a ranked table. The
other modes exist as explicit refusals rather than missing commands, so running
them tells you which phase they arrive in instead of a stack trace.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime

import pandas as pd

from agents.backtest_agent import BacktestAgent
from agents.data_agent import DataAgent
from backtester.engine import results_frame
from utils.config import load_config
from utils.logging_setup import get_logger, setup_logging

log = get_logger("orchestrator")

BANNER = r"""
   MERIDIAN CAPITAL
   multi-agent trading research desk · v1.0 · Phase 1
"""

PHASE_PENDING = {
    "paper": ("Phase 4", "PaperBroker with a persisted $5,000 virtual ledger"),
    "live": ("Phase 5", "LiveBroker stub plus the three-gate refusal logic"),
    "killswitch": ("Phase 4", "position flattening and the halt flag"),
}


def run_research(config, symbols: list[str] | None = None) -> pd.DataFrame:
    """Fetch, backtest, report. No execution, safe to run any time."""
    wong = DataAgent(config)
    market = wong.fetch_universe(symbols)

    leo = BacktestAgent(config)
    results = leo.run_all(market)
    benchmarks = leo.benchmarks(market)

    frame = results_frame(results)
    _print_report(config, market, frame, benchmarks)
    return frame


def _print_report(config, market, frame: pd.DataFrame, benchmarks: dict) -> None:
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

    if frame.empty:
        print("\n  No results produced.")
        return

    print("\n  RESULTS (ranked by Sharpe)")
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

    synthetic = [s for s, d in market.items() if d.is_synthetic]
    if synthetic:
        print(f"\n  ⚠ {', '.join(synthetic)} used synthetic data. These numbers describe"
              "\n    a random walk, not a market. Do not act on them.")

    print("\n  Phase 2 adds walk-forward validation. Until then a high Sharpe here"
          "\n  is an in-sample number and should be treated as curve-fit.")
    print("=" * 78 + "\n")


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
