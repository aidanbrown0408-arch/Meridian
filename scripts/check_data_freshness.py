"""Read-only check: does Wong get the newest completed session right now?

    python3 scripts/check_data_freshness.py

Fetches every stock ticker exactly the way the paper run does, but with the
cache off and NOTHING written -- no ledger, no cache, no dashboard, no Slack.
Safe to run any time.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.compliance_agent import ComplianceAgent  # noqa: E402
from agents.data_agent import DataAgent  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.market_calendar import expected_last_bar  # noqa: E402


def main() -> int:
    config = load_config()
    wong = DataAgent(config)
    wong.use_cache = False                     # read fresh, write nothing
    david = ComplianceAgent(config)
    stocks = [s for s in config.universe if config.asset_class(s) != "crypto"]
    expected = expected_last_bar("stocks", config)
    print(f"\nNewest completed session should be: {expected}\n")
    bad = 0
    for symbol in stocks:
        md = wong.fetch(symbol)
        last = md.bars.index[-1].date() if not md.bars.empty else None
        rep = david.check(md)
        built = any("built from Yahoo intraday" in n for n in md.notes)
        status = "SYNTHETIC -- Yahoo unreachable" if md.is_synthetic else \
            ("BLOCKED: " + rep.reason) if rep.blocked else (
            "ok (built from intraday + official close)" if built else "ok (daily history)")
        close = float(md.bars["close"].iloc[-1]) if last else float("nan")
        print(f"  {symbol:6} source={md.data_source:9} last={last}  close={close:,.2f}  {status}")
        bad += rep.blocked or md.is_synthetic
    print("\nAll current." if not bad else f"\n{bad} ticker(s) not current -- paste this output to Claude.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
