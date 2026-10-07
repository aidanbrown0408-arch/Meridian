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


def prove_intraday(config, stocks, expected) -> int:
    """--intraday: build the newest session's bar from intraday data for every
    stock, whether or not daily history already has it, and show both side by
    side. Proves the fallback path works on this machine tonight."""
    import pandas as pd
    print(f"\nBuilding the {expected} bar from Yahoo intraday + official close:\n")
    bad = 0
    for symbol in stocks:
        try:
            bar, detail = DataAgent._session_bar_from_intraday(symbol, expected)
        except Exception as exc:
            bar, detail = None, f"error: {exc}"
        try:
            daily = DataAgent._normalize(DataAgent(config)._fetch_stock(symbol))
            dclose = (float(daily.loc[pd.Timestamp(expected), "close"])
                      if pd.Timestamp(expected) in daily.index else None)
        except Exception:
            dclose = None
        if bar is None:
            bad += 1
            print(f"  {symbol:6} FAILED ({detail})")
            continue
        b = bar.iloc[0]
        cmp = (f"daily history close {dclose:,.2f} (diff {b['close'] - dclose:+.2f})"
               if dclose is not None else "daily history: not published yet")
        print(f"  {symbol:6} O {b['open']:,.2f} H {b['high']:,.2f} L {b['low']:,.2f} "
              f"C {b['close']:,.2f} [{detail}] | {cmp}")
    print("\nIntraday path works." if not bad else f"\n{bad} failed -- paste this output to Claude.")
    return 1 if bad else 0


def main() -> int:
    config = load_config()
    wong = DataAgent(config)
    wong.use_cache = False                     # read fresh, write nothing
    david = ComplianceAgent(config)
    stocks = [s for s in config.universe if config.asset_class(s) != "crypto"]
    expected = expected_last_bar("stocks", config)
    if "--intraday" in sys.argv:
        return prove_intraday(config, stocks, expected)
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
