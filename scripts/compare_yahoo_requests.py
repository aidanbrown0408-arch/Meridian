"""Read-only: which kind of Yahoo request has today's bar right after the close?

    python3 scripts/compare_yahoo_requests.py

Run it any time after 4:00 PM ET on a trading day (the earlier the better --
the gap showed at 8:15 PM and was gone by ~10 PM). Writes nothing.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import yfinance as yf

ET = ZoneInfo("America/New_York")
SYMBOLS = ["SPY", "AAPL", "MSFT"]


def last(df):
    if df is None or df.empty:
        return "EMPTY"
    ts = df.index[-1]
    return f"{ts.date()} close {float(df['Close'].iloc[-1]):,.2f}"


def main():
    now = datetime.now(ET)
    start = (datetime.now(timezone.utc) - timedelta(days=740)).date().isoformat()
    end = (now.date() + timedelta(days=1)).isoformat()
    print(f"\nyfinance {yf.__version__} · now {now:%Y-%m-%d %H:%M %Z}\n")
    for sym in SYMBOLS:
        t = yf.Ticker(sym)
        print(sym)
        print("  A start/end (what Wong uses) ", last(t.history(start=start, end=end, interval="1d", auto_adjust=True)))
        print("  B start only                 ", last(t.history(start=start, interval="1d", auto_adjust=True)))
        print("  C period=5d                  ", last(t.history(period="5d", interval="1d", auto_adjust=True)))
        print("  D period=2y                  ", last(t.history(period="2y", interval="1d", auto_adjust=True)))
        intr = t.history(period="1d", interval="5m")
        print("  E intraday 5m, last bar      ", f"{intr.index[-1]:%Y-%m-%d %H:%M}" if not intr.empty else "EMPTY")
        try:
            meta = t.get_history_metadata()
            print("  F quote                      ", meta.get("regularMarketPrice"), meta.get("regularMarketTime"))
        except Exception as exc:
            print("  F quote                       error:", exc)
    print("\nPaste this whole output to Claude.")


if __name__ == "__main__":
    main()
