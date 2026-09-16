"""Dashboard view model for George's tabbed report.

The pipeline hands George plain, JSON-safe context (the same dict that is
cached to reports/dashboard_context.json). Everything the tabbed template
needs beyond that -- account cards, grouped alerts, the strategy roster,
agent standup cards, The Floor roster -- is derived here, at render time,
from that context. Deriving at render time (instead of caching the derived
shapes) means an older cached context still renders, and `paper-options`
can re-render the page without re-running the stock pipeline.

Presentation only: nothing here touches a ledger, a signal, or a risk
decision.
"""

from __future__ import annotations

import base64
from datetime import datetime
from pathlib import Path

from jinja2 import pass_context

TEAM_PHOTO_DIR = Path(__file__).resolve().parent.parent / "static" / "team" / "thumb"

# Display style per callsign -- fallback for cached contexts written before
# `strategy_styles` was added to the context.
DEFAULT_STYLES = {
    "ORBIT": "Slow trend-follower", "FLUX": "Fast trend-follower",
    "REVERT": "RSI mean-reversion", "SURGE": "Slow breakout",
    "SPARK": "Fast breakout", "ANCHOR": "Bollinger mean-reversion",
}

# The Floor. (slug, name, role, desk, description). A headshot is picked up
# automatically from static/team/thumb/<slug>.jpg; without one the card
# shows a gold monogram.
TEAM_ROSTER = [
    ("wong", "Wong", "Data Agent", "stock",
     "Sources daily OHLCV history across the universe: yfinance for equities, ccxt against "
     "Binance for crypto. When a feed fails, falls back to a seeded random walk so the pipeline "
     "never breaks, and marks the result synthetic so nothing downstream mistakes it for a "
     "market."),
    ("david", "David", "Compliance Agent", "stock",
     "Runs data-quality checks before any strategy touches the tape: multi-day gaps, zero or "
     "negative prices, stale feeds, and implausible moves that suggest an unadjusted corporate "
     "action. Holds standing authority to block a ticker for the day."),
    ("leo", "Leo", "Backtest Agent", "stock",
     "Runs every strategy and ticker combination through the vectorized engine with full cost "
     "modeling: commission and slippage charged on every position change, so a round trip pays "
     "both ways."),
    ("charles", "Charles", "Risk Agent", "stock",
     "Walk-forward validates every candidate across five folds, rejects anything breaching the "
     "drawdown limit, and caps the live roster. Sizes survivors by inverse-volatility risk "
     "parity, with correlated mean-reversion traders sharing one slot."),
    ("greg", "Greg", "Regime Agent", "stock",
     "Classifies each ticker as trending, choppy, or undecided from ADX, moving-average slope "
     "and realized-volatility percentile. Tilts allocation against mismatched traders; never "
     "benches outright."),
    ("edwin", "Edwin", "Sentiment Agent", "stock",
     "Will surface news-sentiment flags via Alpha Vantage. Strictly informational: never sizes "
     "a position, never vetoes a strategy. Not wired into the pipeline yet."),
    ("cornelius", "Cornelius", "Execution Agent", "stock",
     "Runs the executor across research, paper and live modes. Simulates fills against the "
     "persisted $5k ledger, marks positions to market, and enforces the halt conditions."),
    ("george", "George", "Reporting Agent", "stock",
     "Assembles this dashboard from every stage of the pipeline and delivers the daily standup "
     "to Slack. Flags synthetic data and empty-roster days instead of reporting nothing."),
    ("augustus", "Augustus", "Options Data Agent", "options",
     "Pulls live SPY and QQQ option chains from yfinance, keeps the nearest expirations inside "
     "the DTE window, and converts per-share quotes to per-contract premiums. Mid price, "
     "falling back to last trade when the book is empty."),
    ("theo", "Theo", "Options Risk Agent", "options",
     "Gatekeeper for the $1,000 bucket: $150 max premium per contract, $400 total at risk, two "
     "contracts at once, long calls and puts only. Halts the bucket at a 50% loss."),
    ("joseph", "Joseph", "Options Broker", "options",
     "Executes Theo-approved SPARK-calls entries against the separate options ledger, marks "
     "positions daily, closes at expiration, and honors the options killswitch."),
]

_photo_cache: dict[str, str] = {}


def _photo(slug: str) -> str:
    if slug not in _photo_cache:
        path = TEAM_PHOTO_DIR / f"{slug}.jpg"
        _photo_cache[slug] = (base64.b64encode(path.read_bytes()).decode("ascii")
                              if path.exists() else "")
    return _photo_cache[slug]


def _has_data(svg: str | None) -> bool:
    return bool(svg) and ">no data<" not in svg


def _money(x: float, decimals: int = 0) -> str:
    return f"${x:,.{decimals}f}"


def _signed(x: float, decimals: int = 0) -> str:
    sign = "+" if x >= 0 else "−"
    return f"{sign}${abs(x):,.{decimals}f}"


def _cls(x: float | None) -> str:
    if x is None:
        return ""
    return "up" if x > 0 else ("dn" if x < 0 else "")


def make_view_global(config=None):
    """Jinja global for the template: `{% set v = view() %}`. Reads the whole
    render context, so a plain `template.render(**context)` works too."""
    @pass_context
    def view(ctx) -> dict:
        return build_view(ctx.get_all(), config)
    return view


def build_view(c: dict, config=None) -> dict:
    get = lambda k, d=None: c.get(k) if c.get(k) is not None else d  # noqa: E731
    cfg = (lambda k, d: config.get(k, d)) if config is not None else (lambda k, d: d)

    ledger = get("ledger")
    options = get("options")
    live = list(get("live_traders", []))
    benched = dict(tuple(x) for x in get("benched_traders", []))
    validation = get("validation_rows", [])
    regimes = get("regime_rows", [])
    recs = get("recommendation_rows", [])
    synthetic = get("synthetic_symbols", [])
    blocks = get("compliance_rows", [])
    weights = get("capital_weights", {})
    styles = {**DEFAULT_STYLES, **get("strategy_styles", {})}
    starting = float(get("starting_capital", 0.0) or 0.0)

    # ---------------------------------------------------------------- accounts
    if ledger:
        stock = {
            "value": _money(ledger["equity"]),
            "pnl": _signed(ledger["total_pnl"]),
            "pnl_pct": f"{ledger['total_pnl_pct']:+.2f}%",
            "cls": _cls(ledger["total_pnl"]),
            "rows": [("Cash", _money(ledger["cash"])),
                     ("Open positions", str(len(ledger["positions"]))),
                     ("Trades", str(ledger["trade_count"]))],
            "halted": ledger["halted"],
            "status": "HALTED" if ledger["halted"] else "Paper",
            "foot": (f"Started at {_money(ledger['starting_capital'])} · "
                     f"{'trading ' + ', '.join(live) if live else 'no trader live today'} · "
                     f"{cfg('risk.max_portfolio_drawdown', 0.15):.0%} max drawdown cap"),
        }
    else:
        stock = {
            "value": _money(starting), "pnl": "—", "pnl_pct": "not trading yet", "cls": "",
            "rows": [("Mode", "Research only"), ("Open positions", "0"), ("Trades", "0")],
            "halted": False, "status": "Research",
            "foot": "Paper trading starts with `python main.py paper`.",
        }

    opt = None
    caps = []
    if options:
        start = float(options["starting_capital"])
        max_open = float(cfg("options.caps.max_total_open_premium", 400.0))
        max_contracts = int(cfg("options.caps.max_concurrent_contracts", 2))
        floor = options.get("halt_floor")
        open_contracts = sum(p["contracts"] for p in options["positions"])
        opt = {
            "value": _money(options["equity"], 2),
            "pnl": _signed(options["total_pnl"], 2),
            "pnl_pct": f"{options['total_pnl_pct']:+.2f}%",
            "cls": _cls(options["total_pnl"]),
            "rows": [("Cash", _money(options["cash"])),
                     ("Premium at risk", f"{_money(options['open_premium'])} / {_money(max_open)}"),
                     ("Halt floor", _money(floor) if floor is not None else "—")],
            "halted": options["halted"],
            "status": "HALTED" if options["halted"] else "Paper",
            "updated": options.get("updated_at", ""),
            "foot": (f"Started at {_money(start)} · SPARK-calls on "
                     f"{', '.join(cfg('options.underlyings', ['SPY', 'QQQ']))} · long calls/puts only, "
                     "never blended with the stock book"),
        }
        caps = [
            {"label": "Premium at risk", "used": options["open_premium"], "limit": max_open,
             "text": f"{_money(options['open_premium'])} of {_money(max_open)}"},
            {"label": "Open contracts", "used": open_contracts, "limit": max_contracts,
             "text": f"{open_contracts} of {max_contracts}"},
        ]
        if floor is not None and start > floor:
            lost = max(0.0, start - options["equity"])
            caps.append({"label": "Loss budget used", "used": lost, "limit": start - floor,
                         "text": f"{_money(lost)} of {_money(start - floor)}"})
        for cap in caps:
            cap["pct"] = round(min(100.0, 100.0 * cap["used"] / cap["limit"]), 1) if cap["limit"] else 0.0
            cap["level"] = "dn" if cap["pct"] >= 90 else ("warn" if cap["pct"] >= 60 else "ok")

    # ---------------------------------------------------------------- stat strip
    spy = next((b for b in get("benchmark_rows", []) if b["symbol"] == "SPY"), None)
    regime_set = {r["regime"] for r in regimes}
    if len(regime_set) == 1:
        regime_label = next(iter(regime_set)).capitalize()
    else:
        regime_label = "Mixed" if regime_set else "—"
    all_traders = sorted(set(styles) | {v["strategy"] for v in validation} | set(live) | set(benched))
    signals = (options or {}).get("signals", [])
    stats = [
        {"label": "Live traders", "value": f"{len(live)} / {len(all_traders)}",
         "sub": ", ".join(live) if live else "none cleared validation", "cls": ""},
        {"label": "Regime", "value": regime_label,
         "sub": " · ".join(f"{r['symbol']} {r['regime']}" for r in regimes) or "no read",
         "cls": "gld"},
        {"label": "SPY buy & hold", "value": f"{spy['cagr']:.1f}%" if spy else "n/a",
         "sub": "CAGR benchmark", "cls": _cls(spy["cagr"]) if spy else ""},
    ]

    # ---------------------------------------------------------------- alerts
    alerts = []
    if ledger and ledger["halted"]:
        alerts.append({"level": "halt", "title": "Stock account halted",
                       "body": ledger["halt_reason"], "cmd": "python main.py clear-halt"})
    if options and options["halted"]:
        alerts.append({"level": "halt", "title": "Options bucket halted",
                       "body": options["halt_reason"], "cmd": "python main.py clear-halt --options"})
    if synthetic:
        alerts.append({"level": "warn", "title": f"{', '.join(synthetic)} on synthetic data",
                       "body": "Every number touching it describes a random walk, not a market. "
                               "Don't act on it.", "cmd": ""})
    for b in blocks:
        alerts.append({"level": "warn", "title": f"{b['symbol']} blocked by David",
                       "body": b["reason"], "cmd": ""})
    groups: dict[tuple, list] = {}
    for r in recs:
        groups.setdefault((r["trigger"], r["action"], r["detail"]), []).append(r["trader"])
    for (trigger, action, detail), traders in groups.items():
        who = ", ".join(traders)
        alerts.append({
            "level": "ask",
            "title": (f"{len(traders)} traders · {trigger.replace('_', ' ')} → {action}"
                      if len(traders) > 1 else f"{who} · {trigger.replace('_', ' ')} → {action}"),
            "who": who if len(traders) > 1 else "",
            "body": detail + " Recommendation only; nothing changes without your approval.",
            "cmd": "python main.py bench --trader X --trigger ...",
        })
    options_open = [p for p in (options or {}).get("positions", [])]

    # ---------------------------------------------------------------- strategies
    by_strategy: dict[str, list] = {}
    for v in validation:
        by_strategy.setdefault(v["strategy"], []).append(v)
    regime_by_symbol = {r["symbol"]: r["regime"] for r in regimes}
    strategy_rows, strats = [], {}
    for callsign in all_traders:
        cands = by_strategy.get(callsign, [])
        if callsign in live:
            status = "live"
        elif "passed validation" in benched.get(callsign, ""):
            status = "watch"
        else:
            status = "bench"
        best = max(cands, key=lambda v: v["sharpe"]) if cands else None
        passed = sum(1 for v in cands if v["eligible"])
        row = {
            "callsign": callsign, "style": styles.get(callsign, ""), "status": status,
            "symbol": best["symbol"] if best else "—",
            "sharpe": best["sharpe"] if best else None,
            "cagr": best.get("cagr") if best else None,
            "max_drawdown": best["max_drawdown"] if best else None,
            "walkforward": f"{best['folds_passed']}/{best['folds_total']}" if best else "—",
            "symbols_passed": f"{passed}/{len(cands)}" if cands else "—",
            "position_pct": round(weights.get(callsign, 0.0) * 100, 1) if status == "live" else None,
        }
        strategy_rows.append(row)
        strats[callsign] = {
            **row,
            "regime": regime_by_symbol.get(row["symbol"], "—"),
            "reason": benched.get(callsign, ""),
            "per_symbol": [{"symbol": v["symbol"], "sharpe": v["sharpe"],
                            "folds": f"{v['folds_passed']}/{v['folds_total']}",
                            "dd": v["max_drawdown"], "eligible": v["eligible"]}
                           for v in sorted(cands, key=lambda v: v["symbol"])],
        }
    order = {"live": 0, "watch": 1, "bench": 2}
    strategy_rows.sort(key=lambda r: (order[r["status"]],
                                      -(r["sharpe"] if r["sharpe"] is not None else -99)))

    # ---------------------------------------------------------------- agents
    data_rows = get("data_rows", [])
    eligible = sum(1 for v in validation if v["eligible"])
    n_results = get("results_count", len(validation))
    stock_desk = [
        _card("Wong", "Data",
              (f"{', '.join(r['symbol'] for r in data_rows)} pulled. {', '.join(synthetic)} on "
               "synthetic fallback, not trusted." if synthetic else
               f"Pulled {', '.join(r['symbol'] for r in data_rows) or 'the universe'}. All "
               "sources live."),
              "Synthetic fallback in use" if synthetic else "All live",
              "warn" if synthetic else "ok"),
        _card("David", "Compliance",
              "; ".join(f"{b['symbol']} blocked ({b['reason']})" for b in blocks)
              or "Every symbol cleared the data-quality checks.",
              f"{len(blocks)} blocked" if blocks else "All clear", "warn" if blocks else "ok"),
        _card("Leo", "Backtest",
              f"Ran {n_results} strategy/ticker combinations. {eligible} cleared walk-forward.",
              f"{eligible}/{len(validation)} eligible", "ok"),
        _card("Charles", "Risk",
              (", ".join(f"{t} {weights.get(t, 0.0):.0%}" for t in live) if live else
               "No strategy cleared validation today. Nothing is sized."),
              f"{len(live)} live" if live else "Nothing live", "ok" if live else "warn"),
        _card("Greg", "Regime",
              ", ".join(f"{r['symbol']} {r['regime']} (ADX {r['adx']})" for r in regimes)
              or "No regime read.", "Read complete" if regimes else "No read", "ok"),
        _card("Edwin", "Sentiment", "Not wired into the pipeline yet. No sentiment reads.",
              "Offline", "off"),
        _card("Cornelius", "Execution",
              (f"{len(ledger['positions'])} open position(s), {ledger['trade_count']} trade(s) "
               f"since inception. Equity {_money(ledger['equity'])}." if ledger else
               "Research mode. No execution today."),
              ("Halted" if ledger and ledger["halted"] else
               "Paper trading" if ledger else "Research only"),
              "bad" if ledger and ledger["halted"] else "ok"),
        _card("George", "Reporting",
              (f"Report assembled. {len(recs)} lifecycle recommendation(s) waiting on you."
               if recs else "Report assembled."),
              f"{len(recs)} awaiting review" if recs else "Complete", "warn" if recs else "ok"),
    ]
    options_desk = []
    if options:
        chains = options.get("chains", [])
        syn_chains = [ch["underlying"] for ch in chains if ch.get("synthetic")]
        options_desk = [
            _card("Augustus", "Options Data",
                  ("; ".join(f"{ch['underlying']} {ch['expirations']} expirations, "
                             f"{ch['calls'] + ch['puts']:,} contracts ({ch['source']})"
                             for ch in chains) or "No chains pulled on the last run."),
                  "Synthetic chain" if syn_chains else ("Chains live" if chains else "No chains"),
                  "warn" if syn_chains or not chains else "ok"),
            _card("Theo", "Options Risk",
                  " · ".join(cap["label"] + " " + cap["text"] for cap in caps),
                  "Bucket halted" if options["halted"] else "Within caps",
                  "bad" if options["halted"] else "ok"),
            _card("Joseph", "Options Broker",
                  (f"{len(options_open)} open contract position(s), {options['trade_count']} "
                   f"trade(s). SPARK trigger: "
                   + (", ".join(f"{s['underlying']} {'LONG' if s['signal_long'] else 'flat'}"
                                for s in signals) or "not checked") + "."),
                  ("Halted" if options["halted"] else
                   "Signal live" if any(s["signal_long"] for s in signals) else "Waiting on signal"),
                  "bad" if options["halted"] else "ok"),
        ]

    # ---------------------------------------------------------------- floor
    status_by_name = {a["name"]: a for a in stock_desk + options_desk}
    team = {"stock": [], "options": []}
    for slug, name, role, desk, desc in TEAM_ROSTER:
        a = status_by_name.get(name)
        team[desk].append({
            "name": name, "role": role, "desc": desc, "photo": _photo(slug),
            "initial": name[0],
            "status": a["status_text"] if a else "No run yet",
            "level": a["level"] if a else "off",
        })

    return {
        "hdr_date": datetime.now().strftime("%b %d, %Y · %H:%M").replace(" 0", " "),
        "stock": stock, "opt": opt, "stats": stats, "alerts": alerts, "caps": caps,
        "signals": signals,
        "charts": {k: get(k, "") if _has_data(get(k)) else ""
                   for k in ("equity_chart", "drawdown_chart", "trader_bar_chart")},
        "options_chart": (options or {}).get("equity_chart", ""),
        "strategy_rows": strategy_rows, "strats": strats,
        "stock_desk": stock_desk, "options_desk": options_desk,
        "team": team,
        "counts": {"alerts": len(alerts),
                   "positions": (len(ledger["positions"]) if ledger else 0) + len(options_open)},
    }


def _card(name, role, message, status_text, level) -> dict:
    return {"name": name, "role": role, "message": message,
            "status_text": status_text, "level": level}
