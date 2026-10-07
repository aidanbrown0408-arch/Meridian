"""Chart data for the dashboard's ACTIVE TRADES tab.

For every position the desk holds (and every one it has closed) this builds
one JSON-safe "card": daily candles for the symbol, the exact indicator the
trader behind the position trades on (moving averages, Donchian channel,
Bollinger bands, RSI), where the strategy signalled, where the ledger
actually filled, and the position's P&L replayed bar by bar from the fills.

Accuracy rules this module holds itself to:
  * Indicators use the strategies' own parameters (from config) and the same
    formulas as `generate_signals`. `tests/test_trade_charts.py` rebuilds every
    strategy's signal from these overlay series and checks it matches
    `generate_signals` bar for bar, so a chart can't quietly drift from what
    the trader actually decides on.
  * Fills come from the ledgers, never from signals. A fill is pinned to the
    price bar it was priced off (`bar_date`, or for older trades the bar whose
    close matches the fill price) and the card says so when that differs from
    the run date.
  * Nothing is guessed silently: inferred trader attribution, stale price
    data, synthetic data, a missing option quote -- each shows as a note on
    the card.

Read-only: nothing here writes a ledger or changes a decision.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from strategies.anchor import AnchorStrategy
from strategies.base import Strategy, build_strategies
from strategies.orbit import OrbitStrategy
from strategies.revert import RevertStrategy, rsi
from strategies.surge import SurgeStrategy
from utils.logging_setup import get_logger
from utils.market_calendar import expected_last_bar as _expected_last_bar

log = get_logger("trade_charts", agent="George")

MIN_BARS_SHOWN = 130      # ~6 months of sessions in the default "Trade" view
PAD_BEFORE_ENTRY = 60     # sessions of context before the first fill
MAX_HISTORY_BARS = 756    # ~3 years shipped per card so 1Y / All / weekly views have data
MAX_CLOSED_CARDS = 12     # most recent closed trades kept on the page
PRICE_MATCH_TOL = 5e-4    # 0.05%: a fill "matches" a bar's close within this

# Per-card trader colours, assigned in this fixed order (the house gold first).
# Validated as a set on the navy card surface: worst all-pairs CVD ΔE 13.4,
# normal-vision ΔE 24.2. A 4th+ trader on one symbol repeats a hue, so every
# overlay also carries its callsign in the legend and on its price label.
TRADER_COLORS = ["#D4AF6A", "#3987e5", "#c74a9a"]
GAIN, LOSS = "#7FBF7F", "#C97A7A"


# ------------------------------------------------------------------ helpers

def _d(ts) -> str:
    return pd.Timestamp(ts).date().isoformat()


def _round(x, nd=4):
    if x is None:
        return None
    x = float(x)
    if not np.isfinite(x):
        return None
    return round(x, nd)


def _line(series: pd.Series, nd: int = 4) -> list[dict]:
    """[{time, value}] with warmup NaNs dropped (lightweight-charts wants gaps
    left out, not nulls)."""
    out = []
    for ts, v in series.items():
        if v is None or not np.isfinite(v):
            continue
        out.append({"time": _d(ts), "value": round(float(v), nd)})
    return out


def _money(x: float) -> str:
    return f"${x:,.2f}"


def _signed(x: float) -> str:
    return f"{'+' if x >= 0 else '−'}${abs(x):,.2f}"


def _cls(x: float | None) -> str:
    if x is None:
        return ""
    return "up" if x > 0 else ("dn" if x < 0 else "")


def _pretty_date(iso: str) -> str:
    try:
        return date.fromisoformat(iso).strftime("%a %b %-d")
    except Exception:
        return iso


def _now_local(config, now: datetime | None = None) -> datetime:
    tz = (config.get("system.timezone", "America/New_York") if config is not None
          else "America/New_York") or "America/New_York"
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now.astimezone(ZoneInfo(tz))


def expected_last_bar(asset_class: str, config=None, now: datetime | None = None) -> date:
    """Same rule David blocks on (utils/market_calendar.py), holidays included."""
    return _expected_last_bar(asset_class, config, now)


def data_status(md, config=None, now: datetime | None = None) -> dict:
    last = md.bars.index[-1].date()
    expected = expected_last_bar(md.asset_class, config, now)
    stale = last < expected
    note = ""
    if stale:
        note = (f"Price data ends {_pretty_date(last.isoformat())}; the "
                f"{_pretty_date(expected.isoformat())} "
                f"{'candle' if md.asset_class == 'crypto' else 'close'} isn't in this "
                f"run's data — David blocks trading on it until it catches up.")
    built = next((n for n in (md.notes or []) if "built from Yahoo intraday" in n), "")
    return {"through": last.isoformat(), "expected": expected.isoformat(),
            "stale": bool(stale), "stale_note": note, "built_note": built,
            "source": md.data_source, "synthetic": bool(md.is_synthetic)}


# ------------------------------------------------------------------ indicators

def indicator_frame(strategy: Strategy, bars: pd.DataFrame) -> dict:
    """The series a strategy decides on, computed exactly as its
    `generate_signals` does. Returns {"series": {name: Series}, "kind": str}."""
    close = bars["close"]
    p = strategy.params
    if isinstance(strategy, OrbitStrategy):
        f, s = int(p["fast"]), int(p["slow"])
        return {"kind": "ma_cross", "series": {
            "fast": close.rolling(f, min_periods=f).mean(),
            "slow": close.rolling(s, min_periods=s).mean()}}
    if isinstance(strategy, SurgeStrategy):
        en, ex = int(p["entry_window"]), int(p["exit_window"])
        high = bars["high"].fillna(close)
        low = bars["low"].fillna(close)
        return {"kind": "donchian", "series": {
            "upper": high.rolling(en, min_periods=en).max().shift(1),
            "lower": low.rolling(ex, min_periods=ex).min().shift(1)}}
    if isinstance(strategy, AnchorStrategy):
        w, k = int(p["window"]), float(p["num_std"])
        mid = close.rolling(w, min_periods=w).mean()
        std = close.rolling(w, min_periods=w).std(ddof=0)
        return {"kind": "bollinger", "series": {
            "middle": mid, "upper": mid + k * std, "lower": mid - k * std}}
    if isinstance(strategy, RevertStrategy):
        return {"kind": "rsi", "series": {"rsi": rsi(close, int(p["period"]))}}
    return {"kind": "unknown", "series": {}}


def signal_from_indicators(strategy: Strategy, bars: pd.DataFrame, ind: dict) -> pd.Series:
    """Rebuild the strategy's entry/exit decision from the charted series.
    Used by the tests to prove the overlays are the real decision inputs."""
    close = bars["close"]
    s = ind["series"]
    p = strategy.params
    if ind["kind"] == "ma_cross":
        return (s["fast"] > s["slow"]).astype("float64").where(s["slow"].notna(), 0.0)
    if ind["kind"] == "donchian":
        return Strategy._hold(close > s["upper"], close < s["lower"])
    if ind["kind"] == "bollinger":
        return Strategy._hold(close < s["lower"], close >= s["middle"])
    if ind["kind"] == "rsi":
        return Strategy._hold(s["rsi"] < float(p["entry"]), s["rsi"] > float(p["exit"]))
    raise ValueError(f"no indicator rebuild for {strategy.callsign}")


def put_indicator_frame(strategy: Strategy, bars: pd.DataFrame) -> dict:
    """The put-side Donchian breakdown in strategies/options_strategy.py:
    active below the prior entry_window low, released above the prior
    exit_window high."""
    close = bars["close"]
    en, ex = int(strategy.params["entry_window"]), int(strategy.params["exit_window"])
    high = bars["high"].fillna(close)
    low = bars["low"].fillna(close)
    return {"kind": "donchian_put", "series": {
        "lower": low.rolling(en, min_periods=en).min().shift(1),
        "upper": high.rolling(ex, min_periods=ex).max().shift(1)}}


def _rules(strategy: Strategy, ind: dict, bars: pd.DataFrame, held_long: bool = True) -> dict:
    """Plain-English entry/exit rule plus where today's bar stands against the
    exit, e.g. "Exits on a close below $327.13 (2.0% under the last close)"."""
    close = float(bars["close"].iloc[-1])
    p = strategy.params
    s = ind["series"]
    out = {"entry": "", "exit": "", "status": "", "distance_pct": None}
    if ind["kind"] == "ma_cross":
        f, sl = int(p["fast"]), int(p["slow"])
        fast, slow = float(s["fast"].iloc[-1]), float(s["slow"].iloc[-1])
        gap = (fast / slow - 1) * 100 if slow else None
        out.update(entry=f"Long while the {f}-day average is above the {sl}-day.",
                   exit=f"Exits when the {f}-day average closes at or below the {sl}-day.",
                   status=(f"{f}-day {_money(fast)} vs {sl}-day {_money(slow)}: "
                           f"{abs(gap):.2f}% {'above' if gap >= 0 else 'below'}"
                           if gap is not None else ""),
                   distance_pct=_round(gap, 2))
    elif ind["kind"] == "donchian":
        en, ex = int(p["entry_window"]), int(p["exit_window"])
        low = bars["low"].fillna(bars["close"])
        # Tomorrow's exit line is the lowest low of the last `ex` sessions,
        # *including* today -- the same shift(1) window, one bar on.
        nxt = float(low.rolling(ex, min_periods=ex).min().iloc[-1])
        dist = (close / nxt - 1) * 100 if nxt else None
        out.update(entry=f"Enters on a close above the prior {en}-day high.",
                   exit=f"Exits on a close below the prior {ex}-day low.",
                   status=(f"Next exit line {_money(nxt)} — "
                           f"{abs(dist):.2f}% {'below' if dist >= 0 else 'above'} the last close"
                           if dist is not None else ""),
                   distance_pct=_round(dist, 2))
    elif ind["kind"] == "bollinger":
        w, k = int(p["window"]), float(p["num_std"])
        mid = float(s["middle"].iloc[-1])
        dist = (mid / close - 1) * 100 if close else None
        out.update(entry=f"Buys a close below the lower {w}-day, {k:g}σ Bollinger band.",
                   exit=f"Takes profit at a close at or above the {w}-day average.",
                   status=(f"Middle band {_money(mid)} — {abs(dist):.2f}% "
                           f"{'above' if dist >= 0 else 'below'} the last close"
                           if dist is not None else ""),
                   distance_pct=_round(dist, 2))
    elif ind["kind"] == "rsi":
        val = float(s["rsi"].iloc[-1])
        out.update(entry=f"Buys when RSI({int(p['period'])}) drops below {p['entry']:g}.",
                   exit=f"Exits when RSI rises above {p['exit']:g}.",
                   status=f"RSI now {val:.1f} (exit above {p['exit']:g})")
    elif ind["kind"] == "donchian_put":
        en, ex = int(p["entry_window"]), int(p["exit_window"])
        high = bars["high"].fillna(bars["close"])
        nxt = float(high.rolling(ex, min_periods=ex).max().iloc[-1])
        dist = (nxt / close - 1) * 100 if close else None
        out.update(entry=f"Put trigger: a close below the prior {en}-day low.",
                   exit=f"Breakdown releases on a close above the prior {ex}-day high.",
                   status=(f"Release line {_money(nxt)} — {abs(dist):.2f}% "
                           f"{'above' if dist >= 0 else 'below'} the last close"
                           if dist is not None else ""),
                   distance_pct=_round(dist, 2))
    return out


def _overlays(callsign: str, ind: dict, color: str, start) -> tuple[list, list]:
    """Price-pane overlays and separate indicator panes for one trader."""
    s = {k: v.loc[start:] for k, v in ind["series"].items()}
    kind = ind["kind"]
    if kind == "ma_cross":
        return ([{"label": f"{callsign} fast MA", "color": color, "dash": False,
                  "points": _line(s["fast"])},
                 {"label": f"{callsign} slow MA", "color": color, "dash": True,
                  "points": _line(s["slow"])}], [])
    if kind in ("donchian", "donchian_put"):
        hi = "entry line" if kind == "donchian" else "release line"
        lo = "exit line" if kind == "donchian" else "trigger line"
        return ([{"label": f"{callsign} {hi}", "color": color, "dash": False,
                  "points": _line(s["upper"]), "step": True},
                 {"label": f"{callsign} {lo}", "color": color, "dash": True,
                  "points": _line(s["lower"]), "step": True}], [])
    if kind == "bollinger":
        return ([{"label": f"{callsign} upper band", "color": color, "dash": True,
                  "points": _line(s["upper"])},
                 {"label": f"{callsign} middle (exit)", "color": color, "dash": False,
                  "points": _line(s["middle"])},
                 {"label": f"{callsign} lower (entry)", "color": color, "dash": True,
                  "points": _line(s["lower"])}], [])
    if kind == "rsi":
        return ([], [{"label": f"{callsign} RSI", "color": color,
                      "points": _line(s["rsi"], 2), "min": 0, "max": 100,
                      "levels": []}])
    return [], []


# ------------------------------------------------------------------ fills

def _bar_for_fill(bars: pd.DataFrame, run_date: str, price: float,
                  recorded: str = "") -> tuple[str, str]:
    """(bar date, note). Recorded bar_date wins. Otherwise look back from the
    run date for the bar whose close the fill was priced at."""
    dates = [_d(t) for t in bars.index]
    if recorded and recorded in dates:
        note = ("" if recorded == run_date else
                f"run on {_pretty_date(run_date)}, priced off the "
                f"{_pretty_date(recorded)} close")
        return recorded, note
    eligible = [i for i, d in enumerate(dates) if d <= run_date]
    for i in reversed(eligible[-6:]):
        c = float(bars["close"].iloc[i])
        if c and abs(c / price - 1) <= PRICE_MATCH_TOL:
            d = dates[i]
            note = ("" if d == run_date else
                    f"run on {_pretty_date(run_date)}, filled at the "
                    f"{_pretty_date(d)} close")
            return d, note
    if eligible:
        d = dates[eligible[-1]]
        return d, (f"fill price {_money(price)} doesn't match a close in the current "
                   "price history (history may have been dividend-adjusted since); "
                   f"pinned to {_pretty_date(d)}")
    return dates[0], "fill predates the charted history"


def _episodes(trades: list, min_dollars: float) -> list[list]:
    """Split one symbol's trades into flat-to-flat episodes."""
    episodes, cur, shares = [], [], 0.0
    for t in trades:
        cur.append(t)
        shares += t.shares if t.side == "buy" else -t.shares
        if t.side == "sell" and shares * t.price < min_dollars:
            episodes.append(cur)
            cur, shares = [], 0.0
    if cur:
        episodes.append(cur)
    return episodes


def _window_start(bars: pd.DataFrame, first_bar: str, last_bar: str | None = None):
    """(start, end, focus_from, focus_to). start..end is everything the card
    ships (up to MAX_HISTORY_BARS); focus_from..focus_to is the trade-centred
    window the chart opens on (the "Trade" range button)."""
    idx = list(bars.index)
    dates = [_d(t) for t in idx]
    i = dates.index(first_bar) if first_bar in dates else 0
    j = dates.index(last_bar) if last_bar and last_bar in dates else len(dates) - 1
    focus = max(0, min(i - PAD_BEFORE_ENTRY, j - MIN_BARS_SHOWN + 1))
    end = min(len(dates) - 1, j + (15 if last_bar else 0))
    start = min(focus, max(0, end - MAX_HISTORY_BARS + 1))
    return idx[start], idx[end], dates[focus], dates[end]


def _pnl_stats(pnl: list[dict]) -> dict | None:
    """Current / best / worst / giveback / last change from a P&L series."""
    if not pnl:
        return None
    vals = [p["value"] for p in pnl]
    hi = max(range(len(vals)), key=lambda k: vals[k])
    lo = min(range(len(vals)), key=lambda k: vals[k])
    cur = pnl[-1]
    out = {"current": cur["value"], "current_pct": cur.get("pct"),
           "peak": vals[hi], "peak_time": pnl[hi]["time"], "peak_pct": pnl[hi].get("pct"),
           "trough": vals[lo], "trough_time": pnl[lo]["time"], "trough_pct": pnl[lo].get("pct"),
           "giveback": round(vals[hi] - cur["value"], 2) if vals[hi] > 0 else 0.0,
           "change": None, "change_time": None, "points": len(pnl)}
    if len(pnl) > 1:
        out["change"] = round(vals[-1] - vals[-2], 2)
        out["change_time"] = pnl[-2]["time"]
    return out


def _candles(bars: pd.DataFrame) -> list[dict]:
    out = []
    for ts, r in bars.iterrows():
        o, h, l, c = (float(r.get("open", r["close"])), float(r.get("high", r["close"])),
                      float(r.get("low", r["close"])), float(r["close"]))
        if not all(np.isfinite(x) for x in (o, h, l, c)):
            continue
        out.append({"time": _d(ts), "open": round(o, 4), "high": round(h, 4),
                    "low": round(l, 4), "close": round(c, 4)})
    return out


def _signal_markers(callsign: str, sig: pd.Series, start, end, color: str,
                    bearish: bool = False) -> list[dict]:
    sig = sig.fillna(0.0)
    prev = sig.shift(1).fillna(0.0)
    out = []
    for ts in sig.loc[start:end].index:
        a, b = prev.loc[ts], sig.loc[ts]
        if a <= 0 < b:
            out.append({"time": _d(ts), "position": "aboveBar" if bearish else "belowBar",
                        "shape": "circle", "color": color, "size": 0.6,
                        "text": f"{callsign} {'put trigger' if bearish else 'entry signal'}"})
        elif a > 0 >= b:
            out.append({"time": _d(ts), "position": "belowBar" if bearish else "aboveBar",
                        "shape": "circle", "color": color, "size": 0.6,
                        "text": f"{callsign} {'release' if bearish else 'exit signal'}"})
    return out


# ------------------------------------------------------------------ stock cards

class _Ctx:
    def __init__(self, config, market, now=None):
        self.config = config
        self.market = market
        self.now = now
        try:
            self.strategies = build_strategies(config)
        except Exception as exc:  # pragma: no cover - display only
            log.warning("Strategies unavailable for chart overlays (%s)", exc)
            self.strategies = {}
        self.colors: dict[str, str] = {}

    def color(self, callsign: str) -> str:
        if callsign not in self.colors:
            self.colors[callsign] = TRADER_COLORS[len(self.colors) % len(TRADER_COLORS)]
        return self.colors[callsign]


def _trader_blocks(ctx: _Ctx, symbol: str, traders: list[str], bars: pd.DataFrame,
                   start, end, attribution: str, put_side: set | None = None):
    """Overlays, panes, signal markers and rule text for each trader."""
    overlays, panes, markers, blocks = [], [], [], []
    put_side = put_side or set()
    for callsign in traders:
        strat = ctx.strategies.get(callsign)
        if strat is None:
            blocks.append({"callsign": callsign, "style": "", "attribution": attribution,
                           "signal_now": None, "entry": "", "exit": "",
                           "status": "Strategy not enabled in config — no overlay.",
                           "distance_pct": None, "color": ctx.color(callsign)})
            continue
        color = ctx.color(callsign)
        bearish = callsign in put_side
        if bearish:
            ind = put_indicator_frame(strat, bars)
            from strategies.options_strategy import _donchian_breakdown
            sig = _donchian_breakdown(bars, int(strat.params["entry_window"]),
                                      int(strat.params["exit_window"]))
            label = f"{callsign}-puts"
        else:
            ind = indicator_frame(strat, bars)
            sig = strat.generate_signals(bars) if len(bars) >= strat.warmup else \
                pd.Series(0.0, index=bars.index)
            label = callsign
        ov, pn = _overlays(label, ind, color, start)
        for o in ov:
            o["points"] = [pt for pt in o["points"] if pt["time"] <= _d(end)]
        for p in pn:
            p["points"] = [pt for pt in p["points"] if pt["time"] <= _d(end)]
            if ind["kind"] == "rsi":
                p["levels"] = [
                    {"value": float(strat.params["entry"]), "label": f"entry < {strat.params['entry']:g}"},
                    {"value": float(strat.params["exit"]), "label": f"exit > {strat.params['exit']:g}"}]
        overlays += ov
        panes += pn
        markers += _signal_markers(label, sig, start, end, color, bearish=bearish)
        rules = _rules(strat, ind, bars.loc[:end])
        blocks.append({"callsign": label, "style": strat.style + (" (put mirror)" if bearish else ""),
                       "attribution": attribution,
                       "signal_now": ("active" if float(sig.loc[:end].iloc[-1]) > 0 else "flat"),
                       "color": color, **rules})
    return overlays, panes, markers, blocks


def _stock_card(ctx: _Ctx, symbol: str, trades: list, position, open_: bool,
                inferred: list[str], regime: str | None, idx: int) -> dict | None:
    md = ctx.market.get(symbol)
    if md is None or md.bars.empty:
        return None
    ctx.colors = {}
    bars = md.bars
    fills = []
    for t in trades:
        bar_date, note = _bar_for_fill(bars, str(t.date)[:10], float(t.price),
                                       getattr(t, "bar_date", "") or "")
        fills.append((t, bar_date, note))

    first_bar = fills[0][1]
    last_bar = None if open_ else fills[-1][1]
    start, end, focus_from, focus_to = _window_start(bars, first_bar, last_bar)
    view = bars.loc[start:end]

    # Who the position is held for.
    recorded = []
    for t in trades:
        for c in getattr(t, "traders", []) or []:
            if c not in recorded:
                recorded.append(c)
    if open_ and position is not None and getattr(position, "traders", None):
        recorded = list(position.traders)
    if recorded:
        traders, attribution = recorded, "recorded"
    elif open_ and inferred:
        traders, attribution = inferred, "inferred"
    else:
        traders, attribution = [], "unknown"

    overlays, panes, sig_markers, blocks = _trader_blocks(
        ctx, symbol, traders, bars.loc[:end], start, end, attribution)

    # Fill markers + P&L replay (cash out vs. market value, costs included).
    markers = list(sig_markers)
    notes = []
    flows: dict[str, list] = {}
    for t, bar_date, note in fills:
        buy = t.side == "buy"
        markers.append({"time": bar_date, "position": "belowBar" if buy else "aboveBar",
                        "shape": "arrowUp" if buy else "arrowDown",
                        "color": GAIN if buy else LOSS, "size": 1.4,
                        "text": f"{'BUY' if buy else 'SELL'} {t.shares:.4g} @ {_money(t.price)}"})
        if note:
            notes.append(f"{'Buy' if buy else 'Sell'} {_pretty_date(str(t.date)[:10])}: {note}.")
        flows.setdefault(bar_date, []).append(t)

    # P&L replay: market value minus net cash put in (costs included). The %
    # is against the most capital the episode ever had deployed (cost basis of
    # shares held, at its peak), so a trim doesn't inflate the percentage.
    pnl, shares, invested, basis, peak_basis = [], 0.0, 0.0, 0.0, 0.0
    for ts, row in bars.loc[first_bar:(last_bar or bars.index[-1])].iterrows():
        d = _d(ts)
        for t in flows.get(d, []):
            if t.side == "buy":
                basis += t.shares * t.price + t.cost
                shares += t.shares
                invested += t.shares * t.price + t.cost
            else:
                if shares > 0:
                    basis -= basis * min(1.0, t.shares / shares)
                shares -= t.shares
                invested -= t.shares * t.price - t.cost
            peak_basis = max(peak_basis, basis)
        value = shares * float(row["close"]) - invested
        pnl.append({"time": d, "value": round(value, 2),
                    "pct": round(value / peak_basis * 100, 2) if peak_basis else None})
    # A closed episode's last bar is fully realized: shares ~0.
    markers.sort(key=lambda m: m["time"])

    last_close = float(bars["close"].iloc[-1])
    stats, headline = [], {}
    if open_ and position is not None:
        unreal = position.unrealized_pnl(last_close)
        basis = position.shares * position.entry_price
        net = pnl[-1]["value"] if pnl else unreal
        headline = {"label": "Unrealized P&L", "value": _signed(unreal),
                    "pct": f"{unreal / basis * 100:+.2f}%" if basis else "",
                    "cls": _cls(unreal), "raw": round(unreal, 2)}
        stats = [("Shares", f"{position.shares:.4f}"),
                 ("Avg entry", _money(position.entry_price)),
                 (f"Close {_pretty_date(_d(bars.index[-1]))}", _money(last_close)),
                 ("Market value", _money(position.market_value(last_close))),
                 ("After costs", _signed(net)),
                 ("Opened", _pretty_date(first_bar))]
        price_lines = [{"price": round(position.entry_price, 4), "label": "avg entry",
                        "color": "#E8E3D3", "dash": True}]
    else:
        realized = pnl[-1]["value"] if pnl else 0.0
        bought = sum(t.shares * t.price for t in trades if t.side == "buy")
        headline = {"label": "Realized P&L", "value": _signed(realized),
                    "pct": f"{realized / bought * 100:+.2f}%" if bought else "",
                    "cls": _cls(realized), "raw": round(realized, 2)}
        stats = [("Opened", _pretty_date(first_bar)), ("Closed", _pretty_date(last_bar)),
                 ("Fills", str(len(trades))),
                 ("Costs paid", _money(sum(t.cost for t in trades)))]
        price_lines = []

    if attribution == "inferred":
        notes.insert(0, f"Opened before the ledger recorded which trader a fill was for; "
                        f"attributed to today's contributor{'s' if len(traders) > 1 else ''} "
                        f"on {symbol} ({', '.join(traders)}).")
    elif attribution == "unknown":
        notes.insert(0, "No trader attribution on record for this trade, so no strategy "
                        "overlay is drawn.")
    sig_entries = [m["time"] for m in sig_markers if m["text"].endswith("entry signal")
                   and m["time"] <= first_bar]
    all_dates = [_d(t) for t in bars.index]
    if sig_entries and first_bar in all_dates and (
            all_dates.index(first_bar) - all_dates.index(sig_entries[-1])) > 2:
        notes.append(f"The entry signal fired {_pretty_date(sig_entries[-1])}, before the "
                     f"buy on {_pretty_date(first_bar)}: Cornelius only buys for traders on "
                     "the day's live roster, so a fill can trail the signal (e.g. until the "
                     "trader clears walk-forward validation).")
    for b in blocks:
        if open_ and b["signal_now"] == "flat":
            notes.append(f"{b['callsign']}'s signal is flat as of the last close — "
                         "Cornelius trims or exits on the next run.")
    ds = data_status(md, ctx.config, ctx.now)
    if ds["stale"] and open_:
        notes.append(ds["stale_note"])
    if ds["built_note"] and open_:
        notes.append(f"Latest bar: {ds['built_note']}.")
    if ds["synthetic"]:
        notes.insert(0, "SYNTHETIC DATA — this chart is a random walk, not the market.")

    return {
        "id": f"stk-{symbol.replace('/', '-')}-{'open' if open_ else idx}",
        "book": "stock", "status": "open" if open_ else "closed",
        "symbol": symbol, "title": symbol,
        "subtitle": ("Stock desk · Cornelius" if md.asset_class != "crypto"
                     else "Crypto · Cornelius"),
        "regime": regime or "", "traders": blocks, "attribution": attribution,
        "data": ds, "headline": headline, "stats": [{"label": k, "value": v} for k, v in stats],
        "candles": _candles(view), "overlays": overlays, "panes": panes,
        "markers": markers, "price_lines": price_lines,
        "pnl": pnl, "pnl_label": "Position P&L incl. costs",
        "pnl_basis": "% of peak capital deployed",
        "pnl_stats": _pnl_stats(pnl), "focus": {"from": focus_from, "to": focus_to},
        "notes": notes, "sort_date": last_bar or first_bar,
    }


def build_stock_charts(config, market: dict, ledger, contributors: dict | None = None,
                       regimes: dict | None = None, now: datetime | None = None) -> dict:
    """Cards for Cornelius's book. `contributors` (symbol -> [callsign]) is
    today's netted contributors, used only to attribute positions opened
    before the ledger recorded traders."""
    if ledger is None:
        return {"open": [], "closed": []}
    ctx = _Ctx(config, market, now)
    min_dollars = float(config.get("execution.min_trade_dollars", 25.0)) if config else 25.0
    contributors = contributors or {}
    regimes = regimes or {}
    by_symbol: dict[str, list] = {}
    for t in ledger.trades:
        by_symbol.setdefault(t.symbol, []).append(t)

    open_cards, closed_cards = [], []
    for symbol, trades in by_symbol.items():
        eps = _episodes(trades, min_dollars)
        for i, ep in enumerate(eps):
            is_open = (i == len(eps) - 1) and symbol in ledger.positions
            try:
                card = _stock_card(ctx, symbol, ep, ledger.positions.get(symbol) if is_open else None,
                                   is_open, list(contributors.get(symbol, [])),
                                   regimes.get(symbol), i)
            except Exception as exc:  # never break the dashboard over one chart
                log.warning("Chart for %s failed (%s) -- skipped.", symbol, exc)
                card = None
            if card is None:
                continue
            (open_cards if is_open else closed_cards).append(card)
    # Positions with no trade history on record (shouldn't happen) still get a row elsewhere.
    closed_cards.sort(key=lambda c: c["sort_date"], reverse=True)
    return {"open": open_cards, "closed": closed_cards[:MAX_CLOSED_CARDS]}


# ------------------------------------------------------------------ options cards

def _trigger_of(reason: str, known) -> tuple[str | None, bool]:
    """('SPARK', is_put) from a proposal reason like 'SPARK-puts signals a
    breakdown on SPY' / 'FLUX signals long on QQQ'."""
    head = (reason or "").split(" ")[0]
    put = head.endswith("-puts")
    base = head[:-5] if put else head
    return (base if base in known else None), put


def _option_card(ctx: _Ctx, key: str, pos_like: dict, trades: list, open_: bool,
                 mark: float | None, regime: str | None, idx: int) -> dict | None:
    md = ctx.market.get(pos_like["underlying"])
    if md is None or md.bars.empty:
        return None
    ctx.colors = {}
    bars = md.bars
    open_t = [t for t in trades if t.side == "open"]
    close_t = [t for t in trades if t.side != "open"]
    first = _bar_for_fill(bars, str(open_t[0].date)[:10] if open_t else _d(bars.index[-1]),
                          float("nan"))[0] if open_t else _d(bars.index[-1])
    last = None
    if not open_ and close_t:
        last = _bar_for_fill(bars, str(close_t[-1].date)[:10], float("nan"))[0]
    start, end, focus_from, focus_to = _window_start(bars, first, last)
    view = bars.loc[start:end]

    trigger, is_put = _trigger_of(pos_like.get("reason", "") or
                                  (open_t[0].reason if open_t else ""), ctx.strategies)
    traders = [trigger] if trigger else []
    overlays, panes, sig_markers, blocks = _trader_blocks(
        ctx, pos_like["underlying"], traders, bars.loc[:end], start, end, "recorded",
        put_side={trigger} if (trigger and is_put) else set())

    kind = "C" if pos_like["option_type"] == "long_call" else "P"
    label = f"{pos_like['underlying']} {pos_like['expiration']} {pos_like['strike']:g}{kind}"
    markers = list(sig_markers)
    for t in trades:
        d = _bar_for_fill(bars, str(t.date)[:10], float("nan"))[0]
        o = t.side == "open"
        markers.append({"time": d, "position": "belowBar" if o else "aboveBar",
                        "shape": "arrowUp" if o else "arrowDown",
                        "color": GAIN if o else LOSS, "size": 1.4,
                        "text": (f"OPEN {t.contracts}× @ {_money(t.premium_per_contract)}/ct"
                                 if o else f"CLOSE ({t.reason}) @ {_money(t.premium_per_contract)}/ct")})
    markers.sort(key=lambda m: m["time"])

    strike = float(pos_like["strike"])
    per_share = float(pos_like["entry_premium_per_contract"]) / 100.0
    breakeven = strike + per_share if kind == "C" else strike - per_share
    price_lines = [{"price": round(strike, 2), "label": "strike",
                    "color": "#E8E3D3", "dash": False},
                   {"price": round(breakeven, 2), "label": "breakeven at expiry",
                    "color": "#E8E3D3", "dash": True}]

    notes = []
    contracts = int(pos_like["contracts"])
    paid = float(pos_like["premium_paid"])
    def _pt(t, v):
        return {"time": t, "value": round(v, 2), "pct": round(v / paid * 100, 2) if paid else None}

    pnl = [_pt(m["date"], contracts * float(m["premium"]) - paid)
           for m in pos_like.get("marks", []) or [] if m.get("premium") is not None]
    if open_:
        exp = date.fromisoformat(pos_like["expiration"])
        today = _now_local(ctx.config, ctx.now).date()
        value = contracts * mark if mark is not None else None
        unreal = (value - paid) if value is not None else None
        headline = {"label": "Unrealized P&L",
                    "value": _signed(unreal) if unreal is not None else "no quote",
                    "pct": f"{unreal / paid * 100:+.1f}%" if unreal is not None and paid else "",
                    "cls": _cls(unreal), "raw": round(unreal, 2) if unreal is not None else 0.0}
        stats = [("Contracts", str(contracts)), ("Premium paid", _money(paid)),
                 ("Mark now", _money(mark) + "/ct" if mark is not None else "no quote"),
                 ("Max loss", _money(paid)),
                 ("Expires", f"{_pretty_date(pos_like['expiration'])} ({(exp - today).days}d)")]
        if mark is None:
            notes.append("No live quote for this contract this run — P&L shown is unknown, "
                         "not zero.")
        if len(pnl) < 2:
            notes.append("P&L history starts filling in from this version on: one quoted "
                         "mark per daily run.")
    else:
        # Cash in minus cash out over the whole contract: every commission counted.
        realized = sum(t.cash_delta for t in trades)
        ledger_realized = sum(t.realized_pnl for t in close_t)
        if abs(ledger_realized - realized) >= 0.005:
            notes.append(f"After all commissions. Joseph's realized column shows "
                         f"{_signed(ledger_realized)} because it doesn't count the opening "
                         f"commission.")
        headline = {"label": "Realized P&L", "value": _signed(realized),
                    "pct": f"{realized / paid * 100:+.1f}%" if paid else "",
                    "cls": _cls(realized), "raw": round(realized, 2)}
        stats = [("Contracts", str(contracts)), ("Premium paid", _money(paid)),
                 ("Opened", _pretty_date(str(open_t[0].date)[:10]) if open_t else "—"),
                 ("Closed", (_pretty_date(str(close_t[-1].date)[:10]) + f" · {close_t[-1].reason}")
                  if close_t else "—")]
        if open_t and close_t:
            pnl = pnl or []
            pnl.append(_pt(str(close_t[-1].date)[:10], realized))
    if not trigger:
        notes.insert(0, "Couldn't tell which trigger opened this contract from its ledger "
                        "reason, so no strategy overlay is drawn.")
    ds = data_status(md, ctx.config, ctx.now)
    if ds["stale"] and open_:
        notes.append(ds["stale_note"])
    if ds["synthetic"]:
        notes.insert(0, "SYNTHETIC DATA — the underlying chart is a random walk.")

    return {
        "id": f"opt-{key.replace('|', '-').replace('.', '_')}-{'open' if open_ else idx}",
        "book": "options", "status": "open" if open_ else "closed",
        "symbol": pos_like["underlying"], "title": label,
        "subtitle": "Options desk · Joseph · underlying shown",
        "regime": regime or "", "traders": blocks,
        "attribution": "recorded" if trigger else "unknown",
        "data": ds, "headline": headline, "stats": [{"label": k, "value": v} for k, v in stats],
        "candles": _candles(view), "overlays": overlays, "panes": panes,
        "markers": markers, "price_lines": price_lines,
        "pnl": sorted(pnl, key=lambda p: p["time"]), "pnl_label": "Contract P&L (daily marks)",
        "pnl_basis": "% of premium paid",
        "pnl_stats": _pnl_stats(sorted(pnl, key=lambda p: p["time"])),
        "focus": {"from": focus_from, "to": focus_to},
        "notes": notes, "sort_date": last or first,
    }


def build_options_charts(config, market: dict, ledger, prices: dict | None = None,
                         regimes: dict | None = None, now: datetime | None = None) -> dict:
    if ledger is None:
        return {"open": [], "closed": []}
    ctx = _Ctx(config, market, now)
    prices = prices or {}
    regimes = regimes or {}

    def key_of(t):
        return f"{t.underlying}|{t.option_type}|{t.strike:g}|{t.expiration}"

    grouped: dict[str, list] = {}
    for t in ledger.trades:
        grouped.setdefault(key_of(t), []).append(t)

    open_cards, closed_cards = [], []
    for key, pos in ledger.positions.items():
        try:
            card = _option_card(ctx, key, {**pos.__dict__}, grouped.get(key, []), True,
                                prices.get(key), regimes.get(pos.underlying), 0)
        except Exception as exc:
            log.warning("Chart for %s failed (%s) -- skipped.", key, exc)
            card = None
        if card:
            open_cards.append(card)

    for key, trades in grouped.items():
        # Split into open->close lifecycles.
        cycle, i = [], 0
        for t in trades:
            cycle.append(t)
            if t.side != "open":
                opens = [x for x in cycle if x.side == "open"]
                contracts = sum(x.contracts for x in opens) or t.contracts
                paid = sum(x.premium_per_contract * x.contracts for x in opens)
                pos_like = {"underlying": t.underlying, "option_type": t.option_type,
                            "strike": t.strike, "expiration": t.expiration,
                            "contracts": contracts, "premium_paid": paid,
                            "entry_premium_per_contract": paid / contracts if contracts else 0.0,
                            "reason": opens[0].reason if opens else "", "marks": []}
                try:
                    card = _option_card(ctx, key, pos_like, cycle, False, None,
                                        regimes.get(t.underlying), i)
                except Exception as exc:
                    log.warning("Chart for closed %s failed (%s) -- skipped.", key, exc)
                    card = None
                if card:
                    closed_cards.append(card)
                cycle, i = [], i + 1
    closed_cards.sort(key=lambda c: c["sort_date"], reverse=True)
    return {"open": open_cards, "closed": closed_cards[:MAX_CLOSED_CARDS]}
