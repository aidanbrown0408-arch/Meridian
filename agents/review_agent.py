"""Jonathan — Trade Review Agent.

Once a trade is fully closed, Jonathan writes its after-action review: what
the trade was, what worked, what didn't, and one lesson. Every finding is a
rule applied to numbers from the ledgers and Wong's price bars, so the same
trade always gets the same review and every claim can be checked by hand.

What he looks at
  * Stock desk: one review per flat-to-flat episode on a symbol (the same
    episode split the ACTIVE TRADES charts use), closed once the position is
    gone from Cornelius's ledger. P&L is cash-basis -- exactly what the
    ledger's cash moved by, costs included.
  * Options desk: one review per contract from open to close/expiry on
    Joseph's ledger. Joseph drops a position's daily marks when it closes,
    so Jonathan snapshots open positions' marks every run and uses that
    history once the contract closes.

Reviews are written once. The exception is a review built on missing or
synthetic price data: that one is marked provisional and rebuilt on the next
run that has real bars, so a bad feed day never bakes bad numbers in.

Output, all under reports/trade_reviews/:
  * <exit date>_<desk>_<trade>.md -- one readable review per trade
  * SCORECARD.md                  -- running record by trader
  * index.json                    -- everything above, for George's dashboard

Read-only toward trading: nothing here writes a ledger, a signal, a risk
decision, or a Slack message.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from agents.regime_agent import RegimeAgent
from agents.trade_charts import _bar_for_fill, _d, _episodes, _trigger_of
from strategies.base import build_strategies
from utils.config import Config
from utils.logging_setup import get_logger

log = get_logger("review", agent="Jonathan")

REVIEWS_SUBDIR = "trade_reviews"
INDEX_FILE = "index.json"
SCORECARD_FILE = "SCORECARD.md"
BENCHMARK = "SPY"

# Thresholds behind the findings. Kept together so a reader can see exactly
# where each line is drawn.
MEANINGFUL_GAIN = 0.02     # a best close this far above entry counts as "was in profit"
GAVE_BACK_SHARE = 0.50     # gave back at least half of the best open gain
KEPT_SHARE = 0.70          # kept at least 70% of the best open gain
DEEP_DRAWDOWN = -0.05      # worst close this far below entry
CLEAN_ENTRY = -0.01        # never worse than this below entry
NEVER_WORKED = 0.005       # best close never more than this above entry
CHASE_RUN = 0.05           # up this much in the 5 sessions before buying
FLUSH_RUN = -0.05          # down this much in the 5 sessions before buying
WHIPSAW_SESSIONS = 5       # in and out within this many sessions
LATE_ENTRY_SESSIONS = 3    # trader had been long this long before the desk bought
HEAVY_COSTS = 0.25         # costs above this share of the gross gain
PRE_RUN_SESSIONS = 5

FAMILY_NAMES = {"trending": "trend-following", "breakout": "breakout",
                "mean_reversion": "mean-reversion"}

AREAS = ("Result", "Entry", "Ride", "Exit", "Context", "Costs", "Options", "Data")

LESSONS = {
    "regime_mismatch": ("{trader} took this trade in a {regime} tape, which isn't its regime. "
                        "Greg halves size there; if these keep losing, they may not be worth "
                        "taking at all."),
    "chased": ("Bought after a {pre_run} run in five sessions. Watch whether {trader}'s entries "
               "after extended moves keep losing; that would argue for waiting on a pullback."),
    "whipsaw": ("In and out in {sessions} sessions at a loss: a false signal, the normal cost of a "
                "{family} trader. It only becomes a problem if it repeats on {symbol}."),
    "gave_back": ("The trade was up {mfe} at its best close but kept only {captured}. The exit "
                  "rule is slow to lock in gains here; worth tracking how often that happens."),
    "never_worked": ("The trade was never meaningfully in profit, so the entry itself was the miss. "
                     "Check {trader}'s recent signal quality on {symbol}."),
    "deep_drawdown": ("It sat through a {mae} drawdown before the exit fired. That is a lot of heat "
                      "for this size; watch whether the exit rule should act sooner."),
    "costs_heavy": ("Costs took {cost_share} of the gross gain. Small or short trades on this "
                    "account don't clear their costs by much."),
    "portfolio_exit": ("The desk sold while {trader} was still long, so the exit was a portfolio "
                       "decision rather than the trader's rule. Check whether roster changes are "
                       "cutting trades short."),
    "decay": ("{underlying} moved the right way but the option still lost: with {dte} days to "
              "expiration, time decay outran the move."),
    "wrong_way": ("{underlying} moved against the position. Premium paid was the full risk, as "
                  "designed; nothing to fix unless it repeats."),
    "expired": ("Held to expiration and settled at {settle}. The move didn't arrive within {dte} "
                "days."),
}
LESSON_ORDER = ("regime_mismatch", "decay", "chased", "never_worked", "gave_back",
                "deep_drawdown", "portfolio_exit", "whipsaw", "expired", "wrong_way",
                "costs_heavy")


# ------------------------------------------------------------------ data types

@dataclass
class Finding:
    kind: str           # "worked" | "didnt" | "note"
    area: str           # one of AREAS
    text: str
    tag: str = ""       # short key the scorecard counts, e.g. "gave_back"


@dataclass
class TradeReview:
    id: str
    desk: str                   # "stock" | "options"
    symbol: str                 # ticker, or the option's underlying
    label: str                  # "NVDA" | "SPY 2026-11-21 600C"
    traders: list
    entry_date: str
    exit_date: str
    days_held: int
    sessions_held: int | None
    entry_price: float          # avg share price | premium per contract
    exit_price: float
    size: str                   # "5.45 sh" | "1 contract"
    capital: float              # dollars put in (buys, or premium + commission)
    costs: float
    pnl: float
    pnl_pct: float
    outcome: str                # "win" | "loss" | "flat"
    findings: list = field(default_factory=list)   # Finding
    lesson: str = ""
    metrics: dict = field(default_factory=dict)
    data_source: str = ""
    provisional: bool = False
    reviewed_at: str = ""
    file: str = ""

    @property
    def worked(self) -> list:
        return [f for f in self.findings if f.kind == "worked"]

    @property
    def didnt(self) -> list:
        return [f for f in self.findings if f.kind == "didnt"]

    @property
    def notes(self) -> list:
        return [f for f in self.findings if f.kind == "note"]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TradeReview":
        d = dict(d)
        d["findings"] = [Finding(**f) for f in d.get("findings", [])]
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class ReviewRun:
    new: list = field(default_factory=list)        # TradeReview written this run
    reviews: list = field(default_factory=list)    # every review on file, newest first
    scorecard: dict = field(default_factory=dict)
    skipped: list = field(default_factory=list)    # (id, reason)


# ------------------------------------------------------------------ helpers

def _pct(x: float | None, nd: int = 1) -> str:
    if x is None:
        return "n/a"
    return f"{x * 100:+.{nd}f}%".replace("-", "−")


def _usd(x: float, signed: bool = False) -> str:
    if signed:
        return ("+" if x >= 0 else "−") + f"${abs(x):,.2f}"
    return f"${x:,.2f}"


def _pts(x: float) -> str:
    return f"{x * 100:.1f} pts"


def _pretty(iso: str) -> str:
    try:
        return date.fromisoformat(iso[:10]).strftime("%b %-d")
    except ValueError:
        return iso


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")


def _outcome(pnl: float) -> str:
    if abs(pnl) < 0.005:
        return "flat"
    return "win" if pnl > 0 else "loss"


def _bar_index(bars: pd.DataFrame, day: str) -> int | None:
    dates = [_d(t) for t in bars.index]
    if day in dates:
        return dates.index(day)
    earlier = [i for i, d in enumerate(dates) if d <= day]
    return earlier[-1] if earlier else None


# ------------------------------------------------------------------ the agent

class TradeReviewAgent:
    """Jonathan."""

    name = "Jonathan"
    role = "Trade Review"

    def __init__(self, config: Config, reports_dir: Path | None = None):
        self.config = config
        self.reports_dir = Path(reports_dir) if reports_dir else config.repo_path("reports")
        self.min_trade_dollars = float(config.get("execution.min_trade_dollars", 25.0))
        self._strategies = None
        self._greg = None

    @property
    def dir(self) -> Path:
        return self.reports_dir / REVIEWS_SUBDIR

    @property
    def strategies(self) -> dict:
        if self._strategies is None:
            self._strategies = build_strategies(self.config)
        return self._strategies

    @property
    def greg(self) -> RegimeAgent:
        if self._greg is None:
            self._greg = RegimeAgent(self.config)
        return self._greg

    # -------------------------------------------------------------- index

    def load_index(self) -> dict:
        path = self.dir / INDEX_FILE
        if not path.exists():
            return {"reviews": [], "marks": {}}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            data.setdefault("reviews", [])
            data.setdefault("marks", {})
            return data
        except Exception as exc:
            log.error("Review index at %s is unreadable (%s) -- starting a fresh one; "
                      "existing review files are left in place.", path, exc)
            return {"reviews": [], "marks": {}}

    def _save_index(self, index: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / INDEX_FILE).write_text(json.dumps(index, indent=2), encoding="utf-8")

    # -------------------------------------------------------------- run

    @staticmethod
    def symbols_needed(stock_ledger=None, options_ledger=None) -> list[str]:
        """Bars Jonathan wants for a standalone review run."""
        syms = {BENCHMARK}
        for t in getattr(stock_ledger, "trades", []) or []:
            syms.add(t.symbol)
        for t in getattr(options_ledger, "trades", []) or []:
            syms.add(t.underlying)
        return sorted(syms)

    def run(self, stock_ledger=None, options_ledger=None, market: dict | None = None,
            rebuild: bool = False, now: datetime | None = None) -> ReviewRun:
        market = market or {}
        now = now or datetime.now(timezone.utc)
        index = self.load_index()
        marks = index.get("marks", {})
        if options_ledger is not None:
            self._snapshot_marks(options_ledger, marks)

        existing = {} if rebuild else {
            r["id"]: TradeReview.from_dict(r) for r in index.get("reviews", [])}
        out = ReviewRun()

        episodes = []
        if stock_ledger is not None:
            episodes += [("stock",) + e for e in self.closed_stock_episodes(stock_ledger)]
        if options_ledger is not None:
            episodes += [("options",) + e for e in self.closed_option_episodes(options_ledger)]

        for desk, rid, key, trades in episodes:
            prior = existing.get(rid)
            if prior is not None and not prior.provisional:
                continue
            try:
                if desk == "stock":
                    review = self.review_stock(rid, key, trades, market)
                else:
                    review = self.review_option(rid, key, trades, market, marks.get(key, []))
            except Exception as exc:  # one bad trade never blocks the rest
                log.warning("Review of %s failed (%s) -- skipped this run.", rid, exc)
                out.skipped.append((rid, str(exc)))
                continue
            if prior is not None and prior.provisional and review.provisional:
                continue  # still no real data; keep the first provisional review
            review.reviewed_at = now.isoformat()
            review.file = self._write_review(review)
            existing[rid] = review
            out.new.append(review)
            log.info("Reviewed %s: %s %s (%s)", review.label, review.outcome,
                     _usd(review.pnl, signed=True), ", ".join(review.traders) or "unattributed")

        reviews = sorted(existing.values(), key=lambda r: (r.exit_date, r.id), reverse=True)
        out.reviews = reviews
        out.scorecard = self.scorecard(reviews)
        index = {"updated_at": now.isoformat(), "reviews": [r.to_dict() for r in reviews],
                 "scorecard": out.scorecard, "marks": marks}
        self._save_index(index)
        self._write_scorecard(out.scorecard, reviews)
        if not out.new:
            log.info("No newly closed trades to review (%d on file).", len(reviews))
        return out

    # -------------------------------------------------------------- episodes

    def closed_stock_episodes(self, ledger) -> list[tuple[str, str, list]]:
        by_symbol: dict[str, list] = {}
        for t in ledger.trades:
            by_symbol.setdefault(t.symbol, []).append(t)
        out = []
        for symbol, trades in by_symbol.items():
            eps = _episodes(trades, self.min_trade_dollars)
            for i, ep in enumerate(eps):
                is_open = i == len(eps) - 1 and symbol in ledger.positions
                if is_open or not any(t.side == "sell" for t in ep):
                    continue
                rid = f"stock:{symbol}:{ep[0].date}:{ep[-1].date}"
                out.append((rid, symbol, ep))
        return out

    def closed_option_episodes(self, ledger) -> list[tuple[str, str, list]]:
        by_key: dict[str, list] = {}
        out = []
        for t in ledger.trades:
            key = f"{t.underlying}|{t.option_type}|{t.strike:g}|{t.expiration}"
            cur = by_key.setdefault(key, [])
            cur.append(t)
            if t.side == "close":
                if any(x.side == "open" for x in cur):
                    rid = f"options:{key}:{cur[0].date}:{t.date}"
                    out.append((rid, key, list(cur)))
                by_key[key] = []
        return out

    def _snapshot_marks(self, ledger, marks: dict) -> None:
        for key, pos in ledger.positions.items():
            have = {m.get("date"): m for m in marks.get(key, [])}
            for m in getattr(pos, "marks", []) or []:
                have[m.get("date")] = m
            marks[key] = sorted(have.values(), key=lambda m: m.get("date", ""))[-400:]

    # -------------------------------------------------------------- price context

    def _bars(self, market: dict, symbol: str):
        md = market.get(symbol)
        if md is None or md.bars is None or md.bars.empty:
            return None, "missing"
        return md.bars, md.data_source

    def _window_return(self, market: dict, symbol: str, start: str, end: str) -> float | None:
        bars, source = self._bars(market, symbol)
        if bars is None or source == "synthetic":
            return None
        i, j = _bar_index(bars, start), _bar_index(bars, end)
        if i is None or j is None or j <= i:
            return None
        return float(bars["close"].iloc[j] / bars["close"].iloc[i] - 1)

    def _regime(self, bars: pd.DataFrame, upto: int) -> str:
        try:
            return self.greg.classify(bars.iloc[: upto + 1]).regime
        except Exception:
            return "undecided"

    def _trader_state(self, callsign: str, bars: pd.DataFrame, i: int, j: int) -> dict | None:
        """Was the trader long on the entry bar, how long had it been long,
        and was it still long on the exit bar? Uses `positions()` -- the same
        shifted series Charles nets into the desk's target."""
        strat = self.strategies.get(callsign)
        if strat is None or len(bars) <= strat.warmup:
            return None
        pos = strat.positions(bars.iloc[: j + 1])
        streak = 0
        for v in reversed(pos.iloc[: i + 1].tolist()):
            if v > 0:
                streak += 1
            else:
                break
        return {"long_at_entry": bool(pos.iloc[i] > 0), "streak_at_entry": streak,
                "long_at_exit": bool(pos.iloc[j] > 0), "family": strat.family,
                "best_regimes": list(strat.best_regimes)}

    # -------------------------------------------------------------- stock review

    def review_stock(self, rid: str, symbol: str, trades: list, market: dict) -> TradeReview:
        buys = [t for t in trades if t.side == "buy"]
        sells = [t for t in trades if t.side == "sell"]
        bought = sum(t.shares * t.price for t in buys)
        sold = sum(t.shares * t.price for t in sells)
        costs = sum(t.cost for t in trades)
        gross = sold - bought
        pnl = gross - costs
        buy_sh = sum(t.shares for t in buys)
        sell_sh = sum(t.shares for t in sells)
        avg_in = bought / buy_sh if buy_sh else 0.0
        avg_out = sold / sell_sh if sell_sh else 0.0
        pnl_pct = pnl / bought if bought else 0.0

        traders: list[str] = []
        for t in buys + sells:
            for c in getattr(t, "traders", []) or []:
                if c not in traders:
                    traders.append(c)

        first, last = trades[0], trades[-1]
        entry_day = first.bar_date or first.date
        exit_day = last.bar_date or last.date
        days = max(0, (date.fromisoformat(exit_day[:10]) - date.fromisoformat(entry_day[:10])).days)

        findings: list[Finding] = []
        m: dict = {"bought": round(bought, 2), "sold": round(sold, 2), "gross": round(gross, 2),
                   "fills": len(trades), "buys": len(buys), "sells": len(sells)}
        lesson_ctx = {"symbol": symbol, "trader": " + ".join(traders) or "the desk"}

        findings.append(Finding(
            "worked" if pnl > 0 else ("didnt" if pnl < 0 else "note"), "Result",
            f"Closed {_usd(pnl, signed=True)} ({_pct(pnl_pct)}) on {_usd(bought)} deployed, "
            f"{days} day(s) after the first buy.", "win" if pnl > 0 else "loss"))
        if len(buys) > 1 or len(sells) > 1:
            findings.append(Finding("note", "Result",
                f"{len(buys)} buy(s) and {len(sells)} sell(s): Cornelius resized the position "
                f"along the way. Average in {_usd(avg_in)}, average out {_usd(avg_out)}."))
        if not traders:
            findings.append(Finding("note", "Data",
                "No trader is recorded on these fills (opened before attribution was "
                "logged), so the trader checks below are skipped."))

        bars, source = self._bars(market, symbol)
        sessions = None
        provisional = bars is None or source == "synthetic"
        if provisional:
            findings.append(Finding("note", "Data",
                f"No real price history for {symbol} on this run ({source}), so the ride, "
                "entry and context checks are skipped. Jonathan rebuilds this review on the "
                "next run with real bars."))
        else:
            i_day, _ = _bar_for_fill(bars, first.date, first.price, first.bar_date)
            j_day, _ = _bar_for_fill(bars, last.date, last.price, last.bar_date)
            i, j = _bar_index(bars, i_day), _bar_index(bars, j_day)
            if i is None or j is None or j < i:
                provisional = True
                findings.append(Finding("note", "Data",
                    "The fills don't line up with the price history, so bar-based checks "
                    "are skipped."))
            else:
                sessions = j - i
                close = bars["close"].astype(float)
                self._stock_ride(findings, m, lesson_ctx, close, i, j, avg_in, avg_out, pnl)
                self._stock_entry(findings, m, lesson_ctx, close, i, pnl, traders)
                self._stock_traders(findings, m, lesson_ctx, bars, i, j, traders, pnl, sessions)
                spy = None if symbol == BENCHMARK else self._window_return(
                    market, BENCHMARK, bars.index[i].strftime("%Y-%m-%d"),
                    bars.index[j].strftime("%Y-%m-%d"))
                if spy is not None:
                    move = avg_out / avg_in - 1 if avg_in else 0.0
                    m["spy_return"] = round(spy, 4)
                    edge = move - spy
                    findings.append(Finding(
                        "worked" if edge > 0 else "didnt", "Context",
                        f"{symbol}'s price move while held ({_pct(move)}) "
                        f"{'beat' if edge > 0 else 'trailed'} SPY over the same sessions "
                        f"({_pct(spy)}) by {_pts(abs(edge))}.",
                        "beat_spy" if edge > 0 else "lagged_spy"))

        if gross > 0 and costs > HEAVY_COSTS * gross:
            share = costs / gross
            lesson_ctx["cost_share"] = f"{share:.0%}"
            findings.append(Finding("didnt", "Costs",
                f"Commission and slippage ({_usd(costs)}) took {share:.0%} of the "
                f"{_usd(gross)} gross gain.", "costs_heavy"))
        else:
            findings.append(Finding("note", "Costs",
                f"Commission and slippage: {_usd(costs)} across {len(trades)} fill(s)"
                + (f", {costs / gross:.0%} of the gross gain." if gross > 0 else
                   f", on top of a {_usd(abs(gross))} gross loss." if gross < 0 else ".")))

        review = TradeReview(
            id=rid, desk="stock", symbol=symbol, label=symbol, traders=traders,
            entry_date=entry_day, exit_date=exit_day, days_held=days, sessions_held=sessions,
            entry_price=round(avg_in, 4), exit_price=round(avg_out, 4),
            size=f"{max(buy_sh, sell_sh):,.2f} sh", capital=round(bought, 2),
            costs=round(costs, 2), pnl=round(pnl, 2), pnl_pct=round(pnl_pct, 4),
            outcome=_outcome(pnl), findings=findings, metrics=m,
            data_source=source, provisional=provisional)
        review.lesson = self._lesson(review, lesson_ctx)
        return review

    def _stock_ride(self, findings, m, ctx, close, i, j, avg_in, avg_out, pnl) -> None:
        held = close.iloc[i + 1: j + 1]
        if held.empty or not avg_in:
            return
        mfe = max(0.0, float(held.max() / avg_in - 1))
        mae = min(0.0, float(held.min() / avg_in - 1))
        captured = avg_out / avg_in - 1
        best_day = _pretty(_d(held.idxmax()))
        worst_day = _pretty(_d(held.idxmin()))
        m.update(mfe=round(mfe, 4), mae=round(mae, 4), captured=round(captured, 4),
                 best_close_date=_d(held.idxmax()), worst_close_date=_d(held.idxmin()))
        ctx.update(mfe=_pct(mfe), mae=_pct(mae), captured=_pct(captured))

        if mfe >= MEANINGFUL_GAIN and (mfe - captured) >= GAVE_BACK_SHARE * mfe:
            findings.append(Finding("didnt", "Ride",
                f"Was up {_pct(mfe)} at the best close ({best_day}) but exited at "
                f"{_pct(captured)}, giving back {_pts(mfe - captured)}.", "gave_back"))
        elif mfe >= MEANINGFUL_GAIN and captured >= KEPT_SHARE * mfe:
            findings.append(Finding("worked", "Ride",
                f"Kept most of the move: exited at {_pct(captured)} against a best close of "
                f"{_pct(mfe)} ({best_day}).", "kept_gain"))
        elif mfe < NEVER_WORKED and pnl < 0:
            findings.append(Finding("didnt", "Ride",
                f"Never meaningfully in profit: the best close was only {_pct(mfe)} above the "
                "average entry.", "never_worked"))
        else:
            findings.append(Finding("note", "Ride",
                f"Best close {_pct(mfe)} ({best_day}), worst {_pct(mae)} ({worst_day}), "
                f"exited at {_pct(captured)}."))

        if mae <= DEEP_DRAWDOWN:
            findings.append(Finding("didnt", "Ride",
                f"Sat through a {_pct(mae)} drawdown (worst close {worst_day}).",
                "deep_drawdown"))
        elif pnl > 0 and mae >= CLEAN_ENTRY:
            findings.append(Finding("worked", "Entry",
                f"Clean entry: never more than {_pct(mae)} underwater on a close.",
                "clean_entry"))

    def _stock_entry(self, findings, m, ctx, close, i, pnl, traders) -> None:
        if i < PRE_RUN_SESSIONS:
            return
        pre = float(close.iloc[i] / close.iloc[i - PRE_RUN_SESSIONS] - 1)
        m["pre_entry_run"] = round(pre, 4)
        ctx["pre_run"] = _pct(pre)
        mean_rev = any(self.strategies.get(t) is not None
                       and self.strategies[t].family == "mean_reversion" for t in traders)
        if pre >= CHASE_RUN and pnl < 0:
            findings.append(Finding("didnt", "Entry",
                f"Bought after a {_pct(pre)} run in the prior {PRE_RUN_SESSIONS} sessions and "
                "lost: the entry chased an extended move.", "chased"))
        elif pre <= FLUSH_RUN and pnl > 0 and mean_rev:
            findings.append(Finding("worked", "Entry",
                f"Bought into a {_pct(pre)} slide over {PRE_RUN_SESSIONS} sessions, the setup "
                "a mean-reversion trader wants, and it bounced.", "bought_dip"))
        else:
            findings.append(Finding("note", "Entry",
                f"{_pct(pre)} in the {PRE_RUN_SESSIONS} sessions before the first buy."))

    def _stock_traders(self, findings, m, ctx, bars, i, j, traders, pnl, sessions) -> None:
        regime_in = self._regime(bars, i)
        regime_out = self._regime(bars, j)
        m.update(regime_at_entry=regime_in, regime_at_exit=regime_out)
        ctx["regime"] = regime_in
        states = {}
        for t in traders:
            st = self._trader_state(t, bars, i, j)
            if st is None:
                continue
            states[t] = st
            family = FAMILY_NAMES.get(st["family"], st["family"])
            ctx.setdefault("family", family)
            if regime_in in st["best_regimes"]:
                findings.append(Finding("worked", "Context",
                    f"{t} is built for a {regime_in} tape, and that's what Greg read on the "
                    "entry bar.", "regime_match"))
            elif regime_in == "undecided":
                findings.append(Finding("note", "Context",
                    f"Greg's read on the entry bar was undecided, so {t} traded at full size "
                    "without a regime tailwind."))
            else:
                ctx["trader"] = t
                findings.append(Finding("didnt", "Context",
                    f"{t} is a {family} trader but Greg read the tape "
                    f"as {regime_in} on the entry bar.", "regime_mismatch"))

            if st["streak_at_entry"] > LATE_ENTRY_SESSIONS:
                findings.append(Finding("note", "Entry",
                    f"{t} had already been long for {st['streak_at_entry']} sessions when the "
                    "desk bought, so the position started late (the trader joined the live "
                    "roster or got capital after its signal turned on).", "late_entry"))
            if st["long_at_exit"]:
                ctx["trader"] = t
                findings.append(Finding("note", "Exit",
                    f"{t} was still long on the exit bar: the sale came from the portfolio side "
                    "(roster rotation, Greg's regime tilt, or risk sizing), not the trader's own "
                    "exit rule.", "portfolio_exit"))
            else:
                findings.append(Finding("worked" if pnl > 0 else "note", "Exit",
                    f"{t}'s own exit rule turned it flat, and the desk sold on that bar.",
                    "rule_exit"))
            if (sessions is not None and sessions <= WHIPSAW_SESSIONS and pnl < 0
                    and st["family"] in ("trending", "breakout")):
                ctx["sessions"] = sessions
                findings.append(Finding("didnt", "Exit",
                    f"Whipsaw: in and out in {sessions} session(s) at a loss, a false signal "
                    f"for a {family} trader.", "whipsaw"))
        m["traders"] = states
        if regime_out != regime_in:
            findings.append(Finding("note", "Context",
                f"The regime shifted from {regime_in} to {regime_out} while the trade was on."))

    # -------------------------------------------------------------- options review

    def review_option(self, rid: str, key: str, trades: list, market: dict,
                      marks: list) -> TradeReview:
        opens = [t for t in trades if t.side == "open"]
        closes = [t for t in trades if t.side == "close"]
        t0, tz = opens[0], closes[-1]
        underlying, kind = t0.underlying, t0.option_type
        is_call = kind == "long_call"
        contracts = sum(t.contracts for t in opens)
        paid = sum(t.contracts * t.premium_per_contract + t.cost for t in opens)
        entry_prem = sum(t.contracts * t.premium_per_contract for t in opens) / contracts
        exit_prem = tz.premium_per_contract
        costs = sum(t.cost for t in trades)
        pnl = sum(t.realized_pnl for t in closes)
        pnl_pct = pnl / paid if paid else 0.0
        label = f"{underlying} {t0.expiration} {t0.strike:g}{'C' if is_call else 'P'}"
        exp = date.fromisoformat(t0.expiration)
        dte_in = (exp - date.fromisoformat(t0.date[:10])).days
        dte_out = (exp - date.fromisoformat(tz.date[:10])).days
        days = max(0, (date.fromisoformat(tz.date[:10]) - date.fromisoformat(t0.date[:10])).days)

        known = set(self.strategies)
        trigger, is_put = _trigger_of(t0.reason, known)
        traders = [f"{trigger}-puts" if is_put else trigger] if trigger else []

        findings: list[Finding] = []
        m: dict = {"dte_at_entry": dte_in, "dte_at_exit": dte_out, "contracts": contracts,
                   "close_reason": tz.reason}
        ctx = {"underlying": underlying, "dte": dte_in, "symbol": underlying,
               "trader": traders[0] if traders else "the options desk",
               "settle": _usd(exit_prem)}

        findings.append(Finding(
            "worked" if pnl > 0 else ("didnt" if pnl < 0 else "note"), "Result",
            f"Closed {_usd(pnl, signed=True)} ({_pct(pnl_pct)}) on {_usd(paid)} of premium "
            f"and commission, {days} day(s) after opening.", "win" if pnl > 0 else "loss"))

        if tz.reason == "expired":
            findings.append(Finding("didnt" if pnl < 0 else "note", "Options",
                f"Held to expiration ({t0.expiration}) and settled at {_usd(exit_prem)}/contract"
                + (", worthless." if exit_prem == 0 else "."), "expired"))
        elif tz.reason == "killswitch flatten":
            findings.append(Finding("note", "Exit",
                "Closed by the options killswitch, not by the strategy."))
        else:
            findings.append(Finding("note", "Exit", f"Closed: {tz.reason}."))

        held_marks = [mk for mk in marks if t0.date[:10] <= mk.get("date", "") <= tz.date[:10]]
        if held_marks and entry_prem:
            best = max(held_marks, key=lambda mk: mk["premium"])
            peak = best["premium"] / entry_prem - 1
            final = exit_prem / entry_prem - 1
            m.update(best_mark=best["premium"], best_mark_date=best["date"],
                     best_mark_return=round(peak, 4))
            ctx.update(mfe=_pct(peak), captured=_pct(final))
            if peak >= 0.25 and (peak - final) >= GAVE_BACK_SHARE * peak:
                findings.append(Finding("didnt", "Ride",
                    f"Premium was up {_pct(peak)} at its best mark ({_pretty(best['date'])}) "
                    f"and closed at {_pct(final)}.", "gave_back"))
            else:
                findings.append(Finding("note", "Ride",
                    f"Best recorded mark {_usd(best['premium'])}/contract "
                    f"({_pct(peak)}, {_pretty(best['date'])})."))
        else:
            findings.append(Finding("note", "Data",
                "No daily marks were recorded while this contract was open, so the best and "
                "worst points of the ride aren't known."))

        bars, source = self._bars(market, underlying)
        provisional = bars is None or source == "synthetic"
        sessions = None
        if provisional:
            findings.append(Finding("note", "Data",
                f"No real price history for {underlying} on this run ({source}); underlying "
                "and regime checks are skipped until the next run with real bars."))
        else:
            i = _bar_index(bars, t0.date[:10])
            j = _bar_index(bars, tz.date[:10])
            if i is not None and j is not None and j >= i:
                sessions = j - i
                close = bars["close"].astype(float)
                und_in, und_out = float(close.iloc[i]), float(close.iloc[j])
                move = und_out / und_in - 1
                right_way = move > 0 if is_call else move < 0
                money = (und_in - t0.strike) / t0.strike if is_call else \
                        (t0.strike - und_in) / t0.strike
                m.update(underlying_in=round(und_in, 2), underlying_out=round(und_out, 2),
                         underlying_move=round(move, 4), moneyness_at_entry=round(money, 4))
                findings.append(Finding("note", "Entry",
                    f"Bought the {t0.strike:g} {'call' if is_call else 'put'} with {underlying} "
                    f"at {_usd(und_in)} ({'in' if money > 0 else 'out of'} the money by "
                    f"{abs(money) * 100:.1f}%), {dte_in} days to expiration."))
                if right_way and pnl > 0:
                    lev = (pnl_pct / abs(move)) if move else 0.0
                    findings.append(Finding("worked", "Options",
                        f"Direction right: {underlying} {_pct(move)} while held, the option "
                        f"{_pct(pnl_pct)} (about {lev:.0f}x the underlying's move).",
                        "right_way"))
                elif right_way:
                    findings.append(Finding("didnt", "Options",
                        f"{underlying} moved the right way ({_pct(move)}) but the option still "
                        f"lost {_pct(pnl_pct)}: time decay and premium outran the move.",
                        "decay"))
                else:
                    findings.append(Finding("didnt", "Options",
                        f"Direction wrong: {underlying} {_pct(move)} while held.",
                        "wrong_way"))
                regime_in = self._regime(bars, i)
                m["regime_at_entry"] = regime_in
                ctx["regime"] = regime_in
                if trigger:
                    best_regimes = self.strategies[trigger].best_regimes
                    if regime_in in best_regimes:
                        findings.append(Finding("worked", "Context",
                            f"Greg read {underlying} as {regime_in} at entry, the tape "
                            f"{trigger} is built for.", "regime_match"))
                    else:
                        findings.append(Finding("didnt", "Context",
                            f"Greg read {underlying} as {regime_in} at entry; {trigger} is "
                            "built for a trending tape.", "regime_mismatch"))

        findings.append(Finding("note", "Costs",
            f"Commission: {_usd(costs)} on {contracts} contract(s)."))

        review = TradeReview(
            id=rid, desk="options", symbol=underlying, label=label, traders=traders,
            entry_date=t0.date[:10], exit_date=tz.date[:10], days_held=days,
            sessions_held=sessions, entry_price=round(entry_prem, 4),
            exit_price=round(exit_prem, 4),
            size=f"{contracts} contract{'s' if contracts != 1 else ''}",
            capital=round(paid, 2), costs=round(costs, 2), pnl=round(pnl, 2),
            pnl_pct=round(pnl_pct, 4), outcome=_outcome(pnl), findings=findings,
            metrics=m, data_source=source, provisional=provisional)
        review.lesson = self._lesson(review, ctx)
        return review

    # -------------------------------------------------------------- lesson

    def _lesson(self, review: TradeReview, ctx: dict) -> str:
        tags = {f.tag for f in review.findings if f.kind == "didnt"}
        if review.outcome != "win" and any(f.tag == "portfolio_exit" for f in review.findings):
            tags.add("portfolio_exit")
        for tag in LESSON_ORDER:
            if tag in tags:
                try:
                    return LESSONS[tag].format(**ctx)
                except KeyError:
                    continue
        who = ctx.get("trader", "the desk")
        if review.outcome == "win":
            regime = ctx.get("regime")
            setting = f" in a {regime} tape" if regime and regime != "undecided" else ""
            return (f"A clean win: {who} on {review.symbol}{setting}, and the rules did their "
                    "job. Nothing to change.")
        return ("A loss inside the rules, with no process error found. Losing trades are part "
                "of the design; nothing to change unless the pattern repeats.")

    # -------------------------------------------------------------- scorecard

    @staticmethod
    def scorecard(reviews: list) -> dict:
        rows: dict[str, dict] = {}

        def bump(name, r):
            row = rows.setdefault(name, {"trader": name, "trades": 0, "wins": 0, "losses": 0,
                                         "pnl": 0.0, "returns": [], "best": None,
                                         "worst": None, "tags": {}})
            row["trades"] += 1
            row["wins"] += r.outcome == "win"
            row["losses"] += r.outcome == "loss"
            row["pnl"] += r.pnl
            row["returns"].append(r.pnl_pct)
            row["best"] = r.pnl if row["best"] is None else max(row["best"], r.pnl)
            row["worst"] = r.pnl if row["worst"] is None else min(row["worst"], r.pnl)
            for f in r.findings:
                if f.kind == "didnt" and f.tag and f.tag != "loss":
                    row["tags"][f.tag] = row["tags"].get(f.tag, 0) + 1

        for r in reviews:
            for t in (r.traders or ["Unattributed"]):
                bump(t, r)
            bump("All trades", r)

        out = []
        for row in rows.values():
            n = row["trades"]
            row["win_rate"] = round(row["wins"] / n, 4) if n else 0.0
            row["avg_return"] = round(sum(row["returns"]) / n, 4) if n else 0.0
            row["pnl"] = round(row["pnl"], 2)
            row["top_issues"] = [k for k, _ in sorted(row["tags"].items(),
                                                      key=lambda kv: (-kv[1], kv[0]))][:3]
            del row["returns"]
            out.append(row)
        out.sort(key=lambda r: (r["trader"] == "All trades", -r["trades"], r["trader"]))
        return {"rows": out, "count": len(reviews)}

    # -------------------------------------------------------------- files

    def _write_review(self, r: TradeReview) -> str:
        self.dir.mkdir(parents=True, exist_ok=True)
        trader = "_".join(r.traders) or "unattributed"
        name = f"{r.exit_date}_{r.desk}_{_slug(r.label)}_{_slug(trader)}.md"
        (self.dir / name).write_text(render_markdown(r), encoding="utf-8")
        return name

    def _write_scorecard(self, card: dict, reviews: list) -> None:
        lines = ["# Jonathan's Scorecard", "",
                 f"{card['count']} closed trade(s) reviewed. "
                 f"Updated {datetime.now():%b %-d, %Y %H:%M}.", ""]
        if card["rows"]:
            lines += ["| Trader | Trades | Wins | Win rate | Net P&L | Avg return | Best | Worst "
                      "| Recurring issues |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
            for row in card["rows"]:
                name = f"**{row['trader']}**" if row["trader"] == "All trades" else row["trader"]
                lines.append(
                    f"| {name} | {row['trades']} | {row['wins']} | {row['win_rate']:.0%} | "
                    f"{_usd(row['pnl'], signed=True)} | {_pct(row['avg_return'])} | "
                    f"{_usd(row['best'], signed=True)} | {_usd(row['worst'], signed=True)} | "
                    f"{', '.join(ISSUE_NAMES.get(t, t) for t in row['top_issues']) or '—'} |")
        lines += ["", "## Reviews", ""]
        for r in reviews:
            lines.append(f"- {r.exit_date} · [{r.label} ({', '.join(r.traders) or 'unattributed'})]"
                         f"({r.file}) · {_usd(r.pnl, signed=True)} · {r.outcome}")
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / SCORECARD_FILE).write_text("\n".join(lines) + "\n", encoding="utf-8")


ISSUE_NAMES = {
    "gave_back": "gave back gains", "never_worked": "never in profit",
    "deep_drawdown": "deep drawdown", "chased": "chased entry", "whipsaw": "whipsaw",
    "regime_mismatch": "wrong regime", "lagged_spy": "trailed SPY",
    "costs_heavy": "costs heavy", "decay": "time decay", "wrong_way": "wrong direction",
    "expired": "expired",
}


def render_markdown(r: TradeReview) -> str:
    who = ", ".join(r.traders) or "unattributed"
    verdict = {"win": "WIN", "loss": "LOSS", "flat": "FLAT"}[r.outcome]
    desk = "Stock desk · Cornelius" if r.desk == "stock" else "Options desk · Joseph"
    unit = "avg share price" if r.desk == "stock" else "premium per contract"
    lines = [
        f"# {r.label} · {who} · {verdict} {_usd(r.pnl, signed=True)}", "",
        f"*Trade review by Jonathan · {desk}*", "",
    ]
    if r.provisional:
        lines += ["> **Provisional:** built without real price history. Jonathan rebuilds "
                  "this review on the next run that has real bars.", ""]
    lines += [
        "| | |", "|---|---|",
        f"| Entry | {_pretty(r.entry_date)} at {_usd(r.entry_price)} ({unit}) |",
        f"| Exit | {_pretty(r.exit_date)} at {_usd(r.exit_price)} |",
        f"| Held | {r.days_held} day(s)"
        + (f", {r.sessions_held} session(s)" if r.sessions_held is not None else "") + " |",
        f"| Size | {r.size}, {_usd(r.capital)} in |",
        f"| Realized P&L | {_usd(r.pnl, signed=True)} ({_pct(r.pnl_pct)}) |",
        f"| Costs | {_usd(r.costs)} |", "",
    ]
    for title, items in (("What worked", r.worked), ("What didn't", r.didnt),
                         ("Notes", r.notes)):
        lines += [f"## {title}", ""]
        lines += [f"- **{f.area}:** {f.text}" for f in items] or ["- Nothing flagged."]
        lines.append("")
    lines += ["## Lesson", "", r.lesson, "",
              f"<sub>Review id `{r.id}` · data: {r.data_source} · "
              f"reviewed {r.reviewed_at[:16].replace('T', ' ')} UTC</sub>", ""]
    return "\n".join(lines)


def dashboard_summary(index: dict | None, limit: int = 20) -> dict | None:
    """The slice of Jonathan's index George needs for the dashboard."""
    if not index:
        return None
    reviews = index.get("reviews", [])
    return {"reviews": reviews[:limit], "count": len(reviews),
            "scorecard": index.get("scorecard", {"rows": [], "count": 0}),
            "updated_at": index.get("updated_at", ""), "issue_names": ISSUE_NAMES}
