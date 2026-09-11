"""George — Reporting Agent.

Aggregates everything the pipeline produced into the daily Meridian Capital
dashboard (spec §11-12): an HTML report in the navy/gold Playfair house
style, plus a standup-style Slack message -- one line per agent that has
something notable to say, boring days shorter than loud ones.

There is no live P&L yet (Cornelius's PaperBroker is Phase 4), so the
"positions & trades" section shows today's *target* allocation -- Charles's
risk-parity weights after Greg's regime adjustment -- clearly labeled as a
target, not a fill. Every number sourced from synthetic data is flagged the
same way it is in the console report.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from agents.compliance_agent import ComplianceReport
from agents.data_agent import MarketData
from agents.regime_agent import RegimeReport
from agents.risk_agent import RiskReport
from backtester.engine import BacktestResult
from utils.config import Config
from utils.logging_setup import get_logger
from utils.metrics import PerformanceSummary
from utils.slack import post_message
from utils.svg_charts import bar_chart, line_chart

log = get_logger("reporting", agent="George")

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
TEAM_PHOTO_DIR = Path(__file__).resolve().parent.parent / "static" / "team"
STANDUP_WEEKDAYS = range(0, 5)  # Monday=0 .. Friday=4

# "The Floor" -- the roster shown on the dashboard's own tab. Purely
# descriptive copy about each agent; carries no pipeline behavior.
TEAM_ROSTER = [
    ("wong", "Wong", "Data Agent",
     "Sources daily OHLCV history across the universe: yfinance for equities, ccxt against "
     "Binance for crypto. When a feed fails, falls back to a seeded random walk so the pipeline "
     "never breaks, and marks the result synthetic so no downstream stage mistakes it for a real "
     "result."),
    ("david", "David", "Compliance Agent",
     "Runs data quality checks before any strategy touches the tape: multi-day gaps, zero or "
     "negative prices, stale feeds, and implausible day-over-day moves that suggest an unadjusted "
     "corporate action. Holds standing authority to block a ticker for the day."),
    ("leo", "Leo", "Backtest Agent",
     "Runs every strategy and ticker combination through the vectorized engine with full cost "
     "modeling — commission and slippage charged on position change, so a round trip pays both "
     "ways. Produces the raw in-sample results the rest of the desk works from."),
    ("charles", "Charles", "Risk Agent",
     "Walk-forward validates every candidate across five folds, hard-rejects anything breaching "
     "the drawdown limit, and caps the live roster. Sizes survivors by inverse-volatility risk "
     "parity, with correlated mean-reversion traders sharing a single slot so true exposure isn't "
     "silently doubled."),
    ("greg", "Greg", "Regime Agent",
     "Classifies each ticker independently as trending, choppy, or undecided using ADX, "
     "moving-average slope, and realized volatility percentile. Halves allocation and requires "
     "multi-day confirmation for traders working against their regime — tilts the book, never "
     "benches outright."),
    ("edwin", "Edwin", "Sentiment Agent",
     "Surfaces news sentiment flags across the universe via Alpha Vantage. Strictly "
     "informational: never sizes a position, never vetoes a strategy, never touches capital."),
    ("cornelius", "Cornelius", "Execution Agent",
     "Implements the executor across research, paper, and live modes. Simulates fills against the "
     "persisted ledger, marks positions to market, and enforces the halt conditions that stop "
     "trading before the account does."),
    ("george", "George", "Reporting Agent",
     "Assembles this dashboard from every stage of the pipeline and delivers the daily standup to "
     "Slack. Flags synthetic data and empty-roster days explicitly rather than reporting nothing."),
]


@dataclass
class DashboardResult:
    html: str
    html_path: Path | None
    standup_text: str
    posted_to_slack: bool


class ReportingAgent:
    """George."""

    name = "George"
    role = "Reporting"

    def __init__(self, config: Config):
        self.config = config
        self.reports_dir = config.repo_path("reports")
        self.env = Environment(
            loader=FileSystemLoader(str(TEMPLATE_DIR)),
            autoescape=select_autoescape(["html"]),
        )
        webhook_env_var = config.get("slack.webhook_url_env", "MERIDIAN_SLACK_WEBHOOK_URL")
        self.webhook_url = os.environ.get(webhook_env_var)
        self.post_enabled = bool(config.get("slack.post_daily_standup", True))

    # ------------------------------------------------------------------ run

    def run(self, market: dict[str, MarketData], compliance: dict[str, ComplianceReport],
            results: list[BacktestResult], benchmarks: dict[str, PerformanceSummary],
            risk_report: RiskReport, regime_report: RegimeReport,
            ledger=None, recommendations: list | None = None, strategies: dict | None = None,
            write_file: bool = True, post_slack: bool = True) -> DashboardResult:
        context = self._build_context(market, compliance, results, benchmarks,
                                       risk_report, regime_report, ledger, recommendations,
                                       strategies)
        html = self.env.get_template("report.html.j2").render(**context)

        html_path = None
        if write_file:
            html_path = self._write_html(html)

        standup_text = self.build_standup_text(market, compliance, risk_report, regime_report,
                                                ledger, recommendations)
        posted = False
        if post_slack and self.post_enabled:
            if self._is_standup_day():
                posted = post_message(self.webhook_url, standup_text)
            else:
                log.info("Weekend -- standup post skipped (weekdays only per spec).")

        return DashboardResult(html=html, html_path=html_path,
                               standup_text=standup_text, posted_to_slack=posted)

    def _is_standup_day(self, now: datetime | None = None) -> bool:
        now = now or datetime.now()
        return now.weekday() in STANDUP_WEEKDAYS

    def _write_html(self, html: str) -> Path:
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        dated = self.reports_dir / f"meridian_{datetime.now():%Y%m%d}.html"
        dated.write_text(html, encoding="utf-8")
        latest = self.reports_dir / "latest.html"
        latest.write_text(html, encoding="utf-8")
        log.info("Dashboard written to %s", dated)
        return dated

    # ------------------------------------------------------------------ HTML context

    def _build_context(self, market, compliance, results, benchmarks,
                       risk_report: RiskReport, regime_report: RegimeReport,
                       ledger=None, recommendations: list | None = None,
                       strategies: dict | None = None) -> dict:
        strategies = strategies or {}
        by_key = {(r.strategy, r.symbol): r for r in results}

        data_rows = [{
            "symbol": symbol, "bars": len(data.bars), "source": data.data_source,
            "last_date": data.last_date.date().isoformat() if data.last_date is not None else "n/a",
            "synthetic": data.is_synthetic,
        } for symbol, data in market.items()]

        compliance_rows = [{"symbol": s, "reason": r.reason}
                           for s, r in compliance.items() if r.blocked]

        regime_rows = [{
            "symbol": symbol, "regime": c.regime, "adx": round(c.adx, 1),
            "detail": c.detail,
        } for symbol, c in regime_report.regimes.items()]

        validation_rows = sorted(risk_report.candidates, key=lambda c: (-c.eligible, -c.sharpe))
        validation_rows = [{
            "strategy": c.strategy, "symbol": c.symbol, "eligible": c.eligible,
            "folds_passed": c.walkforward.folds_passed, "folds_total": c.walkforward.folds_total,
            "sharpe": round(c.sharpe, 2), "max_drawdown": round(c.backtest_max_drawdown * 100, 2),
            "reject_reason": c.reject_reason,
        } for c in validation_rows]

        allocation_rows = [{
            "strategy": adj.strategy, "symbol": adj.symbol, "regime": adj.regime,
            "matched": adj.matched, "active": adj.active,
            "base_weight": round(adj.base_weight * 100, 1),
            "adjusted_weight": round(adj.adjusted_weight * 100, 1),
            "note": adj.note,
        } for adj in regime_report.adjustments]

        target_rows = [{
            "symbol": symbol, "target_pct": round(pos.target_weight * 100, 1),
            "target_dollars": round(pos.target_weight * risk_report.starting_capital, 0),
            "capped": pos.capped, "contributors": ", ".join(pos.contributors),
        } for symbol, pos in regime_report.netted_positions.items() if pos.contributors]

        benchmark_rows = [{
            "symbol": symbol, "sharpe": round(s.sharpe, 2),
            "cagr": round(s.cagr * 100, 2), "max_drawdown": round(s.max_drawdown * 100, 2),
        } for symbol, s in benchmarks.items()]

        ledger_summary = self._ledger_summary(ledger, market)
        recommendation_rows = [{
            "trader": r.trader, "trigger": r.trigger, "action": r.action, "detail": r.detail,
        } for r in (recommendations or [])]
        synthetic_symbols = [s for s, d in market.items() if d.is_synthetic]

        strategy_rows, strats_json = self._strategy_rows(strategies, risk_report, by_key)
        all_traders = set(strategies) | {c.strategy for c in risk_report.candidates}

        return {
            "generated_at": datetime.now().strftime("%A, %B %d %Y — %H:%M"),
            "hdr_date": datetime.now().strftime("%b %-d, %Y · %H:%M"),
            "starting_capital": risk_report.starting_capital,
            "universe": list(market),
            "data_rows": data_rows,
            "compliance_rows": compliance_rows,
            "regime_rows": regime_rows,
            "validation_rows": validation_rows,
            "live_traders": risk_report.live_traders,
            "benched_traders": sorted(risk_report.benched_traders.items()),
            "allocation_rows": allocation_rows,
            "target_rows": target_rows,
            "benchmark_rows": benchmark_rows,
            "synthetic_symbols": synthetic_symbols,
            "equity_chart": self._equity_chart(by_key, benchmarks, risk_report, market),
            "drawdown_chart": self._drawdown_chart(by_key, risk_report),
            "trader_bar_chart": self._trader_bar_chart(risk_report),
            "ledger": ledger_summary,
            "recommendation_rows": recommendation_rows,
            "metrics": self._overview_metrics(risk_report, regime_report, benchmarks,
                                               ledger_summary, market, all_traders),
            "strategy_rows": strategy_rows,
            "strats_json": json.dumps(strats_json),
            "agent_cards": self._agent_cards(market, compliance, results, risk_report,
                                              regime_report, ledger_summary, recommendation_rows),
            "sentiment_rows": self._sentiment_rows(market),
            "team": self._team_context(),
        }

    # ------------------------------------------------------------------ overview / strategies / agents

    def _overview_metrics(self, risk_report: RiskReport, regime_report: RegimeReport,
                          benchmarks, ledger_summary, market, all_traders) -> dict:
        if ledger_summary:
            portfolio_value = ledger_summary["equity"]
            portfolio_sub = f"{ledger_summary['total_pnl_pct']:+.1f}% all-time"
            portfolio_sub_class = "up" if ledger_summary["total_pnl_pct"] >= 0 else "dn"
            pnl_value = f"{ledger_summary['total_pnl']:+,.0f}"
            pnl_class = "up" if ledger_summary["total_pnl"] >= 0 else "dn"
            pnl_sub = f"{ledger_summary['trade_count']} trade(s) since inception"
            max_dd = f"{max(0.0, -ledger_summary['total_pnl_pct']):.1f}%"
        else:
            portfolio_value = risk_report.starting_capital
            portfolio_sub = "no live P&L yet"
            portfolio_sub_class = ""
            pnl_value = "—"
            pnl_class = ""
            pnl_sub = "paper trading not started"
            max_dd = "—"

        spy = benchmarks.get("SPY")
        spy_cagr = f"{spy.cagr * 100:.1f}%" if spy else "n/a"
        spy_class = "up" if spy and spy.cagr >= 0 else ("dn" if spy else "")

        max_dd_sub = f"limit {self.config.get('risk.max_strategy_drawdown', 0.25) * 100:.0f}%"

        regimes = [c.regime for c in regime_report.regimes.values()]
        if regimes and len(set(regimes)) == 1:
            regime_label = regimes[0].upper()
        elif regimes:
            regime_label = "MIXED"
        else:
            regime_label = "N/A"

        return {
            "portfolio_value": portfolio_value, "portfolio_sub": portfolio_sub,
            "portfolio_sub_class": portfolio_sub_class,
            "pnl_value": pnl_value, "pnl_class": pnl_class, "pnl_sub": pnl_sub,
            "spy_cagr": spy_cagr, "spy_class": spy_class,
            "max_dd": max_dd, "max_dd_sub": max_dd_sub,
            "traders_live": len(risk_report.live_traders), "traders_total": len(all_traders),
            "regime_label": regime_label, "regime_sub": ", ".join(market),
        }

    def _strategy_rows(self, strategies: dict, risk_report: RiskReport, by_key: dict):
        """One row per callsign -- its best (highest-Sharpe) symbol -- for the
        Strategies tab table, plus a JSON blob the row's detail modal reads."""
        rows, strats = [], {}
        by_strategy: dict[str, list] = {}
        for c in risk_report.candidates:
            by_strategy.setdefault(c.strategy, []).append(c)

        for callsign in sorted(set(strategies) | set(by_strategy)):
            cands = by_strategy.get(callsign, [])
            style = strategies[callsign].style if callsign in strategies else ""
            if callsign in risk_report.live_traders:
                status = "live"
            elif "passed validation" in risk_report.benched_traders.get(callsign, ""):
                status = "watch"
            else:
                status = "bench"

            if not cands:
                rows.append({
                    "callsign": callsign, "style": style, "status": status,
                    "sharpe": "—", "cagr": None, "max_drawdown": 0.0,
                    "walkforward": "0/0", "position_pct": None,
                })
                strats[callsign] = {
                    "style": style, "status": status, "symbol": "—", "sharpe": None,
                    "cagr": None, "max_drawdown": None, "walkforward": "0/0",
                    "position_pct": None, "regime": "—",
                    "reason": risk_report.benched_traders.get(callsign, "no data today"),
                    "reject_reason": "no data today",
                }
                continue

            best = max(cands, key=lambda c: c.sharpe)
            bt = by_key.get((best.strategy, best.symbol))
            cagr = round(bt.summary.cagr * 100, 1) if bt is not None else None
            position_pct = (round(risk_report.capital_weights.get(callsign, 0.0) * 100, 1)
                            if status == "live" else None)

            rows.append({
                "callsign": callsign, "style": style, "status": status,
                "sharpe": round(best.sharpe, 2), "cagr": cagr,
                "max_drawdown": round(best.backtest_max_drawdown * 100, 1),
                "walkforward": f"{best.walkforward.folds_passed}/{best.walkforward.folds_total}",
                "position_pct": position_pct,
            })
            strats[callsign] = {
                "style": style, "status": status, "symbol": best.symbol,
                "sharpe": round(best.sharpe, 2), "cagr": cagr,
                "max_drawdown": round(best.backtest_max_drawdown * 100, 1),
                "walkforward": f"{best.walkforward.folds_passed}/{best.walkforward.folds_total}",
                "position_pct": position_pct, "regime": "—",
                "reason": risk_report.benched_traders.get(callsign, ""),
                "reject_reason": best.reject_reason,
            }
        rows.sort(key=lambda r: ({"live": 0, "watch": 1, "bench": 2}[r["status"]],
                                 -(r["sharpe"] if isinstance(r["sharpe"], (int, float)) else -999)))
        return rows, strats

    def _agent_cards(self, market, compliance, results, risk_report: RiskReport,
                     regime_report: RegimeReport, ledger_summary, recommendation_rows) -> list:
        synthetic = [s for s, d in market.items() if d.is_synthetic]
        blocked = {s: r.reason for s, r in compliance.items() if r.blocked}
        eligible_count = sum(1 for c in risk_report.candidates if c.eligible)

        cards = [
            {"name": "Wong", "role": "Data",
             "message": (f"{', '.join(market)} pulled -- {', '.join(synthetic)} on synthetic "
                        f"fallback, not trusted." if synthetic else
                        f"Pulled {', '.join(market)} -- all sources live, no synthetic fallback "
                        "today."),
             "status_text": "Synthetic fallback in use" if synthetic else "All live",
             "warn": bool(synthetic)},
            {"name": "David", "role": "Compliance",
             "message": ("; ".join(f"{s} blocked ({r})" for s, r in blocked.items())
                        if blocked else "All symbols cleared the data-quality checks. No blocks "
                        "today."),
             "status_text": f"{len(blocked)} blocked" if blocked else "All clear",
             "warn": bool(blocked)},
            {"name": "Leo", "role": "Backtest",
             "message": (f"Ran {len(results)} strategy/ticker combination(s); {eligible_count} "
                        "cleared walk-forward validation today."),
             "status_text": f"{eligible_count}/{len(risk_report.candidates)} eligible",
             "warn": False},
            {"name": "Charles", "role": "Risk",
             "message": ((", ".join(f"{t} {risk_report.capital_weights.get(t, 0.0):.0%}"
                                    for t in risk_report.live_traders))
                        if risk_report.live_traders else
                        "No strategy cleared validation today -- see the Strategies tab."),
             "status_text": (f"{len(risk_report.live_traders)} live" if risk_report.live_traders
                             else "Nothing live"),
             "warn": not risk_report.live_traders},
            {"name": "Greg", "role": "Regime",
             "message": (", ".join(f"{s} {c.regime}" for s, c in regime_report.regimes.items())
                        or "No regime read today."),
             "status_text": "Regime read complete", "warn": False},
            {"name": "Edwin", "role": "Sentiment",
             "message": "Sentiment unavailable today (see full report). Informational only.",
             "status_text": "No reads", "warn": False},
            {"name": "Cornelius", "role": "Execution",
             "message": (f"Paper mode active. {len(ledger_summary['positions'])} open "
                        f"position(s), {ledger_summary['trade_count']} trade(s) since "
                        f"inception. Equity ${ledger_summary['equity']:,.0f}."
                        if ledger_summary else
                        "Research mode -- no execution today. Run 'python main.py paper' to "
                        "trade this allocation."),
             "status_text": "Paper trading" if ledger_summary else "Research only",
             "warn": bool(ledger_summary and ledger_summary["halted"])},
            {"name": "George", "role": "Reporting",
             "message": (f"Daily report assembled and written to disk. "
                        f"{len(recommendation_rows)} lifecycle recommendation(s) awaiting "
                        "operator review." if recommendation_rows else
                        "Daily report assembled and written to disk."),
             "status_text": (f"{len(recommendation_rows)} inquiry(ies)" if recommendation_rows
                             else "Report complete"),
             "warn": bool(recommendation_rows)},
        ]
        return cards

    def _sentiment_rows(self, market: dict[str, MarketData]) -> list:
        """Edwin isn't wired into the pipeline yet (no SentimentAgent exists
        in this codebase) -- shown honestly as unavailable rather than
        fabricating scores."""
        has_key = bool(os.environ.get("ALPHA_VANTAGE_API_KEY"))
        note = "Sentiment agent not yet integrated" if has_key else "ALPHA_VANTAGE_API_KEY not set"
        return [{"symbol": s, "status": "unavailable", "score": "—", "articles": "—", "note": note}
               for s in market]

    def _team_context(self) -> list:
        team = []
        for slug, name, role, desc in TEAM_ROSTER:
            photo_path = TEAM_PHOTO_DIR / "thumb" / f"{slug}.jpg"
            photo = base64.b64encode(photo_path.read_bytes()).decode("ascii") if photo_path.exists() else ""
            team.append({"name": name, "role": role, "desc": desc, "photo": photo})
        return team

    def _ledger_summary(self, ledger, market: dict[str, MarketData]) -> dict | None:
        """Real paper P&L, once Cornelius has actually traded -- None in
        research mode, where there is nothing to show but a target."""
        if ledger is None:
            return None
        prices = {s: float(d.bars["close"].iloc[-1]) for s, d in market.items() if not d.bars.empty}
        equity = ledger.mark_to_market(prices)
        position_rows = [{
            "symbol": symbol, "shares": round(pos.shares, 4),
            "entry_price": round(pos.entry_price, 2),
            "price": round(prices.get(symbol, pos.entry_price), 2),
            "days_held": pos.days_held(),
            "market_value": round(pos.market_value(prices.get(symbol, pos.entry_price)), 2),
            "unrealized_pnl": round(pos.unrealized_pnl(prices.get(symbol, pos.entry_price)), 2),
        } for symbol, pos in ledger.positions.items()]
        total_pnl = equity - ledger.starting_capital
        return {
            "cash": round(ledger.cash, 2), "equity": round(equity, 2),
            "starting_capital": ledger.starting_capital,
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round((total_pnl / ledger.starting_capital) * 100, 2)
                            if ledger.starting_capital else 0.0,
            "positions": position_rows,
            "halted": ledger.halted, "halt_reason": ledger.halt_reason,
            "trade_count": len(ledger.trades),
        }

    # ------------------------------------------------------------------ charts

    def _pick_headline(self, risk_report: RiskReport):
        """The clearest single (trader, symbol) story for the equity/drawdown
        charts: the best-Sharpe live, eligible candidate, if any."""
        eligible = [c for c in risk_report.candidates
                   if c.eligible and c.strategy in risk_report.live_traders]
        if not eligible:
            return None
        return max(eligible, key=lambda c: c.sharpe)

    def _equity_chart(self, by_key, benchmarks, risk_report, market) -> str:
        series = {}
        headline = self._pick_headline(risk_report)
        if headline is not None:
            bt = by_key.get((headline.strategy, headline.symbol))
            if bt is not None and len(bt.equity) > 1:
                series[f"{headline.strategy}/{headline.symbol}"] = bt.equity.round(4).tolist()
        title = "Equity curve (backtest, in-sample) vs. SPY buy-and-hold"
        if "SPY" in market and len(market["SPY"].bars) > 1:
            close = market["SPY"].bars["close"]
            series["SPY buy-and-hold"] = (close / close.iloc[0]).round(4).tolist()
        return line_chart(series, title=title)

    def _drawdown_chart(self, by_key, risk_report) -> str:
        headline = self._pick_headline(risk_report)
        if headline is None:
            return line_chart({}, title="Drawdown (backtest, in-sample)")
        bt = by_key.get((headline.strategy, headline.symbol))
        if bt is None or bt.drawdown.empty:
            return line_chart({}, title="Drawdown (backtest, in-sample)")
        series = {f"{headline.strategy}/{headline.symbol}": (bt.drawdown * 100).round(2).tolist()}
        return line_chart(series, title="Drawdown %, backtest in-sample")

    def _trader_bar_chart(self, risk_report: RiskReport) -> str:
        live = risk_report.live_traders
        if not live:
            return bar_chart([], [], title="Live trader CAGR (backtest, in-sample)")
        best_by_trader: dict[str, float] = {}
        for c in risk_report.candidates:
            if c.strategy in live and c.eligible:
                best_by_trader[c.strategy] = max(best_by_trader.get(c.strategy, float("-inf")),
                                                 c.sharpe)
        labels = list(best_by_trader)
        values = [round(best_by_trader[t], 2) for t in labels]
        return bar_chart(labels, values, title="Live trader Sharpe (backtest, in-sample)", unit="")

    # ------------------------------------------------------------------ standup

    def build_standup_text(self, market: dict[str, MarketData],
                           compliance: dict[str, ComplianceReport],
                           risk_report: RiskReport, regime_report: RegimeReport,
                           ledger=None, recommendations: list | None = None) -> str:
        lines = []

        synthetic = [s for s, d in market.items() if d.is_synthetic]
        if synthetic:
            lines.append(f"\U0001F50D Wong (Data): {', '.join(market)} pulled -- "
                         f"{', '.join(synthetic)} on synthetic fallback, not trusted.")
        else:
            lines.append(f"\U0001F50D Wong (Data): Pulled {', '.join(market)} -- "
                         f"all sources live, no synthetic fallback today.")

        blocked = {s: r.reason for s, r in compliance.items() if r.blocked}
        if blocked:
            detail = "; ".join(f"{s} ({r})" for s, r in blocked.items())
            lines.append(f"\U0001F6D1 David (Compliance): Blocked {detail}.")
        else:
            lines.append("\U0001F6D1 David (Compliance): No blocks.")

        if risk_report.live_traders:
            weights = ", ".join(f"{t} {risk_report.capital_weights.get(t, 0.0):.0%}"
                                for t in risk_report.live_traders)
            lines.append(f"⚖️ Charles (Risk): Live today: {weights}.")
        else:
            lines.append("⚖️ Charles (Risk): No strategy cleared validation today "
                         "-- closest contenders in the full report.")

        if regime_report.regimes:
            regimes = ", ".join(f"{s} {c.regime}" for s, c in regime_report.regimes.items())
            lines.append(f"\U0001F9ED Greg (Regime): {regimes} today.")

        for rec in (recommendations or []):
            lines.append(f"\U0001FA91 Lifecycle ({rec.trigger}): {rec.trader} -- "
                         f"{rec.action.upper()}. {rec.detail} Operator approval required "
                         "before anything actually changes.")

        if ledger is not None:
            prices = {s: float(d.bars["close"].iloc[-1]) for s, d in market.items()
                      if not d.bars.empty}
            equity = ledger.mark_to_market(prices)
            pnl_pct = ((equity - ledger.starting_capital) / ledger.starting_capital
                      if ledger.starting_capital else 0.0)
            halt = "  ⛔ HALTED" if ledger.halted else ""
            lines.append(f"\U0001F4CA George (Reporting): Paper equity ${equity:,.0f} "
                         f"({pnl_pct:+.1%} since inception), {len(ledger.positions)} open "
                         f"position(s).{halt} Full report attached.")
        elif risk_report.live_traders:
            lines.append("\U0001F4CA George (Reporting): Target allocation ready -- no live "
                         "P&L yet (paper trading begins Phase 4). Full report attached.")
        else:
            lines.append("\U0001F4CA George (Reporting): Full report attached.")

        return "\n".join(lines)
