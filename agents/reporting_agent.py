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
from strategies.base import Strategy
from utils.config import Config
from utils.logging_setup import get_logger
from utils.metrics import PerformanceSummary
from utils.slack import post_message
from utils.svg_charts import bar_chart, line_chart

log = get_logger("reporting", agent="George")

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
STANDUP_WEEKDAYS = range(0, 5)  # Monday=0 .. Friday=4


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
            sentiment_report: dict | None = None, ledger=None,
            recommendations: list | None = None, strategies: dict[str, Strategy] | None = None,
            write_file: bool = True, post_slack: bool = True) -> DashboardResult:
        sentiment_report = sentiment_report or {}
        context = self._build_context(market, compliance, results, benchmarks,
                                       risk_report, regime_report, sentiment_report,
                                       ledger, recommendations, strategies or {})
        html = self.env.get_template("report.html.j2").render(**context)

        html_path = None
        if write_file:
            html_path = self._write_html(html)

        standup_text = self.build_standup_text(market, compliance, risk_report, regime_report,
                                                sentiment_report, ledger, recommendations)
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
                       sentiment_report: dict | None = None,
                       ledger=None, recommendations: list | None = None,
                       strategies: dict[str, Strategy] | None = None) -> dict:
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

        sentiment_rows = [{
            "symbol": symbol, "available": snap.available, "label": snap.label,
            "score": round(snap.score, 2), "article_count": snap.article_count,
            "top_headline": snap.top_headline, "note": snap.note,
        } for symbol, snap in (sentiment_report or {}).items()]

        ledger_summary = self._ledger_summary(ledger, market)
        recommendation_rows = [{
            "trader": r.trader, "trigger": r.trigger, "action": r.action, "detail": r.detail,
        } for r in (recommendations or [])]

        roster_rows, strategy_detail = self._roster(by_key, risk_report, regime_report,
                                                     strategies, recommendations or [])
        trade_rows = self._trade_rows(ledger, market, regime_report)
        overview = self._overview_metrics(ledger, risk_report, regime_report, benchmarks)
        agent_notes = self._agent_notes(market, compliance, risk_report, regime_report,
                                        sentiment_report, ledger, recommendations or [])

        return {
            "generated_at": datetime.now().strftime("%A, %B %d %Y — %H:%M"),
            "generated_short": datetime.now().strftime("%b %d, %Y · %H:%M"),
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
            "sentiment_rows": sentiment_rows,
            "synthetic_symbols": [s for s, d in market.items() if d.is_synthetic],
            "equity_chart": self._equity_chart(by_key, benchmarks, risk_report, market),
            "drawdown_chart": self._drawdown_chart(by_key, risk_report),
            "trader_bar_chart": self._trader_bar_chart(risk_report),
            "ledger": ledger_summary,
            "recommendation_rows": recommendation_rows,
            "roster_rows": roster_rows,
            "strategy_detail_json": json.dumps(strategy_detail),
            "trade_rows": trade_rows,
            "overview": overview,
            "agent_notes": agent_notes,
        }

    # ------------------------------------------------------------------ roster (Strategies tab)

    def _roster(self, by_key, risk_report: RiskReport, regime_report: RegimeReport,
               strategies: dict[str, Strategy], recommendations: list) -> tuple[list, dict]:
        """One row per known trader -- LIVE/WATCH/BENCHED/FLAT -- plus a
        JSON-able detail blob per callsign for the strategy modal. WATCH means
        it cleared walk-forward on at least one symbol but isn't sized today
        (capped roster or a correlated slot already spoken for); FLAT means it
        hasn't cleared validation on anything right now."""
        by_strategy: dict[str, list] = {}
        for c in risk_report.candidates:
            by_strategy.setdefault(c.strategy, []).append(c)
        regime_by_strategy = {adj.strategy: adj for adj in regime_report.adjustments}
        rec_by_trader = {r.trader: r for r in recommendations}

        rows, detail = [], {}
        for callsign, strat in strategies.items():
            candidates = by_strategy.get(callsign, [])
            eligible = [c for c in candidates if c.eligible]
            best = (max(eligible, key=lambda c: c.sharpe) if eligible else
                   (max(candidates, key=lambda c: c.sharpe) if candidates else None))
            bt = by_key.get((callsign, best.symbol)) if best else None

            if callsign in risk_report.live_traders:
                status = "live"
            elif callsign in risk_report.benched_traders:
                status = "benched"
            elif eligible:
                status = "watch"
            else:
                status = "flat"

            adj = regime_by_strategy.get(callsign)
            regime_note = (f"{adj.regime} — {'matched' if adj.matched else 'mismatched'}"
                          if adj else "—")
            sharpe = round(best.sharpe, 2) if best else None
            cagr = round(bt.summary.cagr * 100, 1) if bt else None
            max_dd = round(best.backtest_max_drawdown * 100, 1) if best else None
            wf = f"{best.walkforward.folds_passed}/{best.walkforward.folds_total}" if best else "—"
            pos_pct = risk_report.capital_weights.get(callsign, 0.0) * 100
            reason = (risk_report.benched_traders.get(callsign, "") if status == "benched" else
                      (rec_by_trader[callsign].detail if callsign in rec_by_trader else ""))

            rows.append({
                "callsign": callsign, "style": strat.style or strat.family or "—",
                "status": status, "symbol": best.symbol if best else "—",
                "sharpe": sharpe, "cagr": cagr, "max_drawdown": max_dd, "walkforward": wf,
                "position_pct": round(pos_pct, 1) if status == "live" else None,
                "regime": regime_note,
            })
            detail[callsign] = {
                "style": strat.style or strat.family or "—", "status": status,
                "symbol": best.symbol if best else "—",
                "sharpe": sharpe, "cagr": cagr, "max_drawdown": max_dd, "walkforward": wf,
                "position_pct": round(pos_pct, 1) if status == "live" else None,
                "regime": regime_note, "reason": reason,
                "reject_reason": best.reject_reason if best and not best.eligible else "",
            }
        rows.sort(key=lambda r: ({"live": 0, "watch": 1, "benched": 2, "flat": 3}[r["status"]],
                                  -(r["sharpe"] or float("-inf"))))
        return rows, detail

    # ------------------------------------------------------------------ trades tab

    def _trade_rows(self, ledger, market: dict[str, MarketData], regime_report: RegimeReport) -> list:
        """Open paper positions -- the only thing Meridian actually has a fill
        for. Contributors and target weight come straight from Greg's netted
        positions so the card explains *why* the position is sized as it is."""
        if ledger is None or not ledger.positions:
            return []
        prices = {s: float(d.bars["close"].iloc[-1]) for s, d in market.items() if not d.bars.empty}
        equity = ledger.mark_to_market(prices)
        rows = []
        for symbol, pos in ledger.positions.items():
            price = prices.get(symbol, pos.entry_price)
            market_value = pos.market_value(price)
            pnl = pos.unrealized_pnl(price)
            net = regime_report.netted_positions.get(symbol)
            rows.append({
                "symbol": symbol, "contributors": ", ".join(net.contributors) if net else "—",
                "shares": round(pos.shares, 4), "entry_price": round(pos.entry_price, 2),
                "price": round(price, 2), "days_held": pos.days_held(),
                "market_value": round(market_value, 2), "unrealized_pnl": round(pnl, 2),
                "unrealized_pnl_pct": round(pnl / (pos.shares * pos.entry_price) * 100, 2)
                                     if pos.shares and pos.entry_price else 0.0,
                "portfolio_pct": round(market_value / equity * 100, 1) if equity else 0.0,
                "target_pct": round(net.target_weight * 100, 1) if net else None,
            })
        return sorted(rows, key=lambda r: -abs(r["market_value"]))

    # ------------------------------------------------------------------ overview tab

    def _overview_metrics(self, ledger, risk_report: RiskReport, regime_report: RegimeReport,
                          benchmarks: dict) -> dict:
        live_n = len(risk_report.live_traders)
        total_n = live_n + len(risk_report.benched_traders)
        regimes = list(regime_report.regimes.values())
        if regimes and len(set(c.regime for c in regimes)) == 1:
            regime_headline = regimes[0].regime
        elif regimes:
            regime_headline = "mixed"
        else:
            regime_headline = "undecided"

        spy = benchmarks.get("SPY")
        vs_spy = round(spy.cagr * 100, 2) if spy else None

        if ledger is not None:
            history = ledger.equity_history
            equity = history[-1]["equity"] if history else ledger.starting_capital
            today_pnl = (equity - history[-2]["equity"]) if len(history) >= 2 else 0.0
            today_pnl_pct = (today_pnl / history[-2]["equity"] * 100
                            if len(history) >= 2 and history[-2]["equity"] else 0.0)
            running_max = 0.0
            max_dd = 0.0
            for row in history:
                running_max = max(running_max, row["equity"])
                if running_max:
                    max_dd = min(max_dd, (row["equity"] - running_max) / running_max)
            total_pnl_pct = ((equity - ledger.starting_capital) / ledger.starting_capital * 100
                             if ledger.starting_capital else 0.0)
        else:
            equity = risk_report.starting_capital
            today_pnl = today_pnl_pct = 0.0
            max_dd = 0.0
            total_pnl_pct = 0.0

        return {
            "has_ledger": ledger is not None,
            "portfolio_value": round(equity, 0),
            "total_pnl_pct": round(total_pnl_pct, 2),
            "today_pnl": round(today_pnl, 2),
            "today_pnl_pct": round(today_pnl_pct, 2),
            "vs_spy": vs_spy,
            "max_drawdown_pct": round(max_dd * 100, 2),
            "max_drawdown_limit_pct": round(float(self.config.get("risk.max_portfolio_drawdown")) * 100, 0),
            "live_traders": live_n, "total_traders": total_n,
            "regime_headline": regime_headline,
        }

    # ------------------------------------------------------------------ agents tab

    def _agent_notes(self, market, compliance, risk_report: RiskReport,
                     regime_report: RegimeReport, sentiment_report: dict,
                     ledger, recommendations: list) -> list:
        """One card per desk agent -- the same facts as the Slack standup,
        split out and given a status dot instead of squeezed into one line."""
        notes = []

        synthetic = [s for s, d in market.items() if d.is_synthetic]
        notes.append({
            "name": "Wong", "role": "Data", "warn": bool(synthetic),
            "message": (f"{', '.join(synthetic)} on synthetic fallback -- not trusted."
                       if synthetic else f"Pulled {', '.join(market)}. All sources live, "
                       "no synthetic fallback today."),
            "status": "Synthetic fallback in use" if synthetic else "All feeds live",
        })

        blocked = {s: r.reason for s, r in compliance.items() if r.blocked}
        notes.append({
            "name": "David", "role": "Compliance", "warn": bool(blocked),
            "message": ("; ".join(f"{s}: {r}" for s, r in blocked.items()) if blocked
                       else "All symbols cleared the data-quality checks. No blocks today."),
            "status": f"{len(blocked)} blocked" if blocked else "All clear",
        })

        n_candidates = len(risk_report.candidates)
        n_eligible = sum(1 for c in risk_report.candidates if c.eligible)
        notes.append({
            "name": "Leo", "role": "Backtest", "warn": False,
            "message": f"Ran {n_candidates} strategy/ticker combination(s); "
                      f"{n_eligible} cleared walk-forward validation today.",
            "status": f"{n_eligible}/{n_candidates} eligible",
        })

        if risk_report.live_traders:
            weights = ", ".join(f"{t} {risk_report.capital_weights.get(t, 0.0):.0%}"
                                for t in risk_report.live_traders)
            charles_msg = f"Live today: {weights}."
        else:
            charles_msg = "No strategy cleared validation today -- see the Strategies tab."
        notes.append({
            "name": "Charles", "role": "Risk", "warn": not risk_report.live_traders,
            "message": charles_msg,
            "status": f"{len(risk_report.live_traders)} live" if risk_report.live_traders
                     else "Nothing live",
        })

        regimes = ", ".join(f"{s} {c.regime}" for s, c in regime_report.regimes.items())
        notes.append({
            "name": "Greg", "role": "Regime", "warn": False,
            "message": f"{regimes}." if regimes else "No symbols to classify today.",
            "status": "Regime read complete",
        })

        available = {s: snap for s, snap in (sentiment_report or {}).items() if snap.available}
        if available:
            reads = ", ".join(f"{s} {snap.label}" for s, snap in available.items())
            edwin_msg = f"{reads}. Informational only -- never gates a trade."
        elif sentiment_report:
            edwin_msg = "Sentiment unavailable today (see full report). Informational only."
        else:
            edwin_msg = "Sentiment disabled or not configured."
        notes.append({
            "name": "Edwin", "role": "Sentiment", "warn": False,
            "message": edwin_msg, "status": f"{len(available)} read(s)" if available else "No reads",
        })

        if ledger is not None:
            prices = {s: float(d.bars["close"].iloc[-1]) for s, d in market.items()
                     if not d.bars.empty}
            equity = ledger.mark_to_market(prices)
            cornelius_msg = (f"Paper mode active. {len(ledger.positions)} open position(s), "
                            f"{len(ledger.trades)} trade(s) since inception. Equity ${equity:,.0f}.")
            if ledger.halted:
                cornelius_msg += f" ⛔ HALTED: {ledger.halt_reason}."
        else:
            cornelius_msg = "Research mode -- no execution. Run 'python main.py paper' to trade this allocation."
        notes.append({
            "name": "Cornelius", "role": "Execution", "warn": ledger is not None and ledger.halted,
            "message": cornelius_msg,
            "status": "HALTED" if (ledger is not None and ledger.halted) else
                     ("Paper trading" if ledger is not None else "No execution"),
        })

        n_recs = len(recommendations)
        george_msg = "Daily report assembled and written to disk."
        if n_recs:
            george_msg += f" {n_recs} lifecycle recommendation(s) awaiting operator review."
        notes.append({
            "name": "George", "role": "Reporting", "warn": bool(n_recs),
            "message": george_msg,
            "status": f"{n_recs} inquiry(ies)" if n_recs else "Report ready",
        })

        return notes

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
                           sentiment_report: dict | None = None,
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

        available = {s: snap for s, snap in (sentiment_report or {}).items() if snap.available}
        if available:
            reads = ", ".join(f"{s} {snap.label}" for s, snap in available.items())
            lines.append(f"\U0001F4F0 Edwin (Sentiment): {reads} -- informational only, "
                         "never gates a trade.")
        elif sentiment_report:
            lines.append("\U0001F4F0 Edwin (Sentiment): unavailable today "
                         "(see full report) -- informational only, no impact on trading.")

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
