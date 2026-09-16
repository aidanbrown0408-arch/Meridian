"""George's options section: the summary snapshot, the paper-options refresh
of latest.html, and the stock dashboard picking the snapshot back up.

Every test points George at a temp directory so the real reports/ folder
is never touched.

Run with:  python -m pytest tests -q      (or)  python tests/test_options_dashboard.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.options_broker import OptionsLedger, OptionsPosition  # noqa: E402
from agents.reporting_agent import ReportingAgent  # noqa: E402
from strategies.options_strategy import SignalCheck  # noqa: E402
from utils.config import load_config  # noqa: E402

CONFIG = load_config()


def _george(tmp: str) -> ReportingAgent:
    george = ReportingAgent(CONFIG)
    george.reports_dir = Path(tmp)
    return george


def _ledger_with_position() -> OptionsLedger:
    exp = (date.today() + timedelta(days=20)).isoformat()
    pos = OptionsPosition(underlying="SPY", option_type="long_call", strike=560.0,
                          expiration=exp, contracts=1, premium_paid=120.0,
                          entry_premium_per_contract=120.0,
                          entry_date=datetime.now(timezone.utc).isoformat(),
                          reason="SPARK long")
    ledger = OptionsLedger(starting_capital=1000.0, cash=880.0,
                           positions={pos.key: pos},
                           equity_history=[{"date": "2026-09-15", "equity": 1000.0},
                                           {"date": "2026-09-16", "equity": 1000.0},
                                           {"date": "2026-09-16", "equity": 1010.0}])
    return ledger


def test_summary_marks_positions_and_is_json_safe():
    ledger = _ledger_with_position()
    key = next(iter(ledger.positions))
    checks = [SignalCheck("SPY", "SPARK", True, "close broke 20d high")]
    summary = ReportingAgent.build_options_summary(ledger, {key: 150.0}, checks=checks,
                                                   halt_floor=500.0)
    assert summary["equity"] == 1030.0
    assert summary["total_pnl"] == 30.0
    assert summary["positions"][0]["pnl"] == 30.0 and summary["positions"][0]["quoted"]
    assert summary["signals"][0]["signal_long"] is True
    assert summary["equity_chart"]  # two distinct dates -> a chart
    json.dumps(summary)


def test_unquoted_position_is_marked_at_cost():
    ledger = _ledger_with_position()
    summary = ReportingAgent.build_options_summary(ledger, {})
    row = summary["positions"][0]
    assert not row["quoted"] and row["pnl"] == 0.0
    assert summary["equity"] == 1000.0


def test_refresh_without_stock_run_renders_options_only():
    with tempfile.TemporaryDirectory() as tmp:
        george = _george(tmp)
        summary = ReportingAgent.build_options_summary(_ledger_with_position(), {})
        path = george.refresh_options(summary)
        html = path.read_text()
        assert path.name == "latest.html"
        assert "Options Bucket" in html and "SPY" in html
        assert "No stock run cached yet" in html
        assert (Path(tmp) / "options_status.json").exists()


def test_refresh_reuses_cached_stock_context():
    with tempfile.TemporaryDirectory() as tmp:
        george = _george(tmp)
        stock = george._empty_stock_context()
        stock.pop("stock_note")
        stock["generated_at"] = "Monday, September 14 2026 — 16:30"
        stock["live_traders"] = ["ORBIT"]
        george._save_stock_context(stock)

        summary = ReportingAgent.build_options_summary(_ledger_with_position(), {})
        html = george.refresh_options(summary).read_text()
        assert "from the Monday, September 14 2026 — 16:30 run" in html
        assert "ORBIT" in html
        assert "Options Bucket" in html


def test_no_snapshot_means_no_options_section():
    with tempfile.TemporaryDirectory() as tmp:
        george = _george(tmp)
        assert george._load_options_summary() is None
        context = george._empty_stock_context()
        context["options"] = None
        html = george.env.get_template("report.html.j2").render(**context)
        assert "Options Bucket" not in html


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except Exception as exc:
                failures += 1
                print(f"FAIL  {name}: {exc!r}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
