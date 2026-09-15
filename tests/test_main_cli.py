"""Phase A4 tests — `paper-options` CLI wiring: Augustus -> Theo -> Joseph
end-to-end, plus the `--options` flag on killswitch/clear-halt.

Every test redirects the options ledger to a temp path (same trick
`test_options_broker.py` uses) so nothing here touches a real ledger file.
Network is not mocked: `allow_synthetic_fallback` is on by default, so these
pass identically whether or not yfinance is reachable from this machine.

Run with:  python -m pytest tests/test_main_cli.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main  # noqa: E402
from utils.config import Config, load_config  # noqa: E402

BASE_CONFIG = load_config()


def _options_config(tmp_path, **overrides) -> Config:
    """A full config with the options ledger redirected to a temp file, plus
    any dotted-key overrides under the `options` section."""
    data = BASE_CONFIG.as_dict()
    data["options"]["ledger_path"] = str(tmp_path / "options_ledger.json")
    for key, value in overrides.items():
        data["options"][key] = value
    return Config(data)


# ------------------------------------------------------------------ argparse


def test_parser_accepts_paper_options_mode():
    args = main.build_parser().parse_args(["paper-options"])
    assert args.mode == "paper-options"


def test_parser_accepts_options_flag_on_killswitch():
    args = main.build_parser().parse_args(["killswitch", "--options"])
    assert args.mode == "killswitch"
    assert args.options is True


def test_options_flag_defaults_false():
    args = main.build_parser().parse_args(["killswitch"])
    assert args.options is False


# ------------------------------------------------------------------ run_paper_options


def test_run_paper_options_is_noop_when_disabled(tmp_path):
    cfg = _options_config(tmp_path, enabled=False)
    assert main.run_paper_options(cfg, symbols=["SPY"]) is None


def test_run_paper_options_runs_end_to_end_with_no_strategy(tmp_path):
    """Phase A4: empty pipeline. No proposals exist yet, so the run must
    fetch data, settle nothing (fresh ledger), and persist a ledger with
    zero open positions -- proving the Augustus -> Theo -> Joseph wiring
    works without opening anything."""
    cfg = _options_config(tmp_path, enabled=True)
    ledger = main.run_paper_options(cfg, symbols=["SPY"])
    assert ledger is not None
    assert ledger.positions == {}
    assert ledger.cash == ledger.starting_capital
    assert not ledger.halted


def test_run_paper_options_persists_the_ledger_across_runs(tmp_path):
    cfg = _options_config(tmp_path, enabled=True)
    first = main.run_paper_options(cfg, symbols=["SPY"])
    second = main.run_paper_options(cfg, symbols=["SPY"])
    assert first.created_at == second.created_at  # same ledger file, not recreated


# ------------------------------------------------------------------ killswitch / clear-halt


def test_killswitch_options_flag_routes_to_joseph_not_cornelius(tmp_path):
    """The stock ledger must not exist/move; only the options ledger halts."""
    cfg = _options_config(tmp_path, enabled=True)
    main.run_paper_options(cfg, symbols=["SPY"])  # create the options ledger
    assert main.run_killswitch(cfg, symbols=["SPY"], options=True) == 0

    from agents.options_broker import OptionsBroker
    ledger = OptionsBroker(cfg).load_ledger()
    assert ledger.halted
    assert ledger.halt_reason == "operator killswitch"


def test_clear_halt_options_flag_clears_only_the_options_halt(tmp_path):
    cfg = _options_config(tmp_path, enabled=True)
    main.run_paper_options(cfg, symbols=["SPY"])
    main.run_killswitch(cfg, symbols=["SPY"], options=True)

    assert main.run_clear_halt(cfg, options=True) == 0

    from agents.options_broker import OptionsBroker
    ledger = OptionsBroker(cfg).load_ledger()
    assert not ledger.halted
    assert ledger.halt_reason == ""


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
