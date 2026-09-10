# Meridian Capital — Phase 1 (Foundation)

Multi-agent trading research desk. This repo currently implements **Phase 1** of the
build spec: a runnable research pipeline that fetches data, backtests all six traders
against all three tickers with cost modeling, and prints a ranked results table.

Not financial advice. Backtests overfit. Markets change.

## What works right now

| Component | Status |
|---|---|
| Repo structure, config loader, logging, metrics | Done |
| Wong (DataAgent) — yfinance / ccxt + synthetic fallback | Done |
| Strategy base class + all 6 traders | Done |
| Vectorized backtester with cost modeling | Done |
| Leo (BacktestAgent) — runs the full cross product | Done |
| `research` CLI mode with console report | Done |
| Walk-forward, risk, regime, reporting, execution | Phases 2–5 |

`paper`, `live`, and `killswitch` are wired into the CLI but refuse to run and tell you
which phase they arrive in. Live mode's three gates are not bypassable and are not
implemented yet either way.

## Setup

```bash
pip install -r requirements.txt
python main.py research
```

Optional flags: `--symbols SPY QQQ`, `--config path/to/config.yaml`, `--verbose`.

Tests: `python tests/test_phase1.py` (or `python -m pytest tests -q`).

## Layout

```
main.py                     Orchestrator CLI
config/config.yaml          Every tunable parameter
agents/data_agent.py        Wong — fetch + synthetic fallback
agents/backtest_agent.py    Leo — runs all strategy/ticker combinations
strategies/base.py          Abstract base, registry, next-bar enforcement
strategies/{orbit,flux,revert,surge,spark,anchor}.py
backtester/engine.py        Vectorized engine + CostModel
utils/{config,logging_setup,metrics}.py
tests/test_phase1.py        Invariant tests
```

## Design decisions worth knowing before Phase 2

**Next-bar execution is enforced centrally.** `generate_signals()` returns the decision
made at a bar's close; the one-bar shift happens once in `Strategy.positions()`. No
individual strategy can introduce lookahead by forgetting to shift. `test_no_lookahead`
verifies this by tampering with the final bar and asserting no earlier position moves.

**Costs are charged on position change, not trade count.** Turnover × one-way bps, per
bar. A full round trip therefore pays twice, which is how the spec's 20 bps (stocks) /
40 bps (crypto) round-trip figures come out. The buy-and-hold benchmark is charged one
one-way cost for its initial purchase so it isn't unfairly advantaged.

**Warmup is hard-zeroed.** Each strategy declares how many bars it needs; the engine
forces position to zero across that span regardless of what the indicator says, and
refuses the run outright if history is shorter than warmup. A too-short run comes back
`blocked=True` with a reason rather than as a misleading row of zeros.

**Synthetic data is contagious by design.** If a feed fails, Wong generates a seeded
random walk so the pipeline never breaks, but every result carries
`data_source="synthetic"` and `is_trusted=False`. The seed is mixed with the symbol so
SPY and QQQ get different series, and repeated runs on a broken feed produce identical
output rather than fresh noise each day.

**`_hold()` state machine for asymmetric rules.** REVERT, SURGE, SPARK, and ANCHOR have
entry conditions that differ from the negation of their exit conditions — "RSI is not
above 55" is not the same as "RSI dropped below 30". Those use an explicit hold-state
loop rather than a vectorized comparison.

**Donchian channels exclude the current bar.** SURGE and SPARK compare today's close to
the prior N-day extreme (`.shift(1)`), otherwise today's high would be part of the high
it's supposed to break.

## Caveats on the current output

Every Sharpe the research mode prints is **in-sample**. There is no walk-forward
validation until Phase 2, so a good-looking number here is exactly the curve-fit the
spec warns about. The console report says so on every run.

In a sandboxed environment with no outbound access to Yahoo or Binance, all three
symbols fall back to synthetic data and the report flags them. That path is working as
intended — it is not a data bug.

## Next: Phase 2

`backtester/walkforward.py` (5 folds, 3-of-5 rule), `agents/compliance_agent.py`
(David's four data-quality checks), and `agents/risk_agent.py` (Charles: risk parity,
position sizing, the shared ANCHOR/REVERT slot). The `blocked` parameter on
`BacktestAgent.run_all()` is already in place for David to populate.
