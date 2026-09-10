# Meridian Capital — Phase 2 (Validation & Risk)

Multi-agent trading research desk. This repo now implements **Phases 1–2** of the
build spec: a runnable research pipeline that fetches data, blocks anything with
broken data, backtests all six traders against all three tickers with cost modeling,
walk-forward validates the survivors, and turns passing traders into a risk-parity
capital allocation — all printed as a console report.

Not financial advice. Backtests overfit. Markets change.

## What works right now

| Component | Status |
|---|---|
| Repo structure, config loader, logging, metrics | Done |
| Wong (DataAgent) — yfinance / ccxt + synthetic fallback | Done |
| Strategy base class + all 6 traders | Done |
| Vectorized backtester with cost modeling | Done |
| Leo (BacktestAgent) — runs the full cross product | Done |
| Walk-forward validation — 5 folds, 3-of-5 majority rule | Done |
| David (ComplianceAgent) — four data-quality checks, blocks per symbol | Done |
| Charles (RiskAgent) — walk-forward gate, drawdown filter, max-3-live roster, risk parity, correlated ANCHOR/REVERT slot, same-ticker netting | Done |
| `research` CLI mode with console report | Done |
| Regime detection, HTML/Slack reporting, execution, lifecycle | Phases 3–5 |

`paper`, `live`, and `killswitch` are wired into the CLI but refuse to run and tell you
which phase they arrive in. Live mode's three gates are not bypassable and are not
implemented yet either way.

## Setup

```bash
pip install -r requirements.txt
python main.py research
```

Optional flags: `--symbols SPY QQQ`, `--config path/to/config.yaml`, `--verbose`.

Tests: `python tests/test_phase1.py && python tests/test_phase2.py`
(or `python -m pytest tests -q`).

## Layout

```
main.py                       Orchestrator CLI
config/config.yaml            Every tunable parameter
agents/data_agent.py          Wong — fetch + synthetic fallback
agents/compliance_agent.py    David — data-quality checks, per-symbol blocking
agents/backtest_agent.py      Leo — runs all strategy/ticker combinations
agents/risk_agent.py          Charles — walk-forward gate, risk parity, netting
strategies/base.py            Abstract base, registry, next-bar enforcement
strategies/{orbit,flux,revert,surge,spark,anchor}.py
backtester/engine.py          Vectorized engine + CostModel
backtester/walkforward.py     5-fold walk-forward validator
utils/{config,logging_setup,metrics}.py
tests/test_phase1.py          Phase 1 invariant tests
tests/test_phase2.py          Phase 2 invariant tests
```

## Design decisions worth knowing

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

**Walk-forward folds are scored out-of-sample without a fitting step.** Meridian's
traders have fixed parameters, so "training window" doesn't mean parameter search — it
means the strategy's indicators warm up over history that predates the fold, and only
the fold's own bars are ever scored. `test_walkforward_no_lookahead_across_folds`
verifies a fold's grade doesn't change when only *later* folds' data is tampered with.

**Majority (3-of-5), not strict or lenient.** Strict (5/5) means nothing ever trades;
lenient (average-of-folds) lets one great fold hide four losers. The bar is per-fold
Sharpe/drawdown/trade-count, graded independently — see `WalkForwardValidator._grade`.

**The per-strategy drawdown filter is a hard reject, not folded into walk-forward.**
Per spec §7, a strategy whose full-period max drawdown exceeds `risk.max_strategy_drawdown`
is rejected "before any capital is assigned" even if it cleared 3-of-5 folds — two
independent gates, both must pass.

**A "trader" is a callsign, not a (strategy, symbol) pair.** ORBIT can be validated on
SPY and BTC/USDT independently; the top-3-by-Sharpe cap in `risk.max_live_traders`
counts distinct callsigns, ranked by the best Sharpe among each trader's passing
symbols. A live trader still only sizes positions on the individual symbols it cleared
walk-forward on — being live overall doesn't license it to trade a symbol it failed.

**ANCHOR and REVERT share one risk-parity slot.** Per spec §7, both are mean-reversion
traders that tend to trigger on the same conditions; treating them as independent slots
would silently double true exposure to that bet. `RiskAgent._build_slots` merges them
into one slot when both are live, and their combined weight is capped at what a single
solo trader would receive (`test_risk_agent_correlated_pair_shares_one_slot`).

**Netting happens on each trader's already-shifted position, not a fresh signal.**
`RiskAgent._net_positions` reads the last bar of the same `position` series Leo already
computed (post next-bar-shift), so "today's signal" here means exactly what the
backtester would have traded, not a live re-evaluation.

## Caveats on the current output

Every Sharpe the research mode prints in the RESULTS table is **in-sample** — it's Leo's
raw backtest, not the validated number. The WALK-FORWARD VALIDATION section is the
number that matters: a strategy only gets capital if it clears 3-of-5 folds *and* stays
under the per-strategy drawdown limit. On a short, noisy synthetic series it is normal
for nothing to pass — the report says so explicitly rather than hiding an empty result,
per spec §6 ("no delivery is worse than a delivery of 'nothing today'").

In a sandboxed environment with no outbound access to Yahoo or Binance, all three
symbols fall back to synthetic data and the report flags them. That path is working as
intended — it is not a data bug.

## Next: Phase 3

`agents/regime_agent.py` (Greg: ADX + slope + vol classifier, the regime-mismatch
capital cut and 2-day signal confirmation), `agents/reporting_agent.py` (George: the
navy/gold Playfair HTML dashboard), and the Slack posting utility. Greg's regime call
layers on top of Charles's risk-parity weights from this phase rather than replacing
them.
