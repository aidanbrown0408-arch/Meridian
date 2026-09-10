# Meridian Capital — Phase 4 (Execution & Lifecycle)

Multi-agent trading research desk. This repo now implements **Phases 1–4** of the
build spec: a runnable research pipeline that fetches data, blocks anything with
broken data, backtests all six traders against all three tickers with cost modeling,
walk-forward validates the survivors, turns passing traders into a risk-parity
capital allocation, classifies each ticker's regime and tilts capital accordingly,
renders the whole thing as a navy/gold HTML dashboard plus a Slack standup post, and
— in paper mode — actually trades that allocation against a persisted $5,000 virtual
ledger, with a killswitch and an operator-gated benching lifecycle on top.

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
| Greg (RegimeAgent) — ADX + slope + vol classifier, regime-mismatch capital cut + 2-day confirmation | Done |
| George (ReportingAgent) — navy/gold Playfair HTML dashboard, Slack standup message | Done |
| Cornelius (PortfolioAgent) — ResearchExecutor no-op + PaperBroker, persisted JSON ledger | Done |
| Killswitch — flatten every open paper position + halt, manual `clear-halt` to resume | Done |
| Lifecycle agent — validation-fail / live-drift benching recommendations, operator-only apply | Done |
| `research` and `paper` CLI modes with console report | Done |
| Live gate (three-gate refusal) | Phase 5 |

`live` is wired into the CLI but refuses to run and tells you which phase it arrives
in. Its three gates are not bypassable and are not implemented yet either way.

## Setup

```bash
pip install -r requirements.txt
python main.py research      # backtest + validate + report, no execution
python main.py paper         # same, plus simulated fills against the paper ledger
python main.py killswitch    # flatten every open paper position and halt
python main.py clear-halt    # manually clear a killswitch halt
python main.py bench   --trader ORBIT --trigger validation --reason "..."
python main.py unbench --trader ORBIT --trigger validation
```

Optional flags: `--symbols SPY QQQ`, `--config path/to/config.yaml`, `--no-slack`,
`--verbose`. `bench`/`unbench` also take `--trigger drift` for a live-drift bench,
which (unlike a validation-fail bench) never auto-reinstates.

To actually post the daily standup to Slack, set `MERIDIAN_SLACK_WEBHOOK_URL` in the
environment before running — without it, the message is logged, not sent, and the run
still succeeds. The HTML dashboard always writes to `reports/meridian_YYYYMMDD.html`
and `reports/latest.html`; the paper ledger persists to `reports/paper_ledger.json` and
the bench state to `reports/bench_state.json` (all gitignored).

Tests: `python tests/test_phase1.py && python tests/test_phase2.py && python tests/test_phase3.py && python tests/test_phase4.py`
(or `python -m pytest tests -q`).

## Layout

```
main.py                       Orchestrator CLI
config/config.yaml            Every tunable parameter
agents/data_agent.py          Wong — fetch + synthetic fallback
agents/compliance_agent.py    David — data-quality checks, per-symbol blocking
agents/backtest_agent.py      Leo — runs all strategy/ticker combinations
agents/risk_agent.py          Charles — walk-forward gate, risk parity, netting
agents/regime_agent.py        Greg — ADX/slope/vol classifier, mismatch capital cut
agents/reporting_agent.py     George — HTML dashboard + Slack standup
agents/portfolio_agent.py     Cornelius — ResearchExecutor + PaperBroker + killswitch
agents/lifecycle_agent.py     Charles+George — benching recommendations, operator gate
strategies/base.py            Abstract base, registry, next-bar enforcement
strategies/{orbit,flux,revert,surge,spark,anchor}.py
backtester/engine.py          Vectorized engine + CostModel
backtester/walkforward.py     5-fold walk-forward validator
templates/report.html.j2      Jinja2 dashboard template (navy/gold Playfair)
utils/{config,logging_setup,metrics,svg_charts,slack}.py
tests/test_phase1.py          Phase 1 invariant tests
tests/test_phase2.py          Phase 2 invariant tests
tests/test_phase3.py          Phase 3 invariant tests
tests/test_phase4.py          Phase 4 invariant tests
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
`RiskAgent._net_positions` (and Greg's regime-aware `_net_positions`) reads the last bar
of the same `position` series Leo already computed (post next-bar-shift), so "today's
signal" means exactly what the backtester would have traded, not a live re-evaluation.

**Regime classification requires every sub-condition to hold, or it's undecided.**
Trending needs high ADX *and* a consistently-signed MA slope *and* normal-to-high
realized vol, all at once; choppy needs the mirror image. A clean, high-drift geometric
random walk clears trending; a fast, tight oscillation clears choppy — see
`test_regime_classifies_a_clean_trend_as_trending` / `..._tight_bounce_as_choppy`. A
fixed-dollar-step ramp deliberately is **not** used as the trending fixture: its
*percentage* volatility shrinks as price compounds up, which would starve the
vol-percentile check for reasons that have nothing to do with trendiness.

**Regime-mismatch is a tilt, never a bench.** A mismatched trader's weight is halved
(`regime.capital_cut`) and its position only counts once it has held steady for
`regime.confirmation_days` bars — filtering single-day noise without ever zeroing the
trader out entirely, per spec §8's "every trader stays in the game." Matched traders and
undecided-regime tickers trade at full weight with no extra delay.

**The confirmation window reads the already-shifted position series, not raw signals.**
Since Phase 4's persisted daily execution loop doesn't exist yet, "held steady for N
days" is checked against the same lagged `position` series the backtester produced,
which is the only place "what actually got traded" lives right now. This will get
revisited once Cornelius's paper broker gives us real day-over-day state.

**No charting library.** `utils/svg_charts.py` hand-builds `<svg>` markup (a multi-series
line chart and a signed bar chart) so the dashboard stays a single dependency-free HTML
file. Colors are the spec's navy/gold/parchment/gain/loss palette, not a generic theme.

**Slack posting never raises.** `utils/slack.py` POSTs via stdlib `urllib`, returns
`False` on any failure or missing webhook, and always logs the message it would have
sent — a Slack outage must never take down the research pipeline.

**Paper position sizing scales with current equity, not the original $5,000.**
`PaperBroker._rebalance_one` marks the account to market first and applies each
target weight to *that* equity, so risk-parity sizing compounds the way it would in a
real account rather than staying pinned to the day-one balance forever.

**A rebalance that can't be fully funded gets sized down, never refused outright.**
If three simultaneous targets would overdraw the cash account, each buy is scaled to
what's actually affordable rather than executed on a first-come basis or rejected
(`test_cash_constraint_scales_down_rather_than_overdrawing`). A trade smaller than
`execution.min_trade_dollars` is skipped entirely to stop the ledger from churning on
rounding-sized deltas between today's target and yesterday's.

**Entry price is a running weighted average, not last-fill price.** Adding to an
existing position blends the new fill into the existing entry price by share count,
so "unrealized P&L" reflects the position's true cost basis across multiple partial
buys, the way a real broker statement would.

**Per-trader shadow returns exist solely to make live-drift detectable.** Once
multiple traders' signals are netted into one blended ticker position (spec's own
design), you can no longer read one trader's P&L back out of the ledger. So
`PaperBroker._record_shadow` keeps a second, parallel return series per trader
(`adjusted_weight × that day's asset return`, independent of what anyone else is doing
on the same ticker) purely so the lifecycle agent has something trader-specific to
compare against the backtest's promised Sharpe.

**Benching is a recommendation until an operator types the command.** `LifecycleAgent.
recommend()` never sets `validation_benched` or `drift_benched` itself — crossing the
fail threshold, or drifting below the Sharpe ratio threshold, only produces a
`Recommendation` object George would post as an "important inquiry." Applying it is a
separate, explicit call (`apply_validation_bench` / `apply_drift_bench`), exposed as
`python main.py bench`, per spec §10's "no automatic benching without operator
approval" (`test_bench_is_never_automatic`).

**The two bench types reinstate differently, on purpose.** A validation-fail bench
clears itself the moment the trader passes walk-forward again — normal regime
rotation, no drama. A live-drift bench only clears via `python main.py unbench
--trigger drift`, and a clean validation pass in between does *not* touch it
(`test_drift_bench_reinstatement_is_manual_only`) — drift means something structurally
broke, and that always gets a human look before capital comes back.

**`bench`/`unbench` aren't in the spec's four named CLI entry points.** Section 10
describes Slack-based "important inquiry" messages and an operator with "final say,"
but doesn't specify the mechanism for giving that say — and this system has no
interactive Slack app, only a webhook. A CLI subcommand is the most direct way to
implement "operator approval" without inventing infrastructure the spec never asked
for.

## Caveats on the current output

Every Sharpe the research mode prints in the RESULTS table is **in-sample** — it's Leo's
raw backtest, not the validated number. The WALK-FORWARD VALIDATION section is the
number that matters: a strategy only gets capital if it clears 3-of-5 folds *and* stays
under the per-strategy drawdown limit. On a short, noisy synthetic series it is normal
for nothing to pass — the report says so explicitly rather than hiding an empty result,
per spec §6 ("no delivery is worse than a delivery of 'nothing today'").

In `research` mode there is still no live P&L — the dashboard's "Target Allocation"
section is exactly that, a target, not a fill, and every backtest chart is labeled
"in-sample" accordingly. Run `python main.py paper` to actually trade that allocation
against the persisted ledger and see real (paper) P&L in the "Paper Ledger" section.

Live-drift detection needs a real shadow track record (`lifecycle.drift_min_track_days`,
default 10 daily paper runs) before it will ever flag anything — on a fresh ledger,
`python main.py paper` will never produce a live-drift recommendation on day one, by
design.

In a sandboxed environment with no outbound access to Yahoo or Binance, all three
symbols fall back to synthetic data and the report flags them. That path is working as
intended — it is not a data bug.

## Next: Phase 5

`agents/sentiment_agent.py` (Edwin: Alpha Vantage `NEWS_SENTIMENT`, informational only
— never vetoes a strategy), the `LiveBroker` stub with the three-gate refusal logic
(implementation gate, `enable_live_trading: true`, `--i-understand-the-risk`), and
cron/Task Scheduler setup docs for daily runs.
