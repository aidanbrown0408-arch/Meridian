"""Strategy lifecycle (benching) — spec §10.

A trader that stops working needs to be pulled off capital before it does
damage. Two independent triggers, and in both cases the operator remains the
final decision-maker: nothing gets benched automatically.

  * Validation-fail: trader fails walk-forward on `validation_fail_consecutive_
    days` consecutive daily runs (default 3). Reinstatement is automatic the
    moment it passes again -- normal regime rotation, no drama.
  * Live-drift: live paper performance diverges materially from what the
    backtest predicted -- deliberately not "bad luck," this is meant to catch
    a structural break. Reinstatement is manual-only.

This module tracks the persisted state, produces recommendations (what
`George` would post to Slack as an "important inquiry"), and exposes the
apply/clear methods a CLI subcommand calls on the operator's behalf. It never
benches anyone on its own initiative.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from utils.config import Config
from utils.logging_setup import get_logger
from utils.metrics import sharpe_ratio

log = get_logger("lifecycle", agent="Charles+George")


@dataclass
class TraderLifecycle:
    consecutive_fails: int = 0
    validation_benched: bool = False
    validation_bench_reason: str = ""
    drift_benched: bool = False
    drift_bench_reason: str = ""
    last_updated: str = ""


@dataclass
class Recommendation:
    """What George would post to Slack as an 'important inquiry.' Purely
    informational -- applying it is a separate, explicit operator action."""
    trader: str
    trigger: str          # "validation_fail" | "live_drift"
    action: str            # "bench" | "keep" | "investigate"
    detail: str


class LifecycleAgent:
    """Charles detects the trigger, George drafts the inquiry; jointly,
    neither one benches anybody without the operator's say-so."""

    name = "Charles+George"
    role = "Lifecycle"

    def __init__(self, config: Config):
        self.config = config
        self.state_path: Path = config.repo_path(
            config.get("lifecycle.bench_state_path", "reports/bench_state.json"))
        self.fail_threshold = int(config.get("lifecycle.validation_fail_consecutive_days", 3))
        self.drift_min_days = int(config.get("lifecycle.drift_min_track_days", 10))
        self.drift_lookback = int(config.get("lifecycle.drift_lookback_days", 20))
        self.drift_ratio_threshold = float(config.get("lifecycle.drift_sharpe_ratio_threshold", 0.5))
        self.risk_free_rate = float(config.get("risk.risk_free_rate", 0.0))

    # ------------------------------------------------------------- persistence

    def load_state(self) -> dict[str, TraderLifecycle]:
        if not self.state_path.exists():
            return {}
        try:
            raw = json.loads(self.state_path.read_text())
            return {trader: TraderLifecycle(**fields) for trader, fields in raw.items()}
        except Exception as exc:
            log.error("Bench state at %s is unreadable (%s) -- treating as empty.",
                     self.state_path, exc)
            return {}

    def save_state(self, state: dict[str, TraderLifecycle]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        raw = {trader: asdict(life) for trader, life in state.items()}
        self.state_path.write_text(json.dumps(raw, indent=2))

    # --------------------------------------------------------------- tracking

    def record_validation_results(self, all_traders: set[str],
                                  passed_by_trader: dict[str, bool]) -> dict[str, TraderLifecycle]:
        """Update the consecutive-fail counters for every trader Leo/Charles
        evaluated today. A trader that passes auto-reinstates from a
        validation-fail bench; a live-drift bench is untouched here --
        that one only a human clears."""
        state = self.load_state()
        now = datetime.now(timezone.utc).isoformat()
        for trader in all_traders:
            life = state.setdefault(trader, TraderLifecycle())
            passed = passed_by_trader.get(trader, False)
            if passed:
                if life.consecutive_fails > 0 or life.validation_benched:
                    log.info("%s: validation passed -- consecutive-fail streak reset.", trader)
                life.consecutive_fails = 0
                if life.validation_benched:
                    life.validation_benched = False
                    life.validation_bench_reason = ""
                    log.info("%s: auto-reinstated from validation-fail bench.", trader)
            else:
                life.consecutive_fails += 1
            life.last_updated = now
        self.save_state(state)
        return state

    # ------------------------------------------------------------ recommending

    def recommend(self, state: dict[str, TraderLifecycle],
                  predicted_sharpe: dict[str, float],
                  trader_shadow: dict[str, list]) -> list[Recommendation]:
        recs: list[Recommendation] = []
        for trader, life in state.items():
            if life.consecutive_fails >= self.fail_threshold and not life.validation_benched:
                recs.append(Recommendation(
                    trader, "validation_fail", "bench",
                    f"{life.consecutive_fails} consecutive failed validation runs "
                    f"(threshold {self.fail_threshold}). Recommend bench; auto-reinstates "
                    "the moment it passes again.",
                ))

            drift = self._check_drift(trader, predicted_sharpe.get(trader),
                                      trader_shadow.get(trader, []))
            if drift is not None and not life.drift_benched:
                recs.append(Recommendation(trader, "live_drift", "investigate", drift))
        return recs

    def _check_drift(self, trader: str, predicted_sharpe: float | None,
                     shadow_history: list) -> str | None:
        if predicted_sharpe is None or predicted_sharpe <= 0:
            return None
        if len(shadow_history) < self.drift_min_days:
            return None
        recent = shadow_history[-self.drift_lookback:]
        returns = pd.Series([row["return"] for row in recent])
        live_sharpe = sharpe_ratio(returns, self.risk_free_rate, ppy=252)
        threshold = predicted_sharpe * self.drift_ratio_threshold
        if live_sharpe < threshold:
            return (f"Live shadow Sharpe {live_sharpe:.2f} over the last {len(recent)} days "
                    f"is well below the backtest's {predicted_sharpe:.2f} "
                    f"(below {self.drift_ratio_threshold:.0%} of it). Possible structural break, "
                    "not just bad luck -- recommend a human look.")
        return None

    # ------------------------------------------------------------ operator actions

    def apply_validation_bench(self, trader: str, reason: str) -> None:
        state = self.load_state()
        life = state.setdefault(trader, TraderLifecycle())
        life.validation_benched = True
        life.validation_bench_reason = reason or "operator-approved validation-fail bench"
        self.save_state(state)
        log.warning("%s: validation-fail bench applied by operator.", trader)

    def apply_drift_bench(self, trader: str, reason: str) -> None:
        state = self.load_state()
        life = state.setdefault(trader, TraderLifecycle())
        life.drift_benched = True
        life.drift_bench_reason = reason or "operator-approved live-drift bench"
        self.save_state(state)
        log.warning("%s: live-drift bench applied by operator.", trader)

    def clear_drift_bench(self, trader: str) -> None:
        state = self.load_state()
        life = state.setdefault(trader, TraderLifecycle())
        life.drift_benched = False
        life.drift_bench_reason = ""
        self.save_state(state)
        log.info("%s: live-drift bench manually cleared by operator.", trader)

    def clear_validation_bench(self, trader: str) -> None:
        """Not part of the spec's normal flow (that one auto-reinstates) but
        available for an operator override, e.g. a bench applied by mistake."""
        state = self.load_state()
        life = state.setdefault(trader, TraderLifecycle())
        life.validation_benched = False
        life.validation_bench_reason = ""
        life.consecutive_fails = 0
        self.save_state(state)
        log.info("%s: validation-fail bench manually cleared by operator.", trader)

    def effective_benched(self, trader: str, state: dict[str, TraderLifecycle] | None = None
                          ) -> tuple[bool, str]:
        state = state if state is not None else self.load_state()
        life = state.get(trader)
        if life is None:
            return False, ""
        if life.validation_benched:
            return True, life.validation_bench_reason
        if life.drift_benched:
            return True, life.drift_bench_reason
        return False, ""
