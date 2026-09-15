"""Augustus — Options Data Agent.

Fetches live option chains for the options universe (SPY, QQQ) via
yfinance's `.option_chain()`. Structure mirrors Wong's `agents/data_agent.py`:
same synthetic-fallback shape, same "provenance never lies" contract — a
chain built synthetically is tagged `data_source == "synthetic"` forever,
and `price_lookup()` refuses to hand synthetic quotes to Joseph as if they
were tradable prices.

Premium convention: yfinance quotes options in PER-SHARE dollars (bid, ask,
lastPrice). Augustus is the one and only place in the options pipeline that
multiplies by 100 to get PER-CONTRACT dollars — the unit Theo's caps
(`max_premium_per_contract`, `max_total_open_premium`) and Joseph's ledger
(`premium_paid`, `OptionsPosition.market_value`) both expect. Nothing
downstream of Augustus should ever see a per-share number.

Two things Augustus hands off, matching what Joseph/Theo already expect:
  * `price_lookup(chains)` — dict of Joseph's ledger key -> current mid,
    the exact shape `OptionsBroker.execute(prices=...)` wants.
  * `find_contract(...)` -> `OptionContract`, which a strategy turns into an
    `OptionsProposal` (Theo's dataclass) by reading `.mid` (or `.ask`) as
    `premium_per_contract`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

from utils.config import Config
from utils.logging_setup import get_logger

log = get_logger("options_data", agent="Augustus")

CHAIN_COLUMNS = ["underlying", "option_type", "strike", "expiration", "bid",
                  "ask", "last", "volume", "open_interest", "implied_volatility"]


@dataclass
class OptionContract:
    """One quoted contract. All dollar fields are PER-CONTRACT (already *100)."""

    underlying: str
    option_type: str          # "long_call" | "long_put" — Theo/Joseph's vocabulary
    strike: float
    expiration: str            # ISO date, "2026-10-17"
    bid: float
    ask: float
    last: float
    volume: int
    open_interest: int
    implied_volatility: float

    @property
    def mid(self) -> float:
        """Best available premium estimate: quoted mid, or last-trade if the
        book is empty."""
        if self.bid > 0 and self.ask > 0:
            return round((self.bid + self.ask) / 2, 2)
        return round(self.last, 2)

    @property
    def key(self) -> str:
        """Must match `OptionsPosition.key` in agents/options_broker.py
        exactly — this is how Joseph's mark_to_market and expiration
        settlement look up a live price for an open position."""
        return f"{self.underlying}|{self.option_type}|{self.strike:g}|{self.expiration}"


@dataclass
class OptionsChain:
    """One underlying's chain across the expirations Augustus decided to keep."""

    underlying: str
    expirations: list[str]
    calls: pd.DataFrame        # CHAIN_COLUMNS, option_type == "long_call"
    puts: pd.DataFrame         # CHAIN_COLUMNS, option_type == "long_put"
    data_source: str           # "yfinance" | "synthetic"
    fetched_at: datetime
    notes: list[str] = field(default_factory=list)

    @property
    def is_synthetic(self) -> bool:
        return self.data_source == "synthetic"

    @property
    def is_trusted(self) -> bool:
        return not self.is_synthetic

    def __repr__(self) -> str:
        return (f"<OptionsChain {self.underlying} {len(self.expirations)} exp "
                f"{len(self.calls)}c/{len(self.puts)}p source={self.data_source}>")


class OptionsDataAgent:
    """Augustus."""

    name = "Augustus"
    role = "Data (Options)"

    def __init__(self, config: Config):
        self.config = config
        self.underlyings = list(config.get("options.underlyings"))
        self.allow_synthetic = bool(
            config.get("options.data.allow_synthetic_fallback", True))
        self.max_expirations = int(config.get("options.data.max_expirations", 4))
        self.min_dte = int(config.get("options.data.min_days_to_expiration", 1))
        self.max_dte = int(config.get("options.data.max_days_to_expiration", 60))

    # ------------------------------------------------------------------ public

    def fetch_universe(self, symbols: list[str] | None = None
                       ) -> dict[str, OptionsChain]:
        """Fetch every configured underlying. Never raises for a single failed
        symbol when synthetic fallback is enabled — same shape as Wong's
        `fetch_universe`."""
        symbols = symbols or self.underlyings
        out: dict[str, OptionsChain] = {}
        for symbol in symbols:
            try:
                out[symbol] = self.fetch(symbol)
            except Exception as exc:  # pragma: no cover - defensive
                log.error("Unrecoverable option chain failure for %s: %s", symbol, exc)
                if not self.allow_synthetic:
                    raise
                out[symbol] = self._synthetic(symbol, reason=str(exc))

        synthetic = [s for s, c in out.items() if c.is_synthetic]
        if synthetic:
            log.warning("Synthetic option chain fallback in use for: %s — "
                        "NOT tradable, scaffolding only", ", ".join(synthetic))
        else:
            log.info("Pulled option chains for %s — all live", ", ".join(out))
        return out

    def fetch(self, symbol: str) -> OptionsChain:
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            all_exps = ticker.options
            if not all_exps:
                raise ValueError(f"yfinance returned no expirations for {symbol}")
            exps = self._select_expirations(all_exps)
            if not exps:
                raise ValueError(f"no expirations for {symbol} within "
                                 f"[{self.min_dte}, {self.max_dte}] DTE")

            call_frames, put_frames = [], []
            for exp in exps:
                raw = ticker.option_chain(exp)
                call_frames.append(self._normalize(raw.calls, symbol, exp, "long_call"))
                put_frames.append(self._normalize(raw.puts, symbol, exp, "long_put"))

            calls = pd.concat(call_frames, ignore_index=True) if call_frames \
                else pd.DataFrame(columns=CHAIN_COLUMNS)
            puts = pd.concat(put_frames, ignore_index=True) if put_frames \
                else pd.DataFrame(columns=CHAIN_COLUMNS)
            if calls.empty and puts.empty:
                raise ValueError(f"no usable contracts for {symbol}")

            log.info("Fetched %s: %d expiration(s), %d calls, %d puts (yfinance)",
                     symbol, len(exps), len(calls), len(puts))
            return OptionsChain(symbol, exps, calls, puts, "yfinance",
                                datetime.now(timezone.utc))
        except Exception as exc:
            log.warning("Live option chain fetch failed for %s (%s)", symbol, exc)
            if not self.allow_synthetic:
                raise
            return self._synthetic(symbol, reason=str(exc))

    def contracts(self, chain: OptionsChain, option_type: str) -> list[OptionContract]:
        """Flatten one side (calls/puts) of a chain into OptionContract rows."""
        frame = chain.calls if option_type == "long_call" else chain.puts
        return [self._row_to_contract(row) for _, row in frame.iterrows()]

    def find_contract(self, chain: OptionsChain, option_type: str,
                      target_strike: float, expiration: str | None = None
                      ) -> OptionContract | None:
        """Closest-strike lookup — the building block for the simplest
        strategy pattern: 'buy the near-the-money call/put expiring around
        X'. Returns None if the side/expiration has no contracts."""
        frame = chain.calls if option_type == "long_call" else chain.puts
        if expiration is not None:
            frame = frame[frame["expiration"] == expiration]
        if frame.empty:
            return None
        idx = (frame["strike"] - target_strike).abs().idxmin()
        return self._row_to_contract(frame.loc[idx])

    def price_lookup(self, chains: dict[str, OptionsChain]) -> dict[str, float]:
        """Build the `prices` dict Joseph's `execute()` / `mark_to_market()` /
        `killswitch()` expect: ledger key -> current dollars-per-contract
        mid. Synthetic chains are excluded outright — Joseph should mark an
        open position at cost rather than trust a fabricated quote."""
        prices: dict[str, float] = {}
        for chain in chains.values():
            if chain.is_synthetic:
                continue
            for frame in (chain.calls, chain.puts):
                for _, row in frame.iterrows():
                    contract = self._row_to_contract(row)
                    if contract.mid > 0:
                        prices[contract.key] = contract.mid
        return prices

    # ------------------------------------------------------------------ shaping

    def _select_expirations(self, all_exps) -> list[str]:
        """Keep expirations inside the configured DTE window, nearest first,
        capped at `max_expirations`. Falls back to the first N raw
        expirations if the window happens to exclude everything (e.g. a
        thin/holiday-adjacent chain) so a fetch never dies purely on the
        DTE filter."""
        today = date.today()
        in_window = [exp for exp in all_exps
                    if self.min_dte <= (date.fromisoformat(exp) - today).days <= self.max_dte]
        chosen = in_window or list(all_exps)
        return chosen[: self.max_expirations]

    def _normalize(self, raw: pd.DataFrame, symbol: str, expiration: str,
                   option_type: str) -> pd.DataFrame:
        """yfinance quotes are per-share; multiply by 100 here — and only
        here — so every downstream consumer sees per-contract dollars."""
        if raw is None or raw.empty:
            return pd.DataFrame(columns=CHAIN_COLUMNS)
        return pd.DataFrame({
            "underlying": symbol,
            "option_type": option_type,
            "strike": raw["strike"].astype(float),
            "expiration": expiration,
            "bid": raw["bid"].fillna(0.0).astype(float) * 100,
            "ask": raw["ask"].fillna(0.0).astype(float) * 100,
            "last": raw["lastPrice"].fillna(0.0).astype(float) * 100,
            "volume": raw["volume"].fillna(0).astype(int),
            "open_interest": raw["openInterest"].fillna(0).astype(int),
            "implied_volatility": raw["impliedVolatility"].fillna(0.0).astype(float),
        })[CHAIN_COLUMNS]

    @staticmethod
    def _row_to_contract(row) -> OptionContract:
        return OptionContract(
            underlying=row["underlying"], option_type=row["option_type"],
            strike=float(row["strike"]), expiration=row["expiration"],
            bid=float(row["bid"]), ask=float(row["ask"]), last=float(row["last"]),
            volume=int(row["volume"]), open_interest=int(row["open_interest"]),
            implied_volatility=float(row["implied_volatility"]),
        )

    # ---------------------------------------------------------------- synthetic

    def _synthetic(self, symbol: str, reason: str = "") -> OptionsChain:
        """Deterministic fake chain, seeded the same way Wong seeds synthetic
        bars, so Phase A4's empty-pipeline dry run works with zero market
        connectivity. NEVER treated as tradable: `price_lookup()` skips
        synthetic chains outright, and any proposal a strategy builds from
        one should be flagged the same way Wong's synthetic bars are (this
        agent doesn't gate that itself — George's reporting does, same as
        the stock side)."""
        seed = int(self.config.get("data.synthetic_seed", 20240101))
        rng = np.random.default_rng(seed + (abs(hash(symbol)) % 10_000))

        start_prices = self.config.get("data.synthetic.start_price", {})
        default_spot = 450.0 if symbol == "SPY" else 380.0
        spot = float(start_prices.get(symbol, default_spot))

        today = date.today()
        day_offsets = [self.min_dte + 13, self.min_dte + 29, self.min_dte + 44,
                      self.min_dte + 59][: max(self.max_expirations, 1)]
        exps = [(today + timedelta(days=d)).isoformat() for d in day_offsets]
        strikes = np.round(spot * np.linspace(0.9, 1.1, 9) / 5.0) * 5.0

        def _side(option_type: str) -> pd.DataFrame:
            rows = []
            for exp in exps:
                dte = max((date.fromisoformat(exp) - today).days, 1)
                for strike in strikes:
                    intrinsic = (max(spot - strike, 0.0) if option_type == "long_call"
                                else max(strike - spot, 0.0))
                    time_value = spot * 0.02 * np.sqrt(dte / 30.0) * \
                        (1 + rng.normal(0, 0.1))
                    mid = max(intrinsic + max(time_value, 0.05), 0.05)
                    spread = max(mid * 0.05, 0.02)
                    rows.append({
                        "underlying": symbol, "option_type": option_type,
                        "strike": float(strike), "expiration": exp,
                        "bid": round((mid - spread / 2) * 100, 2),
                        "ask": round((mid + spread / 2) * 100, 2),
                        "last": round(mid * 100, 2),
                        "volume": int(rng.integers(0, 500)),
                        "open_interest": int(rng.integers(0, 5000)),
                        "implied_volatility": round(float(rng.uniform(0.12, 0.35)), 4),
                    })
            return pd.DataFrame(rows, columns=CHAIN_COLUMNS)

        note = f"synthetic option chain fallback (seed={seed})"
        if reason:
            note += f" after: {reason[:180]}"
        log.warning("%s: %s", symbol, note)
        return OptionsChain(symbol, exps, _side("long_call"), _side("long_put"),
                            "synthetic", datetime.now(timezone.utc), [note])
