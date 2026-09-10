"""Wong — Data Agent.

Fetches OHLCV history for the configured universe. If a real source fails, Wong
falls back to a seeded random walk so the rest of the pipeline still has
something to chew on — but the result carries `data_source == "synthetic"`
forever after, and downstream stages must exclude it from anything the operator
would mistake for a real result.

Wong's contract: always return a MarketData object per symbol, or raise only if
synthetic fallback is explicitly disabled.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from utils.config import Config
from utils.logging_setup import get_logger

log = get_logger("data", agent="Wong")

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


@dataclass
class MarketData:
    """One symbol's price history plus the provenance of that history."""
    symbol: str
    asset_class: str
    bars: pd.DataFrame
    data_source: str            # "yfinance" | "ccxt" | "synthetic" | "cache"
    fetched_at: datetime
    notes: list[str] = field(default_factory=list)

    @property
    def is_synthetic(self) -> bool:
        return self.data_source == "synthetic"

    @property
    def is_trusted(self) -> bool:
        """Only real data feeds the 'trusted' section of George's report."""
        return not self.is_synthetic

    @property
    def last_date(self) -> pd.Timestamp | None:
        return None if self.bars.empty else self.bars.index[-1]

    def __len__(self) -> int:
        return len(self.bars)

    def __repr__(self) -> str:
        return (f"<MarketData {self.symbol} {len(self.bars)} bars "
                f"source={self.data_source}>")


class DataAgent:
    """Wong."""

    name = "Wong"
    role = "Data"

    def __init__(self, config: Config):
        self.config = config
        self.history_days = int(config.get("universe.history_days"))
        self.allow_synthetic = bool(config.get("data.allow_synthetic_fallback", True))
        self.use_cache = bool(config.get("data.use_cache", True))
        self.cache_dir = config.repo_path(config.get("data.cache_dir", "data_cache"))
        self.cache_ttl = timedelta(hours=float(config.get("data.cache_ttl_hours", 12)))
        self._exchange = None

    # ------------------------------------------------------------------ public

    def fetch_universe(self, symbols: list[str] | None = None) -> dict[str, MarketData]:
        """Fetch every symbol. Never raises for a single failed symbol when
        synthetic fallback is enabled."""
        symbols = symbols or self.config.universe
        out: dict[str, MarketData] = {}
        for symbol in symbols:
            try:
                out[symbol] = self.fetch(symbol)
            except Exception as exc:  # pragma: no cover - defensive
                log.error("Unrecoverable fetch failure for %s: %s", symbol, exc)
                if not self.allow_synthetic:
                    raise
                out[symbol] = self._synthetic(symbol, reason=str(exc))

        synthetic = [s for s, d in out.items() if d.is_synthetic]
        if synthetic:
            log.warning("Synthetic fallback in use for: %s — results are NOT trusted",
                        ", ".join(synthetic))
        else:
            log.info("Pulled %s — all sources live, no synthetic fallback",
                     ", ".join(out))
        return out

    def fetch(self, symbol: str) -> MarketData:
        asset_class = self.config.asset_class(symbol)

        cached = self._read_cache(symbol) if self.use_cache else None
        if cached is not None:
            log.info("Loaded %s from cache: %d bars", symbol, len(cached))
            return MarketData(symbol, asset_class, cached, "cache",
                              datetime.now(timezone.utc), ["served from local cache"])

        try:
            if asset_class == "crypto":
                bars, source = self._fetch_crypto(symbol), self.config.get("data.crypto_source")
            else:
                bars, source = self._fetch_stock(symbol), self.config.get("data.stock_source")
            bars = self._normalize(bars)
            if bars.empty:
                raise ValueError("source returned no rows")
            if self.use_cache:
                self._write_cache(symbol, bars)
            log.info("Fetched %s: %d bars (%s), last %s",
                     symbol, len(bars), source, bars.index[-1].date())
            return MarketData(symbol, asset_class, bars, source,
                              datetime.now(timezone.utc))
        except Exception as exc:
            log.warning("Live fetch failed for %s (%s)", symbol, exc)
            if not self.allow_synthetic:
                raise
            return self._synthetic(symbol, reason=str(exc))

    # ------------------------------------------------------------------ sources

    def _fetch_stock(self, symbol: str) -> pd.DataFrame:
        import yfinance as yf

        start = datetime.now(timezone.utc) - timedelta(days=self.history_days + 10)
        ticker = yf.Ticker(symbol)
        raw = ticker.history(start=start.date().isoformat(), interval="1d",
                             auto_adjust=True, raise_errors=True)
        if raw is None or raw.empty:
            raise ValueError(f"yfinance returned nothing for {symbol}")
        raw = raw.rename(columns=str.lower)
        return raw[[c for c in OHLCV_COLUMNS if c in raw.columns]]

    def _fetch_crypto(self, symbol: str) -> pd.DataFrame:
        import ccxt

        if self._exchange is None:
            exchange_id = self.config.get("data.crypto_exchange", "binance")
            self._exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})

        since = int((time.time() - (self.history_days + 10) * 86400) * 1000)
        rows: list[list] = []
        cursor = since
        # ccxt caps a page at ~1000 candles; page forward until we reach today.
        while True:
            page = self._exchange.fetch_ohlcv(symbol, timeframe="1d",
                                              since=cursor, limit=1000)
            if not page:
                break
            rows.extend(page)
            if len(page) < 1000:
                break
            cursor = page[-1][0] + 86_400_000
        if not rows:
            raise ValueError(f"ccxt returned nothing for {symbol}")

        frame = pd.DataFrame(rows, columns=["ts", *OHLCV_COLUMNS])
        frame = frame.drop_duplicates(subset="ts")
        frame.index = pd.to_datetime(frame.pop("ts"), unit="ms")
        return frame

    # ---------------------------------------------------------------- synthetic

    def _synthetic(self, symbol: str, reason: str = "") -> MarketData:
        """Seeded geometric random walk. Deterministic per symbol so repeated
        runs on a broken feed produce identical (and identically flagged)
        results rather than fresh noise every day."""
        asset_class = self.config.asset_class(symbol)
        seed = int(self.config.get("data.synthetic_seed", 20240101))
        # Mix the symbol into the seed so SPY and QQQ aren't the same series.
        rng = np.random.default_rng(seed + (abs(hash(symbol)) % 10_000))

        start_prices = self.config.get("data.synthetic.start_price", {})
        start_price = float(start_prices.get(symbol, 100.0))
        vols = self.config.get("data.synthetic.annual_vol", {})
        annual_vol = float(vols.get(symbol, vols.get("default", 0.16)))
        drift = float(self.config.get("data.synthetic.annual_drift", 0.07))

        ppy = 365 if asset_class == "crypto" else 252
        n = self.history_days if asset_class == "crypto" else int(self.history_days * 252 / 365)

        if asset_class == "crypto":
            index = pd.date_range(end=pd.Timestamp.utcnow().normalize().tz_localize(None),
                                  periods=n, freq="D")
        else:
            index = pd.bdate_range(end=pd.Timestamp.utcnow().normalize().tz_localize(None),
                                   periods=n)

        daily_vol = annual_vol / np.sqrt(ppy)
        daily_drift = drift / ppy - 0.5 * daily_vol ** 2
        shocks = rng.normal(daily_drift, daily_vol, size=n)
        close = start_price * np.exp(np.cumsum(shocks))

        # Build a plausible OHLC envelope around the close path.
        open_ = np.concatenate([[start_price], close[:-1]])
        intrabar = np.abs(rng.normal(0, daily_vol * 0.6, size=n))
        high = np.maximum(open_, close) * (1 + intrabar)
        low = np.minimum(open_, close) * (1 - intrabar)
        volume = rng.lognormal(mean=15.0, sigma=0.35, size=n)

        bars = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=index,
        )
        note = f"synthetic fallback (seed={seed})"
        if reason:
            note += f" after: {reason[:180]}"
        log.warning("%s: %s", symbol, note)
        return MarketData(symbol, asset_class, self._normalize(bars), "synthetic",
                          datetime.now(timezone.utc), [note])

    # ------------------------------------------------------------------- helpers

    @staticmethod
    def _normalize(bars: pd.DataFrame) -> pd.DataFrame:
        """Uniform shape for every source: tz-naive DatetimeIndex, lowercase
        OHLCV float columns, sorted, deduplicated."""
        frame = bars.copy()
        frame.columns = [str(c).lower() for c in frame.columns]
        for col in OHLCV_COLUMNS:
            if col not in frame.columns:
                frame[col] = np.nan
        frame = frame[OHLCV_COLUMNS]
        frame.index = pd.to_datetime(frame.index)
        if getattr(frame.index, "tz", None) is not None:
            frame.index = frame.index.tz_convert("UTC").tz_localize(None)
        frame.index = frame.index.normalize()
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        return frame.astype("float64").dropna(subset=["close"])

    def _cache_file(self, symbol: str) -> Path:
        return self.cache_dir / f"{symbol.replace('/', '_')}.csv"

    def _read_cache(self, symbol: str) -> pd.DataFrame | None:
        path = self._cache_file(symbol)
        meta = path.with_suffix(".meta.json")
        if not path.exists() or not meta.exists():
            return None
        try:
            written = datetime.fromisoformat(json.loads(meta.read_text())["written_at"])
            if datetime.now(timezone.utc) - written > self.cache_ttl:
                return None
            frame = pd.read_csv(path, index_col=0, parse_dates=True)
            return self._normalize(frame)
        except Exception:
            return None

    def _write_cache(self, symbol: str, bars: pd.DataFrame) -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            path = self._cache_file(symbol)
            bars.to_csv(path)
            path.with_suffix(".meta.json").write_text(json.dumps({
                "symbol": symbol,
                "rows": int(len(bars)),
                "written_at": datetime.now(timezone.utc).isoformat(),
            }))
        except Exception as exc:  # pragma: no cover
            log.debug("Cache write skipped for %s: %s", symbol, exc)
