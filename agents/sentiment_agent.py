"""Edwin — Sentiment Agent.

News sentiment via Alpha Vantage's `NEWS_SENTIMENT` endpoint (spec §9,
Phase 5). Purely informational: Edwin's output is displayed on the
dashboard and the Slack standup, but nothing downstream ever reads it to
gate a strategy, adjust a risk-parity weight, or block a trade. Charles
(walk-forward + drawdown) and David (data quality) are the only agents with
veto power in this system -- Edwin only reports what the news is saying.

Same defensive posture as `utils/slack.py`: a missing API key, a network
failure, or a malformed response must never take down the research
pipeline. Every failure mode returns an "unavailable" snapshot with a
`note` explaining why, rather than raising.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from agents.data_agent import MarketData
from utils.config import Config
from utils.logging_setup import get_logger

log = get_logger("sentiment", agent="Edwin")

NEWS_SENTIMENT_URL = "https://www.alphavantage.co/query"

# Alpha Vantage's own bucket thresholds for `overall_sentiment_score`.
_BUCKETS = (
    (-1.00, -0.35, "bearish"),
    (-0.35, -0.15, "somewhat-bearish"),
    (-0.15, 0.15, "neutral"),
    (0.15, 0.35, "somewhat-bullish"),
    (0.35, 1.01, "bullish"),
)


def _label_for(score: float) -> str:
    for low, high, label in _BUCKETS:
        if low <= score < high:
            return label
    return "neutral"


@dataclass
class SentimentSnapshot:
    """One symbol's news-sentiment read. Informational only."""
    symbol: str
    available: bool
    label: str = "unavailable"
    score: float = 0.0
    article_count: int = 0
    top_headline: str = ""
    note: str = ""
    fetched_at: datetime | None = None


class SentimentAgent:
    """Edwin."""

    name = "Edwin"
    role = "Sentiment"

    def __init__(self, config: Config):
        self.config = config
        self.enabled = bool(config.get("sentiment.enabled", True))
        self.api_key_env = config.get("sentiment.api_key_env", "ALPHA_VANTAGE_API_KEY")
        self.lookback_days = int(config.get("sentiment.lookback_days", 3))
        self.timeout = float(config.get("sentiment.request_timeout_seconds", 10))
        self.cache_dir: Path = config.repo_path(config.get("sentiment.cache_dir", "data_cache"))
        self.cache_ttl = timedelta(hours=float(config.get("sentiment.cache_ttl_hours", 6)))
        self.crypto_ticker_map: dict = config.get("sentiment.crypto_ticker_map", {})

    # ------------------------------------------------------------------ public

    def run(self, market: dict[str, MarketData]) -> dict[str, SentimentSnapshot]:
        """One snapshot per symbol in `market`. Never raises -- a symbol
        Edwin can't get a read on comes back `available=False` with a note,
        same shape as every other symbol."""
        import os

        snapshots: dict[str, SentimentSnapshot] = {}
        if not self.enabled:
            for symbol in market:
                snapshots[symbol] = SentimentSnapshot(
                    symbol, available=False, note="sentiment.enabled is false in config")
            return snapshots

        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            log.info("No %s set -- sentiment unavailable this run (informational feature only).",
                     self.api_key_env)
            for symbol in market:
                snapshots[symbol] = SentimentSnapshot(
                    symbol, available=False,
                    note=f"{self.api_key_env} not set")
            return snapshots

        for symbol in market:
            snapshots[symbol] = self._fetch_one(symbol, api_key)
        return snapshots

    # ------------------------------------------------------------------ per-symbol

    def _fetch_one(self, symbol: str, api_key: str) -> SentimentSnapshot:
        ticker = self.crypto_ticker_map.get(symbol, symbol)

        cached = self._read_cache(ticker)
        if cached is not None:
            log.debug("%s: sentiment served from cache", symbol)
            return SentimentSnapshot(symbol=symbol, **cached)

        try:
            payload = self._request(ticker, api_key)
            snapshot = self._parse(symbol, ticker, payload)
            self._write_cache(ticker, snapshot)
            return snapshot
        except Exception as exc:  # pragma: no cover - defensive, mirrors utils/slack.py
            log.warning("%s: sentiment fetch failed (%s) -- informational feature only, "
                       "pipeline continues.", symbol, exc)
            return SentimentSnapshot(symbol, available=False, note=str(exc)[:200])

    def _request(self, ticker: str, api_key: str) -> dict:
        time_from = (datetime.now(timezone.utc) - timedelta(days=self.lookback_days)
                    ).strftime("%Y%m%dT0000")
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "time_from": time_from,
            "sort": "LATEST",
            "limit": "50",
            "apikey": api_key,
        }
        url = f"{NEWS_SENTIMENT_URL}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(url, headers={"User-Agent": "meridian-capital/1.0"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = response.read()
        return json.loads(body)

    def _parse(self, symbol: str, ticker: str, payload: dict) -> SentimentSnapshot:
        now = datetime.now(timezone.utc)
        if "Information" in payload or "Note" in payload or "Error Message" in payload:
            reason = payload.get("Information") or payload.get("Note") or payload.get("Error Message")
            return SentimentSnapshot(symbol, available=False, note=str(reason)[:200], fetched_at=now)

        articles = payload.get("feed", [])
        if not articles:
            return SentimentSnapshot(symbol, available=True, label="neutral", score=0.0,
                                     article_count=0, note="no recent articles",
                                     fetched_at=now)

        scores = []
        for article in articles:
            ticker_sentiments = article.get("ticker_sentiment", [])
            match = next((t for t in ticker_sentiments
                         if t.get("ticker", "").upper() == ticker.upper()), None)
            raw = (match or {}).get("ticker_sentiment_score",
                                    article.get("overall_sentiment_score"))
            try:
                scores.append(float(raw))
            except (TypeError, ValueError):
                continue

        avg_score = sum(scores) / len(scores) if scores else 0.0
        top = articles[0]
        return SentimentSnapshot(
            symbol=symbol, available=True, label=_label_for(avg_score),
            score=round(avg_score, 3), article_count=len(articles),
            top_headline=str(top.get("title", ""))[:200], fetched_at=now,
        )

    # ------------------------------------------------------------------ cache

    def _cache_file(self, ticker: str) -> Path:
        return self.cache_dir / f"sentiment_{ticker.replace('/', '_').replace(':', '_')}.json"

    def _read_cache(self, ticker: str) -> dict | None:
        path = self._cache_file(ticker)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text())
            written = datetime.fromisoformat(raw["written_at"])
            if datetime.now(timezone.utc) - written > self.cache_ttl:
                return None
            data = raw["snapshot"]
            if data.get("fetched_at"):
                data["fetched_at"] = datetime.fromisoformat(data["fetched_at"])
            data.pop("symbol", None)
            return data
        except Exception:
            return None

    def _write_cache(self, ticker: str, snapshot: SentimentSnapshot) -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            data = {
                "available": snapshot.available, "label": snapshot.label,
                "score": snapshot.score, "article_count": snapshot.article_count,
                "top_headline": snapshot.top_headline, "note": snapshot.note,
                "fetched_at": snapshot.fetched_at.isoformat() if snapshot.fetched_at else None,
            }
            self._cache_file(ticker).write_text(json.dumps({
                "written_at": datetime.now(timezone.utc).isoformat(),
                "snapshot": data,
            }))
        except Exception as exc:  # pragma: no cover
            log.debug("Sentiment cache write skipped for %s: %s", ticker, exc)
