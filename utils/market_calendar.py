"""Which daily bar *should* exist right now.

Wong and David use this to refuse yesterday's price on today's run: after
4:15 PM Eastern on a trading day the newest stock bar must be today's; before
that (or on a weekend/holiday) it must be the previous session's. Crypto
candles close at 00:00 UTC, so the newest complete one is yesterday's UTC
candle.

NYSE holidays are listed here (from NYSE Group's published 2026-2028
calendar) and can be extended in config under `data.market_holidays`. Early
closes (1:00 PM) don't matter: by the 8:15 PM run those sessions are closed
too. When the calendar runs out (2029+), weekends are still handled and an
unlisted holiday would raise a stale-data block, never a silent stale trade --
add the new year's dates when NYSE publishes them.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

EXCHANGE_TZ = "America/New_York"
# A daily bar is final a little after the 4:00 PM close.
SESSION_FINAL = time(16, 15)

NYSE_HOLIDAYS = {
    # 2026
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
    "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    # 2027
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31",
    "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
    # 2028 (no New Year's Day holiday: Jan 1 2028 is a Saturday)
    "2028-01-17", "2028-02-21", "2028-04-14", "2028-05-29", "2028-06-19",
    "2028-07-04", "2028-09-04", "2028-11-23", "2028-12-25",
}


def _holidays(config=None) -> set[str]:
    extra = config.get("data.market_holidays", []) if config is not None else []
    return NYSE_HOLIDAYS | {str(d) for d in (extra or [])}


def is_session(d: date, config=None) -> bool:
    return d.weekday() < 5 and d.isoformat() not in _holidays(config)


def previous_session(d: date, config=None) -> date:
    d -= timedelta(days=1)
    while not is_session(d, config):
        d -= timedelta(days=1)
    return d


def next_session(d: date, config=None) -> date:
    d += timedelta(days=1)
    while not is_session(d, config):
        d += timedelta(days=1)
    return d


def _utc(now: datetime | None) -> datetime:
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now


def exchange_now(now: datetime | None = None) -> datetime:
    return _utc(now).astimezone(ZoneInfo(EXCHANGE_TZ))


def expected_last_bar(asset_class: str, config=None, now: datetime | None = None) -> date:
    """The newest complete daily bar that should exist at `now`."""
    if asset_class == "crypto":
        return _utc(now).astimezone(timezone.utc).date() - timedelta(days=1)
    local = exchange_now(now)
    today = local.date()
    if is_session(today, config) and local.time() >= SESSION_FINAL:
        return today
    return previous_session(today, config)


def sessions_between(last: date, expected: date, asset_class: str, config=None) -> int:
    """How many bars are missing: sessions after `last` up to `expected`."""
    if last >= expected:
        return 0
    if asset_class == "crypto":
        return (expected - last).days
    n, d = 0, last
    while d < expected:
        d = next_session(d, config)
        n += 1
    return n
