"""Operator-local calendar dates for ledger labels.

Ledger marks and trade rows are labelled with the operator's calendar date
(`system.timezone`, America/New_York), not the UTC date. The daily run fires
at 8:15 PM Eastern -- already the next day in UTC -- so labelling with the
UTC date would stamp every scheduled run with tomorrow's date.

Only *labels* use this. Expiration and holding-period logic keep comparing
UTC datetimes, unchanged.
"""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

DEFAULT_TIMEZONE = "America/New_York"


def local_date(config=None, when: datetime | None = None) -> str:
    """ISO date (YYYY-MM-DD) of `when` (default: now) in the operator's timezone."""
    tz_name = DEFAULT_TIMEZONE
    if config is not None:
        tz_name = config.get("system.timezone", DEFAULT_TIMEZONE) or DEFAULT_TIMEZONE
    when = when or datetime.now(timezone.utc)
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when.astimezone(ZoneInfo(tz_name)).date().isoformat()
