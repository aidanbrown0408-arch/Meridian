"""Slack webhook posting.

Everything ships through a single Slack webhook (spec §11) -- no email, no
SMS, one integration to configure. This module never raises: a Slack outage
or a missing webhook URL must never take down the research pipeline. Callers
get a bool back and a log line explains what happened.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from utils.logging_setup import get_logger

log = get_logger("slack", agent="George")


def post_message(webhook_url: str | None, text: str, blocks: list[dict] | None = None,
                 timeout: float = 10.0) -> bool:
    """POST a message to a Slack incoming webhook. Returns True on a 2xx
    response, False (never raises) on any failure or a missing URL."""
    if not webhook_url:
        log.info("No Slack webhook configured -- message logged, not sent:\n%s", text)
        return False

    payload: dict = {"text": text}
    if blocks:
        payload["blocks"] = blocks

    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        webhook_url, data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            ok = 200 <= response.status < 300
            if ok:
                log.info("Posted to Slack (%d bytes).", len(body))
            else:
                log.warning("Slack webhook returned status %d", response.status)
            return ok
    except urllib.error.URLError as exc:
        log.warning("Slack post failed: %s", exc)
        return False
