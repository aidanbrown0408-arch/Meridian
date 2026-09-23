"""Slack transport: the incoming webhook, plus per-bot posting for agent
identities.

The webhook (spec §11) remains the default -- no email, no SMS, one
integration to configure. `post_as()` is the optional bot-mode transport
layered on top of it: one Slack app per agent, authenticated with its own
`xoxb-...` token, posting through `chat.postMessage` so the message shows up
under that agent's own name and avatar rather than the shared webhook's.
Both functions never raise: a Slack outage, a bad token, or a missing
webhook/channel must never take down the research pipeline. Callers get a
value back (bool, or a response dict / None) and a log line explains what
happened.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from utils.logging_setup import get_logger

log = get_logger("slack", agent="George")

CHAT_POST_MESSAGE_URL = "https://slack.com/api/chat.postMessage"


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


def post_as(bot_token: str | None, channel: str | None, text: str,
           blocks: list[dict] | None = None, thread_ts: str | None = None,
           timeout: float = 10.0) -> dict | None:
    """POST a message to Slack's `chat.postMessage` Web API, authenticated as
    one bot user -- this is what makes a message show up as that agent
    rather than the shared webhook. `thread_ts` replies into an existing
    thread, which is how the standup becomes a real conversation.

    Returns the parsed JSON response (it carries `ts`, needed to open or
    continue a thread) on success, or `None` on any failure -- a missing
    token/channel, a network error, or Slack rejecting the call (bad token,
    bot not invited to the channel, etc.). Never raises, so a caller can
    always fall back to `post_message` without a try/except of its own.
    """
    if not bot_token or not channel:
        log.info("Bot-mode Slack post skipped -- no token/channel configured.")
        return None

    payload: dict = {"channel": channel, "text": text}
    if blocks:
        payload["blocks"] = blocks
    if thread_ts:
        payload["thread_ts"] = thread_ts

    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        CHAT_POST_MESSAGE_URL, data=body, method="POST",
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {bot_token}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, ValueError) as exc:
        log.warning("Slack bot post failed: %s", exc)
        return None

    if not data.get("ok"):
        log.warning("Slack bot post rejected: %s", data.get("error", "unknown error"))
        return None

    log.info("Posted to Slack as bot (%d bytes, ts=%s).", len(body), data.get("ts"))
    return data
