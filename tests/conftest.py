"""Shared test setup: the test suite must never post to a real Slack
workspace, even when the operator's shell has the webhook or bot-mode
variables set."""

import pytest

_BOT_MODE_VARS = (
    "MERIDIAN_SLACK_CHANNEL_ID", "MERIDIAN_SLACK_ALERTS_CHANNEL_ID",
    "MERIDIAN_SLACK_TOKEN_WONG", "MERIDIAN_SLACK_TOKEN_DAVID",
    "MERIDIAN_SLACK_TOKEN_LEO", "MERIDIAN_SLACK_TOKEN_CHARLES",
    "MERIDIAN_SLACK_TOKEN_GREG", "MERIDIAN_SLACK_TOKEN_CORNELIUS",
    "MERIDIAN_SLACK_TOKEN_GEORGE", "MERIDIAN_SLACK_TOKEN_AUGUSTUS",
    "MERIDIAN_SLACK_TOKEN_THEO", "MERIDIAN_SLACK_TOKEN_JOSEPH",
)


@pytest.fixture(autouse=True)
def _no_real_slack(monkeypatch):
    for var in ("MERIDIAN_SLACK_WEBHOOK_URL", "MERIDIAN_SLACK_ALERTS_WEBHOOK_URL",
               *_BOT_MODE_VARS):
        monkeypatch.delenv(var, raising=False)
