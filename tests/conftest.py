"""Shared test setup: the test suite must never post to a real Slack
workspace, even when the operator's shell has the webhook variables set."""

import pytest


@pytest.fixture(autouse=True)
def _no_real_slack(monkeypatch):
    for var in ("MERIDIAN_SLACK_WEBHOOK_URL", "MERIDIAN_SLACK_ALERTS_WEBHOOK_URL"):
        monkeypatch.delenv(var, raising=False)
