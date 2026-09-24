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


@pytest.fixture(autouse=True)
def _isolated_reports_dir(monkeypatch, tmp_path):
    """ReportingAgent.reports_dir defaults to config.repo_path("reports"),
    which always resolves to the real repo -- independent of any tmp_path
    override a test uses for ledger paths. Production code (main._options_leg)
    builds its own ReportingAgent(config) internally and calls
    refresh_options() unconditionally, so any test that exercises the options
    leg (test_main_cli.py has a dozen of these) was writing real halt/test
    scenarios straight into reports/options_status.json and reports/latest.html
    in the actual repo -- that's how a "bucket halted, equity $400" test
    fixture ended up baked into a real dashboard. Patch the constructor
    itself so no call site, present or future, direct or indirect, can slip
    through."""
    from agents.reporting_agent import ReportingAgent

    real_init = ReportingAgent.__init__
    isolated_dir = tmp_path / "reports"

    def patched_init(self, config, *args, **kwargs):
        real_init(self, config, *args, **kwargs)
        self.reports_dir = isolated_dir

    monkeypatch.setattr(ReportingAgent, "__init__", patched_init)
