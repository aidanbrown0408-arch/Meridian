# Scheduling daily runs

Meridian doesn't include its own scheduler — `python main.py paper` is a
single, idempotent run that fetches data, rebalances the persisted paper
ledger, and posts the standup. Running it once a day, every trading day, is
what turns the paper ledger into the forward-tested track record the
live-trading gate (spec §13) eventually needs. This doc covers the two
common ways to do that: cron on Linux/macOS, and Task Scheduler on Windows.

Run in **paper** mode, not `research`, for the daily job — `research` never
touches the ledger, so a cron job set to `research` builds no track record
at all.

## Before scheduling anything

Run it by hand first and confirm it succeeds:

```bash
cd /path/to/Meridian
python main.py paper --no-slack
```

Drop `--no-slack` once you're ready to post the standup for real, and set
`MERIDIAN_SLACK_WEBHOOK_URL` (and `ALPHA_VANTAGE_API_KEY`, if you want
Edwin's sentiment read populated) in whatever environment the scheduler
runs the job in — cron and Task Scheduler jobs do **not** inherit your
interactive shell's environment variables by default.

## Linux/macOS — cron

Wrap the run in a small shell script so cron's minimal environment doesn't
trip over a missing `PATH` or virtualenv, and so you get a log file per run:

```bash
#!/usr/bin/env bash
# meridian_daily.sh
set -euo pipefail
cd /path/to/Meridian
source .venv/bin/activate          # if you're using a virtualenv
export MERIDIAN_SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
export ALPHA_VANTAGE_API_KEY="..."  # optional -- Edwin degrades gracefully without it
python main.py paper >> reports/cron.log 2>&1
```

```bash
chmod +x meridian_daily.sh
crontab -e
```

Add a line for weekdays only (market data is stale on weekends anyway, and
`slack.post_daily_standup`'s weekday check would skip the post regardless):

```cron
# Run at 6:15pm local time, Mon-Fri, after US markets close
15 18 * * 1-5 /path/to/Meridian/meridian_daily.sh
```

Crypto never closes, so if BTC/USDT is part of your universe and you want a
same-day read on it too, consider a second daily run, or drop the `1-5` and
run every day — the pipeline handles weekends fine, `_check_gaps` and
`_check_staleness` already account for stocks vs. crypto calendars
separately (`agents/compliance_agent.py`).

## Windows — Task Scheduler

1. Create `meridian_daily.bat`:

   ```bat
   @echo off
   cd /d C:\path\to\Meridian
   call .venv\Scripts\activate.bat
   set MERIDIAN_SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
   set ALPHA_VANTAGE_API_KEY=...
   python main.py paper >> reports\cron.log 2>&1
   ```

2. Open **Task Scheduler** → **Create Task** (not "Basic Task" — you want
   the extra options):
   - **General**: run whether the user is logged in or not; "Run with
     highest privileges" is not needed.
   - **Triggers**: New → Daily, set the time, then under "Advanced
     settings" restrict repeat to weekdays if desired (Task Scheduler's
     daily trigger doesn't have a native "weekdays only" toggle — use a
     **Weekly** trigger instead and check Mon–Fri).
   - **Actions**: New → Start a program → `C:\path\to\Meridian\meridian_daily.bat`.
   - **Conditions**: uncheck "Start the task only if the computer is on AC
     power" if this runs on a laptop.

3. Test it once with **Run** in the Task Scheduler UI before trusting the
   schedule, and check `reports\cron.log` and `reports\latest.html`.

## What a missed or double-run does

- **Missed day**: nothing breaks. `run_paper` always rebalances *toward*
  today's target from whatever the ledger currently holds — there's no
  "yesterday's fill" it depends on having happened. The lifecycle agent's
  consecutive-fail counters and Edwin's cache both key off calendar dates,
  not run count, so a skipped day doesn't corrupt either.
- **Double run same day**: the second run reads live prices again, and
  Cornelius rebalances toward the same target weights — if the price
  hasn't moved much, the second run is mostly a no-op (`min_trade_dollars`
  filters out the resulting dust delta). It is not designed to be run more
  than a few times a day; running it every few minutes would churn on
  cost for no benefit and burn through Alpha Vantage's free-tier rate
  limit for Edwin.

## Killswitch and halts are never scheduled

`python main.py killswitch` and `python main.py bench`/`unbench` are
operator actions on purpose (spec §10) — do not wire them into a cron job.
If the scheduled `paper` run finds the ledger halted, it logs that and
skips trading for the day; it does not clear the halt itself, and neither
should a script.
