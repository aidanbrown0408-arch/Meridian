#!/bin/bash
# Daily paper-trading run (stock desk + options leg), invoked by launchd (see
# install_daily_options.sh). `paper-options` no longer runs standalone -- SPY/QQQ
# options entries fire inside `main.py paper` -- so this runs `main.py paper`
# from the repo root and appends output to
# reports/launchd_paper_options.log (gitignored via reports/*.log).
set -u

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${MERIDIAN_PYTHON:-/Library/Frameworks/Python.framework/Versions/3.14/bin/python3}"
LOG_FILE="$REPO_DIR/reports/launchd_paper_options.log"

cd "$REPO_DIR" || exit 1
mkdir -p "$REPO_DIR/reports"

# Slack bot-mode credentials (MERIDIAN_SLACK_CHANNEL_ID + per-agent bot tokens).
# The launchd plist only sets PATH, so the vars have to come from here.
# Gitignored; absent on a fresh clone -- skip quietly rather than fail the run.
SLACK_ENV_FILE="$REPO_DIR/scripts/slack_env.sh"
if [ -f "$SLACK_ENV_FILE" ]; then
  # shellcheck source=/dev/null
  . "$SLACK_ENV_FILE"
else
  echo "$(date '+%Y-%m-%d %H:%M:%S') slack_env.sh not found -- Slack bot mode disabled" >> "$LOG_FILE"
fi

# Detect a scheduler gap: this job going silent (unloaded, machine asleep
# through the whole window, etc.) used to look identical to "nothing to
# report" -- the log went quiet for four straight trading days before
# anyone noticed. A weekday-to-weekday gap is well under 24h; give a wide
# berth for a single missed weekend (Fri -> Mon is ~76h) before flagging.
# Based on the log file's mtime, so it costs nothing when everything's fine.
if [ -f "$LOG_FILE" ]; then
  last_touch_epoch=$(date -r "$LOG_FILE" +%s 2>/dev/null || echo 0)
  now_epoch=$(date +%s)
  if [ "$last_touch_epoch" -gt 0 ]; then
    gap_hours=$(( (now_epoch - last_touch_epoch) / 3600 ))
    if [ "$gap_hours" -gt 76 ]; then
      gap_days=$(( gap_hours / 24 ))
      echo "$(date '+%Y-%m-%d %H:%M:%S') WARNING: no run logged in ~${gap_days}d -- scheduler may have gone silent" >> "$LOG_FILE"
      "$PYTHON_BIN" -c "
from utils.config import load_config
from utils.notifications import Notifier
try:
    Notifier(load_config()).operator_action(
        'Scheduler resumed after a gap',
        'No paper run was logged in roughly ${gap_days} day(s) before this one -- '
        'the launchd job may have been unloaded, or the Mac was asleep through '
        'every scheduled window. Worth checking it is still installed: '
        'launchctl print gui/\$(id -u)/com.meridian.paper-options')
except Exception:
    pass
" >> "$LOG_FILE" 2>&1
    fi
  fi
fi

# Skip weekends even if the job is kicked manually or fires late after sleep.
dow=$(date +%u)   # 1=Mon ... 7=Sun
if [ "$dow" -ge 6 ]; then
  echo "$(date '+%Y-%m-%d %H:%M:%S') weekend -- skipping paper run" >> "$LOG_FILE"
  exit 0
fi

# Idempotent against a duplicate fire the same day (RunAtLoad catching up
# right after the scheduled StartCalendarInterval already ran, a manual
# `launchctl kickstart` on top of the normal 8:15 PM fire, etc.) -- running
# main.py paper twice in one day would double-count that day's fills.
today="$(date '+%Y-%m-%d')"
if [ -f "$LOG_FILE" ] && grep -q "^===== $today .* paper exit 0 =====" "$LOG_FILE"; then
  echo "$(date '+%Y-%m-%d %H:%M:%S') already ran successfully today -- skipping duplicate fire" >> "$LOG_FILE"
  exit 0
fi

# Wait for Yahoo. Its daily history publishes a session's bar hours after
# the 4 PM close (Mon 10/05's appeared between 8:15 and 11:11 PM; Tue 10/06's
# was missing at 8:15 PM). Wong also builds the missing bar from intraday
# data, but this doesn't depend on that working: before trading, a read-only
# check (no ledger/cache/Slack writes) confirms every stock has the newest
# completed session. If not, wait and re-check, up to FRESHNESS_TRIES times.
# If it's still stale after that, run anyway: David blocks the stale tickers,
# Cornelius freezes them, and David posts a Slack alert -- never a trade on
# an old price.
FRESHNESS_TRIES="${MERIDIAN_FRESHNESS_TRIES:-8}"        # 8 x 25 min ~= until 11:35 PM
FRESHNESS_WAIT_SECS="${MERIDIAN_FRESHNESS_WAIT_SECS:-1500}"
attempt=1
while true; do
  if "$PYTHON_BIN" scripts/check_data_freshness.py >> "$LOG_FILE" 2>&1; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') price data current (check $attempt) -- trading" >> "$LOG_FILE"
    break
  fi
  if [ "$attempt" -ge "$FRESHNESS_TRIES" ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') price data still not current after $attempt checks -- running anyway; stale tickers will be frozen and alerted" >> "$LOG_FILE"
    break
  fi
  echo "$(date '+%Y-%m-%d %H:%M:%S') price data not current yet (check $attempt/$FRESHNESS_TRIES) -- retrying in $((FRESHNESS_WAIT_SECS / 60)) min" >> "$LOG_FILE"
  attempt=$((attempt + 1))
  sleep "$FRESHNESS_WAIT_SECS"
done

{
  echo "===== $(date '+%Y-%m-%d %H:%M:%S %Z') paper start ====="
  "$PYTHON_BIN" main.py paper
  status=$?
  echo "===== $(date '+%Y-%m-%d %H:%M:%S %Z') paper exit $status ====="
} >> "$LOG_FILE" 2>&1

exit ${status:-1}
