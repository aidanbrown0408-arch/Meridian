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

# Skip weekends even if the job is kicked manually or fires late after sleep.
dow=$(date +%u)   # 1=Mon ... 7=Sun
if [ "$dow" -ge 6 ]; then
  echo "$(date '+%Y-%m-%d %H:%M:%S') weekend -- skipping paper run" >> "$LOG_FILE"
  exit 0
fi

{
  echo "===== $(date '+%Y-%m-%d %H:%M:%S %Z') paper start ====="
  "$PYTHON_BIN" main.py paper
  status=$?
  echo "===== $(date '+%Y-%m-%d %H:%M:%S %Z') paper exit $status ====="
} >> "$LOG_FILE" 2>&1

exit ${status:-1}
