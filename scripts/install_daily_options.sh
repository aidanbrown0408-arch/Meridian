#!/bin/bash
# Install (or remove) a launchd job that runs `main.py paper` (stock desk +
# options leg) every weekday at 4:15 PM local time (after the US close, so SPARK
# sees a finished daily bar).
#
#   bash scripts/install_daily_options.sh            # install / reinstall
#   bash scripts/install_daily_options.sh uninstall  # remove
#   launchctl kickstart gui/$(id -u)/com.meridian.paper-options   # run now
set -euo pipefail

# Label and log name kept from the old standalone options job so an existing
# install keeps working without a reinstall.
LABEL="com.meridian.paper-options"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RUNNER="$REPO_DIR/scripts/run_daily_options.sh"
DOMAIN="gui/$(id -u)"

if [ "${1:-}" = "uninstall" ]; then
  launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
  rm -f "$PLIST"
  echo "Removed $LABEL."
  exit 0
fi

# Resolve the same python3 you use in Terminal (launchd has a bare PATH).
PYTHON_BIN="${MERIDIAN_PYTHON:-$(command -v python3 || true)}"
if [ -z "$PYTHON_BIN" ]; then
  echo "Could not find python3 on PATH. Re-run with MERIDIAN_PYTHON=/path/to/python3." >&2
  exit 1
fi
if ! "$PYTHON_BIN" -c "import pandas, yaml, yfinance, jinja2" 2>/dev/null; then
  echo "$PYTHON_BIN is missing Meridian's dependencies (pip install -r requirements.txt)." >&2
  exit 1
fi

# Bake the python path into the runner.
sed -i '' "s#^PYTHON_BIN=.*#PYTHON_BIN=\"\${MERIDIAN_PYTHON:-$PYTHON_BIN}\"#" "$RUNNER"
chmod +x "$RUNNER"

mkdir -p "$HOME/Library/LaunchAgents" "$REPO_DIR/reports"

{
cat <<PLISTHEAD
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>$RUNNER</string>
  </array>
  <key>WorkingDirectory</key><string>$REPO_DIR</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key><string>$(dirname "$PYTHON_BIN"):/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
  </dict>
  <key>StartCalendarInterval</key>
  <array>
PLISTHEAD
for wd in 1 2 3 4 5; do
  echo "    <dict><key>Weekday</key><integer>$wd</integer><key>Hour</key><integer>16</integer><key>Minute</key><integer>15</integer></dict>"
done
cat <<PLISTTAIL
  </array>
  <key>StandardOutPath</key><string>$REPO_DIR/reports/launchd_stdout.log</string>
  <key>StandardErrorPath</key><string>$REPO_DIR/reports/launchd_stderr.log</string>
</dict>
</plist>
PLISTTAIL
} > "$PLIST"

plutil -lint "$PLIST"
launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
launchctl bootstrap "$DOMAIN" "$PLIST"

echo
echo "Installed $LABEL -- weekdays at 4:15 PM local time."
echo "  python: $PYTHON_BIN"
echo "  log:    $REPO_DIR/reports/launchd_paper_options.log"
echo "Test it now with:  launchctl kickstart $DOMAIN/$LABEL"
