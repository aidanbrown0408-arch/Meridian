#!/bin/bash
# Install (or remove) a launchd job that serves reports/ (the dashboard
# George writes) over plain HTTP on your local network, so the "latest.html"
# link actually works from a phone or another device on the same Wi-Fi --
# not just by opening the file on this Mac. Stays LAN-only: bound to all
# interfaces so it's reachable by IP/hostname on your network, but your
# home router still blocks it from the open internet (no port forwarding
# is set up here). The repo is public on GitHub; this deliberately never
# touches git -- nothing here gets pushed or becomes internet-visible.
#
#   bash scripts/install_dashboard_server.sh            # install / reinstall
#   bash scripts/install_dashboard_server.sh uninstall   # remove
#   bash scripts/install_dashboard_server.sh status      # is it running?
set -euo pipefail

LABEL="com.meridian.dashboard-server"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPORTS_DIR="$REPO_DIR/reports"
DOMAIN="gui/$(id -u)"
PORT="${MERIDIAN_DASHBOARD_PORT:-8765}"

if [ "${1:-}" = "uninstall" ]; then
  launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
  rm -f "$PLIST"
  echo "Removed $LABEL."
  exit 0
fi

if [ "${1:-}" = "status" ]; then
  if launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1; then
    echo "$LABEL is loaded."
    launchctl print "$DOMAIN/$LABEL" | grep -E "state =|pid =" || true
  else
    echo "$LABEL is NOT loaded -- it will not serve anything until you reinstall it:"
    echo "  bash scripts/install_dashboard_server.sh"
  fi
  exit 0
fi

PYTHON_BIN="${MERIDIAN_PYTHON:-$(command -v python3 || true)}"
if [ -z "$PYTHON_BIN" ]; then
  echo "Could not find python3 on PATH. Re-run with MERIDIAN_PYTHON=/path/to/python3." >&2
  exit 1
fi

mkdir -p "$HOME/Library/LaunchAgents" "$REPORTS_DIR"

# Prefer the Mac's Bonjour/mDNS name (stable, works from other devices on
# the LAN as <name>.local) over a raw IP, which can change with DHCP.
HOST="$(scutil --get LocalHostName 2>/dev/null || true)"
if [ -n "$HOST" ]; then
  HOST="$HOST.local"
else
  HOST="$(hostname -s 2>/dev/null || hostname)"
fi
DASHBOARD_URL="http://$HOST:$PORT/latest.html"

{
cat <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>$PYTHON_BIN</string>
    <string>-m</string>
    <string>http.server</string>
    <string>$PORT</string>
    <string>--bind</string>
    <string>0.0.0.0</string>
    <string>--directory</string>
    <string>$REPORTS_DIR</string>
  </array>
  <key>WorkingDirectory</key><string>$REPO_DIR</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$REPORTS_DIR/dashboard_server_stdout.log</string>
  <key>StandardErrorPath</key><string>$REPORTS_DIR/dashboard_server_stderr.log</string>
</dict>
</plist>
PLIST
} > "$PLIST"

plutil -lint "$PLIST"
launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
launchctl bootstrap "$DOMAIN" "$PLIST"

# Point the Slack standup footer at the real link instead of a local path.
# config.yaml stays committed to a PUBLIC repo, so only ever write the LAN
# hostname/port here -- never an external URL, a token, or anything else
# sensitive.
CONFIG="$REPO_DIR/config/config.yaml"
if [ -f "$CONFIG" ]; then
  sed -i '' "s#^  dashboard_url: .*#  dashboard_url: \"$DASHBOARD_URL\"#" "$CONFIG"
fi

sleep 1
if launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1; then
  echo
  echo "Installed and verified running: $LABEL"
  echo "  Dashboard: $DASHBOARD_URL"
  echo "  (reachable from any device on your Wi-Fi; not from the open internet)"
  echo "  config/config.yaml's slack.dashboard_url updated -- the Slack standup"
  echo "  footer will link here starting with the next paper run."
  echo "Check on it later with:  bash scripts/install_dashboard_server.sh status"
else
  echo
  echo "WARNING: bootstrap did not report an error, but $LABEL is NOT loaded." >&2
  echo "Check Console.app (search 'com.meridian') or re-run this script." >&2
  exit 1
fi
