#!/usr/bin/env bash
###############################################################################
# install_until_ready.sh
#
# Keep retrying the installers for Node/npm and Google-Chrome, and
# run the Selenium check, until *all three* succeed.
###############################################################################
set -euo pipefail                       # abort on script syntax errors
IFS=$'\n\t'

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
NODE_SCRIPT="${DIR}/install_node.sh"
CHROME_SCRIPT="${DIR}/install_chrome.sh"
SELENIUM_CHECK="${DIR}/check_selenium.py"

SLEEP_BETWEEN_TRIES=30                 # seconds

echo "▶️  Starting repeated install/check loop … (Ctrl-C to abort)"

while true; do
  NEED_RETRY=false

  # ────────────────── npm ──────────────────
  if ! command -v npm >/dev/null 2>&1; then
    echo "→ npm not found – running install_node.sh"
    bash "$NODE_SCRIPT" || true              # keep looping if it fails
    NEED_RETRY=true
  else
    echo "✓ npm $(npm -v)"
  fi

  # ─────────────── Google-Chrome ───────────
  if ! command -v google-chrome >/dev/null 2>&1; then
    echo "→ google-chrome not found – running install_chrome.sh"
    bash "$CHROME_SCRIPT" || true
    NEED_RETRY=true
  else
    echo "✓ $(google-chrome --version)"
  fi

  # ───────────── Selenium sanity check ─────────────
  # Only meaningful if Chrome exists; but running twice hurts nothing.
  if python "$SELENIUM_CHECK"; then
    echo "✓ Selenium check passed"
  else
    echo "→ Selenium check failed – will retry"
    NEED_RETRY=true
  fi

  # ───────────── exit or sleep ─────────────
  if [ "$NEED_RETRY" = false ]; then
    echo "✅ npm, Chrome, and Selenium are all ready. Done."
    exit 0
  fi

  echo "⚠️  At least one step failed; retrying in ${SLEEP_BETWEEN_TRIES}s …"
  sleep "$SLEEP_BETWEEN_TRIES"
done
