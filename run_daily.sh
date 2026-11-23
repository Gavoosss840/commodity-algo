#!/usr/bin/env bash
# Daily run on macOS/Linux (cron example: 30 21 * * 1-5 /path/to/run_daily.sh)
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
python -m commodity_algo run
git add output && git commit -m "Daily fiche $(date +%F)" >/dev/null 2>&1 && git push >/dev/null 2>&1 || true
