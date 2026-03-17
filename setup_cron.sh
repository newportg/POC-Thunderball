#!/bin/bash
# Setup automated lottery result fetching via cron
# Runs on: Tuesday, Wednesday, Friday, Saturday at 9:00 PM (21:00)

SCRIPT_PATH="/workspaces/POC-Thunderball/fetch_lottery_results.py"
PYTHON_PATH="/workspaces/POC-Thunderball/.venv/bin/python"
LOG_FILE="/workspaces/POC-Thunderball/fetch_lottery.log"

# Create cron job that runs at 21:00 on Tue, Wed, Fri, Sat
CRON_SCHEDULE="0 21 * * 2,3,5,6"
CRON_COMMAND="$PYTHON_PATH $SCRIPT_PATH >> $LOG_FILE 2>&1"

# Install cron job
(crontab -l 2>/dev/null | grep -v "$SCRIPT_PATH"; echo "$CRON_SCHEDULE $CRON_COMMAND") | crontab -

echo "✓ Cron job installed to run lottery fetcher at 9 PM on Tue/Wed/Fri/Sat"
echo ""
echo "To verify installation, run: crontab -l"
echo "To view logs, run: tail -f $LOG_FILE"
