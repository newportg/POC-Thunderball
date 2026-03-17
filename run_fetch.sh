#!/bin/bash
# Simple wrapper to run the lottery fetcher
# Usage: ./run_fetch.sh

cd "$(dirname "$0")"
/workspaces/POC-Thunderball/.venv/bin/python fetch_lottery_results.py
