# Automated Lottery Results Fetcher

## Overview
This system automatically pulls the latest Thunderball draw results from the National Lottery and updates your local CSV file.

**Draw Schedule:** Tuesday, Wednesday, Friday, Saturday  
**Results Published:** ~10:00 PM GMT

## Files
- `fetch_lottery_results.py` - Main fetcher script
- `setup_cron.sh` - Cron job installer (for Linux/macOS)

## Quick Start

### Manual Run
```bash
cd /workspaces/POC-Thunderball
python fetch_lottery_results.py
```

Output example:
```
✓ Added 2 new draw(s) to data/thunderball-draw-history.csv
  Draw 3870: 15-Mar-2026 | 3-7-15-28-34 TB:11
  Draw 3871: 17-Mar-2026 | 1-12-20-29-38 TB:8
```

## Production Deployment

### Option 1: Linux/macOS with Cron (Recommended)
```bash
bash setup_cron.sh
```
This installs a cron job that runs at **21:00 (9 PM) on Tue/Wed/Fri/Sat**.

Verify:
```bash
crontab -l
```

View logs:
```bash
tail -f fetch_lottery.log
```

### Option 2: Docker/Container
Add to your Dockerfile:
```dockerfile
# Install cron daemon
RUN apt-get install -y cron

# Copy fetcher
COPY fetch_lottery_results.py /app/
COPY setup_cron.sh /app/

# Setup cron on container start
RUN bash /app/setup_cron.sh

# Start cron in foreground
CMD ["cron", "-f"]
```

Or use a supervisor service to run the script periodically.

### Option 3: GitHub Actions (Cloud)
Create `.github/workflows/fetch-lottery.yml`:
```yaml
name: Fetch Lottery Results

on:
  schedule:
    # Cron: 21:00 UTC on Tue(2), Wed(3), Fri(5), Sat(6)
    - cron: "0 21 * * 2,3,5,6"

jobs:
  fetch:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install lxml requests
      - run: python fetch_lottery_results.py
      - name: Commit changes
        run: |
          git config --local user.email "action@github.com"
          git config --local user.name "GitHub Action"
          git add data/thunderball-draw-history.csv
          git diff --quiet && git diff --staged --quiet || git commit -m "Update lottery results"
          git push
```

### Option 4: AWS Lambda
Package the script and:
1. Create Lambda function with Python 3.11 runtime
2. Add `lxml` and `requests` to deployment package
3. Set CloudWatch Events to trigger on Tue/Wed/Fri/Sat at 21:00 UTC
4. Update S3 bucket with new CSV file on completion

### Option 5: Systemd Timer (Linux)
Create `/etc/systemd/system/fetch-lottery.Service`:
```ini
[Unit]
Description=Fetch National Lottery Thunderball Results
After=network-online.target

[Service]
Type=oneshot
ExecStart=/workspaces/POC-Thunderball/.venv/bin/python /workspaces/POC-Thunderball/fetch_lottery_results.py
User=ubuntu
StandardOutput=journal
StandardError=journal
```

Create `/etc/systemd/system/fetch-lottery.timer`:
```ini
[Unit]
Description=Run Fetch Lottery at 9 PM on draw days

[Timer]
OnCalendar=Tue,Wed,Fri,Sat *-*-* 21:00:00
Unit=fetch-lottery.service

[Install]
WantedBy=timers.target
```

Enable and start:
```bash
sudo systemctl enable fetch-lottery.timer
sudo systemctl start fetch-lottery.timer
systemctl status fetch-lottery.timer
```

## How It Works

1. **Fetch:** Retrieves XML from `https://www.national-lottery.co.uk/results/thunderball/draw-history/xml`
2. **Parse:** Extracts draw number, date, 5 main balls, thunderball, ball set, and machine
3. **Deduplicate:** Checks if draw already exists in CSV (by DrawNumber)
4. **Update:** Prepends new draws to CSV (keeps latest first)
5. **Log:** Prints summary of added draws and any errors

## Data Format

Input (XML):
```xml
<draw>
  <draw-number>3869</draw-number>
  <draw-date>2026-03-14</draw-date>
  <draw-machine>Excalibur4</draw-machine>
  <balls>
    <set>T6</set>
    <ball number="1">5</ball>
    <ball number="2">8</ball>
    ...
    <bonus-ball type="thunderball" number="1">5</bonus-ball>
  </balls>
</draw>
```

Output (CSV):
```csv
DrawDate,Ball 1,Ball 2,Ball 3,Ball 4,Ball 5,Thunderball,Ball Set,Machine,DrawNumber
14-Mar-2026,5,8,24,26,32,5,T6,Excalibur4,3869
```

## Troubleshooting

### SSL Certificate Error
The script disables SSL verification for development. For production, install CA certificates or use a trusted environment.

### Network Issues
The script has a 10-second timeout. If the National Lottery site is slow, increase `timeout=10` to `timeout=30` in the code.

### Date Format Issues
Ensure your system locale supports "DD-Mon-YYYY" format (e.g., 14-Mar-2026). The script uses Python's `strftime("%d-%b-%Y")`.

### CSV Corruption
The script reads the entire CSV into memory before writing. If this fails, a backup is recommended:
```bash
cp data/thunderball-draw-history.csv data/thunderball-draw-history.csv.bak
```

## Integration with Streamlit App

Once new draws are added to the CSV:
1. Restart your Streamlit app (or it auto-reloads on file change)
2. Upload the updated CSV in the UI
3. Run the Rolling 9-Draw Timeline with fresh data

No code changes needed—the app automatically reads the latest data.
