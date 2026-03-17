#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

import requests

SUBJECT_FILE = Path("reports/latest_email_subject.txt")
BODY_FILE = Path("reports/latest_prediction_report.txt")

# Sandbox: https://sandbox.api.mailtrap.io/api/send/<inbox_id>
# Production: https://send.api.mailtrap.io/api/send
MAILTRAP_SANDBOX_INBOX_ID = os.getenv("MAILTRAP_SANDBOX_INBOX_ID", "4467174").strip()


def main() -> int:
    token = os.getenv("MAILTRAP_API_TOKEN", "").strip()
    to_email = os.getenv("MAILTRAP_EMAIL_TO", "").strip() or os.getenv("ALERT_EMAIL_TO", "").strip()
    from_email = os.getenv("MAILTRAP_EMAIL_FROM", "").strip() or os.getenv("ALERT_EMAIL_FROM", "").strip()
    use_sandbox = os.getenv("MAILTRAP_USE_SANDBOX", "true").strip().lower() not in {"false", "0", "no"}

    if not token or not to_email or not from_email:
        print("Skipping Mailtrap email: missing MAILTRAP_API_TOKEN and/or recipient/sender secrets")
        return 0

    subject = SUBJECT_FILE.read_text(encoding="utf-8").strip()
    body = BODY_FILE.read_text(encoding="utf-8")

    if use_sandbox:
        url = f"https://sandbox.api.mailtrap.io/api/send/{MAILTRAP_SANDBOX_INBOX_ID}"
        print(f"Sending via Mailtrap sandbox (inbox {MAILTRAP_SANDBOX_INBOX_ID})...")
    else:
        url = "https://send.api.mailtrap.io/api/send"
        print("Sending via Mailtrap production...")

    payload = {
        "from": {"email": from_email, "name": "Thunderball Predictor"},
        "to": [{"email": to_email}],
        "subject": subject,
        "text": body,
    }

    response = requests.post(
        url,
        json=payload,
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )

    if not response.ok:
        raise SystemExit(f"Mailtrap send failed {response.status_code}: {response.text}")

    print("Mailtrap email sent successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

