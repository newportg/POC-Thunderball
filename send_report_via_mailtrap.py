#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

import mailtrap as mt

SUBJECT_FILE = Path("reports/latest_email_subject.txt")
BODY_FILE = Path("reports/latest_prediction_report.txt")

# Inbox ID for sandbox mode (found in your Mailtrap inbox URL)
MAILTRAP_SANDBOX_INBOX_ID = int(os.getenv("MAILTRAP_SANDBOX_INBOX_ID", "4467174"))


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

    mail = mt.Mail(
        sender=mt.Address(email=from_email, name="Thunderball Predictor"),
        to=[mt.Address(email=to_email)],
        subject=subject,
        text=body,
    )

    if use_sandbox:
        client = mt.MailtrapClient(token=token, sandbox=True, inbox_id=str(MAILTRAP_SANDBOX_INBOX_ID))
        print(f"Sending via Mailtrap sandbox (inbox {MAILTRAP_SANDBOX_INBOX_ID})...")
    else:
        client = mt.MailtrapClient(token=token)
        print("Sending via Mailtrap production...")

    client.send(mail)
    print("Mailtrap email sent successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

