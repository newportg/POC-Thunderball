#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

SUBJECT_FILE = Path("reports/latest_email_subject.txt")
BODY_FILE = Path("reports/latest_prediction_report.txt")
#MAILTRAP_ENDPOINT = "https://send.api.mailtrap.io/api/send"
MAILTRAP_ENDPOINT = "https://sandbox.api.mailtrap.io/api/send"  # Updated endpoint for Mailtrap's sandbox environment


def main() -> int:
    token = os.getenv("MAILTRAP_API_TOKEN", "").strip()
    to_email = os.getenv("MAILTRAP_EMAIL_TO", "").strip() or os.getenv("ALERT_EMAIL_TO", "").strip()
    from_email = os.getenv("MAILTRAP_EMAIL_FROM", "").strip() or os.getenv("ALERT_EMAIL_FROM", "").strip()

    if not token or not to_email or not from_email:
        print("Skipping Mailtrap email: missing MAILTRAP_API_TOKEN and/or recipient/sender secrets")
        return 0

    subject = SUBJECT_FILE.read_text(encoding="utf-8").strip()
    body = BODY_FILE.read_text(encoding="utf-8")

    payload = {
        "from": {"email": from_email, "name": "Thunderball Predictor"},
        "to": [{"email": to_email}],
        "subject": subject,
        "text": body,
    }



    request = urllib.request.Request(
        MAILTRAP_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            status = response.status
            response_text = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Mailtrap HTTP error {exc.code}: {error_body}") from exc

    if status < 200 or status >= 300:
        raise SystemExit(f"Mailtrap send failed with status {status}: {response_text}")

    print("Mailtrap email sent successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


#curl --location --request POST 'https://sandbox.api.mailtrap.io/api/send/4467174' --header 'Authorization: Bearer 651630edaf7f9decc33dc8b14c77ef13' --header 'Content-Type: application/json' --data-raw '{"from":{"email":"hello@example.com","name":"Mailtrap Test"},"to":[{"email":"gary.newport@zoomalong.co.uk"}],"subject":"You are awesome!","text":"Congrats for sending test email with Mailtrap!","category":"Integration Test"}'
# 651630edaf7f9decc33dc8b14c77ef13