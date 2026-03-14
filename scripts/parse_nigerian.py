"""
parse_nigerian.py
Parses the Nigerian Fraud / 419 email dataset into CSV.

Usage:
    python scripts/parse_nigerian.py

Output:
    data/processed/nigerian.csv
"""

import mailbox
import csv
from pathlib import Path
from email_utils import extract_subject, extract_body


RAW_FILE = Path("data/raw/nigerian_fraud/fradulent_emails.txt")
OUTPUT = Path("data/processed/nigerian.csv")


def main():
    records = []
    print(f"[Nigerian Fraud] Parsing {RAW_FILE}")

    mbox = mailbox.mbox(str(RAW_FILE))
    count = 0

    for i, msg in enumerate(mbox):
        from_header = msg.get("From", "")
        subject = extract_subject(msg)

        if "MAILER-DAEMON" in from_header or "FOLDER INTERNAL DATA" in subject:
            continue

        body = extract_body(msg)

        if not body or len(body.strip()) < 10:
            continue

        records.append({
            "id": f"nigerian_{i}",
            "source": "nigerian",
            "subject": subject,
            "text": body,
            "label": "phishing"
        })
        count += 1

    # Write CSV
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "source", "subject", "text", "label"])
        writer.writeheader()
        writer.writerows(records)

    print(f"[Nigerian Fraud] Total: {count} phishing emails → {OUTPUT}")


if __name__ == "__main__":
    main()
