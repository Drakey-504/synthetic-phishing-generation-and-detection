"""
parse_nazario.py
Parses the Nazario phishing corpus (mbox files) into CSV.

Usage:
    python scripts/parse_nazario.py

Output:
    data/processed/nazario.csv
"""

import mailbox
import csv
from pathlib import Path
from email_utils import extract_subject, extract_body


RAW_DIR = Path("data/raw/nazario")
OUTPUT = Path("data/processed/nazario.csv")


def main():
    records = []
    mbox_files = sorted(f for f in RAW_DIR.iterdir() if not f.name.startswith("."))
    print(f"[Nazario] Found {len(mbox_files)} files")

    for mbox_path in mbox_files:
        print(f"  Parsing {mbox_path.name}...")
        mbox = mailbox.mbox(str(mbox_path))
        count = 0

        for i, msg in enumerate(mbox):
            from_header = msg.get("From", "")
            subject = extract_subject(msg)

            # Skip internal mbox metadata messages
            if "MAILER-DAEMON" in from_header or "FOLDER INTERNAL DATA" in subject:
                continue

            body = extract_body(msg)

            if not body or len(body.strip()) < 10:
                continue

            records.append({
                "id": f"nazario_{mbox_path.name}_{i}",
                "source": "nazario",
                "subject": subject,
                "text": body,
                "label": "phishing"
            })
            count += 1

        print(f"    → {count} emails")

    # Write CSV
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "source", "subject", "text", "label"])
        writer.writeheader()
        writer.writerows(records)

    print(f"\n[Nazario] Total: {len(records)} phishing emails → {OUTPUT}")


if __name__ == "__main__":
    main()
