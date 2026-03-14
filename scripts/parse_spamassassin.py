"""
parse_spamassassin.py
Parses SpamAssassin easy_ham directories into CSV.

Usage:
    python scripts/parse_spamassassin.py

Output:
    data/processed/spamassassin.csv
"""

import csv
from pathlib import Path
from email_utils import parse_email_file


RAW_DIR = Path("data/raw/spamassassin")
OUTPUT = Path("data/processed/spamassassin.csv")


def main():
    records = []
    print(f"[SpamAssassin] Scanning {RAW_DIR}")

    ham_dirs = sorted(d for d in RAW_DIR.iterdir() if d.is_dir() and not d.name.startswith("."))

    for ham_dir in ham_dirs:
        print(f"  Parsing {ham_dir.name}/...")
        count = 0

        for email_file in sorted(ham_dir.iterdir()):
            if not email_file.is_file() or email_file.name.startswith("."):
                continue
            # Skip the cmds metadata file
            if email_file.name == "cmds":
                continue

            subject, body = parse_email_file(email_file)

            if not body or len(body.strip()) < 10:
                continue

            records.append({
                "id": f"spamassassin_{ham_dir.name}_{email_file.name}",
                "source": "spamassassin",
                "subject": subject,
                "text": body,
                "label": "legitimate"
            })
            count += 1

        print(f"    → {count} emails")

    # Write CSV
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "source", "subject", "text", "label"])
        writer.writeheader()
        writer.writerows(records)

    print(f"\n[SpamAssassin] Total: {len(records)} legitimate emails → {OUTPUT}")


if __name__ == "__main__":
    main()
