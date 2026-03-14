"""
parse_enron.py
Parses the Enron email dataset into CSV.
Samples up to 5000 legitimate emails from inbox/sent folders.

Usage:
    python scripts/parse_enron.py

Output:
    data/processed/enron.csv
"""

import csv
import random
from pathlib import Path
from tqdm import tqdm
from email_utils import parse_email_file


RAW_DIR = Path("data/raw/enron/maildir")
OUTPUT = Path("data/processed/enron.csv")
MAX_EMAILS = 5000
SEED = 42

# Only pull from these folders — real email content, not calendar/contacts/etc.
TARGET_FOLDERS = {"inbox", "sent_items", "sent", "_sent_mail"}


def main():
    records = []
    print(f"[Enron] Scanning {RAW_DIR}")

    # Collect all email file paths from target folders
    all_files = []
    users = sorted(RAW_DIR.iterdir())

    for user_dir in users:
        if not user_dir.is_dir() or user_dir.name.startswith("."):
            continue
        for folder in user_dir.iterdir():
            if not folder.is_dir() or folder.name not in TARGET_FOLDERS:
                continue
            for email_file in folder.iterdir():
                if email_file.is_file() and not email_file.name.startswith("."):
                    all_files.append(email_file)

    print(f"  Found {len(all_files)} email files in target folders")

    # Randomly sample
    random.seed(SEED)
    if len(all_files) > MAX_EMAILS:
        all_files = random.sample(all_files, MAX_EMAILS)
        print(f"  Sampled {MAX_EMAILS} emails")

    # Parse each file
    for filepath in tqdm(all_files, desc="  Parsing"):
        subject, body = parse_email_file(filepath)

        if not body or len(body.strip()) < 10:
            continue

        relative = filepath.relative_to(RAW_DIR)
        record_id = f"enron_{str(relative).replace('/', '_')}"

        records.append({
            "id": record_id,
            "source": "enron",
            "subject": subject,
            "text": body,
            "label": "legitimate"
        })

    # Write CSV
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "source", "subject", "text", "label"])
        writer.writeheader()
        writer.writerows(records)

    print(f"\n[Enron] Total: {len(records)} legitimate emails → {OUTPUT}")


if __name__ == "__main__":
    main()
