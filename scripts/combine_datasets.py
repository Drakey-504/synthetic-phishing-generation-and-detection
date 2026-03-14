"""
combine_datasets.py
Combines the four parsed CSVs into a single unified dataset.

Usage:
    Run the four parsers first, then:
    python scripts/combine_datasets.py

Output:
    data/processed/emails_raw.csv
"""

import pandas as pd
from pathlib import Path


PROCESSED_DIR = Path("data/processed")
OUTPUT = PROCESSED_DIR / "emails_raw.csv"

SOURCES = [
    PROCESSED_DIR / "nazario.csv",
    PROCESSED_DIR / "nigerian.csv",
    PROCESSED_DIR / "enron.csv",
    PROCESSED_DIR / "spamassassin.csv",
]


def main():
    print("Combining parsed datasets...\n")

    dfs = []
    for path in SOURCES:
        if not path.exists():
            print(f"  WARNING: {path} not found — skipping. Run the parser first.")
            continue
        df = pd.read_csv(path)
        print(f"  {path.name}: {len(df)} emails")
        dfs.append(df)

    if not dfs:
        print("No datasets found. Run the individual parsers first.")
        return

    combined = pd.concat(dfs, ignore_index=True)

    # Summary
    print(f"\n{'=' * 40}")
    print(f"Total: {len(combined)} emails")
    print(f"\nBy label:")
    print(combined["label"].value_counts().to_string())
    print(f"\nBy source:")
    print(combined["source"].value_counts().to_string())

    # Save
    combined.to_csv(OUTPUT, index=False)
    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    main()
