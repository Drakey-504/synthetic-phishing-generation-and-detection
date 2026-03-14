"""
split_data.py
Adds a train/test split column to the cleaned email dataset.

Input:
    data/processed/emails_clean.csv

Output:
    data/processed/emails_clean.csv (updated with 'split' column)

Split: 80/20 stratified by label.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path


INPUT = Path("data/processed/emails_clean.csv")


def main():
    df = pd.read_csv(INPUT)
    print(f"Loaded {len(df)} emails")

    # Stratified 80/20 split
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=0.2,
        stratify=df["label"],
        random_state=42
    )

    df["split"] = ""
    df.loc[train_idx, "split"] = "train"
    df.loc[test_idx, "split"] = "test"

    # Summary
    print(f"\n{'='*50}")
    print("SPLIT SUMMARY")
    print(f"{'='*50}")

    for split in ["train", "test"]:
        subset = df[df["split"] == split]
        phishing = (subset["label"] == "phishing").sum()
        legit = (subset["label"] == "legitimate").sum()
        print(f"\n{split.upper()}: {len(subset)} emails")
        print(f"  Phishing:   {phishing} ({phishing/len(subset)*100:.1f}%)")
        print(f"  Legitimate: {legit} ({legit/len(subset)*100:.1f}%)")

    # Save
    df.to_csv(INPUT, index=False)
    print(f"\nSaved to {INPUT}")


if __name__ == "__main__":
    main()
