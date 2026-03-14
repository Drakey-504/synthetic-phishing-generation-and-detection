"""
preprocess.py
Cleans and preprocesses the combined raw email dataset.

Input:
    data/processed/emails_raw.csv

Output:
    data/processed/emails_clean.csv

Steps:
    1. Strip any remaining HTML tags
    2. Remove header leakage (X-Spam-*, etc.)
    3. Normalize whitespace
    4. Drop non-English emails
    5. Remove exact text duplicates
    6. Filter very short emails (< 10 words)
    7. Filter very long emails (> 5000 chars)
    8. Print class distribution summary
"""

import pandas as pd
import re
from bs4 import BeautifulSoup
from pathlib import Path


INPUT = Path("data/processed/emails_raw.csv")
OUTPUT = Path("data/processed/emails_clean.csv")


def strip_html(text):
    """Remove any remaining HTML tags."""
    if not isinstance(text, str):
        return ""
    if "<" in text and ">" in text:
        try:
            return BeautifulSoup(text, "html.parser").get_text(separator=" ")
        except Exception:
            return text
    return text


def remove_header_leakage(text):
    """Remove leaked email headers from body text."""
    if not isinstance(text, str):
        return ""
    # Remove lines that look like leaked headers
    leaked_patterns = [
        r"^X-Spam-\S+:.*$",
        r"^X-Virus-\S+:.*$",
        r"^X-HE-\S+:.*$",
        r"^X-Filterd-\S+:.*$",
        r"^X-FDA:.*$",
        r"^X-IMAP:.*$",
        r"^Status: \w+$",
        r"^X-Status:.*$",
        r"^X-Keywords:.*$",
        r"^X-UID: \d+$",
    ]
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        is_leaked = False
        for pattern in leaked_patterns:
            if re.match(pattern, line.strip()):
                is_leaked = True
                break
        if not is_leaked:
            cleaned.append(line)
    return "\n".join(cleaned)


def normalize_whitespace(text):
    """Collapse multiple spaces/newlines, strip edges."""
    if not isinstance(text, str):
        return ""
    # Replace tabs with spaces
    text = text.replace("\t", " ")
    # Collapse multiple spaces into one
    text = re.sub(r" {2,}", " ", text)
    # Collapse 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip each line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    return text.strip()


def is_english(text):
    """
    Simple heuristic: check if the majority of characters are ASCII.
    Emails with >10% non-ASCII are likely non-English.
    """
    if not isinstance(text, str) or len(text) == 0:
        return False
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return (ascii_chars / len(text)) > 0.9


def word_count(text):
    """Count words in text."""
    if not isinstance(text, str):
        return 0
    return len(text.split())


def main():
    print("=" * 60)
    print("EMAIL PREPROCESSING")
    print("=" * 60)

    df = pd.read_csv(INPUT)
    print(f"\nLoaded {len(df)} emails from {INPUT}")
    print(f"  Phishing:   {(df['label'] == 'phishing').sum()}")
    print(f"  Legitimate: {(df['label'] == 'legitimate').sum()}")

    # ── Step 1: Strip remaining HTML ────────────────────────────────────
    print(f"\n[1/7] Stripping remaining HTML...")
    df["text"] = df["text"].apply(strip_html)

    # ── Step 2: Remove header leakage ───────────────────────────────────
    print("[2/7] Removing header leakage...")
    df["text"] = df["text"].apply(remove_header_leakage)

    # ── Step 3: Normalize whitespace ────────────────────────────────────
    print("[3/7] Normalizing whitespace...")
    df["text"] = df["text"].apply(normalize_whitespace)
    df["subject"] = df["subject"].fillna("").apply(normalize_whitespace)

    # ── Step 4: Drop non-English emails ─────────────────────────────────
    print("[4/7] Dropping non-English emails...")
    before = len(df)
    df = df[df["text"].apply(is_english)]
    dropped = before - len(df)
    print(f"  Dropped {dropped} non-English emails")

    # ── Step 5: Remove exact text duplicates ────────────────────────────
    print("[5/7] Removing duplicate emails...")
    before = len(df)
    df = df.drop_duplicates(subset="text", keep="first")
    dropped = before - len(df)
    print(f"  Dropped {dropped} exact duplicates")

    # ── Step 6: Filter very short emails (< 10 words) ──────────────────
    print("[6/7] Filtering very short emails (< 10 words)...")
    before = len(df)
    df = df[df["text"].apply(word_count) >= 10]
    dropped = before - len(df)
    print(f"  Dropped {dropped} short emails")

    # ── Step 7: Filter very long emails (> 5000 chars) ──────────────────
    print("[7/7] Truncating very long emails (> 5000 chars)...")
    long_count = (df["text"].str.len() > 5000).sum()
    df["text"] = df["text"].apply(lambda x: x[:5000] if isinstance(x, str) and len(x) > 5000 else x)
    print(f"  Truncated {long_count} emails")

    # ── Reset index ─────────────────────────────────────────────────────
    df = df.reset_index(drop=True)

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"FINAL DATASET")
    print(f"{'=' * 60}")
    print(f"Total: {len(df)} emails")

    print(f"\nBy label:")
    label_counts = df["label"].value_counts()
    for label, count in label_counts.items():
        pct = count / len(df) * 100
        print(f"  {label}: {count} ({pct:.1f}%)")

    print(f"\nBy source:")
    source_counts = df.groupby(["source", "label"]).size()
    print(source_counts.to_string())

    print(f"\nText length stats:")
    print(f"  Mean:   {df['text'].str.len().mean():.0f} chars")
    print(f"  Median: {df['text'].str.len().median():.0f} chars")
    print(f"  Min:    {df['text'].str.len().min()} chars")
    print(f"  Max:    {df['text'].str.len().max()} chars")

    print(f"\nWord count stats:")
    wc = df["text"].apply(word_count)
    print(f"  Mean:   {wc.mean():.0f} words")
    print(f"  Median: {wc.median():.0f} words")

    # Class balance check
    ratio = label_counts.min() / label_counts.max()
    if ratio < 0.6:
        print(f"\n⚠ Class imbalance detected (ratio: {ratio:.2f})")
        print(f"  Consider downsampling the majority class or using class weights during training.")
    else:
        print(f"\n✓ Class balance OK (ratio: {ratio:.2f})")

    # ── Save ────────────────────────────────────────────────────────────
    df.to_csv(OUTPUT, index=False)
    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    main()
