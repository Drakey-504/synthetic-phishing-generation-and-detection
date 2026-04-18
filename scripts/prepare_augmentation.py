"""
prepare_augmentation.py
Build the augmented training dataset and held-out synthetic test set for the
Phase 4 augmentation experiment.

The augmentation experiment asks: if ML detectors are retrained on real
phishing + some synthetic phishing, does the detection gap close?

To answer that credibly we need:
  - A held-out synthetic test set the retrained model has NEVER seen.
  - The existing real test set, kept untouched so we can verify that adding
    synthetic data didn't degrade real-phishing detection.

Inputs:
    data/processed/emails_clean.csv              (real emails with train/test splits)
    data/processed/synthetic_phishing_clean.csv  (457 synthetic phishing emails)

Outputs (to --outdir, default data/processed/):
    synthetic_split.csv            Source-of-truth: 457 rows + 'aug_split' column.
    synthetic_test.csv             Held-out synthetic phishing (~228 rows).
                                   Keeps method/sophistication/scenario columns
                                   for per-strata breakdowns at eval time.
    emails_augmented_train.csv     Real-train rows + ~229 synthetic-train rows
                                   labeled as phishing. Same schema as
                                   emails_clean.csv so trainers need no changes.

Evaluation targets for the augmented model:
    - Real test:       emails_clean.csv where split == 'test' (unchanged, 2,647 rows).
    - Synthetic test:  synthetic_test.csv (228 rows).

Usage:
    python scripts/prepare_augmentation.py

Reproducibility: SEED=42 pins the stratified split. Running this script
twice produces byte-identical outputs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# Design constants (see README / writeup section on augmentation design).
SYNTHETIC_TEST_SIZE = 0.50  # 50/50 split of the 457 synthetic emails.
SEED = 42


def stratified_split(df: pd.DataFrame) -> pd.DataFrame:
    """Split synthetic emails 50/50, stratified by method × sophistication.

    Returns df with a new column 'aug_split' ∈ {'train', 'test'}.
    """
    strata = df["method"].astype(str) + "|" + df["sophistication"].astype(str)

    train_idx, test_idx = train_test_split(
        df.index,
        test_size=SYNTHETIC_TEST_SIZE,
        stratify=strata,
        random_state=SEED,
    )

    df = df.copy()
    df["aug_split"] = ""
    df.loc[train_idx, "aug_split"] = "train"
    df.loc[test_idx, "aug_split"] = "test"
    return df


def print_strata_summary(df: pd.DataFrame) -> None:
    """Verify stratification produced balanced splits."""
    ct = df.groupby(["method", "sophistication", "aug_split"]).size() \
           .unstack(fill_value=0)
    ct["total"] = ct.sum(axis=1)
    ct["test_pct"] = (ct["test"] / ct["total"] * 100).round(1)
    print("\nStrata balance (should be ~50% test in every row):")
    print(ct.to_string())


def build_augmented_train(real_df: pd.DataFrame,
                          synth_train: pd.DataFrame) -> pd.DataFrame:
    """Real-train + synthetic-train, conformed to the emails_clean.csv schema."""
    real_train = real_df[real_df["split"] == "train"].copy()

    # Conform synthetic rows to the real-email schema:
    # emails_clean.csv columns: id, source, subject, text, label, split
    synth_conformed = synth_train.copy()
    synth_conformed["source"] = "synthetic:" + synth_conformed["method"]
    synth_conformed["label"] = "phishing"
    synth_conformed["split"] = "train"
    synth_conformed = synth_conformed[["id", "source", "subject", "text", "label", "split"]]

    combined = pd.concat([real_train, synth_conformed], ignore_index=True)

    # Shuffle so synthetic rows aren't clustered at the end of the file —
    # matters for any trainer that doesn't shuffle internally (and for
    # anyone who eyeballs the CSV).
    combined = combined.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return combined


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", type=Path,
                        default=Path("data/processed/emails_clean.csv"))
    parser.add_argument("--synthetic", type=Path,
                        default=Path("data/processed/synthetic_phishing_clean.csv"))
    parser.add_argument("--outdir", type=Path,
                        default=Path("data/processed"))
    args = parser.parse_args()

    if not args.real.exists():
        print(f"ERROR: real dataset not found: {args.real}", file=sys.stderr)
        return 2
    if not args.synthetic.exists():
        print(f"ERROR: synthetic dataset not found: {args.synthetic}", file=sys.stderr)
        return 2

    args.outdir.mkdir(parents=True, exist_ok=True)

    # -- Load --------------------------------------------------------------
    real = pd.read_csv(args.real)
    synth = pd.read_csv(args.synthetic)

    print(f"Real dataset:      {len(real)} rows  "
          f"(train={(real['split']=='train').sum()}, "
          f"test={(real['split']=='test').sum()})")
    print(f"Synthetic dataset: {len(synth)} rows")

    # -- Stratified 50/50 split of synthetic ------------------------------
    synth_split = stratified_split(synth)
    synth_split.to_csv(args.outdir / "synthetic_split.csv", index=False)
    print(f"\n✓ Wrote {args.outdir / 'synthetic_split.csv'}  "
          f"(source of truth: 457 rows + aug_split column)")

    print_strata_summary(synth_split)
    n_synth_train = (synth_split["aug_split"] == "train").sum()
    n_synth_test = (synth_split["aug_split"] == "test").sum()
    print(f"\nTotals: synthetic train = {n_synth_train}, synthetic test = {n_synth_test}")

    # -- Held-out synthetic test set --------------------------------------
    # Keep all original columns so eval scripts can do per-method /
    # per-sophistication / per-scenario breakdowns on the held-out set.
    synth_test = synth_split[synth_split["aug_split"] == "test"].copy()
    synth_test.to_csv(args.outdir / "synthetic_test.csv", index=False)
    print(f"\n✓ Wrote {args.outdir / 'synthetic_test.csv'}  ({len(synth_test)} rows)")

    # -- Augmented training file ------------------------------------------
    synth_train = synth_split[synth_split["aug_split"] == "train"].copy()
    combined = build_augmented_train(real, synth_train)

    n_real = (~combined["source"].str.startswith("synthetic:")).sum()
    n_synth = combined["source"].str.startswith("synthetic:").sum()
    pct_synth = n_synth / len(combined) * 100
    phish_n = (combined["label"] == "phishing").sum()
    legit_n = (combined["label"] == "legitimate").sum()

    combined.to_csv(args.outdir / "emails_augmented_train.csv", index=False)
    print(f"\n✓ Wrote {args.outdir / 'emails_augmented_train.csv'}  "
          f"({len(combined)} rows)")
    print(f"    - {n_real} real + {n_synth} synthetic "
          f"({pct_synth:.1f}% synthetic)")
    print(f"    - {phish_n} phishing + {legit_n} legitimate "
          f"({phish_n/len(combined)*100:.1f}% phishing)")

    # -- How to use -------------------------------------------------------
    print("\n" + "=" * 66)
    print("HOW TO USE")
    print("=" * 66)
    print("""
For each ML detector (LR, XGBoost, DistilBERT), train TWO models:

  1. Baseline:  trained on emails_clean.csv where split=='train'
  2. Augmented: trained on emails_augmented_train.csv (no split filter —
                all rows are training data)

Evaluate BOTH models on BOTH test sets:

  Real test:       emails_clean.csv where split=='test'  (2,647 rows)
  Synthetic test:  synthetic_test.csv                    (228 rows)

Headline result: detection rate on synthetic_test
  - Baseline model → the 'detection gap' number from Phase 3.
  - Augmented model → should be much higher if augmentation worked.

Sanity check: detection rate on real test should stay the same or improve
              for the augmented model. If it drops, the model over-fit to
              synthetic patterns at the cost of real detection.

Note: SpamAssassin is rule-based and isn't retrained — the augmentation
experiment only applies to the ML detectors.
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
