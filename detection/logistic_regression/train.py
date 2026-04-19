"""
detection/logistic_regression/train.py

Train a TF-IDF + Logistic Regression phishing detector.

Two modes:
    --mode baseline    Train on real training data only (Phase 3 baseline).
    --mode augmented   Train on emails_augmented_train.csv (Phase 4 augmented).

Saves a single joblib file containing the fitted vectorizer + classifier
to models/logistic_regression/<mode>.joblib.

Usage:
    python detection/logistic_regression/train.py --mode baseline
    python detection/logistic_regression/train.py --mode augmented
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Allow imports from detection/_shared.py when run from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _shared  # noqa: E402


SEED = 42


def build_pipeline() -> Pipeline:
    """TF-IDF (unigrams + bigrams, 5k features) + Logistic Regression."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=SEED,
        )),
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=["baseline", "augmented"])
    parser.add_argument("--output-dir", type=Path,
                        default=Path("models/logistic_regression"))
    args = parser.parse_args()

    # Load training data per mode ------------------------------------------
    if args.mode == "baseline":
        train_df, _ = _shared.load_real()
        print(f"Baseline mode: loaded {len(train_df)} real training rows "
              f"from emails_clean.csv (split=='train').")
    else:
        train_df = _shared.load_augmented()
        n_synth = train_df["source"].str.startswith("synthetic:").sum()
        n_real = len(train_df) - n_synth
        print(f"Augmented mode: loaded {len(train_df)} training rows "
              f"({n_real} real + {n_synth} synthetic).")

    # Clean text, encode labels -------------------------------------------
    X_train = train_df["text"].fillna("").apply(_shared.clean_text).values
    y_train = _shared.encode_labels(train_df["label"])

    print(f"Label distribution: phishing={int((y_train==1).sum())}, "
          f"legitimate={int((y_train==0).sum())}")

    # Fit pipeline ---------------------------------------------------------
    pipe = build_pipeline()
    print("\nFitting TF-IDF + LR pipeline...")
    pipe.fit(X_train, y_train)
    print("✓ Training complete.")

    # Save -----------------------------------------------------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"{args.mode}.joblib"
    joblib.dump(pipe, out_path)
    print(f"\n✓ Saved model to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
