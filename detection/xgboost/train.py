"""
detection/xgboost/train.py

Train a TF-IDF + XGBoost phishing detector.

Two modes:
    --mode baseline    Train on real training data only (Phase 3 baseline).
    --mode augmented   Train on emails_augmented_train.csv (Phase 4 augmented).

Saves a single joblib file containing the fitted vectorizer + classifier
to models/xgboost/<mode>.joblib.

Usage:
    python detection/xgboost/train.py --mode baseline
    python detection/xgboost/train.py --mode augmented
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _shared  # noqa: E402


SEED = 42


def build_pipeline() -> Pipeline:
    """TF-IDF (unigrams + bigrams, 5k features) + XGBoost.

    Hyperparameters from Sathwika's baseline notebook: 300 trees, depth 6,
    lr 0.1, 0.8 row/feature subsampling.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
        )),
        ("clf", XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=-1,
        )),
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=["baseline", "augmented"])
    parser.add_argument("--output-dir", type=Path,
                        default=Path("models/xgboost"))
    args = parser.parse_args()

    if args.mode == "baseline":
        train_df, _ = _shared.load_real()
        print(f"Baseline mode: loaded {len(train_df)} real training rows.")
    else:
        train_df = _shared.load_augmented()
        n_synth = train_df["source"].str.startswith("synthetic:").sum()
        n_real = len(train_df) - n_synth
        print(f"Augmented mode: loaded {len(train_df)} rows "
              f"({n_real} real + {n_synth} synthetic).")

    X_train = train_df["text"].fillna("").apply(_shared.clean_text).values
    y_train = _shared.encode_labels(train_df["label"])

    print(f"Label distribution: phishing={int((y_train==1).sum())}, "
          f"legitimate={int((y_train==0).sum())}")

    pipe = build_pipeline()
    print("\nFitting TF-IDF + XGBoost pipeline...")
    pipe.fit(X_train, y_train)
    print("✓ Training complete.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"{args.mode}.joblib"
    joblib.dump(pipe, out_path)
    print(f"\n✓ Saved model to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
