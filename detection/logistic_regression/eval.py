"""
detection/logistic_regression/eval.py

Evaluate a trained LR model on the real test set and the synthetic phishing
set. Writes the four standard result CSVs:

    results/logistic_regression/<mode>/real_test.csv
    results/logistic_regression/<mode>/synthetic.csv
    results/logistic_regression/<mode>/baseline.csv
    results/logistic_regression/<mode>/detection_gap.csv

Also prints a summary matching the SpamAssassin gap.py output format.

The synthetic set evaluated depends on --mode:
    baseline   -> synthetic_phishing_clean.csv (all 457)
    augmented  -> synthetic_split.csv where aug_split=='test' (229 held-out)

This difference is critical: the augmented model has SEEN the synthetic
train rows, so evaluating against the full 457 would leak training data.

Usage:
    python detection/logistic_regression/eval.py --mode baseline
    python detection/logistic_regression/eval.py --mode augmented
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _shared  # noqa: E402


DETECTOR = "logistic_regression"
THRESHOLD = 0.5  # LR default decision threshold.


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=["baseline", "augmented"])
    parser.add_argument("--model-dir", type=Path,
                        default=Path("models/logistic_regression"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/logistic_regression"))
    args = parser.parse_args()

    # Load model -----------------------------------------------------------
    model_path = args.model_dir / f"{args.mode}.joblib"
    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}", file=sys.stderr)
        print(f"Run: python detection/logistic_regression/train.py --mode {args.mode}",
              file=sys.stderr)
        return 2
    pipe = joblib.load(model_path)
    print(f"Loaded model: {model_path}")

    out_dir = args.output_dir / args.mode

    # Real test set --------------------------------------------------------
    _, real_test_df = _shared.load_real()
    X_real = real_test_df["text"].fillna("").apply(_shared.clean_text).values
    y_real = _shared.encode_labels(real_test_df["label"])

    real_proba = pipe.predict_proba(X_real)[:, 1]
    real_pred = (real_proba >= THRESHOLD).astype(int)

    real_metrics = _shared.classification_metrics(y_real, real_pred)
    print(f"\nReal test set ({len(real_test_df)} rows):")
    print(f"  Accuracy:  {real_metrics['accuracy']:.4f}")
    print(f"  Precision: {real_metrics['precision']:.4f}")
    print(f"  Recall:    {real_metrics['recall']:.4f}")
    print(f"  F1:        {real_metrics['f1']:.4f}")
    print(f"  Confusion: TP={real_metrics['tp']} FP={real_metrics['fp']} "
          f"TN={real_metrics['tn']} FN={real_metrics['fn']}")

    _shared.write_per_email_results(
        real_test_df, real_pred, real_proba,
        out_dir / "real_test.csv",
    )
    _shared.write_baseline(DETECTOR, THRESHOLD, real_metrics,
                           out_dir / "baseline.csv")

    # Synthetic set --------------------------------------------------------
    if args.mode == "baseline":
        # Evaluate on full cleaned set (Phase 3).
        synth_df = _shared.load_synthetic()
        print(f"\nSynthetic set: full {len(synth_df)} cleaned rows "
              f"(Phase 3 detection gap).")
    else:
        # Evaluate on held-out only (Phase 4).
        _, synth_df = _shared.load_synthetic_split()
        print(f"\nSynthetic set: {len(synth_df)} held-out rows "
              f"from synthetic_split.csv where aug_split=='test' (Phase 4).")

    X_synth = synth_df["text"].fillna("").apply(_shared.clean_text).values
    synth_proba = pipe.predict_proba(X_synth)[:, 1]
    synth_pred = (synth_proba >= THRESHOLD).astype(int)
    synth_caught = int(synth_pred.sum())
    synth_rate = synth_caught / len(synth_pred) if len(synth_pred) else 0.0

    print(f"  Caught:         {synth_caught} / {len(synth_df)}")
    print(f"  Detection rate: {synth_rate:.4f}")
    print(f"  Gap vs real:    {real_metrics['recall'] - synth_rate:+.4f}")

    _shared.write_per_email_results(
        synth_df, synth_pred, synth_proba,
        out_dir / "synthetic.csv",
    )

    # Reload the per-email CSVs to ensure detection_gap has the full
    # method/sophistication/scenario metadata merged in.
    real_out = pd.read_csv(out_dir / "real_test.csv")
    synth_out = pd.read_csv(out_dir / "synthetic.csv")
    _shared.write_detection_gap(
        DETECTOR, real_out, synth_out,
        out_dir / "detection_gap.csv",
    )

    print(f"\n✓ Wrote results to {out_dir}/")
    for fname in ["real_test.csv", "synthetic.csv", "baseline.csv", "detection_gap.csv"]:
        print(f"    {out_dir / fname}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
