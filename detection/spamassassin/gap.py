"""
gap.py
Compute SpamAssassin's baseline metrics and detection gap.

SpamAssassin's default threshold of 5.0 is tuned for bulk spam and significantly
under-recalls on phishing. Rather than use the default, this script:

  1. Tunes the score threshold on a labeled real-training set (F1-optimal by
     default). This mirrors how the ML detectors use training data to set
     parameters, giving SpamAssassin a fair comparison.
  2. Re-classifies the real test set and synthetic set at the tuned threshold.
  3. Writes the four standard output files matching the schema used by the
     ML detectors (detection/README.md).

Inputs:
    --train      Per-email results on the real training set (from eval.py
                 with --split train). Used to pick the threshold.
    --real       Per-email results on the real test set.
    --synthetic  Per-email results on the synthetic phishing set.
    --synthetic-meta Original synthetic CSV (for method/sophistication/scenario).

Outputs (to --outdir):
    real_test.csv      Per-email with standardized columns (id, label, subject,
                       score, prediction). Reclassified at the tuned threshold.
    synthetic.csv      Same schema, reclassified at the tuned threshold.
    baseline.csv       One row: detector, threshold, confusion matrix + metrics
                       on the real test set.
    detection_gap.csv  Long-form table with groupings: real_baseline, overall,
                       method, sophistication, method_scenario. Pivots cleanly
                       for cross-detector comparison.

Also prints a human-readable summary.

Usage:
    python detection/spamassassin/gap.py \
        --train     results/spamassassin/real_train.csv \
        --real      results/spamassassin/real_test.csv \
        --synthetic results/spamassassin/synthetic.csv \
        --synthetic-meta data/processed/synthetic_phishing_clean.csv \
        --outdir    results/spamassassin/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


DETECTOR_NAME = "spamassassin"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_results(path: Path) -> pd.DataFrame:
    """Load an eval.py output CSV. Drops rows with no sa_score (spamd errors)."""
    df = pd.read_csv(path)
    df["sa_score"] = pd.to_numeric(df["sa_score"], errors="coerce")
    errored = df["sa_score"].isna().sum()
    if errored:
        print(f"  WARNING: {errored} rows in {path.name} had no score and were dropped.")
    df = df.dropna(subset=["sa_score"]).copy()
    return df


# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------
def tune_threshold(train: pd.DataFrame, objective: str,
                   grid: np.ndarray) -> tuple[float, dict]:
    """Pick the threshold that optimizes the chosen objective on train.

    Objectives:
        f1           maximize F1 (recommended default)
        precision_95 maximize recall subject to precision >= 0.95
        precision_98 maximize recall subject to precision >= 0.98
    """
    y_true = (train["label"] == "phishing").astype(int).values
    scores = train["sa_score"].values

    best = None
    for t in grid:
        y_pred = (scores >= t).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
        stats = dict(threshold=float(t), f1=f1, precision=prec, recall=rec,
                     tp=tp, fp=fp, fn=fn)

        if objective == "f1":
            if best is None or f1 > best["f1"]:
                best = stats
        else:
            target = {"precision_95": 0.95, "precision_98": 0.98}[objective]
            if prec >= target:
                if best is None or rec > best["recall"]:
                    best = stats

    if best is None:
        raise RuntimeError(f"No threshold in grid met objective {objective}")
    return best["threshold"], best


# ---------------------------------------------------------------------------
# Metrics (phishing = positive class)
# ---------------------------------------------------------------------------
def metrics_at_threshold(df: pd.DataFrame, threshold: float) -> dict:
    """Accuracy / precision / recall / F1 with 'phishing' as positive class."""
    y_true = df["label"].values
    y_pred = np.where(df["sa_score"].values >= threshold, "phishing", "legitimate")

    tp = int(((y_true == "phishing") & (y_pred == "phishing")).sum())
    fp = int(((y_true == "legitimate") & (y_pred == "phishing")).sum())
    tn = int(((y_true == "legitimate") & (y_pred == "legitimate")).sum())
    fn = int(((y_true == "phishing") & (y_pred == "legitimate")).sum())
    n = tp + fp + tn + fn

    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0

    return dict(n=n, tp=tp, fp=fp, tn=tn, fn=fn,
                accuracy=acc, precision=prec, recall=rec, f1=f1)


def reclassify(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Return a copy with standardized 'score' and 'prediction' columns."""
    out = df.copy()
    out["score"] = out["sa_score"].astype(float)
    out["prediction"] = np.where(
        out["score"] >= threshold, "phishing", "legitimate"
    )
    return out


# ---------------------------------------------------------------------------
# Result writers (matching the LR/XGBoost/DistilBERT schema)
# ---------------------------------------------------------------------------
def write_per_email(df: pd.DataFrame, out_path: Path) -> None:
    """Write id, label, subject, score, prediction — the standard schema."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["id", "label", "subject", "score", "prediction"]
    df[cols].to_csv(out_path, index=False)


def write_baseline(threshold: float, metrics: dict, out_path: Path) -> None:
    row = {"detector": DETECTOR_NAME, "threshold": threshold, **metrics}
    pd.DataFrame([row]).to_csv(out_path, index=False)


def group_stats(df: pd.DataFrame, grouping: str, group_value: str) -> dict:
    n = len(df)
    caught = int((df["prediction"] == "phishing").sum())
    return {
        "detector": DETECTOR_NAME,
        "grouping": grouping,
        "group_value": group_value,
        "n": n,
        "caught": caught,
        "detection_rate": caught / n if n else 0.0,
        "mean_score": float(df["score"].mean()) if n else float("nan"),
    }


def write_detection_gap(real: pd.DataFrame, synth: pd.DataFrame,
                        out_path: Path) -> None:
    """Long-form table with all grouping levels."""
    rows: list[dict] = []

    # real_baseline: recall on real phishing only
    real_phish = real[real["label"] == "phishing"]
    rows.append({
        "detector": DETECTOR_NAME,
        "grouping": "real_baseline",
        "group_value": "real_phishing",
        "n": len(real_phish),
        "caught": int((real_phish["prediction"] == "phishing").sum()),
        "detection_rate": float((real_phish["prediction"] == "phishing").mean())
                          if len(real_phish) else 0.0,
        "mean_score": float(real_phish["score"].mean()) if len(real_phish) else float("nan"),
    })

    # overall synthetic
    rows.append(group_stats(synth, "overall", "synthetic_all"))

    # by method, sophistication, method × scenario
    for method, grp in synth.groupby("method"):
        rows.append(group_stats(grp, "method", str(method)))
    for soph, grp in synth.groupby("sophistication"):
        rows.append(group_stats(grp, "sophistication", str(soph)))
    for (method, scen), grp in synth.groupby(["method", "scenario"]):
        rows.append(group_stats(grp, "method_scenario", f"{method}:{scen}"))

    pd.DataFrame(rows).to_csv(out_path, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path,
                        help="SA-scored real training set (for threshold tuning).")
    parser.add_argument("--real", required=True, type=Path,
                        help="SA-scored real test set.")
    parser.add_argument("--synthetic", required=True, type=Path,
                        help="SA-scored synthetic phishing set.")
    parser.add_argument("--synthetic-meta", required=True, type=Path,
                        help="Synthetic source CSV (for method/sophistication/scenario).")
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--objective", default="f1",
                        choices=["f1", "precision_95", "precision_98"],
                        help="Tuning objective on training set (default: f1).")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override: skip tuning and use this threshold.")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    # ----- Load all three scored sets ------------------------------------
    print("=" * 64)
    print("Loading scored inputs ...")
    print("=" * 64)
    train = load_results(args.train)
    real = load_results(args.real)
    synth = load_results(args.synthetic)
    print(f"  Train:     {len(train)} rows "
          f"({(train['label']=='phishing').sum()} phishing, "
          f"{(train['label']=='legitimate').sum()} legit)")
    print(f"  Real test: {len(real)} rows")
    print(f"  Synthetic: {len(synth)} rows")

    # ----- Tune threshold on training set --------------------------------
    print("\n" + "=" * 64)
    if args.threshold is not None:
        print(f"Using fixed threshold: {args.threshold}")
        threshold = args.threshold
    else:
        print(f"Tuning threshold on training set (objective: {args.objective}) ...")
        grid = np.arange(-2.0, 15.01, 0.1)
        threshold, stats = tune_threshold(train, args.objective, grid)
        print(f"  Selected threshold:  {threshold:.2f}")
        print(f"  Training F1:         {stats['f1']:.4f}")
        print(f"  Training precision:  {stats['precision']:.4f}")
        print(f"  Training recall:     {stats['recall']:.4f}")

        # For context, what does default 5.0 give?
        default_metrics = metrics_at_threshold(train, 5.0)
        print(f"\n  (vs default 5.0:     train F1 = {default_metrics['f1']:.4f}, "
              f"recall = {default_metrics['recall']:.4f})")
    print("=" * 64)

    # ----- Evaluate on real test set -------------------------------------
    test_metrics = metrics_at_threshold(real, threshold)
    print(f"\nReal test @ threshold={threshold:.2f}:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}  "
          f"<- phishing detection rate on real")
    print(f"  F1:        {test_metrics['f1']:.4f}")
    print(f"  Confusion: TP={test_metrics['tp']} FP={test_metrics['fp']} "
          f"TN={test_metrics['tn']} FN={test_metrics['fn']}")

    # ----- Attach method/sophistication/scenario to synthetic -----------
    meta = pd.read_csv(args.synthetic_meta)
    merge_cols = ["id", "method", "sophistication", "scenario"]
    missing = set(merge_cols) - set(meta.columns)
    if missing:
        print(f"ERROR: synthetic-meta missing columns: {missing}", file=sys.stderr)
        return 2
    synth = synth.merge(meta[merge_cols], on="id", how="left")
    unmatched = synth["method"].isna().sum()
    if unmatched:
        print(f"  WARNING: {unmatched} synthetic rows didn't match metadata.")
    synth = synth.dropna(subset=["method"]).copy()

    # ----- Reclassify both sets at tuned threshold ----------------------
    real_out = reclassify(real, threshold)
    synth_out = reclassify(synth, threshold)

    # ----- Synthetic detection summary ----------------------------------
    caught = int((synth_out["prediction"] == "phishing").sum())
    synth_rate = caught / len(synth_out) if len(synth_out) else 0.0

    print(f"\nSynthetic @ threshold={threshold:.2f}:")
    print(f"  Caught:         {caught} / {len(synth_out)}")
    print(f"  Detection rate: {synth_rate:.4f}")
    print(f"  Detection gap:  {test_metrics['recall'] - synth_rate:+.4f}")
    print(f"  (positive = detector is WORSE on synthetic)")

    # ----- Write all four result files ----------------------------------
    write_per_email(real_out, args.outdir / "real_test.csv")
    write_per_email(synth_out, args.outdir / "synthetic.csv")
    write_baseline(threshold, test_metrics, args.outdir / "baseline.csv")
    write_detection_gap(real_out, synth_out, args.outdir / "detection_gap.csv")

    # ----- By-method breakdown in stdout --------------------------------
    gap_df = pd.read_csv(args.outdir / "detection_gap.csv")
    print("\nBy generation method:")
    print(gap_df[gap_df["grouping"] == "method"]
          [["group_value", "n", "caught", "detection_rate", "mean_score"]]
          .to_string(index=False, float_format="%.4f"))

    print("\nBy sophistication:")
    print(gap_df[gap_df["grouping"] == "sophistication"]
          [["group_value", "n", "caught", "detection_rate", "mean_score"]]
          .to_string(index=False, float_format="%.4f"))

    print(f"\nWrote:")
    for fname in ["real_test.csv", "synthetic.csv", "baseline.csv", "detection_gap.csv"]:
        print(f"  {args.outdir / fname}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
