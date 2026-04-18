"""
gap.py
Compute SpamAssassin's baseline metrics and detection gap.

Inputs:
    --real      Per-email results on the real test set (from eval.py)
    --synthetic Per-email results on the synthetic phishing set
    --synthetic-meta The original synthetic CSV (for method/sophistication/scenario)

Outputs (to --outdir):
    baseline.csv       One row per detector with confusion matrix + metrics on
                       the real test set.
    detection_gap.csv  Long-form table. Each row is one (detector, grouping,
                       group_value) with n / caught / detection_rate /
                       mean_score. The 'grouping' column takes values:
                         - overall          (one row)
                         - method           (one row per generation method)
                         - sophistication   (one row per sophistication level)
                         - method_scenario  (one row per method x scenario)
                         - real_baseline    (one row: recall on real phishing)
                       This format pivots cleanly for cross-detector comparison
                       in Phase 4.

Also prints a human-readable summary to stdout.

Usage:
    python detection/spamassassin/gap.py \
        --real      results/spamassassin/real_test.csv \
        --synthetic results/spamassassin/synthetic.csv \
        --synthetic-meta data/processed/synthetic_phishing_clean.csv \
        --outdir    results/spamassassin/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


SA_THRESHOLD = 5.0
DETECTOR_NAME = "spamassassin"


def metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Compute accuracy / precision / recall / F1 treating 'phishing' as the
    positive class."""
    tp = ((y_true == "phishing") & (y_pred == "phishing")).sum()
    fp = ((y_true == "legitimate") & (y_pred == "phishing")).sum()
    tn = ((y_true == "legitimate") & (y_pred == "legitimate")).sum()
    fn = ((y_true == "phishing") & (y_pred == "legitimate")).sum()
    n = tp + fp + tn + fn

    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return dict(n=int(n), tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
                accuracy=acc, precision=prec, recall=rec, f1=f1)


def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Coerce score to numeric (empty string -> NaN).
    df["sa_score"] = pd.to_numeric(df["sa_score"], errors="coerce")
    errored = df["sa_score"].isna().sum()
    if errored:
        print(f"  WARNING: {errored} rows in {path.name} had no score and were dropped.")
    df = df.dropna(subset=["sa_score"]).copy()
    return df


def group_stats(df: pd.DataFrame, grouping: str, group_value: str) -> dict:
    """One long-form row: group-level n/caught/detection_rate/mean_score."""
    n = len(df)
    caught = int((df["sa_prediction"] == "phishing").sum())
    return {
        "detector": DETECTOR_NAME,
        "grouping": grouping,
        "group_value": group_value,
        "n": n,
        "caught": caught,
        "detection_rate": caught / n if n else 0.0,
        "mean_score": float(df["sa_score"].mean()) if n else float("nan"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", required=True, type=Path)
    parser.add_argument("--synthetic", required=True, type=Path)
    parser.add_argument("--synthetic-meta", required=True, type=Path,
                        help="Original synthetic CSV (for method/sophistication/scenario).")
    parser.add_argument("--outdir", required=True, type=Path)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    # -- Baseline on real test set ----------------------------------------
    print("=" * 64)
    print("Loading real test results ...")
    print("=" * 64)
    real = load_results(args.real)
    print(f"  Real test rows: {len(real)}")
    print(f"  Label distribution: {dict(real['label'].value_counts())}")

    base = metrics(real["label"], real["sa_prediction"])
    print(f"\nBaseline @ threshold={SA_THRESHOLD}:")
    print(f"  Accuracy:  {base['accuracy']:.4f}")
    print(f"  Precision: {base['precision']:.4f}")
    print(f"  Recall:    {base['recall']:.4f}  <- phishing detection rate on real")
    print(f"  F1:        {base['f1']:.4f}")
    print(f"  Confusion: TP={base['tp']} FP={base['fp']} TN={base['tn']} FN={base['fn']}")

    # baseline.csv ---------------------------------------------------------
    baseline_row = {
        "detector": DETECTOR_NAME,
        "threshold": SA_THRESHOLD,
        **base,
    }
    pd.DataFrame([baseline_row]).to_csv(args.outdir / "baseline.csv", index=False)

    real_phish_recall = base["recall"]

    # -- Synthetic detection gap ------------------------------------------
    print("\n" + "=" * 64)
    print("Loading synthetic results + metadata ...")
    print("=" * 64)
    synth = load_results(args.synthetic)
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
    print(f"  Synthetic rows scored: {len(synth)}")

    # Build the long-form detection_gap table -----------------------------
    rows: list[dict] = []

    # Row 1: real-phishing baseline (for cross-reference in Phase 4).
    real_phish = real[real["label"] == "phishing"]
    rows.append({
        "detector": DETECTOR_NAME,
        "grouping": "real_baseline",
        "group_value": "real_phishing",
        "n": len(real_phish),
        "caught": int((real_phish["sa_prediction"] == "phishing").sum()),
        "detection_rate": real_phish_recall,
        "mean_score": float(real_phish["sa_score"].mean()),
    })

    # Row 2: synthetic overall.
    rows.append(group_stats(synth, "overall", "synthetic_all"))

    # Rows 3-5: synthetic by method.
    for method, grp in synth.groupby("method"):
        rows.append(group_stats(grp, "method", str(method)))

    # Rows 6-8: synthetic by sophistication.
    for soph, grp in synth.groupby("sophistication"):
        rows.append(group_stats(grp, "sophistication", str(soph)))

    # Rows 9+: synthetic by method x scenario (fine-grained).
    for (method, scen), grp in synth.groupby(["method", "scenario"]):
        rows.append(group_stats(grp, "method_scenario", f"{method}:{scen}"))

    gap_df = pd.DataFrame(rows)
    gap_df.to_csv(args.outdir / "detection_gap.csv", index=False)

    # -- Human-readable summary -------------------------------------------
    overall_rate = gap_df.loc[gap_df["grouping"] == "overall",
                              "detection_rate"].iloc[0]
    gap = real_phish_recall - overall_rate

    print("\n" + "=" * 64)
    print("DETECTION GAP SUMMARY")
    print("=" * 64)
    print(f"\nBaseline recall on real phishing:   {real_phish_recall:.4f}")
    print(f"Detection rate on synthetic overall: {overall_rate:.4f}")
    print(f"Detection gap (real - synth):        {gap:+.4f}")
    print("  (positive = detector is WORSE on synthetic)")

    print("\nBy generation method:")
    print(gap_df[gap_df["grouping"] == "method"]
          [["group_value", "n", "caught", "detection_rate", "mean_score"]]
          .to_string(index=False, float_format="%.4f"))

    print("\nBy sophistication:")
    print(gap_df[gap_df["grouping"] == "sophistication"]
          [["group_value", "n", "caught", "detection_rate", "mean_score"]]
          .to_string(index=False, float_format="%.4f"))

    print(f"\nWrote:")
    print(f"  {args.outdir / 'baseline.csv'}")
    print(f"  {args.outdir / 'detection_gap.csv'}  ({len(gap_df)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
