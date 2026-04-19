"""
_shared.py
Utilities shared across the LR / XGBoost / DistilBERT detectors.

Responsibilities:
  - Load the canonical train/test splits exactly as Phase 1 defined them.
  - Load synthetic phishing and the augmented training set produced by
    scripts/prepare_augmentation.py.
  - Provide a consistent text-cleaning function for classical ML detectors
    (LR / XGBoost). DistilBERT uses its own tokenizer and skips this step.
  - Write result CSVs in the schema documented in detection/README.md so
    all four detectors (including SpamAssassin) produce comparable output.

Output file schema (all detectors produce these four files):

    results/<detector>/real_test.csv
        Per-email predictions on the real test set.
        Columns: id, label, subject, <score_col>, <prediction_col>

    results/<detector>/synthetic.csv
        Per-email predictions on the synthetic phishing set.
        Same columns as real_test.csv.

    results/<detector>/baseline.csv
        One row: detector, threshold, n, tp, fp, tn, fn,
        accuracy, precision, recall, f1.

    results/<detector>/detection_gap.csv
        Long-form breakdown. Columns:
        detector, grouping, group_value, n, caught, detection_rate, mean_score.
        'grouping' values: real_baseline, overall, method, sophistication,
        method_scenario.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_real(path: Path = Path("data/processed/emails_clean.csv")
              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load emails_clean.csv and return (train, test) DataFrames based on the
    canonical 'split' column produced by scripts/split_data.py.

    Null subjects are coerced to empty string; null texts are dropped (should
    never happen, but defensive).
    """
    df = pd.read_csv(path)
    df["subject"] = df["subject"].fillna("").astype(str)
    df = df.dropna(subset=["text"]).copy()
    train = df[df["split"] == "train"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    return train, test


def load_synthetic(path: Path = Path("data/processed/synthetic_phishing_clean.csv")
                   ) -> pd.DataFrame:
    """Load the full 457-row cleaned synthetic phishing set (Phase 3 eval target).

    This is the pre-augmentation-split file — every row is used as part of the
    Phase 3 detection-gap measurement. For Phase 4 use load_synthetic_split().
    """
    df = pd.read_csv(path)
    df["subject"] = df["subject"].fillna("").astype(str)
    df = df.dropna(subset=["text"]).copy()
    return df


def load_synthetic_split(path: Path = Path("data/processed/synthetic_split.csv")
                         ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load synthetic_split.csv and return (synth_train, synth_test) based on
    the 'aug_split' column produced by scripts/prepare_augmentation.py.
    """
    df = pd.read_csv(path)
    df["subject"] = df["subject"].fillna("").astype(str)
    df = df.dropna(subset=["text"]).copy()
    train = df[df["aug_split"] == "train"].reset_index(drop=True)
    test = df[df["aug_split"] == "test"].reset_index(drop=True)
    return train, test


def load_augmented(path: Path = Path("data/processed/emails_augmented_train.csv")
                   ) -> pd.DataFrame:
    """Load the augmented training set (real train + 228 synthetic train rows,
    all labeled). No split filter — every row is training data.
    """
    df = pd.read_csv(path)
    df["subject"] = df["subject"].fillna("").astype(str)
    df = df.dropna(subset=["text"]).copy()
    return df


# ---------------------------------------------------------------------------
# Text cleaning (classical ML only)
# ---------------------------------------------------------------------------
_URL_RE = re.compile(r"http\S+|www\.\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normalize text for TF-IDF feature extraction.

    Matches the cleaning Sathwika's baseline notebooks used:
      - lowercase
      - URLs → 'URL' token
      - email addresses → 'EMAIL' token
      - collapse whitespace

    DistilBERT does NOT use this — its tokenizer handles these cases natively
    and collapsing them throws away signal the transformer can use.
    """
    text = str(text).lower()
    text = _URL_RE.sub("URL", text)
    text = _EMAIL_RE.sub("EMAIL", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Metrics (phishing = positive class)
# ---------------------------------------------------------------------------
def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    """Return TP/FP/TN/FN counts treating 1 (phishing) as the positive class.

    y_true and y_pred should be integer arrays with values in {0, 1}.
    """
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return dict(tp=tp, fp=fp, tn=tn, fn=fn)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Accuracy, precision, recall, F1 (phishing = positive class)."""
    cc = confusion_counts(y_true, y_pred)
    tp, fp, tn, fn = cc["tp"], cc["fp"], cc["tn"], cc["fn"]
    n = tp + fp + tn + fn
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {"n": n, **cc, "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1}


# ---------------------------------------------------------------------------
# Result writers
# ---------------------------------------------------------------------------
# Column names in the per-email CSVs are standardized to:
#   id, label, subject, score, prediction
# Regardless of which detector produced them. This is what Phase 4
# aggregation will load.

PRED_COLS = ["id", "label", "subject", "score", "prediction"]


def _label_str(y: int | str) -> str:
    """Normalize label to the string form used in emails_clean.csv."""
    if isinstance(y, str):
        return y
    return "phishing" if int(y) == 1 else "legitimate"


def write_per_email_results(df: pd.DataFrame, y_pred: np.ndarray,
                            score: np.ndarray, out_path: Path,
                            label_col: str = "label") -> None:
    """Write a per-email result CSV (real_test.csv or synthetic.csv).

    df must contain 'id', 'subject', and label_col. Rows in df, y_pred, and
    score must be aligned.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels_out = df[label_col].apply(_label_str).values if label_col in df else ""
    pred_labels = np.where(np.asarray(y_pred) == 1, "phishing", "legitimate")
    result = pd.DataFrame({
        "id": df["id"].values,
        "label": labels_out,
        "subject": df["subject"].fillna("").values,
        "score": np.asarray(score, dtype=float),
        "prediction": pred_labels,
    })
    result.to_csv(out_path, index=False)


def write_baseline(detector: str, threshold: float,
                   metrics: dict[str, Any], out_path: Path) -> None:
    """Write results/<detector>/baseline.csv (one row)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    row = {"detector": detector, "threshold": threshold, **metrics}
    pd.DataFrame([row]).to_csv(out_path, index=False)


def write_detection_gap(detector: str, real_test: pd.DataFrame,
                        synthetic: pd.DataFrame, out_path: Path,
                        score_col: str = "score",
                        prediction_col: str = "prediction") -> None:
    """Write results/<detector>/detection_gap.csv.

    Long-form: one row per (grouping, group_value). Groupings are:
      real_baseline, overall, method, sophistication, method_scenario.

    real_test and synthetic are the per-email DataFrames already written to
    disk (so they have score/prediction columns). synthetic must have
    method/sophistication/scenario columns; if missing, this function merges
    from synthetic_phishing_clean.csv.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Need method/sophistication/scenario on synthetic. Merge if absent.
    need_merge = any(c not in synthetic.columns
                     for c in ["method", "sophistication", "scenario"])
    if need_merge:
        meta = pd.read_csv("data/processed/synthetic_phishing_clean.csv")
        synthetic = synthetic.merge(
            meta[["id", "method", "sophistication", "scenario"]],
            on="id", how="left",
        )

    def group_stats(df: pd.DataFrame, grouping: str, group_value: str) -> dict:
        n = len(df)
        caught = int((df[prediction_col] == "phishing").sum())
        return {
            "detector": detector,
            "grouping": grouping,
            "group_value": group_value,
            "n": n,
            "caught": caught,
            "detection_rate": caught / n if n else 0.0,
            "mean_score": float(df[score_col].mean()) if n else float("nan"),
        }

    rows: list[dict] = []

    # Real-phishing baseline (for cross-reference).
    real_phish = real_test[real_test["label"] == "phishing"]
    rows.append({
        "detector": detector,
        "grouping": "real_baseline",
        "group_value": "real_phishing",
        "n": len(real_phish),
        "caught": int((real_phish[prediction_col] == "phishing").sum()),
        "detection_rate": float((real_phish[prediction_col] == "phishing").mean())
                          if len(real_phish) else 0.0,
        "mean_score": float(real_phish[score_col].mean()) if len(real_phish) else float("nan"),
    })

    # Overall synthetic.
    rows.append(group_stats(synthetic, "overall", "synthetic_all"))

    # By method.
    for method, grp in synthetic.groupby("method"):
        rows.append(group_stats(grp, "method", str(method)))

    # By sophistication.
    for soph, grp in synthetic.groupby("sophistication"):
        rows.append(group_stats(grp, "sophistication", str(soph)))

    # By method × scenario.
    for (method, scen), grp in synthetic.groupby(["method", "scenario"]):
        rows.append(group_stats(grp, "method_scenario", f"{method}:{scen}"))

    pd.DataFrame(rows).to_csv(out_path, index=False)


# ---------------------------------------------------------------------------
# Label encoding
# ---------------------------------------------------------------------------
LABEL_TO_INT = {"legitimate": 0, "phishing": 1}


def encode_labels(series: pd.Series) -> np.ndarray:
    """Map 'legitimate'/'phishing' -> 0/1. Raises on unknown labels."""
    encoded = series.map(LABEL_TO_INT)
    if encoded.isna().any():
        bad = series[encoded.isna()].unique()
        raise ValueError(f"Unknown labels: {bad}")
    return encoded.astype(int).values
