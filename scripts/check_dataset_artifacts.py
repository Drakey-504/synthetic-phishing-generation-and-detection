"""
Dataset-artifact check: do the classical ML detectors rely on words that only
identify which dataset an email came from, rather than whether it is phishing?

The legitimate and phishing corpora come from different sources and eras
(Enron / SpamAssassin ham are ~2000-2002; Nazario phishing is 2020-2025), so
some tokens identify the source rather than phishing behaviour:
  - "monkey", "jose"   -> the Nazario corpus owner's address (jose@monkey.org)
  - "enron", "spamassassin" -> names of the legitimate corpora
  - four-digit years   -> the era gap between corpora

This script retrains Logistic Regression and XGBoost with the exact pipelines
from detection/*/train.py, once as-is and once with those tokens removed, and
reports how much the results move. DistilBERT is not covered (needs a GPU).

Usage (from the project root):
    python scripts/check_dataset_artifacts.py
"""

import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "detection"))
import _shared  # noqa: E402

ARTIFACT_TOKENS = re.compile(r"monkey|jose|spamassassin|enron|\b(?:19|20)\d\d\b", re.I)


def load_pipeline_builder(detector: str):
    spec = importlib.util.spec_from_file_location(
        f"{detector}_train", ROOT / "detection" / detector / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_pipeline


def prepare(texts: pd.Series, scrub: bool) -> np.ndarray:
    cleaned = texts.fillna("").map(_shared.clean_text)
    if scrub:
        cleaned = cleaned.map(lambda t: ARTIFACT_TOKENS.sub(" ", t))
    return cleaned.values


def artifact_feature_ranks(pipe, top_n: int = 25) -> str:
    """Artifact words among the model's top_n features, with their rank.

    Logistic Regression is ranked by absolute coefficient (influence in either
    direction); XGBoost by feature importance.
    """
    names = pipe.named_steps["tfidf"].get_feature_names_out()
    clf = pipe.named_steps["clf"]
    weights = np.abs(clf.coef_[0]) if hasattr(clf, "coef_") else clf.feature_importances_
    order = np.argsort(-weights)[:top_n]
    hits = [f"{names[i]} (#{rank + 1})" for rank, i in enumerate(order)
            if ARTIFACT_TOKENS.search(names[i])]
    return "; ".join(hits)


def main() -> int:
    real = pd.read_csv(ROOT / "data/processed/emails_clean.csv")
    synth = pd.read_csv(ROOT / "data/processed/synthetic_phishing_clean.csv")
    train, test = real[real.split == "train"], real[real.split == "test"]
    y_train = (train.label == "phishing").astype(int).values
    y_test = (test.label == "phishing").astype(int).values

    rows = []
    for detector in ["logistic_regression", "xgboost"]:
        build = load_pipeline_builder(detector)
        for scrub in [False, True]:
            pipe = build()
            pipe.fit(prepare(train.text, scrub), y_train)
            pred_test = pipe.predict(prepare(test.text, scrub))
            pred_synth = pd.Series(pipe.predict(prepare(synth.text, scrub)), index=synth.index)
            recall = pred_test[y_test == 1].mean()
            rows.append({
                "detector": detector,
                "artifact_words": "removed" if scrub else "kept",
                "real_f1": round(float(f1_score(y_test, pred_test)), 4),
                "real_recall": round(float(recall), 4),
                "synth_caught": round(float(pred_synth.mean()), 4),
                "zero_shot": round(float(pred_synth[synth.method == "zero_shot"].mean()), 3),
                "few_shot": round(float(pred_synth[synth.method == "few_shot"].mean()), 3),
                "rephrasing": round(float(pred_synth[synth.method == "rephrasing"].mean()), 3),
                "spear": round(float(pred_synth[synth.sophistication == "spear"].mean()), 3),
                "gap": round(float(recall - pred_synth.mean()), 4),
                "artifact_words_in_top25": "" if scrub else artifact_feature_ranks(pipe),
            })
            print(rows[-1], flush=True)

    out = ROOT / "results/dataset_artifact_check.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
