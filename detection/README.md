# Phase 3 — Detection

Detectors evaluated in this project:

| Detector | Type | Owner | Status |
|---|---|---|---|
| Logistic Regression (TF-IDF) | classical ML | teammate | in progress |
| XGBoost (TF-IDF) | classical ML | teammate | pending |
| DistilBERT (fine-tuned) | transformer | teammate | pending |
| SpamAssassin | rule-based | me | done / scripts ready |

Each detector lives in its own subdirectory with a `train.py` / `eval.py`
pair (or just `eval.py` for rule-based), a local README, and any
detector-specific config.

## Layout

```
detection/
├── README.md                    <- this file
├── spamassassin/
│   ├── README.md                <- run instructions
│   ├── setup.md                 <- macOS install steps
│   ├── eval.py                  <- scores a CSV of emails
│   └── gap.py                   <- baseline + detection-gap analysis
├── logistic_regression/         <- teammate (notebooks/02_Baseline_Logistic.ipynb promoted here)
├── xgboost/                     <- teammate
└── distilbert/                  <- teammate
```

## Shared conventions (for all detectors)

All detectors should write results to `results/<detector_name>/` with the
same filenames so the final cross-detector comparison (Phase 4) can load
them uniformly:

- `real_test.csv` — per-email results on `emails_clean.csv` where
  `split == "test"`. Columns: `id, label, subject, <score>, <prediction>`
- `synthetic.csv` — per-email results on `synthetic_phishing_clean.csv`.
  Same column layout.
- `baseline.csv` — one row with accuracy/precision/recall/F1 and
  confusion-matrix counts (TP/FP/TN/FN) on the real test set.
- `detection_gap.csv` — long-form table with columns
  `detector, grouping, group_value, n, caught, detection_rate, mean_score`
  where `grouping` takes values `real_baseline`, `overall`, `method`,
  `sophistication`, `method_scenario`. One file per detector,
  concatenated in Phase 4.

## Run order

1. Each detector writes its per-email CSVs (`real_test.csv`, `synthetic.csv`).
2. Each detector's analysis script produces its own gap tables.
3. Phase 4 script (not yet written) loads all four detectors' outputs and
   produces the cross-detector comparison table that's the headline result.
