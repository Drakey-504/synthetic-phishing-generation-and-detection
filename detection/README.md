# Phase 3 & Phase 4 — Detection

Four detectors are evaluated in this project, each in its own subdirectory:

| Detector | Type | Where | Runtime |
|---|---|---|---|
| Logistic Regression (TF-IDF) | classical ML | `logistic_regression/` | ~30s local |
| XGBoost (TF-IDF) | classical ML | `xgboost/` | ~2 min local |
| DistilBERT (fine-tuned) | transformer | `distilbert/` | ~15 min Colab GPU |
| SpamAssassin | rule-based | `spamassassin/` | ~30 min + Docker (scores train+test+synth) |

Each ML detector (LR, XGBoost, DistilBERT) produces **two models**:
- `baseline` — trained on real phishing only (Phase 3)
- `augmented` — trained on real + synthetic phishing (Phase 4)

SpamAssassin is rule-based so only has one evaluation.

## Shared conventions

All detectors write the same 4-file output schema to `results/<detector>/<mode>/`:

```
results/
├── logistic_regression/
│   ├── baseline/
│   │   ├── real_test.csv          per-email predictions on real test
│   │   ├── synthetic.csv          per-email predictions on synthetic
│   │   ├── baseline.csv           one-row summary (metrics on real test)
│   │   └── detection_gap.csv      long-form breakdown
│   └── augmented/
│       └── ...same 4 files...
├── xgboost/
│   └── ...same...
├── distilbert/
│   └── ...same...
└── spamassassin/
    ├── real_test.csv
    ├── synthetic.csv
    ├── baseline.csv
    └── detection_gap.csv
```

### Per-email CSV columns (`real_test.csv` / `synthetic.csv`)

`id, label, subject, score, prediction`

- `score` is the detector's raw numeric output (probability for ML models, SpamAssassin score for SA)
- `prediction` is the binary decision ('phishing' or 'legitimate')

### Baseline CSV columns (`baseline.csv`)

`detector, threshold, n, tp, fp, tn, fn, accuracy, precision, recall, f1`

One row per file.

### Detection gap CSV columns (`detection_gap.csv`)

`detector, grouping, group_value, n, caught, detection_rate, mean_score`

Long-form. `grouping` takes values:
- `real_baseline` — one row, phishing recall on real test
- `overall` — one row, detection rate on all synthetic
- `method` — three rows, one per generation method (zero_shot / few_shot / rephrasing)
- `sophistication` — three rows (bulk / spear / rephrased)
- `method_scenario` — ~27 rows, finest-grained

Concatenating the four detectors' `detection_gap.csv` files gives the master table for the Phase 4 writeup.

## Shared code

Detection utilities (data loading, text cleaning, metrics, result writers) live in `_shared.py` at this directory's root. All three ML detector scripts import from it via `sys.path` manipulation at the top of each file.

## Full run sequence

Assuming `data/processed/emails_clean.csv`, `synthetic_phishing_clean.csv`, `synthetic_split.csv`, and `emails_augmented_train.csv` already exist:

```bash
# SpamAssassin (tuned-threshold workflow)
python detection/spamassassin/eval.py \
    --input  data/processed/emails_clean.csv \
    --output results/spamassassin/real_train.csv \
    --split  train                           # needed to tune the threshold
python detection/spamassassin/eval.py \
    --input  data/processed/emails_clean.csv \
    --output results/spamassassin/real_test.csv \
    --split  test
python detection/spamassassin/eval.py \
    --input  data/processed/synthetic_phishing_clean.csv \
    --output results/spamassassin/synthetic.csv
python detection/spamassassin/gap.py \
    --train     results/spamassassin/real_train.csv \
    --real      results/spamassassin/real_test.csv \
    --synthetic results/spamassassin/synthetic.csv \
    --synthetic-meta data/processed/synthetic_phishing_clean.csv \
    --outdir    results/spamassassin/

# Logistic Regression
python detection/logistic_regression/train.py --mode baseline
python detection/logistic_regression/eval.py  --mode baseline
python detection/logistic_regression/train.py --mode augmented
python detection/logistic_regression/eval.py  --mode augmented

# XGBoost
python detection/xgboost/train.py --mode baseline
python detection/xgboost/eval.py  --mode baseline
python detection/xgboost/train.py --mode augmented
python detection/xgboost/eval.py  --mode augmented

# DistilBERT (on Colab GPU — see detection/distilbert/README.md)
python detection/distilbert/train.py --mode baseline
python detection/distilbert/eval.py  --mode baseline
python detection/distilbert/train.py --mode augmented
python detection/distilbert/eval.py  --mode augmented
```

## Gitignore recommendation

Add to `.gitignore`:

```
models/
results/*/hf_trainer_workdir/
```

Trained model files can be large (DistilBERT checkpoints ~250 MB each). Only commit the result CSVs.