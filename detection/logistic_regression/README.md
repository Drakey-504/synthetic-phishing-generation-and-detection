# Logistic Regression Detector

TF-IDF + Logistic Regression baseline. Classical bag-of-words approach.

## Files

- `train.py` — fits TF-IDF + LR pipeline on real or augmented training data
- `eval.py` — loads a saved pipeline and writes the four standard result CSVs

## Run

From repo root:

```bash
# Phase 3 baseline: train on real only
python detection/logistic_regression/train.py --mode baseline
python detection/logistic_regression/eval.py  --mode baseline

# Phase 4 augmented: train on real + synthetic train
python detection/logistic_regression/train.py --mode augmented
python detection/logistic_regression/eval.py  --mode augmented
```

Each pair takes about 30 seconds on a laptop.

## Outputs

- `models/logistic_regression/{baseline,augmented}.joblib` — fitted pipelines
- `results/logistic_regression/{baseline,augmented}/{real_test,synthetic,baseline,detection_gap}.csv`

## Hyperparameters

- TF-IDF: 5,000 features, unigrams + bigrams, sublinear TF, min_df=2
- LogReg: `class_weight='balanced'`, `max_iter=1000`, seed 42
- Decision threshold: 0.5 (default)
- Text cleaning (before TF-IDF): lowercase, URL/EMAIL tokenization, whitespace collapse

Matches Sathwika's original baseline notebook configuration.

## Eval-target difference between modes

- `baseline` evaluates against the full 457-row `synthetic_phishing_clean.csv` (Phase 3 detection gap measurement)
- `augmented` evaluates against the 229 held-out rows in `synthetic_split.csv` where `aug_split=='test'`

Evaluating the augmented model against the full 457 would leak the 228 synthetic-train rows that were used to fit it, so the scripts enforce this automatically.
