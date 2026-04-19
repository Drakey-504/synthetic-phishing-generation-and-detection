# XGBoost Detector

TF-IDF + XGBoost baseline. Gradient-boosted trees on the same feature space as Logistic Regression.

## Files

- `train.py` — fits TF-IDF + XGBoost pipeline on real or augmented training data
- `eval.py` — loads a saved pipeline and writes the four standard result CSVs

## Run

From repo root:

```bash
# Phase 3 baseline: train on real only
python detection/xgboost/train.py --mode baseline
python detection/xgboost/eval.py  --mode baseline

# Phase 4 augmented: train on real + synthetic train
python detection/xgboost/train.py --mode augmented
python detection/xgboost/eval.py  --mode augmented
```

Training takes about 1-2 minutes on a laptop (XGBoost parallelizes across CPU cores).

## Outputs

- `models/xgboost/{baseline,augmented}.joblib` — fitted pipelines
- `results/xgboost/{baseline,augmented}/{real_test,synthetic,baseline,detection_gap}.csv`

## Hyperparameters

- TF-IDF: 5,000 features, unigrams + bigrams, sublinear TF, min_df=2 (matches LR)
- XGBoost: 300 trees, depth 6, lr 0.1, subsample 0.8, colsample 0.8, seed 42
- Decision threshold: 0.5

Matches Sathwika's original baseline notebook configuration.

## Eval-target difference between modes

Same as logistic_regression — `baseline` evaluates against the full 457-row cleaned set, `augmented` evaluates against the 229-row held-out subset.
