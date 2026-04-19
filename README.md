# synthetic-phishing-generation-and-detection
Synthetic Phishing Generation and Detection Project for AI Applications in Information Security, Spring 2026

## Setup

### Prerequisites
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/download)
- [Ollama](https://ollama.com) (installed separately as a system application)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (for running SpamAssassin in Phase 3)

### Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Drakey-504/synthetic-phishing-generation-and-detection.git
   cd synthetic-phishing-generation-and-detection
   ```

2. Create the conda environment from the lockfile:
   ```bash
   conda env create -f environment.yml
   conda activate phishing
   ```

3. Pull the LLM model for phishing email generation:
   ```bash
   ollama pull llama3
   ```

4. Verify the setup:
   ```bash
   python -c "import pandas, sklearn, xgboost, transformers, torch, spacy, ollama; print('All packages OK')"
   ```

## Datasets

All raw datasets are stored in `data/raw/` and excluded from version control via `.gitignore`. Download them manually using the links below.

### Phishing Sources

**Nazario Phishing Corpus**
- Source: http://monkey.org/~jose/phishing/
- Download the yearly archives: `phishing-2020` through `phishing-2025`
- Skip the older `phishing0.mbox` through `phishing3.mbox` (2005–2007 era, outdated phishing language) and `private-phishing4.mbox` (may have usage restrictions)
- Save to `data/raw/nazario/`

**Nigerian Fraud / 419 Email Dataset**
- Source: https://www.kaggle.com/datasets/rtatman/fraudulent-email-corpus
- Save to `data/raw/nigerian_fraud/`

### Legitimate Email Sources

**Enron Email Dataset**
- Source: https://www.cs.cmu.edu/~enron/
- Download the May 7, 2015 version (standard)
- This is large (~1.7GB compressed). We only use a random subset of ~5,000 emails.
- Save to `data/raw/enron/`
- Note: This dataset requires high memory to process.

**SpamAssassin Public Corpus**
- Source: https://spamassassin.apache.org/old/publiccorpus/
- Download only the easy\_ham archives: `20030228_easy_ham.tar.bz2`, `20030228_easy_ham_2.tar.bz2`
- Skip `hard_ham` (borderline spam-like legitimate emails that could confuse the classifier) and `spam` archives (generic junk mail, not phishing)
- Gives ~3,900 clean legitimate emails to supplement Enron
- Note: strip any leftover SpamAssassin headers (e.g., `X-Spam-Status`) during preprocessing
- Save to `data/raw/spamassassin/`

### Benchmark Dataset

**Opara et al. GPT-4o Phishing Dataset**
- Source: https://www.kaggle.com/datasets/guchiopara/ai-generated-emails
- Used as a pre-built benchmark to compare our synthetic emails against
- Save to `data/raw/opara/`

## Phase 1: Data Pipeline

Run the following scripts in order from the project root to go from raw datasets to a clean, split dataset ready for model training.

### Step 1: Parse Raw Emails

Each script parses one dataset and outputs a CSV to `data/processed/`.

```bash
python scripts/parse_nazario.py
python scripts/parse_nigerian.py
python scripts/parse_enron.py
python scripts/parse_spamassassin.py
```

Then combine all four into a single CSV:

```bash
python scripts/combine_datasets.py
```

**Output:** `data/processed/emails_raw.csv` (~14,600 emails)

### Step 2: Clean and Preprocess

Strips remaining HTML, removes header leakage (`X-Spam-*` etc.), normalizes whitespace, drops non-English emails, removes exact duplicates, filters very short emails (<10 words), and truncates very long emails (>5,000 chars).

```bash
python scripts/preprocess.py
```

**Output:** `data/processed/emails_clean.csv` (~13,200 emails)

### Step 3: Train/Test Split

Adds a stratified 80/20 `split` column to the cleaned dataset.

```bash
python scripts/split_data.py
```

**Output:** Updates `data/processed/emails_clean.csv` with `split` column (train/test)

### Step 4: Exploratory Data Analysis

Open and run the EDA notebook to inspect the dataset:

```bash
jupyter lab notebooks/01_eda.ipynb
```

The notebook covers class distribution, text length analysis, top words per class, leakage checks, source distribution, and sample emails.

## Phase 2: Synthetic Phishing Generation

Uses Llama 3 (8B) via Ollama to generate synthetic phishing emails across three methods simulating escalating attacker capability.

### Step 1: Generate

```bash
python scripts/generate_phishing.py
```

Produces 460 synthetic phishing emails:
- **zero_shot** (130): no phishing examples in prompt — simulates a low-effort attacker
- **few_shot** (130): prompt includes real phishing examples — simulates an attacker with access to a phishing corpus
- **rephrasing** (200): takes real Nazario/Nigerian emails and has the LLM rewrite them professionally — simulates an attacker polishing proven phishing

Bulk and spear-phishing variants are generated across 13 scenarios (account_suspension, ceo_wire_transfer, document_share, hr_benefits, linkedin_recruiter, password_reset, etc.).

**Output:** `data/processed/synthetic_phishing.csv` (460 rows)

### Step 2: Clean

Strips generation artifacts: trailing meta-commentary from the LLM ("Note: I rewrote the email to..."), unfilled template placeholders (`[Recipient]`, `[amount]`), and exact duplicates. Placeholders are filled with randomized plausible values drawn from per-category pools using deterministic per-row sampling.

```bash
python scripts/cleanup_synthetic_phishing.py \
    --input  data/processed/synthetic_phishing.csv \
    --output data/processed/synthetic_phishing_clean.csv \
    --seed 42
```

**Output:** `data/processed/synthetic_phishing_clean.csv` (457 rows after deduplication)

## Phase 3: Detection

Four detectors are evaluated on both the real test set (baseline) and the synthetic phishing set (detection gap). Each detector lives in its own subdirectory under `detection/` and writes standardized output CSVs to `results/<detector>/`.

| Detector | Type | Status |
|---|---|---|
| SpamAssassin | rule-based | done |
| Logistic Regression (TF-IDF) | classical ML | done |
| XGBoost (TF-IDF) | classical ML | done |
| DistilBERT (fine-tuned) | transformer | done |

See `detection/README.md` for shared conventions (output filenames, column layouts) that all four detectors follow.

### SpamAssassin (completed)

Runs SpamAssassin in Docker, communicates via the SPAMC protocol over TCP directly from Python (no local `spamc` binary needed). SA's default threshold of 5.0 is designed for bulk spam and significantly under-recalls on phishing, so we tune the decision threshold on the real training set (F1-optimal) to give SA a fair comparison with the ML detectors.

One-time setup:
```bash
docker run -d --name spamd -p 783:783 --restart unless-stopped instantlinux/spamassassin
```

Run evaluation:
```bash
# Score the real training set — used to tune the decision threshold (~20-30 min)
python detection/spamassassin/eval.py \
    --input  data/processed/emails_clean.csv \
    --output results/spamassassin/real_train.csv \
    --split  train

# Score the real test set
python detection/spamassassin/eval.py \
    --input  data/processed/emails_clean.csv \
    --output results/spamassassin/real_test.csv \
    --split  test

# Score the synthetic phishing set
python detection/spamassassin/eval.py \
    --input  data/processed/synthetic_phishing_clean.csv \
    --output results/spamassassin/synthetic.csv

# Tune threshold on train, re-classify, compute baseline and detection gap
python detection/spamassassin/gap.py \
    --train     results/spamassassin/real_train.csv \
    --real      results/spamassassin/real_test.csv \
    --synthetic results/spamassassin/synthetic.csv \
    --synthetic-meta data/processed/synthetic_phishing_clean.csv \
    --outdir    results/spamassassin/
```

Full setup instructions: `detection/spamassassin/setup.md`.
Run instructions: `detection/spamassassin/README.md`.

**Key result:** at the tuned threshold (1.7), SpamAssassin catches 71.2% of real phishing but only 38.7% of synthetic phishing — a 32-point detection gap. Few-shot synthetic phishing is hardest to detect (18.1% caught), followed by zero-shot (27.7%); rephrasing-based synthetic catches 59.0% because vocabulary from real source emails survives the rewrite. Spear-phishing variants evade SA most (14.0% caught). Full breakdown in `results/spamassassin/detection_gap.csv`.

### ML detectors (Logistic Regression, XGBoost, DistilBERT)

Each ML detector has its own subdirectory under `detection/` with `train.py`, `eval.py`, and a README. All three follow the same interface: `--mode baseline` trains on real phishing only, `--mode augmented` trains on real + synthetic phishing (the Phase 4 experiment).

```bash
# Logistic Regression (~30s local)
python detection/logistic_regression/train.py --mode baseline
python detection/logistic_regression/eval.py  --mode baseline

# XGBoost (~2 min local)
python detection/xgboost/train.py --mode baseline
python detection/xgboost/eval.py  --mode baseline

# DistilBERT (~15 min on Colab GPU — see detection/distilbert/README.md)
python detection/distilbert/train.py --mode baseline
python detection/distilbert/eval.py  --mode baseline
```

Each run writes four result CSVs to `results/<detector>/baseline/`. Swap `baseline` for `augmented` to produce the Phase 4 results.

**Key results on real test set (baseline):**

| Detector | Real F1 | Synth detection | Detection gap |
|---|---|---|---|
| SpamAssassin (tuned) | 0.823 | 38.7% | 32.4 pts |
| Logistic Regression | 0.988 | 88.8% | 9.2 pts |
| XGBoost | 0.987 | 91.9% | 6.0 pts |
| DistilBERT | 0.993 | 90.8% | 8.7 pts |

Classical ML detectors are much more robust to AI-generated phishing than rule-based detection, but all three still have a measurable gap the Phase 4 experiment closes.

## Phase 4: Augmentation Experiment (Mitigation)

Tests whether retraining ML detectors on real + synthetic phishing closes the detection gap.

### Step 1: Prepare Augmentation Data

```bash
python scripts/prepare_augmentation.py
```

Splits the 457 synthetic emails 50/50 (stratified by method × sophistication) into:
- 228 emails added to the training set
- 229 emails held out as the augmentation evaluation target

**Outputs:**
- `data/processed/synthetic_split.csv` — all 457 synthetic emails with a new `aug_split` column (values: `train` / `test`). Filter by `aug_split` at load time, same pattern as `emails_clean.csv`'s `split` column.
- `data/processed/emails_augmented_train.csv` — real train + synthetic train combined (10,814 rows).

### Step 2: Retrain and Evaluate (per ML detector)

For each ML detector, train both `--mode baseline` and `--mode augmented`, then evaluate both on the real test set and on the held-out synthetic test set:

```bash
python detection/logistic_regression/train.py --mode augmented
python detection/logistic_regression/eval.py  --mode augmented

python detection/xgboost/train.py --mode augmented
python detection/xgboost/eval.py  --mode augmented

python detection/distilbert/train.py --mode augmented
python detection/distilbert/eval.py  --mode augmented
```

Each `eval.py` automatically evaluates against the right synthetic set (full 457-row `synthetic_phishing_clean.csv` for baseline; 229-row held-out subset of `synthetic_split.csv` for augmented — enforced by the scripts to prevent leakage).

SpamAssassin is rule-based so it isn't retrained — the augmentation experiment applies only to the ML detectors.

### Step 3: Cross-Detector Analysis

Run the analysis notebook to produce the headline table and publication-quality figures:

```bash
jupyter lab notebooks/02_results_analysis.ipynb
# Run All
```

The notebook reads every detector's `baseline.csv` and `detection_gap.csv`, produces 5 figures (saved to `results/figures/`), and writes `results/headline_table.csv` — the cross-detector summary that goes in the writeup.

**Augmentation results:**

| Detector | Synth detection (baseline → augmented) | Real F1 (baseline → augmented) |
|---|---|---|
| Logistic Regression | 88.8% → 97.4% | 0.988 → 0.987 |
| XGBoost | 91.9% → 99.1% | 0.987 → 0.983 |
| DistilBERT | 90.8% → 100.0% | 0.993 → 0.992 |

Augmentation with ~5.6% synthetic phishing in training closes the detection gap across all three ML detectors with negligible impact on real-phishing detection.

## Project Structure

```
├── data/
│   ├── raw/                    # Raw downloaded datasets (gitignored)
│   └── processed/              # Cleaned CSVs
├── detection/                  # Classifier scripts
│   ├── README.md               # Shared conventions for all detectors
│   ├── _shared.py              # Shared data-loading / metrics / writers
│   ├── spamassassin/           # Rule-based detector
│   │   ├── README.md
│   │   ├── setup.md
│   │   ├── eval.py             # Scores emails via spamd
│   │   └── gap.py              # Tunes threshold, writes result CSVs
│   ├── logistic_regression/    # TF-IDF + LR
│   │   ├── README.md
│   │   ├── train.py
│   │   └── eval.py
│   ├── xgboost/                # TF-IDF + XGBoost
│   │   ├── README.md
│   │   ├── train.py
│   │   └── eval.py
│   └── distilbert/             # Fine-tuned DistilBERT (Colab)
│       ├── README.md
│       ├── train.py
│       └── eval.py
├── generation/                 # Phase 2 prompt templates
│   └── prompts/
│       ├── zero_shot.json
│       ├── few_shot.json
│       └── rephrasing.json
├── notebooks/
│   ├── 01_eda.ipynb            # Phase 1 exploratory analysis
│   └── 02_results_analysis.ipynb  # Phase 3/4 cross-detector analysis
├── results/
│   ├── spamassassin/           # 4 result CSVs (single mode)
│   ├── logistic_regression/
│   │   ├── baseline/           # 4 result CSVs per mode
│   │   └── augmented/
│   ├── xgboost/{baseline,augmented}/
│   ├── distilbert/{baseline,augmented}/
│   ├── figures/                # PNGs produced by the results notebook
│   └── headline_table.csv      # Cross-detector summary
├── scripts/
│   ├── email_utils.py          # Shared email parsing utilities
│   ├── parse_nazario.py
│   ├── parse_nigerian.py
│   ├── parse_enron.py
│   ├── parse_spamassassin.py
│   ├── combine_datasets.py
│   ├── preprocess.py
│   ├── split_data.py
│   ├── generate_phishing.py            # Phase 2 generation
│   ├── cleanup_synthetic_phishing.py   # Phase 2 cleanup
│   └── prepare_augmentation.py         # Phase 4 augmentation data prep
├── .gitignore
├── environment.yml             # Conda environment specification
└── README.md
```

## Dataset Summary

After running the full pipeline:

### Real dataset (Phase 1)

| Metric | Value |
|--------|-------|
| Total emails | 13,233 |
| Train set | 10,586 |
| Test set | 2,647 |
| Phishing | 4,835 (36.5%) |
| Legitimate | 8,398 (63.5%) |
| Avg word count | 238 |
| Median word count | 163 |

| Source | Label | Count |
|--------|-------|-------|
| Enron | Legitimate | 4,626 |
| SpamAssassin | Legitimate | 3,772 |
| Nazario | Phishing | 1,546 |
| Nigerian Fraud | Phishing | 3,289 |

### Synthetic dataset (Phase 2)

| Method | Count | Description |
|---|---|---|
| zero_shot | 130 | 80 bulk + 50 spear, no examples in prompt |
| few_shot | 127 | 80 bulk + 47 spear, prompt includes real phishing examples |
| rephrasing | 200 | 80 Nazario + 120 Nigerian sources, rewritten by LLM |
| **Total** | **457** | |

### Augmented dataset (Phase 4)

| Metric | Value |
|---|---|
| Synthetic train (added to real train) | 228 |
| Synthetic test (held out) | 229 |
| Total augmented training set | 10,814 rows (10,586 real + 228 synthetic) |
| Synthetic proportion of training data | 2.1% (5.6% of phishing) |