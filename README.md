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

| Detector | Type | Owner | Status |
|---|---|---|---|
| Logistic Regression (TF-IDF) | classical ML | teammate | in progress |
| XGBoost (TF-IDF) | classical ML | teammate | pending |
| DistilBERT (fine-tuned) | transformer | teammate | pending |
| SpamAssassin | rule-based | Karthik | done |

See `detection/README.md` for shared conventions (output filenames, column layouts) that all four detectors follow.

### SpamAssassin (completed)

Runs SpamAssassin 4.0.1 in Docker, communicates via the SPAMC protocol over TCP directly from Python (no local `spamc` binary needed).

One-time setup:
```bash
docker run -d --name spamd -p 783:783 --restart unless-stopped instantlinux/spamassassin
```

Run evaluation:
```bash
# Score the real test set
python detection/spamassassin/eval.py \
    --input  data/processed/emails_clean.csv \
    --output results/spamassassin/real_test.csv \
    --split  test

# Score the synthetic phishing set
python detection/spamassassin/eval.py \
    --input  data/processed/synthetic_phishing_clean.csv \
    --output results/spamassassin/synthetic.csv

# Compute baseline metrics and detection gap
python detection/spamassassin/gap.py \
    --real results/spamassassin/real_test.csv \
    --synthetic results/spamassassin/synthetic.csv \
    --synthetic-meta data/processed/synthetic_phishing_clean.csv \
    --outdir results/spamassassin/
```

Full setup instructions: `detection/spamassassin/setup.md`.
Run instructions: `detection/spamassassin/README.md`.

**Key result:** SpamAssassin catches 50.7% of real phishing but only 13.6% of synthetic phishing — a 37-point detection gap. Few-shot synthetic phishing is completely invisible (0% caught). Full breakdown in `results/spamassassin/detection_gap.csv`.

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
- `data/processed/synthetic_split.csv` — source of truth, 457 rows + `aug_split` column
- `data/processed/synthetic_test.csv` — held-out synthetic phishing (229 rows)
- `data/processed/emails_augmented_train.csv` — real train + synthetic train combined (10,814 rows)

### Step 2: Retrain and Evaluate (per ML detector)

For LR, XGBoost, and DistilBERT:
1. Train a **baseline** model on `emails_clean.csv` (split='train')
2. Train an **augmented** model on `emails_augmented_train.csv` (all rows)
3. Evaluate both on `synthetic_test.csv` (primary) and `emails_clean.csv` (split='test', sanity check)

SpamAssassin is rule-based so it isn't retrained — the augmentation experiment applies only to the ML detectors.

### Step 3: Cross-Detector Analysis

(Not yet written) Aggregates all four detectors' `detection_gap.csv` files into the headline cross-detector comparison table and the before/after augmentation table.

## Project Structure

```
├── data/
│   ├── raw/                    # Raw downloaded datasets (gitignored)
│   └── processed/              # Cleaned CSVs (gitignored)
├── detection/                  # Classifier scripts + saved models
│   ├── README.md               # Shared conventions for all detectors
│   ├── spamassassin/           # Rule-based detector (done)
│   │   ├── README.md
│   │   ├── setup.md
│   │   ├── eval.py
│   │   └── gap.py
│   ├── logistic_regression/    # (teammate)
│   ├── xgboost/                # (teammate)
│   └── distilbert/             # (teammate)
├── generation/                 # Prompt templates and generation configs
│   └── prompts/
│       ├── zero_shot.json
│       ├── few_shot.json
│       └── rephrasing.json
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory data analysis
│   └── 02_Baseline_Logistic.ipynb  # LR baseline (teammate)
├── results/
│   └── spamassassin/           # Phase 3 SpamAssassin results
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