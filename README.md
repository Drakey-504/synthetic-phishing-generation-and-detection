# Can LLM-Generated Phishing Evade Today's Detectors?

Attackers can now generate polished phishing emails with a local LLM in seconds. This project measures how well four detectors (rule-based, classical ML, and transformer) hold up against LLM-generated phishing, then tests whether a small amount of synthetic training data closes the gap.

**Course:** 14-795 AI Applications in Information Security, Carnegie Mellon University, Spring 2026  
**Team:** Karthik Chakkalakal, Sathwika Manda

## Key findings

- **Rule-based filtering breaks down.** SpamAssassin (threshold tuned on real training data) catches 71.2% of real phishing but only 38.7% of LLM-generated phishing, a 32-point gap. It catches 18.1% of few-shot emails and 14.0% of spear-phishing.
- **ML detectors are more robust, but spear-phishing is their blind spot.** Logistic Regression, XGBoost, and DistilBERT reach F1 of 0.987–0.993 on real phishing and catch 89–92% of synthetic phishing overall (a 6–9 point gap). On targeted spear-phishing they catch only 49–64%. Colleague-impersonation (0–20% caught) and CEO wire-transfer (0–60%) are the hardest scenarios.
- **A little synthetic data closes the gap.** Adding 228 synthetic emails (~5.6% of phishing training examples) raises detection on a held-out set of 229 synthetic emails from 90–93% to 97–100%, and spear-phishing detection from 54–68% to 88–100%. Real-phishing F1 drops by less than 0.005.

| Detector | Real-phishing F1 | Synthetic caught (all 457) | Spear-phishing caught | Held-out 229: baseline → augmented |
|---|---|---|---|---|
| SpamAssassin (tuned) | 0.823 | 38.7% | 14.0% | n/a (rule-based) |
| Logistic Regression | 0.988 | 88.8% | 49.0% | 90.0% → 97.4% |
| XGBoost | 0.987 | 91.9% | 64.0% | 92.6% → 99.1% |
| DistilBERT | 0.993 | 90.8% | 60.0% | 91.3% → 100.0% |

All figures come from the course-era result files in `results/`. The held-out baseline column uses the baseline models' saved per-email predictions, restricted to the same 229 emails the augmented models were scored on, so the comparison is like-for-like.

## What we built

1. **Data pipeline:** parsers for four corpora (Nazario and Nigerian-fraud phishing; Enron and SpamAssassin ham), preprocessing that strips leaked `X-Spam-*` and mail-client headers, deduplicates, and filters; an EDA check for label-revealing terms; and a stratified 80/20 split (13,233 emails).
2. **Synthetic generation:** Llama 3 8B via Ollama produces 457 phishing emails across three attacker levels: zero-shot and few-shot emails cover 13 scenarios (8 bulk, 5 spear-phishing), and rephrasing rewrites 200 real phishing emails. A cleanup step then removes LLM meta-commentary and fills template placeholders.
3. **Detection:** SpamAssassin in Docker, queried over the SPAMC protocol from Python, with an F1-tuned threshold; TF-IDF Logistic Regression and XGBoost; fine-tuned DistilBERT.
4. **Augmentation experiment:** synthetic emails split 50/50 (stratified by method and sophistication), with the eval scripts enforcing that augmented models are only scored on the held-out half.

## Follow-up after the course (October 2026)

*Added after the course; not part of the submitted project.*

**What this checks.** Whether the Logistic Regression and XGBoost detectors learned to recognize phishing, or partly just learned which dataset each email came from. `scripts/check_dataset_artifacts.py` retrains both models twice, once as-is and once with dataset-identifying words removed, and compares the results.

**Why it matters.** The phishing class mixes two sources. Most of it (Nigerian fraud, about two-thirds) is from the same early-2000s era as the legitimate emails (Enron and SpamAssassin, ~2000–2002), but the Nazario phishing (about a third) is from 2020–2025 and was collected at a single inbox, `jose@monkey.org`, which about 70% of those emails name in the text. So some words separate the classes only because of where the emails came from: "monkey" appears in over 1,000 phishing emails and about 30 legitimate ones, and "2023" and "2024" appear only in Nazario phishing. A model can score well by spotting words like these without learning anything about phishing, and they would be useless on real mail.

**Result.** These words do rank among the top Logistic Regression and XGBoost features ("enron" is #3 for Logistic Regression, "monkey" #5 for XGBoost; full list in `results/dataset_artifact_check.csv`). Removing them changes real F1 by ≤ 0.005 and synthetic detection by ≤ 0.5 points, so the models are relying on real phishing signal and the results above hold. DistilBERT was not re-tested.

**Remaining limitation.** The legitimate mail is about two decades old, so real-test scores likely overstate performance on modern inboxes.

## Contributions

**Karthik Chakkalakal** ([@Drakey-504](https://github.com/Drakey-504))
- Built the data pipeline: dataset parsers, preprocessing and header-leakage removal, train/test split, and EDA
- Designed and ran the synthetic generation pipeline (prompt templates, generation, and cleanup)
- Built the SpamAssassin evaluation: Docker spamd, SPAMC client, threshold tuning, and per-method/per-sophistication gap analysis
- Designed the augmentation experiment and its leakage-safe held-out split
- Refactored the ML detectors into a shared train/eval interface (`--mode baseline|augmented`), ran the experiments, and produced the cross-detector analysis and figures

**Sathwika Manda** ([@sathwikamanda](https://github.com/sathwikamanda))
- Built and trained the initial Logistic Regression, XGBoost, and DistilBERT models in Colab (`detection/02_Baseline_Logistic.ipynb`, `detection/03_XGBoost.ipynb`, `detection/04_DistilBERT.ipynb`); the shared detector scripts reuse her text cleaning, TF-IDF, Logistic Regression, XGBoost, and DistilBERT training settings

## Stack

Python · Ollama (Llama 3 8B) · SpamAssassin (Docker) · scikit-learn · XGBoost · PyTorch / Hugging Face Transformers · pandas · Jupyter · Google Colab (T4 GPU)

---

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

Zero-shot and few-shot emails cover 13 scenarios: 8 bulk (account_suspension, password_reset, document_share, etc.) and 5 spear-phishing (ceo_wire_transfer, colleague_impersonation, hr_benefits, linkedin_recruiter, vendor_impersonation).

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

| Detector | Type |
|---|---|
| SpamAssassin | rule-based |
| Logistic Regression (TF-IDF) | classical ML |
| XGBoost (TF-IDF) | classical ML |
| DistilBERT (fine-tuned) | transformer |

See `detection/README.md` for shared conventions (output filenames, column layouts) that all four detectors follow.

### SpamAssassin

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

**Key result:** at the tuned threshold (1.7), SpamAssassin catches 71.2% of real phishing but only 38.7% of synthetic phishing — a 32-point detection gap. Few-shot synthetic phishing is hardest to detect (18.1% caught), followed by zero-shot (27.7%); rephrasing-based synthetic catches 59.0%, likely because vocabulary from the real source emails survives the rewrite. Spear-phishing variants evade SA most (14.0% caught). Full breakdown in `results/spamassassin/detection_gap.csv`.

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

The ML detectors (including DistilBERT) are much more robust to AI-generated phishing than rule-based detection, but all three still have a measurable gap the Phase 4 experiment closes.

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

The detection columns compare the baseline and augmented models on the same 229 held-out synthetic emails; real F1 is on the real test set.

| Detector | Synth detection (baseline → augmented) | Spear-phishing (baseline → augmented) | Real F1 (baseline → augmented) |
|---|---|---|---|
| Logistic Regression | 90.0% → 97.4% | 54% → 88% | 0.988 → 0.986 |
| XGBoost | 92.6% → 99.1% | 68% → 96% | 0.987 → 0.983 |
| DistilBERT | 91.3% → 100.0% | 62% → 100% | 0.993 → 0.992 |

Augmentation with ~5.6% synthetic phishing in training closes the detection gap across all three ML detectors with negligible impact on real-phishing detection.

## Project Structure

```
├── data/
│   ├── raw/                    # Raw downloaded datasets (gitignored)
│   └── processed/              # Cleaned CSVs
├── detection/                  # Classifier scripts
│   ├── README.md               # Shared conventions for all detectors
│   ├── 02_Baseline_Logistic.ipynb  # Original Colab baselines (Sathwika)
│   ├── 03_XGBoost.ipynb
│   ├── 04_DistilBERT.ipynb
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
│   ├── spamassassin/           # 5 result CSVs (incl. real_train.csv)
│   ├── logistic_regression/
│   │   ├── baseline/           # 4 result CSVs per mode
│   │   └── augmented/
│   ├── xgboost/{baseline,augmented}/
│   ├── distilbert/{baseline,augmented}/
│   ├── figures/                # PNGs produced by the results notebook
│   ├── headline_table.csv      # Cross-detector summary
│   └── dataset_artifact_check.csv # Dataset-artifact check output
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
│   ├── prepare_augmentation.py         # Phase 4 augmentation data prep
│   └── check_dataset_artifacts.py      # Dataset-artifact check
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
| few_shot | 127 | 77 bulk + 50 spear, prompt includes real phishing examples |
| rephrasing | 200 | 80 Nazario + 120 Nigerian sources, rewritten by LLM |
| **Total** | **457** | |

### Augmented dataset (Phase 4)

| Metric | Value |
|---|---|
| Synthetic train (added to real train) | 228 |
| Synthetic test (held out) | 229 |
| Total augmented training set | 10,814 rows (10,586 real + 228 synthetic) |
| Synthetic proportion of training data | 2.1% (5.6% of phishing) |