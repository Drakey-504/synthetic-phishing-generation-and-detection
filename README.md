### synthetic-phishing-generation-and-detection
Synthetic Phishing Generation and Detection Project for AI Applications in Information Security, Spring 2026

## Setup

### Prerequisites
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/download)
- [Ollama](https://ollama.com) (installed separately as a system application)

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

## Data Pipeline

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

## Project Structure

```
├── data/
│   ├── raw/                  # Raw downloaded datasets (gitignored)
│   └── processed/            # Cleaned CSVs (gitignored)
├── detection/                # Classifier training scripts and saved models
├── generation/               # Prompt templates and generation scripts
├── notebooks/
│   └── 01_eda.ipynb          # Exploratory data analysis
├── results/                  # Output metrics, confusion matrices, charts
├── scripts/
│   ├── email_utils.py        # Shared email parsing utilities
│   ├── parse_nazario.py      # Nazario corpus parser
│   ├── parse_nigerian.py     # Nigerian fraud dataset parser
│   ├── parse_enron.py        # Enron email dataset parser
│   ├── parse_spamassassin.py # SpamAssassin easy_ham parser
│   ├── combine_datasets.py   # Combines all parsed CSVs
│   ├── preprocess.py         # Cleaning and preprocessing
│   └── split_data.py         # Train/test split
├── .gitignore
├── environment.yml           # Conda environment specification
└── README.md
```

## Dataset Summary

After running the full pipeline:

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