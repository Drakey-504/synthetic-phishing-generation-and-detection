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