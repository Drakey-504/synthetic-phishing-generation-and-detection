# DistilBERT Detector

Fine-tuned `distilbert-base-uncased` for phishing classification. Transformer-based, much more compute-intensive than LR/XGBoost.

**Designed to run on Google Colab with a GPU.** CPU fine-tuning will take many hours per mode.

## Files

- `train.py` — fine-tunes DistilBERT on real or augmented training data
- `eval.py` — loads a saved model checkpoint and writes the four standard result CSVs

## Colab setup

In a Colab notebook, at the start of the session:

```python
# 1. Runtime → Change runtime type → GPU (T4 is free, enough)
# 2. Clone the repo
!git clone https://github.com/Drakey-504/synthetic-phishing-generation-and-detection.git
%cd synthetic-phishing-generation-and-detection

# 3. Install deps (conda environment.yml won't work on Colab; install the few extras needed)
!pip install -q transformers datasets accelerate joblib scikit-learn

# 4. Verify GPU
import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
```

## Run

```bash
# Baseline (Phase 3): ~10-15 minutes on T4 GPU
!python detection/distilbert/train.py --mode baseline
!python detection/distilbert/eval.py  --mode baseline

# Augmented (Phase 4): similar runtime
!python detection/distilbert/train.py --mode augmented
!python detection/distilbert/eval.py  --mode augmented
```

The trained model checkpoints are large (~250 MB each). You probably don't want to commit them to git — add `models/distilbert/` to `.gitignore`, and either download them from Colab after training or re-run on Colab when needed. Only the result CSVs need to be committed.

## Saving results back to the repo

After running in Colab, download the `results/distilbert/` folder to your local machine and push it:

```python
# In Colab, zip and download results
!zip -r distilbert_results.zip results/distilbert/
from google.colab import files
files.download('distilbert_results.zip')
```

Then unzip locally into the repo's `results/` directory and commit.

## Outputs

- `models/distilbert/{baseline,augmented}/` — fine-tuned model (config + weights + tokenizer)
- `results/distilbert/{baseline,augmented}/{real_test,synthetic,baseline,detection_gap}.csv`

## Hyperparameters

- Base model: `distilbert-base-uncased`
- 5 epochs max, batch size 16, learning rate 2e-5, warmup ratio 0.1, weight decay 0.01
- Early stopping: patience 2 epochs on validation ROC-AUC
- Max sequence length: 512 tokens (enough for most phishing emails)
- FP16 mixed precision when GPU available
- Seed 42

No text cleaning is applied before tokenization — DistilBERT's tokenizer handles URLs and email addresses natively, and stripping them would throw away signal the transformer can use. This is a deliberate difference from the LR/XGBoost pipeline.

## Eval-target difference between modes

Same as the other detectors — `baseline` against the full 457, `augmented` against the 229 held-out subset.
