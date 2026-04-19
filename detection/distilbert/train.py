"""
detection/distilbert/train.py

Fine-tune DistilBERT for phishing classification.

Two modes:
    --mode baseline    Fine-tune on real training data only (Phase 3 baseline).
    --mode augmented   Fine-tune on emails_augmented_train.csv (Phase 4).

Saves the fine-tuned model (tokenizer + weights) to
    models/distilbert/<mode>/

Designed to run on Colab GPU. On CPU this will take many hours per mode —
avoid unless necessary.

Colab usage:
    # Mount Drive or clone the repo first, then cd to repo root.
    !pip install transformers datasets accelerate -q
    !python detection/distilbert/train.py --mode baseline
    !python detection/distilbert/train.py --mode augmented

Local usage (same):
    python detection/distilbert/train.py --mode baseline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _shared  # noqa: E402


SEED = 42
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 512  # enough for most emails without aggressive truncation
VAL_FRACTION = 0.1  # carved out of training data for early stopping


def tokenize_dataset(df: pd.DataFrame, tokenizer) -> Dataset:
    """Convert DataFrame with 'text' and 'label' columns to a tokenized HF Dataset."""
    # Labels must be int for the Trainer.
    labels = _shared.encode_labels(df["label"])
    ds = Dataset.from_dict({
        "text": df["text"].fillna("").astype(str).tolist(),
        "label": labels.tolist(),
    })

    def tok(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    ds = ds.map(tok, batched=True, remove_columns=["text"])
    ds.set_format(type="torch",
                  columns=["input_ids", "attention_mask", "label"])
    return ds


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    proba = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
    return {
        "accuracy": float((preds == labels).mean()),
        "roc_auc": float(roc_auc_score(labels, proba)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=["baseline", "augmented"])
    parser.add_argument("--output-dir", type=Path,
                        default=Path("models/distilbert"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    set_seed(SEED)
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load training data ---------------------------------------------------
    if args.mode == "baseline":
        train_all, _ = _shared.load_real()
        print(f"Baseline mode: {len(train_all)} real training rows.")
    else:
        train_all = _shared.load_augmented()
        n_synth = train_all["source"].str.startswith("synthetic:").sum()
        n_real = len(train_all) - n_synth
        print(f"Augmented mode: {len(train_all)} rows "
              f"({n_real} real + {n_synth} synthetic).")

    # Split out a validation set for early stopping. Stratified by label.
    train_df, val_df = train_test_split(
        train_all, test_size=VAL_FRACTION, random_state=SEED,
        stratify=train_all["label"],
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    print(f"Train: {len(train_df)}  Val: {len(val_df)}")

    # Tokenize -------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("Tokenizing...")
    train_ds = tokenize_dataset(train_df, tokenizer)
    val_ds = tokenize_dataset(val_df, tokenizer)

    # Model ----------------------------------------------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2,
    )

    # Trainer --------------------------------------------------------------
    model_out_dir = args.output_dir / args.mode
    model_out_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(model_out_dir / "hf_trainer_workdir"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc",
        greater_is_better=True,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        seed=SEED,
        report_to="none",  # no wandb/tensorboard
        save_total_limit=2,  # keep only best + latest to save disk
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("\nStarting fine-tuning...")
    trainer.train()
    print("✓ Training complete.")

    # Save best model + tokenizer ------------------------------------------
    trainer.save_model(str(model_out_dir))
    tokenizer.save_pretrained(str(model_out_dir))
    print(f"\n✓ Saved model to {model_out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
